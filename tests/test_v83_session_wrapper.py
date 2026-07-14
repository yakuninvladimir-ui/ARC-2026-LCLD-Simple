from dataclasses import asdict

import pytest

from v8_agent.config import V8Config
from v8_agent.session import GameSession, LevelRunLimitReached


def _obs(frame, *, state="PLAYING", grid=None, actions=None, level=0, completed=0):
    return {
        "grid": grid or [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "metadata": {
            "game_id": "g",
            "level_index": level,
            "frame_index": frame,
            "available_actions": list(actions or ["ACTION1", "ACTION6"]),
            "state": state,
            "levels_completed": completed,
        },
    }


def test_explicit_transition_commit_updates_telemetry_and_rejects_duplicate():
    session = GameSession(V8Config(qwen_backend="disabled"))
    action = session.act(_obs(0, actions=["ACTION1"]))
    assert action["action_id"] == "ACTION1"
    assert session.harness_telemetry()["pending_official_transition"] is True
    assert session.observe_action_result(_obs(1, actions=["ACTION1"])) is True
    telemetry = session.harness_telemetry()
    assert telemetry["pending_official_transition"] is False
    assert telemetry["observed_transition_ingestions"] == 1
    assert len(session.memory.events) == 1
    assert session.observe_action_result(_obs(1, actions=["ACTION1"])) is False
    assert session.harness_telemetry()["observed_transition_duplicate_skips"] == 1


def test_runtime_timeout_update_reaches_active_session_config():
    session = GameSession(V8Config(qwen_timeout_seconds=120))
    session.update_runtime_config({"qwen_timeout_seconds": 17})
    assert session.config.qwen_timeout_seconds == 17


def test_session_synthesizes_monotonic_step_when_wrapper_frame_index_is_stale():
    session = GameSession(V8Config(qwen_backend="disabled"))
    session.act(_obs(0, actions=["ACTION1"], grid=[[0, 0], [0, 1]]))
    assert session.observe_action_result(_obs(0, actions=["ACTION1"], grid=[[0, 0], [1, 1]])) is True

    assert session.memory.events[-1].step_index == 1
    assert session._latest_snapshot.step_index == 1


def test_game_over_is_replayable_reset_not_terminal():
    session = GameSession(V8Config(qwen_backend="disabled", reset_on_game_over=True))
    action = session.act(_obs(0, state="GAME_OVER", actions=[]))
    assert action["action_id"] == "RESET"
    assert session.harness_telemetry()["pending_official_transition"] is True
    assert session.harness_telemetry()["game_over_reset_count"] == 1
    # Official reset produces the replayed level. The reset transition must be committed.
    assert session.observe_action_result(_obs(1, state="PLAYING", actions=["ACTION1"])) is True
    assert session.harness_telemetry()["pending_official_transition"] is False
    # A second GAME_OVER remains replayable; no reset-count terminal condition exists.
    action2 = session.act(_obs(2, state="GAME_OVER", actions=[]))
    assert action2["action_id"] == "RESET"
    assert session.harness_telemetry()["game_over_reset_count"] == 2


def test_fourth_failed_attempt_is_terminal_without_a_fourth_retry_reset():
    session = GameSession(V8Config(
        qwen_backend="disabled",
        reset_on_game_over=True,
        max_level_attempts=4,
    ))

    for attempt_index in range(3):
        action = session.act(_obs(attempt_index * 2, state="GAME_OVER", actions=[]))
        assert action["action_id"] == "RESET"
        session.observe_action_result(_obs(attempt_index * 2 + 1, actions=["ACTION1"]))

    with pytest.raises(LevelRunLimitReached) as raised:
        session.act(_obs(7, state="GAME_OVER", actions=[]))

    assert raised.value.reason_code == "level_attempt_limit_reached"
    telemetry = session.harness_telemetry()
    assert telemetry["game_over_reset_count"] == 3
    assert telemetry["level_attempt_index_by_level"] == {0: 3}
    assert len(telemetry["level_attempt_records"]) == 4
    assert telemetry["terminal_level_limit"]["reason_code"] == "level_attempt_limit_reached"


def test_level_action_limit_stops_before_emitting_action_over_budget():
    session = GameSession(V8Config(
        qwen_backend="disabled",
        max_actions_per_level=2,
    ))

    first = session.act(_obs(0, actions=["ACTION1", "ACTION2"]))
    assert first["action_id"] == "ACTION1"
    session.observe_action_result(_obs(1, grid=[[0, 0], [1, 1]], actions=["ACTION1", "ACTION2"]))
    second = session.act(_obs(2, grid=[[0, 0], [1, 1]], actions=["ACTION1", "ACTION2"]))
    assert second["action_id"] == "ACTION2"
    session.observe_action_result(_obs(3, grid=[[0, 1], [1, 1]], actions=["ACTION1", "ACTION2"]))

    with pytest.raises(LevelRunLimitReached) as raised:
        session.act(_obs(4, grid=[[0, 1], [1, 1]], actions=["ACTION1", "ACTION2"]))

    assert raised.value.reason_code == "level_action_limit_reached"
    assert session.harness_telemetry()["action_count_by_level"] == {0: 2}


def test_action_attempt_suppression_is_state_scoped():
    session = GameSession(V8Config(qwen_backend="disabled"))
    session.act(_obs(0, actions=["ACTION1"]))
    pending = session.pending_action
    assert pending is not None
    sig = pending.action.suppression_signature
    state_sig = pending.before_snapshot.semantic_state_signature
    assert session.memory.action_attempt_count(sig, state_sig) == 1
    assert session.memory.action_attempt_count(sig, "different_state") == 0


def test_epistemic_exhaustion_stops_without_liveness_revisit():
    session = GameSession(V8Config(qwen_backend="disabled", max_same_state_action_repeats=1))
    actions = []
    for frame in range(2):
        action = session.act(_obs(frame, actions=["ACTION1", "ACTION2"]))
        actions.append((action["action_id"], action["reasoning"]["source"]))
        session.observe_action_result(_obs(frame + 1, actions=["ACTION1", "ACTION2"]))
    assert actions == [("ACTION1", "initial_action_probe"), ("ACTION2", "initial_action_probe")]
    assert session.harness_telemetry()["exhaustion_revisit_count"] == 0
    with pytest.raises(RuntimeError, match="deterministic fallback is disabled"):
        session.act(_obs(3, actions=["ACTION1", "ACTION2"]))


def test_action7_is_not_used_as_probe():
    session = GameSession(V8Config(qwen_backend="disabled"))
    action = session.act(_obs(0, actions=["ACTION1", "ACTION7"]))
    assert action["action_id"] == "ACTION1"
    assert action["reasoning"]["source"] == "initial_action_probe"
    session.observe_action_result(_obs(1, actions=["ACTION1", "ACTION7"]))
    with pytest.raises(RuntimeError, match="deterministic fallback is disabled"):
        session.act(_obs(2, actions=["ACTION1", "ACTION7"]))
