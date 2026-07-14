from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config
from v8_agent.deliberation import choose_qwen_role
from v8_agent.game_adapter import GameAdapter
from v8_agent.hypothesis_bank import HypothesisBank
from v8_agent.memory import GameMemory
from v8_agent.types import ActionEffectRecord, QwenBudgetState, QwenRole


def _state_snap(actions=("ACTION1","ACTION6"), step=0):
    obs={"grid":[[0,0],[0,1]], "metadata":{"game_id":"g","available_actions":list(actions),"frame_index":step}}
    state=GameAdapter().to_world_state(obs)
    snap=ARGALiteBuilder().build(state, GameMemory(), V8Config(min_steps_between_qwen_calls=0))
    return state, snap


def _record_effect(memory, action_id, *, level=0, step=0):
    memory.action_effects[(action_id, None)] = ActionEffectRecord(
        action_id, None, "effect", 1, 0.57, level, step
    )


def test_globally_possible_coordinate_action_does_not_override_last_frame_availability():
    obs = {
        "grid": [[0, 0], [0, 1]],
        "metadata": {
            "game_id": "g",
            "available_actions": ["ACTION1"],
            "possible_actions": ["ACTION1", "ACTION6"],
            "frame_index": 0,
        },
    }
    state = GameAdapter().to_world_state(obs)
    memory = GameMemory()
    snapshot = ARGALiteBuilder().build(state, memory, V8Config(min_steps_between_qwen_calls=0))
    _record_effect(memory, "ACTION1")

    role = choose_qwen_role(
        state,
        snapshot,
        memory,
        HypothesisBank(),
        QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
    )

    assert snapshot.coordinate_action_ids == ()
    assert role is QwenRole.PRIMARY


def test_new_level_defers_primary_until_simple_action_memory_exists():
    state, snap = _state_snap()
    memory = GameMemory()
    role = choose_qwen_role(state, snap, memory, HypothesisBank(), QwenBudgetState(), V8Config(min_steps_between_qwen_calls=0), is_new_level=True)
    assert role is None
    _record_effect(memory, "ACTION1")
    role = choose_qwen_role(state, snap, memory, HypothesisBank(), QwenBudgetState(), V8Config(min_steps_between_qwen_calls=0), is_new_level=True)
    assert role is QwenRole.COORDINATE


def test_primary_follows_observed_coordinate_research_effect():
    state, snap = _state_snap(step=5)
    memory = GameMemory()
    _record_effect(memory, "ACTION1")
    _record_effect(memory, "ACTION6")
    budget = QwenBudgetState(calls_this_game=1, coordinate_calls_by_level={0:1}, total_calls_by_level={0:1}, last_qwen_step=0)
    role = choose_qwen_role(state, snap, memory, HypothesisBank(), budget, V8Config(min_steps_between_qwen_calls=0), is_new_level=False)
    assert role is QwenRole.PRIMARY


def test_coordinate_only_surface_calls_coordinate_before_primary():
    state, snap = _state_snap(actions=("ACTION6",), step=0)
    role = choose_qwen_role(
        state,
        snap,
        GameMemory(),
        HypothesisBank(),
        QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
        is_new_level=True,
    )

    assert role is QwenRole.COORDINATE


def test_empty_bank_does_not_request_a_second_semantic_call():
    state, snap = _state_snap(actions=("ACTION1",), step=2)
    memory = GameMemory()
    _record_effect(memory, "ACTION1")
    budget = QwenBudgetState(calls_this_game=1, primary_calls_by_level={0:1}, total_calls_by_level={0:1}, last_qwen_step=1)
    role = choose_qwen_role(state, snap, memory, HypothesisBank(), budget, V8Config(min_steps_between_qwen_calls=3), is_new_level=False)
    assert role is None


def test_primary_waits_for_every_action_in_the_current_frame_surface():
    state, snap = _state_snap(actions=("ACTION1", "ACTION2", "ACTION6", "ACTION7"))
    memory = GameMemory()
    _record_effect(memory, "ACTION1")

    role = choose_qwen_role(
        state, snap, memory, HypothesisBank(), QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
    )
    assert role is None
    assert memory.action_research_status(snap)["missing_action_ids"] == ["ACTION2", "ACTION6"]

    _record_effect(memory, "ACTION2")
    role = choose_qwen_role(
        state, snap, memory, HypothesisBank(), QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
    )
    assert role is QwenRole.COORDINATE

    _record_effect(memory, "ACTION6")
    role = choose_qwen_role(
        state, snap, memory, HypothesisBank(), QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
    )
    assert role is QwenRole.PRIMARY
    status = memory.action_research_status(snap)
    assert status["researched_action_ids"] == list(snap.available_actions)
    assert status["intrinsically_known_undo_action_ids"] == ["ACTION7"]


def test_newly_available_action_reopens_research_before_primary():
    state1, snap1 = _state_snap(actions=("ACTION1",), step=1)
    memory = GameMemory()
    _record_effect(memory, "ACTION1", step=1)
    assert choose_qwen_role(
        state1, snap1, memory, HypothesisBank(), QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
    ) is QwenRole.PRIMARY

    state2, snap2 = _state_snap(actions=("ACTION1", "ACTION5"), step=2)
    assert choose_qwen_role(
        state2, snap2, memory, HypothesisBank(), QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
    ) is None
    assert memory.unprobed_action_effect_ids(snap2, V8Config()) == ["ACTION5"]

    _record_effect(memory, "ACTION5", step=2)
    assert choose_qwen_role(
        state2, snap2, memory, HypothesisBank(), QwenBudgetState(),
        V8Config(min_steps_between_qwen_calls=0),
    ) is QwenRole.PRIMARY
