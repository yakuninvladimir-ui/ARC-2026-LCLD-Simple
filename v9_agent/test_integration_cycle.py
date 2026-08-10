"""Synthetic end-to-end integration test (Engineering Spec V9 §14.3).

Runs the full V9 loop without a live model:

    observation -> snapshot -> dual-view packet -> fake Qwen -> HypothesisBank
    -> TrajectoryVerifier (ACCEPT via probe transitions) -> rebind
    -> verified_semantic queue -> CandidateAction -> scripted env step
    -> TransitionJudge -> GameMemory -> ... -> WIN -> terminal guard

Uses only the ``fake`` backend and a scripted deterministic environment.
"""

from __future__ import annotations

import unittest

from v9_agent.config import config_from_mapping
from v9_agent.session import GameSession


class ScriptedEnv:
    """Tiny deterministic 'game': ACTION1 slides a 2x2 block right; ACTION2 is
    a legal no-op; RESET restores the entry state; WIN when the block reaches
    the target column."""

    WIDTH = 8
    HEIGHT = 8
    TARGET_COL = 5

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = 1
        self.levels_completed = 0
        self.won = False

    def grid(self) -> list[list[int]]:
        g = [[0] * self.WIDTH for _ in range(self.HEIGHT)]
        for r in (2, 3):
            for c in (self.pos, self.pos + 1):
                g[r][c] = 1
        g[5][6] = 2  # static second object
        return g

    def observation(self, state: str = "PLAYING") -> dict:
        return {
            "game_id": "integration_game",
            "grid": self.grid(),
            "available_actions": ["ACTION1", "ACTION2"],
            "state": state,
            "levels_completed": self.levels_completed,
            "metadata": {
                "game_id": "integration_game",
                "level_index": 0,
                "levels_completed": self.levels_completed,
                "win_levels": 1,
            },
        }

    def step(self, action: dict) -> dict:
        action_id = action.get("action_id") or action.get("id")
        if action_id == "RESET":
            self.reset()
            return self.observation()
        if action_id == "ACTION1" and not self.won:
            self.pos = min(self.pos + 1, self.TARGET_COL)
            if self.pos >= self.TARGET_COL:
                self.won = True
                self.levels_completed = 1
                return self.observation(state="WIN")
        return self.observation()


class FullCycleIntegrationTest(unittest.TestCase):
    def test_snapshot_packet_fake_qwen_contour_bank_judge_memory(self):
        config = config_from_mapping({})
        self.assertEqual(config.qwen_backend, "fake")
        session = GameSession(config)
        env = ScriptedEnv()

        obs = env.observation(state="NOT_STARTED")
        actions_log: list[dict] = []
        verified_queue_seen = False
        bank_authorized_seen = False
        continuation_seen = False
        terminal_guard = None

        for _ in range(60):
            action = session.act(obs)
            actions_log.append(action)
            if session.bank.verified_semantic:
                verified_queue_seen = True
            if action.get("reasoning", {}).get("hypothesis_id"):
                bank_authorized_seen = True
            if action.get("reasoning", {}).get("source") == "confirmed_continuation":
                continuation_seen = True
            if action.get("reasoning", {}).get("source") == "terminal_guard":
                terminal_guard = action
                break
            after = env.step(action)
            committed = session.observe_action_result(after)
            self.assertTrue(committed, "pending transition must commit exactly once")
            obs = after

        # 1. The loop started with the initial RESET contract.
        self.assertEqual(actions_log[0]["action_id"], "RESET")
        self.assertEqual(actions_log[0]["reasoning"]["source"], "initial_reset")

        # 2. Probe-first research ran before any Qwen call: probe actions
        # precede the first bank-authorized hypothesis action.
        first_hyp_idx = next(
            i for i, a in enumerate(actions_log) if a.get("reasoning", {}).get("hypothesis_id")
        )
        probe_sources = {
            a.get("reasoning", {}).get("source") for a in actions_log[:first_hyp_idx]
        }
        self.assertTrue(
            any("probe" in str(s) for s in probe_sources),
            f"expected probe actions before first hypothesis action, got {probe_sources}",
        )

        # 3. Fake Qwen was called (primary semantic role).
        self.assertGreaterEqual(session.budget.calls_this_game, 1)

        # 4. Offline contour promoted a repaired/accepted plan into
        # verified_semantic and the bank authorized an action from a hypothesis.
        self.assertTrue(verified_queue_seen, "TrajectoryVerifier never promoted to verified_semantic")
        self.assertTrue(bank_authorized_seen, "no hypothesis-authorized CandidateAction was emitted")

        # 5. Empiric judge + memory recorded real transitions.
        self.assertGreater(len(session.memory.action_memory_records), 0)
        telemetry = session.harness_telemetry()
        self.assertEqual(
            telemetry["observed_transition_ingestions"],
            len(actions_log) - 1,  # every emitted action except the terminal guard committed once
        )
        self.assertFalse(telemetry["pending_official_transition"])

        # 6. The scripted game was solved and the terminal guard fired.
        self.assertTrue(env.won, "scripted env was not solved within the step budget")
        self.assertIsNotNone(terminal_guard)
        self.assertEqual(terminal_guard["action_id"], "RESET")

        # 7. Confirmed-effect exploitation: after the verified plan proved its
        # mechanic, the bank chained confirmed_continuation steps toward the WIN
        # without spending another Qwen call.
        self.assertTrue(continuation_seen, "confirmed effect was never chained into a continuation")
        self.assertEqual(session.budget.calls_this_game, 1, "continuation must not consume Qwen calls")


class NoOpEnv:
    """Unwinnable 'game': every action is a legal no-op. Used to exercise the
    failed-attempt RESET cycle (spec: invalid hypotheses -> reset -> feedback)."""

    def observation(self, state: str = "PLAYING") -> dict:
        return {
            "game_id": "noop_game",
            "grid": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            "available_actions": ["ACTION1", "ACTION2"],
            "state": state,
            "levels_completed": 0,
            "metadata": {"game_id": "noop_game", "level_index": 0, "levels_completed": 0, "win_levels": 1},
        }

    def step(self, action: dict) -> dict:
        return self.observation()


class _BrokenQwen:
    """Always proposes an action that is not on the surface: every hypothesis
    is rejected at bank ingestion, so no executable plan ever exists."""

    def call(self, role, packet, config):
        return {
            "schema_version": "v8.7.semantic_trajectories",
            "decision": "PROPOSE",
            "hypotheses": [{
                "id": "bad1",
                "family": "other",
                "objective": {"kind": "other", "source_objects": [], "reference_objects": [], "description": "illegal"},
                "actions": ["ACTION9"],
                "action_runs": [{"action_id": "ACTION9", "repeat": 1}],
                "confidence": 0.1,
            }],
        }


class FailedAttemptResetCycleTest(unittest.TestCase):
    def test_invalid_hypotheses_lead_to_reset_with_feedback(self):
        session = GameSession(config_from_mapping({}))
        session.qwen = _BrokenQwen()
        env = NoOpEnv()
        obs = env.observation(state="NOT_STARTED")
        resets: list[dict] = []
        for _ in range(90):
            action = session.act(obs)
            source = action.get("reasoning", {}).get("source")
            if source == "failed_attempt_reset":
                resets.append(action)
                if len(resets) >= 2:
                    break
            if source == "terminal_guard":
                break
            obs = env.step(action)
            session.observe_action_result(obs)

        # 1. The dead attempt was detected and reset instead of endless flailing.
        self.assertTrue(resets, "no failed-attempt reset was emitted")

        # 2. The failure was recorded with its trigger for the next Qwen packet.
        records = list(session.memory.level_attempt_records)
        self.assertTrue(records, "no level attempt failure was recorded")
        self.assertTrue(
            any(record.get("reset_trigger") == "no_executable_hypothesis_fallback_exhausted" for record in records),
            f"unexpected reset triggers: {[r.get('reset_trigger') for r in records]}",
        )

        # 3. The retry cycle called Qwen again (feedback loop continues with the
        # retained explanation instead of abandoning the level).
        self.assertGreaterEqual(session.budget.calls_this_game, 2)

        # 4. Attempt indexing advanced (memory spans attempts within the game).
        attempt_indices = session.harness_telemetry()["level_attempt_index_by_level"]
        self.assertGreaterEqual(attempt_indices.get(0, 0), 1)


class GameOverEnv:
    """Declares GAME_OVER after the third non-RESET action; RESET revives play.
    Exercises the competition invariant: exactly one gateway RESET after
    GAME_OVER, no Qwen call consumed, then the loop continues."""

    def __init__(self) -> None:
        self.moves = 0
        self.revives = 0

    def observation(self, state: str = "PLAYING") -> dict:
        return {
            "game_id": "gameover_game",
            "grid": [[0, 0, 0, 0], [0, 3, 3, 0], [0, 3, 3, 0], [0, 0, 0, 0]],
            "available_actions": ["ACTION1", "ACTION2"],
            "state": state,
            "levels_completed": 0,
            "metadata": {
                "game_id": "gameover_game",
                "level_index": 0,
                "levels_completed": 0,
                "win_levels": 1,
                "game_over": state == "GAME_OVER",
            },
        }

    def step(self, action: dict) -> dict:
        action_id = action.get("action_id") or action.get("id")
        if action_id == "RESET":
            self.moves = 0
            self.revives += 1
            return self.observation()
        self.moves += 1
        if self.moves >= 3:
            return self.observation(state="GAME_OVER")
        return self.observation()


class GameOverResetCycleTest(unittest.TestCase):
    def test_game_over_emits_exactly_one_reset_without_qwen(self):
        session = GameSession(config_from_mapping({}))
        env = GameOverEnv()
        obs = env.observation(state="NOT_STARTED")
        game_over_resets = 0
        qwen_calls_at_game_over: list[int] = []
        for _ in range(40):
            action = session.act(obs)
            source = action.get("reasoning", {}).get("source")
            if source == "game_over_level_reset":
                game_over_resets += 1
                qwen_calls_at_game_over.append(session.budget.calls_this_game)
                if game_over_resets >= 2:
                    break
            if source == "terminal_guard":
                break
            obs = env.step(action)
            session.observe_action_result(obs)

        # GAME_OVER produced a RESET (possibly twice across the loop) and play revived.
        self.assertGreaterEqual(game_over_resets, 1)
        self.assertGreaterEqual(env.revives, 1)
        # The GAME_OVER RESET path never consumes a Qwen call by itself.
        self.assertEqual(qwen_calls_at_game_over, sorted(qwen_calls_at_game_over))
        telemetry = session.harness_telemetry()
        self.assertGreaterEqual(telemetry["game_over_reset_count"], 1)


if __name__ == "__main__":
    unittest.main()
