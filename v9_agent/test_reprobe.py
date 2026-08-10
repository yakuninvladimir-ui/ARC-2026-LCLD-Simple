"""Reprobe policy tests (owner directive 2026-08-09).

Rule: an action whose probe revealed NO visible effect on a finished level
("no_effect" / "not_moved" / "unchanged") must be scheduled for reprobe on the
next level — effects may be conditional on level state (ar25: click/cycle
selection only becomes meaningful once selectable pieces exist). Actions with
a revealed mechanic (any visible effect) stay researched for the whole game.
Newly available actions are always researched before primary planning.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from v9_agent.action_effects import merge_effect_record
from v9_agent.memory import GameMemory


def _snapshot(level_index: int, actions=("ACTION5",)):
    return SimpleNamespace(
        available_actions=list(actions),
        coordinate_action_ids=[],
        undo_action_ids=[],
        level_index=level_index,
    )


def _record(action_id: str, outcome: str, level_index: int):
    return merge_effect_record(
        None,
        action_id=action_id,
        target_signature=None,
        outcome=outcome,
        level_index=level_index,
        step_index=1,
    )


class ReprobePolicyTest(unittest.TestCase):
    def test_no_effect_expires_at_level_boundary(self):
        memory = GameMemory()
        memory.reset_game("g")
        memory.action_effects[("ACTION5", None)] = _record("ACTION5", "no_effect", 0)
        # Same level: already probed here, counts as researched.
        status = memory.action_research_status(_snapshot(0))
        self.assertIn("ACTION5", status["researched_action_ids"])
        self.assertEqual(status["missing_action_ids"], [])
        # Next level: unrevealed effect knowledge expires, reprobe required.
        status = memory.action_research_status(_snapshot(1))
        self.assertIn("ACTION5", status["missing_action_ids"])

    def test_no_effect_reprobe_only_once_per_level(self):
        memory = GameMemory()
        memory.reset_game("g")
        memory.action_effects[("ACTION5", None)] = _record("ACTION5", "no_effect", 0)
        self.assertIn("ACTION5", memory.action_research_status(_snapshot(1))["missing_action_ids"])
        # Reprobe on level 1 also finds nothing: researched for the rest of level 1.
        memory.action_effects[("ACTION5", None)] = _record("ACTION5", "no_effect", 1)
        status = memory.action_research_status(_snapshot(1))
        self.assertIn("ACTION5", status["researched_action_ids"])
        # ...but level 2 asks again.
        self.assertIn("ACTION5", memory.action_research_status(_snapshot(2))["missing_action_ids"])

    def test_revealing_effect_persists_across_levels(self):
        memory = GameMemory()
        memory.reset_game("g")
        for outcome in ("effect", "negative_effect", "no_progress"):
            memory.action_effects[("ACTION1", None)] = _record("ACTION1", outcome, 0)
            status = memory.action_research_status(_snapshot(3, actions=("ACTION1",)))
            self.assertIn("ACTION1", status["researched_action_ids"], outcome)
            self.assertEqual(status["missing_action_ids"], [], outcome)

    def test_mixed_records_any_revealing_wins(self):
        memory = GameMemory()
        memory.reset_game("g")
        memory.action_effects[("ACTION5", None)] = _record("ACTION5", "no_effect", 0)
        memory.action_effects[("ACTION5", "sig_b")] = _record("ACTION5", "effect", 0)
        status = memory.action_research_status(_snapshot(1))
        self.assertIn("ACTION5", status["researched_action_ids"])

    def test_new_action_is_always_missing(self):
        memory = GameMemory()
        memory.reset_game("g")
        memory.action_effects[("ACTION1", None)] = _record("ACTION1", "effect", 0)
        status = memory.action_research_status(_snapshot(1, actions=("ACTION1", "ACTION6")))
        self.assertIn("ACTION6", status["missing_action_ids"])
        self.assertIn("ACTION1", status["researched_action_ids"])


if __name__ == "__main__":
    unittest.main()
