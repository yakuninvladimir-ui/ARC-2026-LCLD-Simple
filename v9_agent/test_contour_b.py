"""Unit-level checks for trajectory contour B helpers (no package install required)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ContourBSourceContractTest(unittest.TestCase):
    def test_trajectory_exposes_partial_hybrid_and_rebind(self):
        text = (ROOT / "trajectory.py").read_text(encoding="utf-8")
        for needle in (
            "def resolve_goal_hash",
            "def rebind_plan_to_test_steps",
            'status="PARTIAL"',
            "hybrid_empiric_execution",
            "prefix_simulated_until_unknown_transition",
            "_objective_is_grounded",
            "_first_action_legal",
        ):
            self.assertIn(needle, text)

    def test_session_applies_partial_and_rebind(self):
        text = (ROOT / "session.py").read_text(encoding="utf-8")
        self.assertIn("rebind_plan_to_test_steps", text)
        self.assertIn('"PARTIAL"', text)
        self.assertIn("semantic_binding=getattr(hyp, \"semantic_binding\", None)", text)

    def test_modules_parse(self):
        for name in ("trajectory.py", "session.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            ast.parse(source)


if __name__ == "__main__":
    unittest.main()
