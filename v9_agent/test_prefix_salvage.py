"""Tests for Contour B prefix salvage order and dual-image projection.

Locks two V9 contract points:
1. A memory-supported, surface-legal trajectory prefix executes (PARTIAL)
   before any hard REJECT, even when a later step is surface-illegal.
2. The model receives at most one image (annotated preferred) and the
   offline verifier_packet never enters the text prompt.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from v9_agent.llm import _packet_for_text_prompt, _packet_image_payloads
from v9_agent.trajectory import TrajectoryVerifier


def _snapshot():
    return SimpleNamespace(
        grid_hash="g1",
        full_grid_hex_rows=("aa", "bb"),
        available_actions=("A", "C"),
        coordinate_action_ids=(),
        coordinate_targets=(),
        objects=(),
    )


def _item(*action_ids):
    return SimpleNamespace(
        hypothesis_id="h1",
        test_plan=tuple(
            SimpleNamespace(action_id=a, coordinate_candidate_id=None, repeat=1)
            for a in action_ids
        ),
        goal_spec=None,
        expected_final_state_hash=None,
        created_state_signature="s1",
        semantic_binding=None,
    )


def _memory():
    return SimpleNamespace(action_memory_records=[], trajectory_evaluations=[], action_effects={})


def _packet(allowed=("A", "C")):
    return {
        "packet_fingerprint": "vfy_test",
        "dual_view_contract": {"planning_ids": "tracked_objects_only"},
        "state": {"grid_hash": "g1"},
        "full_grid_hex_rows": ["aa", "bb"],
        "planning_objects": [{"id": "o0"}],
        "action_probe_summaries": [
            {"action_id": "A", "grid_hash_before": "g1", "grid_hash_after": "g2"},
        ],
        "execution_constraints": {
            "allowed_action_ids": list(allowed),
            "allowed_coordinate_candidate_ids": [],
        },
    }


class PrefixSalvageTest(unittest.TestCase):
    def test_legal_prefix_survives_illegal_tail(self):
        # A simulates (g1->g2); B is not on the action surface. The A prefix
        # must execute as PARTIAL instead of rejecting the whole trajectory.
        verifier = TrajectoryVerifier()
        result = verifier.verify_hypothesis(
            _item("A", "B"), _snapshot(), _memory(), None,
            verifier_packet=_packet(allowed=("A", "C")),
        )
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.reason, "prefix_simulated_clipped_before_illegal_step")
        self.assertEqual([s.action_id for s in result.plan.steps], ["A"])
        self.assertEqual(result.details.get("unavailable_action"), "B")

    def test_illegal_first_step_rejects(self):
        verifier = TrajectoryVerifier()
        result = verifier.verify_hypothesis(
            _item("B"), _snapshot(), _memory(), None,
            verifier_packet=_packet(allowed=("A", "C")),
        )
        self.assertEqual(result.status, "REJECT")
        self.assertEqual(result.reason, "unknown_action:B")

    def test_unknown_transition_partial_unchanged(self):
        # A simulates; C is legal but has no recorded transition -> PARTIAL prefix.
        verifier = TrajectoryVerifier()
        result = verifier.verify_hypothesis(
            _item("A", "C"), _snapshot(), _memory(), None,
            verifier_packet=_packet(allowed=("A", "C")),
        )
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.reason, "prefix_simulated_until_unknown_transition")
        self.assertEqual([s.action_id for s in result.plan.steps], ["A"])
        self.assertEqual(result.details.get("remaining_action_ids"), ["C"])

    def test_full_simulation_accept_unchanged(self):
        verifier = TrajectoryVerifier()
        result = verifier.verify_hypothesis(
            _item("A"), _snapshot(), _memory(), None,
            verifier_packet=_packet(allowed=("A", "C")),
        )
        self.assertEqual(result.status, "ACCEPT")
        self.assertEqual(result.reason, "simulated_from_memory")


class DualImageProjectionTest(unittest.TestCase):
    def _packet_with_images(self):
        return {
            "dual_view": {"qwen_visual": "annotated"},
            "verifier_packet": {"full_grid_hex_rows": ["aa"], "secret": "offline"},
            "current_frame_png": {
                "data_base64": "UkFX",  # RAW
                "available_to_configured_backend": True,
            },
            "annotated_frame_png": {
                "data_base64": "QU5OT1RBVEVE",  # ANNOTATED
                "available_to_configured_backend": True,
            },
        }

    def test_raw_and_annotated_pair_attached_in_order(self):
        payloads = _packet_image_payloads(self._packet_with_images())
        self.assertEqual(payloads, ["UkFX", "QU5OT1RBVEVE"])

    def test_raw_fallback_when_annotated_missing(self):
        packet = self._packet_with_images()
        packet["annotated_frame_png"] = {"data_base64": "", "available_to_configured_backend": True}
        payloads = _packet_image_payloads(packet)
        self.assertEqual(payloads, ["UkFX"])

    def test_verifier_packet_never_in_text_prompt(self):
        projected = _packet_for_text_prompt(self._packet_with_images())
        self.assertNotIn("verifier_packet", projected)
        dumped = json.dumps(projected)
        self.assertNotIn("data_base64", dumped)
        self.assertNotIn('"secret"', dumped)
        self.assertNotIn("full_grid_hex_rows", dumped)
        self.assertEqual(
            projected["current_frame_png"]["attachment_status"],
            "ATTACHED_IMAGE_1:current_frame_png",
        )
        self.assertEqual(
            projected["annotated_frame_png"]["attachment_status"],
            "ATTACHED_IMAGE_2:annotated_frame_png",
        )
        self.assertEqual(
            projected["dual_view"]["verifier_packet_status"],
            "offline_only_not_in_model_prompt",
        )


if __name__ == "__main__":
    unittest.main()
