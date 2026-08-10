"""Smoke tests for dual-view annotated PNG and verifier_packet."""

from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace

from v9_agent.frame_media import annotated_frame_png, current_frame_png
from v9_agent.trajectory import TrajectoryVerifier
from v9_agent.types import HypothesisItem, Progress, Relevance, TriTruth, Validity, TestStep


class DualViewFrameMediaTest(unittest.TestCase):
    def test_annotated_png_differs_from_raw_and_lists_labels(self):
        rows = ("00011", "00011", "22200")
        raw = current_frame_png(rows, cell_scale=4)
        annotated = annotated_frame_png(
            rows,
            [
                {"label": "obj0", "bbox_xyxy": [3, 0, 4, 1]},
                {"label": "obj1", "bbox_xyxy": [0, 2, 2, 2]},
            ],
            cell_scale=4,
        )
        self.assertEqual(raw["attachment_id"], "current_frame_png")
        self.assertEqual(annotated["attachment_id"], "annotated_frame_png")
        self.assertNotEqual(raw["sha256"], annotated["sha256"])
        self.assertEqual(len(annotated["annotations"]), 2)
        self.assertEqual(annotated["annotations"][0]["label"], "obj0")
        # PNG magic
        self.assertTrue(base64.b64decode(annotated["data_base64"]).startswith(b"\x89PNG"))


class VerifierPacketTrajectoryTest(unittest.TestCase):
    def test_verifier_uses_packet_constraints_and_probe_summaries(self):
        snapshot = SimpleNamespace(
            grid_hash="g1",
            full_grid_hex_rows=("aa", "bb"),
            available_actions=("A", "B"),
            objects=(),
        )
        item = SimpleNamespace(
            hypothesis_id="h1",
            test_plan=(SimpleNamespace(action_id="A", coordinate_candidate_id=None, repeat=1),),
            goal_spec=None,
            expected_final_state_hash=None,
            created_state_signature="s1",
        )
        memory = SimpleNamespace(action_memory_records=[], trajectory_evaluations=[])
        verifier_packet = {
            "packet_fingerprint": "vfy_test",
            "dual_view_contract": {"planning_ids": "tracked_objects_only"},
            "state": {"grid_hash": "g1"},
            "full_grid_hex_rows": ["aa", "bb"],
            "planning_objects": [{"id": "obj0"}],
            "action_probe_summaries": [
                {
                    "action_id": "A",
                    "grid_hash_before": "g1",
                    "grid_hash_after": "g2",
                }
            ],
            "execution_constraints": {
                "allowed_action_ids": ["A", "B"],
                "allowed_coordinate_candidate_ids": [],
            },
        }
        verifier = TrajectoryVerifier()
        result = verifier.verify_hypothesis(item, snapshot, memory, None, verifier_packet=verifier_packet)
        self.assertEqual(result.status, "ACCEPT")
        self.assertEqual(result.details.get("verifier_packet_fingerprint"), "vfy_test")
        self.assertEqual(result.details.get("planning_object_count"), 1)


if __name__ == "__main__":
    unittest.main()
