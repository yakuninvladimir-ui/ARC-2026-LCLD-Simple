import unittest
from types import SimpleNamespace

from v9_agent.trajectory import HexStateComparator, TrajectoryPlanner, TrajectoryStep, TrajectoryPlan, TrajectoryVerifier
from v9_agent.types import HypothesisItem, Progress, Relevance, TriTruth, Validity, TestStep


class TrajectoryPlannerTest(unittest.TestCase):
    def test_planner_finds_alternative_path_in_memory_graph(self):
        start = SimpleNamespace(
            snapshot_id="s1",
            grid_hash="g1",
            full_grid_hex_rows=("aa", "bb"),
            available_actions=("A", "B"),
            coordinate_action_ids=(),
            coordinate_targets=(),
            objects=(),
        )
        memory = SimpleNamespace(
            action_memory_records=[
                {"grid_hash_before": "g1", "grid_hash_after": "g2", "action_id": "A", "coordinate_candidate_id": None},
                {"grid_hash_before": "g2", "grid_hash_after": "g3", "action_id": "B", "coordinate_candidate_id": None},
            ]
        )
        planner = TrajectoryPlanner()
        plan = planner.plan({"expected_final_state_hash": "g3"}, start, memory, SimpleNamespace(max_qwen_trajectory_steps=10))
        self.assertIsNotNone(plan)
        self.assertEqual([step.action_id for step in plan.steps], ["A", "B"])
        self.assertEqual(plan.expected_final_state_hash, "g3")
 
    def test_planner_uses_reverse_distance_heuristic(self):
        planner = TrajectoryPlanner()
        memory = SimpleNamespace(
            action_memory_records=[
                {"grid_hash_before": "g1", "grid_hash_after": "g2", "action_id": "A", "coordinate_candidate_id": None},
                {"grid_hash_before": "g2", "grid_hash_after": "g4", "action_id": "B", "coordinate_candidate_id": None},
                {"grid_hash_before": "g1", "grid_hash_after": "g3", "action_id": "C", "coordinate_candidate_id": None},
                {"grid_hash_before": "g3", "grid_hash_after": "g4", "action_id": "D", "coordinate_candidate_id": None},
            ]
        )
        transitions, _ = planner._build_transition_indexes(memory)
        reverse = planner._build_reverse_transition_indexes(transitions)
        distances = planner._compute_reverse_distances(["g4"], reverse, max_steps=10)
        self.assertEqual(distances.get("g4"), 0)
        self.assertEqual(distances.get("g2"), 1)
        self.assertEqual(distances.get("g3"), 1)
        self.assertEqual(distances.get("g1"), 2)
 
    def test_planner_returns_zero_step_plan_when_start_already_matches_goal(self):
        start = SimpleNamespace(
            snapshot_id="s1",
            grid_hash="g1",
            full_grid_hex_rows=("aa", "bb"),
            available_actions=("A",),
            coordinate_action_ids=(),
            coordinate_targets=(),
            objects=(),
        )
        memory = SimpleNamespace(action_memory_records=[], action_effects={})
        planner = TrajectoryPlanner()
        plan = planner.plan({"expected_final_state_hash": "g1"}, start, memory, SimpleNamespace(max_qwen_trajectory_steps=10))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps, [])
        self.assertEqual(plan.total_steps, 0)
        self.assertEqual(plan.expected_final_state_hash, "g1")

    def test_apply_action_uses_no_effect_memory_when_no_transition_exists(self):
        planner = TrajectoryPlanner()
        memory = SimpleNamespace(
            action_memory_records=[],
            action_effects={("A", None): SimpleNamespace(outcome="no_effect")},
        )
        next_states = planner._apply_action("g1", "A", None, {}, memory)
        self.assertEqual(next_states, ["g1"])


class TrajectoryVerifierTest(unittest.TestCase):
    def test_verifier_accepts_direct_simulation(self):
        snapshot = SimpleNamespace(
            grid_hash="g1",
            full_grid_hex_rows=("aa", "bb"),
            available_actions=("A",),
            objects=(),
        )
        item = HypothesisItem(
            hypothesis_id="h1",
            source="test",
            claim="claim",
            truth=TriTruth.UNKNOWN,
            relevance=Relevance.UNDECIDED,
            validity=Validity.UNCHECKED,
            progress=Progress.UNKNOWN,
            test_plan=(TestStep(kind="step", action_id="A"),),
            cursor=0,
            priority=0.0,
            confidence=0.0,
            expiry_step=None,
            evidence_refs=(),
            suppression_signature="sig",
            created_state_signature="s1",
            proposal_batch_id="",
            semantic_objective=None,
            semantic_binding=None,
            expected_final_state_hash=None,
            goal_spec=None,
        )
        memory = SimpleNamespace(action_memory_records=[{"grid_hash_before": "g1", "grid_hash_after": "g2", "action_id": "A", "coordinate_candidate_id": None}], trajectory_evaluations=[])
        verifier = TrajectoryVerifier()
        result = verifier.verify_hypothesis(item, snapshot, memory, None)
        self.assertEqual(result.status, "ACCEPT")
        self.assertEqual(result.plan.expected_final_state_hash, "g2")

    def test_verifier_returns_corrected_when_alternative_path_exists(self):
        snapshot = SimpleNamespace(
            grid_hash="g1",
            full_grid_hex_rows=("aa", "bb"),
            available_actions=("A", "B"),
            coordinate_action_ids=(),
            coordinate_targets=(),
            objects=(),
        )
        item = HypothesisItem(
            hypothesis_id="h2",
            source="test",
            claim="claim",
            truth=TriTruth.UNKNOWN,
            relevance=Relevance.UNDECIDED,
            validity=Validity.UNCHECKED,
            progress=Progress.UNKNOWN,
            test_plan=(TestStep(kind="step", action_id="C"),),
            cursor=0,
            priority=0.0,
            confidence=0.0,
            expiry_step=None,
            evidence_refs=(),
            suppression_signature="sig",
            created_state_signature="s1",
            proposal_batch_id="",
            semantic_objective=None,
            semantic_binding=None,
            expected_final_state_hash=None,
            goal_spec={"expected_final_state_hash": "g3"},
        )
        memory = SimpleNamespace(
            action_memory_records=[
                {"grid_hash_before": "g1", "grid_hash_after": "g2", "action_id": "A", "coordinate_candidate_id": None},
                {"grid_hash_before": "g2", "grid_hash_after": "g3", "action_id": "B", "coordinate_candidate_id": None},
            ],
            trajectory_evaluations=[],
        )
        verifier = TrajectoryVerifier()
        result = verifier.verify_hypothesis(item, snapshot, memory, None)
        self.assertEqual(result.status, "CORRECTED")
        self.assertIsNotNone(result.plan)
        self.assertEqual([step.action_id for step in result.plan.steps], ["A", "B"])

    def test_trajectory_step_conversion_to_test_step(self):
        trajectory_step = TrajectoryStep(action_id="A", coordinate_candidate_id="c1")
        test_step = trajectory_step.to_test_step()
        self.assertEqual(test_step.action_id, "A")
        self.assertEqual(test_step.coordinate_candidate_id, "c1")
        self.assertEqual(test_step.kind, "corrected_trajectory_step")


class HexStateComparatorTest(unittest.TestCase):
    def test_compare_snapshots_reports_motion_and_color_changes(self):
        expected_obj = SimpleNamespace(
            object_id="o1",
            centroid_rc=(1.0, 1.0),
            bbox_rc=(0, 0, 2, 2),
            color_histogram={1: 3},
            local_mask_hex_rows=("111", "111"),
            colors=(1,),
        )
        observed_obj = SimpleNamespace(
            object_id="o1",
            centroid_rc=(2.0, 3.0),
            bbox_rc=(1, 2, 3, 4),
            color_histogram={2: 3},
            local_mask_hex_rows=("222", "222"),
            colors=(2,),
        )
        expected = SimpleNamespace(objects=(expected_obj,), full_grid_hex_rows=("aa", "bb"), width=2, height=2)
        observed = SimpleNamespace(objects=(observed_obj,), full_grid_hex_rows=("aa", "bc"), width=2, height=2)
        comparison = HexStateComparator.compare_snapshots(expected, observed)
        self.assertIn("o1", comparison["object_deltas"])
        self.assertEqual(comparison["object_deltas"]["o1"]["dx"], 2.0)
        self.assertEqual(comparison["object_deltas"]["o1"]["dy"], 1.0)
        self.assertEqual(comparison["object_deltas"]["o1"]["color_from"], 1)
        self.assertEqual(comparison["object_deltas"]["o1"]["color_to"], 2)
        self.assertEqual(comparison["hex_delta"]["changed_cell_count"], 1)


if __name__ == "__main__":
    unittest.main()
