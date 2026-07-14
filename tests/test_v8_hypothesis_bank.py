from types import SimpleNamespace

from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config
from v8_agent.game_adapter import GameAdapter
from v8_agent.hypothesis_bank import HypothesisBank
import v8_agent.hypothesis_bank as hypothesis_bank
from v8_agent.memory import GameMemory
from v8_agent.types import Progress, QwenRole, Relevance, TriTruth, Validity, VerificationContractKind


def _snap():
    obs={"grid":[[0,0,0],[0,1,0],[0,0,0]], "metadata":{"game_id":"g","available_actions":["ACTION1","ACTION6"],"frame_index":0}}
    return ARGALiteBuilder().build(GameAdapter().to_world_state(obs), GameMemory(), V8Config())


def test_rejects_invented_ids_and_raw_coordinates():
    snap = _snap()
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.PRIMARY, {"schema_version":"v8.2.semantic_output", "hypotheses":[{"claim":"bad", "target_object_ids":["obj_missing"], "target_relation_ids":[], "test_plan":[{"action_id":"ACTION1"}]}]}, snap, V8Config())
    assert bank.invalid_rejections
    assert not bank.has_valid_action_candidates()
    bank.add_qwen_output(QwenRole.COORDINATE, {"schema_version":"v8.2.coordinate_output", "coordinate_hypotheses":[{"coordinate_action_id":"ACTION6", "candidate_target_ids":[snap.coordinate_targets[0].candidate_id], "x":1, "y":1}]}, snap, V8Config())
    assert any(r["reason"] == "raw_coordinates_rejected" for r in bank.invalid_rejections)


def test_bank_allows_execution_without_qwen_while_candidate_remains():
    snap = _snap()
    cid = snap.coordinate_targets[0].candidate_id
    bank = HypothesisBank()
    out = {"schema_version":"v8.2.coordinate_output", "coordinate_hypotheses":[{"claim":"try target", "coordinate_action_id":"ACTION6", "candidate_target_ids":[cid], "expected_effect":"change", "probe_priority":1.0, "confidence":0.5}]}
    bank.add_qwen_output(QwenRole.COORDINATE, out, snap, V8Config())
    assert bank.has_valid_action_candidates()
    action = bank.next_candidate_action(snap, "coordinate")
    assert action is not None
    assert action.action_id == "ACTION6"
    assert action.coordinate_candidate_id == cid


def test_v84_semantic_plan_accepts_valid_action_and_rejects_invented_action():
    snap = _snap()
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.4.semantic_plan",
        "decision": "PLAN",
        "goal": {"operation": "select", "object_ids": [], "relation_ids": [], "description": "change target"},
        "steps": [{"action_id": "ACTION1"}],
        "completion_criterion": "target changed",
        "confidence": 0.7,
    }, snap, V8Config())
    assert bank.has_executable_candidate(snap)

    bad = HypothesisBank()
    bad.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.4.semantic_plan",
        "decision": "PLAN",
        "goal": {"operation": "select", "object_ids": [], "relation_ids": [], "description": "bad action"},
        "steps": [{"action_id": "ACTION99"}],
        "completion_criterion": "bad",
        "confidence": 0.1,
    }, snap, V8Config())
    assert any(item["reason"] == "unknown_action_id" for item in bad.invalid_rejections)


def test_v84_abstain_is_accepted_and_raw_coordinates_rejected():
    snap = _snap()
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.PRIMARY, {"schema_version": "v8.4.semantic_plan", "decision": "ABSTAIN", "abstain_reason": "insufficient mechanics"}, snap, V8Config())
    assert not bank.has_valid_action_candidates()
    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.4.semantic_plan",
        "decision": "PLAN",
        "x": 1,
        "goal": {"operation": "select", "object_ids": [], "relation_ids": [], "description": "raw"},
        "steps": [],
        "completion_criterion": "raw",
        "confidence": 0.1,
    }, snap, V8Config())
    assert any(item["reason"] == "raw_coordinates_rejected" for item in bank.invalid_rejections)


def test_v85_semantic_hypotheses_accept_multiple_candidates_and_reject_bad_ids():
    snap = _snap()
    obj_id = snap.objects[0].object_id
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.5.semantic_trajectory_hypotheses",
        "decision": "PROPOSE",
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "rank": 1,
                "family": "object_selection",
                "goal": {"operation": "select", "object_ids": [obj_id], "relation_ids": [], "description": "change the visible object"},
                "evidence": {"supporting_fact_ids": [obj_id], "supporting_action_ids": ["ACTION1"], "summary": "object exists and action is available"},
                "trajectory": [{"action_id": "ACTION1"}],
                "trajectory_status": "candidate_prefix_until_reobservation",
                "reobserve_after_step": 1,
                "expected_progress": {"description": "tracked object should move or change", "tracked_object_ids": [obj_id], "tracked_relation_ids": []},
                "completion_criterion": "object changes or level advances",
                "main_uncertainty": "level goal unknown",
                "confidence": 0.6,
            },
            {
                "hypothesis_id": "h2",
                "rank": 2,
                "family": "other",
                "goal": {"operation": "other", "object_ids": [], "relation_ids": [], "description": "bad action candidate"},
                "evidence": {"supporting_fact_ids": [], "supporting_action_ids": ["ACTION99"], "summary": "bad action"},
                "trajectory": [{"action_id": "ACTION99"}],
                "trajectory_status": "candidate_prefix_until_reobservation",
                "reobserve_after_step": 1,
                "expected_progress": {"description": "bad", "tracked_object_ids": [], "tracked_relation_ids": []},
                "completion_criterion": "bad",
                "main_uncertainty": "bad",
                "confidence": 0.1,
            },
        ],
    }, snap, V8Config())

    assert bank.has_executable_candidate(snap)
    assert any(item["reason"] == "invented_evidence_action_id" for item in bank.invalid_rejections)

    abstain = HypothesisBank()
    abstain.add_qwen_output(QwenRole.PRIMARY, {"schema_version": "v8.5.semantic_trajectory_hypotheses", "decision": "ABSTAIN", "abstain_reason": "no grounded hypothesis"}, snap, V8Config())
    assert not abstain.has_valid_action_candidates()


def test_v86_semantic_hypotheses_accept_alias_ids_and_compact_actions():
    snap = _snap()
    obj_id = snap.objects[0].object_id
    bank = HypothesisBank()

    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.6.semantic_trajectory_hypotheses",
        "decision": "PROPOSE",
        "hypotheses": [{
            "id": "h1",
            "family": "object_selection",
            "objects": ["o0"],
            "relations": [],
            "basis": "object exists and action is available",
            "actions": ["ACTION1"],
            "status": "prefix_until_reobservation",
            "uncertainty": "goal unknown",
            "confidence": 0.5,
        }],
    }, snap, V8Config())

    assert bank.semantic_test_queue
    step = bank.semantic_test_queue[0].test_plan[0]
    assert step.action_id == "ACTION1"
    assert step.target_object_ids == (obj_id,)


def test_v86_trajectory_progress_rejects_net_zero_motion_sequence():
    packet = {
        "control_model": {
            "actions": {
                "ACTION1": {"effects": [{"object_id": "o0", "delta_xy": [0, -3], "direction": "up"}]},
                "ACTION2": {"effects": [{"object_id": "o0", "delta_xy": [0, 3], "direction": "down"}]},
            }
        },
        "neutral_relations": [],
    }

    assert not hypothesis_bank._v86_trajectory_has_progress({"objects": ["o0"], "relations": [], "actions": ["ACTION1", "ACTION2"]}, packet)
    assert hypothesis_bank._v86_trajectory_has_progress({"objects": ["o0"], "relations": [], "actions": ["ACTION1", "ACTION1"]}, packet)


def test_v87_full_trajectory_stays_active_and_uses_effect_contracts():
    obs = {
        "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "metadata": {"game_id": "g", "available_actions": ["ACTION1", "ACTION2"], "frame_index": 0},
    }
    snap = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), GameMemory(), V8Config())
    packet = {
        "scene": {"priority_facts_not_goals": []},
        "action_model": {
            "actions": {
                "ACTION1": {"observed": True, "effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [1, 0]}]},
                "ACTION2": {"observed": True, "effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [0, 1]}]},
            }
        },
        "execution_constraints": {"allowed_action_ids": ["ACTION1", "ACTION2"]},
    }
    output = {
        "schema_version": "v8.7.semantic_trajectories",
        "decision": "PROPOSE",
        "hypotheses": [
            {
                "id": "h1",
                "family": "spatial_configuration",
                "objective": {"kind": "relative_arrangement", "source_objects": ["o0"], "reference_objects": [], "description": "move the source twice"},
                "relations": [],
                "basis": "The observed action translates the selected source.",
                "actions": ["ACTION1", "ACTION1"],
                "status": "complete_candidate",
                "uncertainty": "boundary unknown",
                "confidence": 0.9,
            },
            {
                "id": "h2",
                "family": "spatial_configuration",
                "objective": {"kind": "relative_arrangement", "source_objects": ["o0"], "reference_objects": [], "description": "move the source down"},
                "relations": [],
                "basis": "The other observed action translates the source.",
                "actions": ["ACTION2"],
                "status": "complete_candidate",
                "uncertainty": "goal unknown",
                "confidence": 0.1,
            },
        ],
    }
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.PRIMARY, output, snap, V8Config(), packet=packet)

    first = bank.next_candidate_action(snap, "semantic")
    assert first is not None and first.action_id == "ACTION1"
    assert first.verification_contract.kind is VerificationContractKind.OBJECT_DISPLACEMENT
    bank.update(SimpleNamespace(
        hypothesis_id=first.hypothesis_id,
        truth=TriTruth.TRUE,
        relevance=Relevance.RELEVANT,
        validity=Validity.VALID,
        progress=Progress.POSITIVE,
        reason_code="target_object_displaced",
        action=first,
    ))

    # The active full trajectory remains first, while lower-ranked alternatives
    # stay queued for later execution if the active hypothesis is exhausted.
    assert len(bank.semantic_test_queue) == 2
    second = bank.next_candidate_action(snap, "semantic")
    assert second is not None and second.action_id == "ACTION1"
    assert second.hypothesis_id == first.hypothesis_id


def test_v87_spatial_trajectory_rejects_unexplained_net_zero_cycle():
    packet = {
        "action_model": {"actions": {
            "ACTION1": {"effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [0, -1]}]},
            "ACTION2": {"effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [0, 1]}]},
        }}
    }
    raw = {
        "family": "spatial_configuration",
        "objective": {"kind": "match_or_overlap", "source_objects": ["o0"], "reference_objects": [], "description": "spatial match"},
        "actions": ["ACTION1", "ACTION2"],
    }

    assert not hypothesis_bank._v87_trajectory_has_progress(raw, packet)
    assert hypothesis_bank._v87_has_unjustified_immediate_inverse_pair(raw, packet)
    raw["actions"] = ["ACTION1", "ACTION1"]
    assert not hypothesis_bank._v87_has_unjustified_immediate_inverse_pair(raw, packet)
    raw["actions"] = ["ACTION1", "ACTION2"]
    raw["family"] = "interaction_sequence"
    raw["objective"]["kind"] = "select_or_activate"
    assert hypothesis_bank._v87_trajectory_has_progress(raw, packet)
    assert not hypothesis_bank._v87_has_unjustified_immediate_inverse_pair(raw, packet)


def test_v87_surface_objective_requires_an_observed_surface_effect():
    packet = {"action_model": {"actions": {
        "ACTION2": {
            "observed": True,
            "effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [0, 3]}],
            "surface_added": [],
            "surface_removed": [],
        },
        "ACTION5": {
            "observed": True,
            "effects": [{"kind": "local_visual_change", "object_id": "o0"}],
            "surface_added": ["ACTION1"],
            "surface_removed": ["ACTION5"],
        },
    }}}
    raw = {
        "family": "action_surface_change",
        "objective": {"kind": "surface_change", "source_objects": ["o0"], "reference_objects": []},
        "actions": ["ACTION2"] * 4,
    }

    assert not hypothesis_bank._v87_trajectory_uses_observed_effect(raw, packet)
    assert not hypothesis_bank._v87_trajectory_has_progress(raw, packet)
    raw["actions"] = ["ACTION5"]
    assert hypothesis_bank._v87_trajectory_uses_observed_effect(raw, packet)
    assert hypothesis_bank._v87_trajectory_has_progress(raw, packet)


def test_v87_trajectory_must_switch_before_using_other_control_group_vectors():
    packet = {
        "scene": {"control_state_transition_candidates": [{
            "type": "SUPPORTED_CONTROL_GROUP_SWITCH",
            "trigger_action_id": "ACTION5",
            "before_control_group_id": "cg0",
            "after_control_group_id": "cg1",
            "current_inferred_control_group_id": "cg1",
        }]},
        "action_model": {"actions": {
            "ACTION1": {"observed_control_group_ids": ["cg1"]},
            "ACTION3": {"observed_control_group_ids": ["cg0"]},
            "ACTION5": {"observed_control_group_ids": []},
        }},
    }
    raw = {"actions": ["ACTION1", "ACTION3"]}
    assert not hypothesis_bank._v87_trajectory_respects_control_context(raw, packet)
    raw["actions"] = ["ACTION1", "ACTION5", "ACTION3"]
    assert hypothesis_bank._v87_trajectory_respects_control_context(raw, packet)
    raw["actions"] = ["ACTION5", "ACTION1"]
    assert not hypothesis_bank._v87_trajectory_respects_control_context(raw, packet)
    raw["actions"] = ["ACTION5", "ACTION3", "ACTION5", "ACTION1"]
    assert hypothesis_bank._v87_trajectory_respects_control_context(raw, packet)

    raw["status"] = "complete_candidate"
    raw["actions"] = ["ACTION1", "ACTION5", "ACTION3"]
    assert hypothesis_bank._v87_trajectory_respects_control_context(raw, packet)


def test_v87_exact_correspondence_requires_the_complete_qwen_route():
    packet = {
        "scene": {"priority_facts_not_goals": [{
            "type": "MOVABLE_REFERENCE_EXACT_GEOMETRY_CORRESPONDENCE",
            "movable_object_id": "o0",
            "reference_object_id": "o1",
            "source_to_reference_delta_xy": [-12.0, 24.0],
        }]},
        "action_model": {"actions": {
            "ACTION2": {"effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [0.0, 3.0]}]},
            "ACTION3": {"effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [-6.0, 0.0]}]},
        }},
    }
    raw = {
        "status": "complete_candidate",
        "objective": {
            "kind": "match_or_overlap",
            "source_objects": ["o0"],
            "reference_objects": ["o1"],
        },
        "actions": ["ACTION3", "ACTION3", *(["ACTION2"] * 9)],
    }

    mismatch = hypothesis_bank._v87_incomplete_exact_correspondence(raw, packet)
    assert mismatch == {
        "required_delta_xy": [-12.0, 24.0],
        "planned_delta_xy": [-12.0, 27.0],
        "residual_delta_xy": [0.0, -3.0],
    }

    complete = dict(raw)
    complete["actions"] = ["ACTION3", "ACTION3", *(["ACTION2"] * 8)]
    assert hypothesis_bank._v87_incomplete_exact_correspondence(complete, packet) is None

    wrong_direction_packet = {
        "scene": {"priority_facts_not_goals": [{
            "type": "MOVABLE_REFERENCE_EXACT_GEOMETRY_CORRESPONDENCE",
            "movable_object_id": "o0",
            "reference_object_id": "o1",
            "source_to_reference_delta_xy": [-18.0, 0.0],
        }]},
        "action_model": {"actions": {
            "ACTION3": {"effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [3.0, 0.0]}]},
        }},
    }
    wrong_direction = {
        "status": "complete_candidate",
        "objective": {
            "kind": "match_or_overlap",
            "source_objects": ["o0"],
            "reference_objects": ["o1"],
        },
        "actions": ["ACTION3"] * 6,
    }
    assert hypothesis_bank._v87_incomplete_exact_correspondence(wrong_direction, wrong_direction_packet) == {
        "required_delta_xy": [-18.0, 0.0],
        "planned_delta_xy": [18.0, 0.0],
        "residual_delta_xy": [-36.0, 0.0],
    }


def test_v87_bank_rejects_trajectory_that_only_increases_exact_match_error():
    obs = {
        "grid": [[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]],
        "metadata": {"game_id": "g", "available_actions": ["ACTION3"], "frame_index": 0},
    }
    snap = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), GameMemory(), V8Config())
    assert len(snap.objects) >= 2
    packet = {
        "execution_constraints": {"allowed_action_ids": ["ACTION3"]},
        "scene": {"priority_facts_not_goals": [{
            "type": "MOVABLE_REFERENCE_EXACT_GEOMETRY_CORRESPONDENCE",
            "movable_object_id": "o0",
            "reference_object_id": "o1",
            "source_to_reference_delta_xy": [-18.0, 0.0],
        }]},
        "action_model": {"actions": {
            "ACTION3": {
                "observed": True,
                "effects": [{"kind": "translation", "object_id": "o0", "delta_xy": [3.0, 0.0]}],
            },
        }},
    }
    output = {
        "schema_version": "v8.7.semantic_trajectories",
        "decision": "PROPOSE",
        "hypotheses": [{
            "id": "h_wrong_sign",
            "family": "object_correspondence",
            "objective": {
                "kind": "match_or_overlap",
                "source_objects": ["o0"],
                "reference_objects": ["o1"],
                "description": "Move the source to the reference.",
            },
            "relations": [],
            "basis": "Required -18, but selected action contributes +3.",
            "actions": ["ACTION3"] * 6,
            "status": "complete_candidate",
            "uncertainty": "none",
            "confidence": 0.9,
        }],
    }
    bank = HypothesisBank()

    bank.add_qwen_output(QwenRole.RESERVE, output, snap, V8Config(), packet=packet)

    assert any(item["reason"] == "trajectory_does_not_complete_claimed_correspondence" for item in bank.invalid_rejections)
    assert not bank.semantic_test_queue


def test_v85_same_shape_alignment_tracks_exact_shape_relation():
    snapshot = SimpleNamespace(
        objects=(
            SimpleNamespace(object_id="pattern", shape_signature="shape_right"),
            SimpleNamespace(object_id="yellow", shape_signature="shape_left"),
            SimpleNamespace(object_id="red", shape_signature="shape_left"),
        ),
        relations=(
            SimpleNamespace(relation_id="rel_shape", relation_type="same_shape", a="yellow", b="red"),
            SimpleNamespace(relation_id="rel_row", relation_type="aligned_row", a="pattern", b="yellow"),
        ),
        available_actions=("ACTION4",),
        undo_action_ids=(),
        coordinate_targets=(),
        level_index=0,
        step_index=0,
        semantic_state_signature="sem_test",
    )
    bank = HypothesisBank()

    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.5.semantic_trajectory_hypotheses",
        "decision": "PROPOSE",
        "hypotheses": [{
            "hypothesis_id": "h1",
            "rank": 1,
            "family": "same_shape_alignment",
            "goal": {"operation": "align", "object_ids": ["pattern", "yellow", "red"], "relation_ids": ["rel_shape", "rel_row"], "description": "align shapes"},
            "evidence": {"supporting_fact_ids": ["pattern", "yellow", "red", "rel_shape", "rel_row"], "supporting_action_ids": ["ACTION4"], "summary": "badly tracks row"},
            "trajectory": [{"action_id": "ACTION4"}],
            "trajectory_status": "candidate_prefix_until_reobservation",
            "reobserve_after_step": 1,
            "expected_progress": {"description": "row gap decreases", "tracked_object_ids": ["pattern", "yellow"], "tracked_relation_ids": ["rel_row"]},
            "completion_criterion": "same shapes align",
            "main_uncertainty": "unknown",
            "confidence": 0.6,
        }],
    }, snapshot, V8Config())

    assert bank.semantic_test_queue
    step = bank.semantic_test_queue[0].test_plan[0]
    assert step.target_relation_ids == ("rel_shape",)
    assert step.target_object_ids == ("yellow", "red")


def test_v85_same_shape_alignment_rejects_without_exact_shape_relation():
    snapshot = SimpleNamespace(
        objects=(
            SimpleNamespace(object_id="pattern", shape_signature="shape_right"),
            SimpleNamespace(object_id="yellow_fragment", shape_signature="shape_rect"),
        ),
        relations=(
            SimpleNamespace(relation_id="rel_near", relation_type="near", a="pattern", b="yellow_fragment"),
        ),
        available_actions=("ACTION4",),
        undo_action_ids=(),
        coordinate_targets=(),
        level_index=0,
        step_index=0,
        semantic_state_signature="sem_test",
    )
    bank = HypothesisBank()

    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.5.semantic_trajectory_hypotheses",
        "decision": "PROPOSE",
        "hypotheses": [{
            "hypothesis_id": "h1",
            "rank": 1,
            "family": "same_shape_alignment",
            "goal": {"operation": "align", "object_ids": ["pattern", "yellow_fragment"], "relation_ids": ["rel_near"], "description": "align blocks"},
            "evidence": {"supporting_fact_ids": ["pattern", "yellow_fragment", "rel_near"], "supporting_action_ids": ["ACTION4"], "summary": "nearby"},
            "trajectory": [{"action_id": "ACTION4"}],
            "trajectory_status": "candidate_prefix_until_reobservation",
            "reobserve_after_step": 1,
            "expected_progress": {"description": "near gap decreases", "tracked_object_ids": ["pattern", "yellow_fragment"], "tracked_relation_ids": ["rel_near"]},
            "completion_criterion": "connect",
            "main_uncertainty": "unknown",
            "confidence": 0.6,
        }],
    }, snapshot, V8Config())

    assert not bank.semantic_test_queue
    assert any(item["reason"] == "same_shape_alignment_without_exact_shape_relation" for item in bank.invalid_rejections)
