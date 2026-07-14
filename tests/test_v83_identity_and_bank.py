from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config
from v8_agent.game_adapter import GameAdapter
from v8_agent.hypothesis_bank import HypothesisBank
from v8_agent.memory import GameMemory
from v8_agent.types import Attribution, CoordinateEffectRecord, Progress, QwenRole, Relevance, TriTruth


def _build(builder, memory, grid, frame):
    obs = {"grid": grid, "metadata": {"game_id": "g", "available_actions": ["ACTION1", "ACTION6"], "frame_index": frame}}
    if memory._current_game_id is None:
        memory.reset_game("g")
    return builder.build(GameAdapter().to_world_state(obs), memory, V8Config())


def test_object_tracks_and_relation_ids_survive_motion():
    builder, memory = ARGALiteBuilder(), GameMemory()
    a = [[0] * 7 for _ in range(7)]
    a[1][1] = 2
    a[5][5] = 2
    b = [[0] * 7 for _ in range(7)]
    b[1][1] = 2
    b[4][4] = 2
    before = _build(builder, memory, a, 0)
    after = _build(builder, memory, b, 1)
    assert {o.object_id for o in before.objects} == {o.object_id for o in after.objects}
    rb = next(r for r in before.relations if r.relation_type == "same_shape")
    ra = next(r for r in after.relations if r.relation_type == "same_shape")
    assert rb.relation_id == ra.relation_id
    assert ra.metric_value < rb.metric_value


def test_stale_coordinate_candidate_does_not_block_replanning():
    builder, memory = ARGALiteBuilder(), GameMemory()
    g1 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    g2 = [[0, 0, 0], [0, 0, 0], [0, 0, 2]]
    first = _build(builder, memory, g1, 0)
    second = _build(builder, memory, g2, 1)
    stale_id = first.coordinate_targets[0].candidate_id
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.COORDINATE, {
        "schema_version": "v8.3.coordinate_output",
        "coordinate_hypotheses": [{
            "claim": "probe stale target",
            "coordinate_action_id": "ACTION6",
            "candidate_target_ids": [stale_id],
            "contract_kind": "LOCAL_TARGET_CHANGE",
            "expected_effect": "target changes",
            "probe_priority": 1,
            "confidence": 1,
        }],
    }, first, V8Config())
    assert bank.has_executable_candidate(first)
    # The target may disappear or rebind. In either case the answer must reflect
    # executability in the current snapshot rather than queue non-emptiness.
    executable = bank.has_executable_candidate(second)
    candidate = bank.next_candidate_action(second, "coordinate", dry_run=True)
    assert executable is (candidate is not None)


def test_qwen_output_rejects_unknown_contract_kind():
    builder, memory = ARGALiteBuilder(), GameMemory()
    snap = _build(builder, memory, [[0, 0], [0, 1]], 0)
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.3.semantic_output",
        "hypotheses": [{
            "claim": "bad contract",
            "test_plan": [{
                "kind": "simple_action_probe",
                "action_id": "ACTION1",
                "contract_kind": "INVENTED_CONTRACT",
            }],
        }],
    }, snap, V8Config())
    assert not bank.has_executable_candidate(snap)
    assert any(item["reason"] == "unknown_contract_kind" for item in bank.invalid_rejections)


def test_coordinate_qwen_output_rejects_unknown_contract_kind():
    builder, memory = ARGALiteBuilder(), GameMemory()
    snap = _build(builder, memory, [[0, 0], [0, 1]], 0)
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.COORDINATE, {
        "schema_version": "v8.3.coordinate_output",
        "coordinate_hypotheses": [{
            "claim": "bad coordinate contract",
            "coordinate_action_id": "ACTION6",
            "candidate_target_ids": [snap.coordinate_targets[0].candidate_id],
            "contract_kind": "CLICK_MAGIC",
        }],
    }, snap, V8Config())
    assert not bank.has_executable_candidate(snap)
    assert any(item["reason"] == "unknown_contract_kind" for item in bank.invalid_rejections)


def test_semantic_multi_object_goal_index_binds_selected_pair():
    builder, memory = ARGALiteBuilder(), GameMemory()
    snap = _build(builder, memory, [[1, 0, 0], [0, 0, 0], [0, 0, 2]], 0)
    object_ids = [obj.object_id for obj in snap.objects]
    assert len(object_ids) >= 2
    bank = HypothesisBank()
    bank.add_qwen_output(QwenRole.PRIMARY, {
        "schema_version": "v8.3.semantic_output",
        "hypotheses": [{
            "level_transition_goal": "apply the same confirmed operation to multiple marked objects",
            "object_goals": [
                {"stage": "target", "operation": "align_with", "source_object_id": object_ids[0]},
                {"stage": "target", "operation": "align_with", "source_object_id": object_ids[1]},
            ],
            "trajectory": [{
                "step_index": 0,
                "object_goal_index": 1,
                "kind": "rule_application",
                "action_id": "ACTION1",
                "contract_kind": "LOCAL_TARGET_CHANGE",
                "expected_observation": "second object changes",
            }],
            "confidence": 0.5,
        }],
    }, snap, V8Config())
    action = bank.next_candidate_action(snap, "semantic")
    assert action is not None
    assert action.verification_contract is not None
    assert action.verification_contract.target_object_ids == (object_ids[1],)


def test_coordinate_research_memory_is_game_scoped():
    memory = GameMemory()
    memory.coordinate_effects.append(CoordinateEffectRecord(
        coordinate_action_id="ACTION6",
        candidate_target_id="ct_old",
        x=1,
        y=1,
        level_index=0,
        step_index=3,
        object_id=None,
        region_signature="old",
        observed_effect="target_local_change_observed",
        truth=TriTruth.TRUE,
        relevance=Relevance.RELEVANT,
        progress=Progress.POSITIVE,
        attribution=Attribution.ACTION_LINKED,
        state_signature="g_old",
        repeat_suppression_signature="sig_old",
    ))
    assert memory.coordinate_research_needed(0) is False
    assert memory.coordinate_research_needed(1) is False
