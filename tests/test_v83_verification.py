from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config
from v8_agent.game_adapter import GameAdapter
from v8_agent.judge import TransitionJudge
from v8_agent.memory import GameMemory
from v8_agent.types import (
    CandidateAction,
    PendingAction,
    Progress,
    Relevance,
    TestStep as VerificationTestStep,
    TriTruth,
    VerificationContractKind,
)
from v8_agent.verification import VerificationBinder


def _snapshot(builder, memory, grid, frame, actions=("ACTION1", "ACTION6"), **metadata):
    payload = {
        "grid": grid,
        "metadata": {
            "game_id": "g",
            "level_index": metadata.pop("level_index", 0),
            "frame_index": frame,
            "available_actions": list(actions),
            **metadata,
        },
    }
    state = GameAdapter().to_world_state(payload)
    if memory._current_game_id is None:
        memory.reset_game("g")
    return builder.build(state, memory, V8Config())


def test_generic_visible_change_is_not_semantic_confirmation():
    builder, memory = ARGALiteBuilder(), GameMemory()
    before = _snapshot(builder, memory, [[0, 0], [0, 1]], 0, actions=("ACTION1",))
    after = _snapshot(builder, memory, [[2, 2], [0, 1]], 1, actions=("ACTION1",))
    action = CandidateAction("ACTION1")
    result = TransitionJudge().evaluate(before, action, after, PendingAction(before, action, None, ""), memory, V8Config())
    assert result.truth is TriTruth.UNKNOWN
    assert result.progress is Progress.UNKNOWN
    assert result.reason_code == "unbound_visible_change_not_semantic_proof"


def test_relation_error_decrease_confirms_only_bound_relation():
    builder, memory = ARGALiteBuilder(), GameMemory()
    grid_before = [[0] * 7 for _ in range(7)]
    grid_before[1][1] = 2
    grid_before[5][5] = 2
    grid_after = [[0] * 7 for _ in range(7)]
    grid_after[1][1] = 2
    grid_after[4][4] = 2
    before = _snapshot(builder, memory, grid_before, 0, actions=("ACTION1",))
    after = _snapshot(builder, memory, grid_after, 1, actions=("ACTION1",))
    relation = next(r for r in before.relations if r.relation_type == "same_shape")
    step = VerificationTestStep(
        kind="relation_probe",
        action_id="ACTION1",
        target_relation_id=relation.relation_id,
        contract_kind="RELATION_ERROR_DECREASE",
        expected_observation="relation error decreases",
    )
    contract = VerificationBinder().bind(step, before, "hyp_relation")
    assert contract is not None
    assert contract.kind is VerificationContractKind.RELATION_ERROR_DECREASE
    action = CandidateAction("ACTION1", hypothesis_id="hyp_relation", verification_contract=contract)
    result = TransitionJudge().evaluate(before, action, after, PendingAction(before, action, "hyp_relation", ""), memory, V8Config())
    assert result.truth is TriTruth.TRUE
    assert result.progress is Progress.POSITIVE
    assert result.error_delta is not None and result.error_delta > 0
    assert result.affected_relations == (relation.relation_id,)


def test_typed_no_effect_has_first_information_gain_then_becomes_irrelevant():
    builder, memory = ARGALiteBuilder(), GameMemory()
    before = _snapshot(builder, memory, [[0, 0], [0, 1]], 0, actions=("ACTION1",))
    after = _snapshot(builder, memory, [[0, 0], [0, 1]], 1, actions=("ACTION1",))
    step = VerificationTestStep(
        kind="simple_action_probe",
        action_id="ACTION1",
        contract_kind="ACTION_EFFECT_DISCOVERY",
        expected_observation="observe typed transition",
    )
    contract = VerificationBinder().bind(step, before, "hyp_effect")
    assert contract is not None
    action = CandidateAction("ACTION1", hypothesis_id="hyp_effect", verification_contract=contract)
    first = TransitionJudge().evaluate(before, action, after, PendingAction(before, action, "hyp_effect", ""), memory, V8Config())
    second = TransitionJudge().evaluate(before, action, after, PendingAction(before, action, "hyp_effect", ""), memory, V8Config())
    assert first.truth is TriTruth.UNKNOWN
    assert first.observed_information_gain > 0
    assert first.relevance is Relevance.RELEVANT
    assert second.observed_information_gain == 0
    assert second.relevance is Relevance.IRRELEVANT
    assert second.reason_code == "repeated_no_effect_irrelevant"


def test_change_outside_bound_target_does_not_confirm_affordance():
    builder, memory = ARGALiteBuilder(), GameMemory()
    before_grid = [[0] * 6 for _ in range(6)]
    before_grid[1][1] = 1
    after_grid = [row[:] for row in before_grid]
    after_grid[5][5] = 2
    before = _snapshot(builder, memory, before_grid, 0, actions=("ACTION1",))
    after = _snapshot(builder, memory, after_grid, 1, actions=("ACTION1",))
    target = next(o for o in before.objects if o.centroid_rc == (1.0, 1.0))
    step = VerificationTestStep(
        kind="object_probe",
        action_id="ACTION1",
        target_object_id=target.object_id,
        contract_kind="LOCAL_TARGET_CHANGE",
    )
    contract = VerificationBinder().bind(step, before, "hyp_local")
    action = CandidateAction("ACTION1", hypothesis_id="hyp_local", verification_contract=contract)
    result = TransitionJudge().evaluate(before, action, after, PendingAction(before, action, "hyp_local", ""), memory, V8Config())
    assert result.truth is TriTruth.UNKNOWN
    assert result.progress is Progress.UNKNOWN
    assert result.reason_code == "change_outside_target"


def test_v84_expected_observation_type_maps_to_contract_kind():
    builder, memory = ARGALiteBuilder(), GameMemory()
    before = _snapshot(builder, memory, [[0, 0], [0, 1]], 0, actions=("ACTION1",))
    target = before.objects[0]
    step = VerificationTestStep(
        kind="object_move",
        action_id="ACTION1",
        target_object_id=target.object_id,
        expected_observation="expected_type=object_move; target should move",
    )
    contract = VerificationBinder().bind(step, before, "hyp_v84")
    assert contract is not None
    assert contract.kind is VerificationContractKind.OBJECT_DISPLACEMENT
