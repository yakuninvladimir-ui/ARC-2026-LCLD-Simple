from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config
from v8_agent.game_adapter import GameAdapter
from v8_agent.judge import TransitionJudge
from v8_agent.memory import GameMemory
from v8_agent.types import (
    Attribution,
    CandidateAction,
    PendingAction,
    Relevance,
    TriTruth,
    VerificationContract,
    VerificationContractKind,
)


def snap(grid):
    obs={"grid":grid, "metadata":{"game_id":"g","available_actions":["ACTION1","ACTION6"],"frame_index":0}}
    return ARGALiteBuilder().build(GameAdapter().to_world_state(obs), GameMemory(), V8Config())


def test_noop_is_unknown_irrelevant_not_invalid():
    before=snap([[0,0],[0,1]])
    after=snap([[0,0],[0,1]])
    action=CandidateAction("ACTION1")
    j=TransitionJudge().evaluate(before, action, after, PendingAction(before, action, None, ""), GameMemory(), V8Config())
    assert j.truth is TriTruth.UNKNOWN
    assert j.relevance is Relevance.IRRELEVANT
    assert j.attribution is Attribution.NO_VISIBLE_CHANGE


def test_no_op_contract_confirmed_by_no_visible_change():
    before=snap([[0,0],[0,1]])
    after=snap([[0,0],[0,1]])
    target=before.objects[0]
    contract=VerificationContract(
        contract_id="vc_noop",
        kind=VerificationContractKind.NO_OP_TEST,
        target_object_ids=(target.object_id,),
        target_signature=target.stable_hash,
    )
    action=CandidateAction("ACTION1", verification_contract=contract)
    j=TransitionJudge().evaluate(before, action, after, PendingAction(before, action, None, ""), GameMemory(), V8Config())
    assert j.truth is TriTruth.TRUE
    assert j.relevance is Relevance.IRRELEVANT
    assert j.reason_code == "no_op_confirmed"


def test_no_op_contract_rejected_by_action_linked_change():
    before=snap([[0,0],[0,1]])
    after=snap([[0,0],[0,2]])
    target=before.objects[0]
    contract=VerificationContract(
        contract_id="vc_noop",
        kind=VerificationContractKind.NO_OP_TEST,
        target_object_ids=(target.object_id,),
        target_signature=target.stable_hash,
    )
    action=CandidateAction("ACTION1", verification_contract=contract)
    j=TransitionJudge().evaluate(before, action, after, PendingAction(before, action, None, ""), GameMemory(), V8Config())
    assert j.truth is TriTruth.FALSE
    assert j.relevance is Relevance.RELEVANT
    assert j.reason_code == "no_op_contradicted_by_action_effect"


def test_passive_unrelated_coordinate_change_unknown():
    before=snap([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
    after=snap([[0,0,2,2],[0,1,2,2],[0,0,0,0],[0,0,0,0]])
    action=CandidateAction("ACTION6", x=0, y=3, coordinate_candidate_id="c")
    j=TransitionJudge().evaluate(before, action, after, PendingAction(before, action, None, ""), GameMemory(), V8Config(passive_change_threshold_cells=2))
    assert j.truth is TriTruth.UNKNOWN
    assert j.attribution in {Attribution.PASSIVE_POSSIBLE, Attribution.MIXED_OR_UNCERTAIN}


def test_object_displacement_contract_accepts_any_moving_target_object():
    before=snap([
        [0,0,0,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,2,0,0],
        [0,0,0,0,0],
    ])
    after=snap([
        [0,0,0,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,2,0],
        [0,0,0,0,0],
    ])
    ids = tuple(obj.object_id for obj in before.objects[:2])
    contract=VerificationContract(
        contract_id="vc_multi_move",
        kind=VerificationContractKind.OBJECT_DISPLACEMENT,
        target_object_ids=ids,
        target_signature="multi",
    )
    action=CandidateAction("ACTION1", verification_contract=contract)

    j=TransitionJudge().evaluate(before, action, after, PendingAction(before, action, "hyp", ""), GameMemory(), V8Config())

    assert j.truth is TriTruth.TRUE
    assert j.reason_code == "target_object_displaced"


def test_raw_visual_summary_preserves_repeated_isolated_center_changes():
    before_grid = [[0 for _ in range(9)] for _ in range(9)]
    for cy, cx in ((2, 2), (6, 6)):
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                before_grid[y][x] = 4
    after_grid = [row[:] for row in before_grid]
    after_grid[2][2] = 5
    after_grid[6][6] = 5

    before = snap(before_grid)
    after = snap(after_grid)
    action = CandidateAction("ACTION1")
    judgment = TransitionJudge().evaluate(
        before,
        action,
        after,
        PendingAction(before, action, None, ""),
        GameMemory(),
        V8Config(),
    )

    visual = judgment.observed_delta["raw_visual_changes"]
    isolated = visual["isolated_center_cell_transition_groups"][0]
    patch = visual["local_3x3_transition_groups"][0]
    assert visual["repeated_isolated_interior_change"] is True
    assert isolated["occurrence_count"] == 2
    assert isolated["locations_xy"] == [[2, 2], [6, 6]]
    assert isolated["locations_bbox_xyxy"] == [2, 2, 6, 6]
    assert patch["changed_offsets_xy"] == [[0, 0]]
    assert patch["before_3x3_rows"] == ["444", "444", "444"]
    assert patch["after_3x3_rows"] == ["444", "454", "444"]


def test_raw_visual_summary_marks_reciprocal_center_value_changes():
    before_grid = [[0 for _ in range(9)] for _ in range(9)]
    after_grid = [row[:] for row in before_grid]
    before_grid[2][2], after_grid[2][2] = 4, 5
    before_grid[6][6], after_grid[6][6] = 5, 4

    before = snap(before_grid)
    after = snap(after_grid)
    action = CandidateAction("ACTION1")
    judgment = TransitionJudge().evaluate(
        before,
        action,
        after,
        PendingAction(before, action, None, ""),
        GameMemory(),
        V8Config(),
    )

    pairs = judgment.observed_delta["raw_visual_changes"]["reciprocal_isolated_interior_transition_pairs"]
    assert len(pairs) == 1
    assert set(pairs[0]["value_pair"]) == {"4", "5"}
    assert pairs[0]["forward_interior_count"] == 1
    assert pairs[0]["reverse_interior_count"] == 1
