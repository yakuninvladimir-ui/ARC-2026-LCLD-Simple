from __future__ import annotations

from collections import Counter
from typing import Any

from .config import V8Config
from .memory import is_coordinate_research_source
from .reverse_semantics import transition_goal_evaluation
from .types import (
    ARGALiteSnapshot,
    Attribution,
    CandidateAction,
    Judgment,
    MechanicResult,
    PendingAction,
    PreflightResult,
    Progress,
    Relevance,
    SemanticJudgment,
    TriTruth,
    Validity,
    VerificationContractKind,
)
from .verification import changed_cells_in_regions, object_displacement, relation_metric_after, target_regions


class PreflightJudge:
    def validate(self, candidate_action: CandidateAction, snapshot: ARGALiteSnapshot, memory: "GameMemory", config: V8Config) -> PreflightResult:
        if candidate_action.action_id == "RESET":
            if snapshot.game_over or snapshot.state_name in {"NOT_STARTED", "NOT_PLAYED"} or "RESET" in snapshot.available_actions:
                return PreflightResult(True, Validity.VALID, "reset_allowed")
            return PreflightResult(False, Validity.INVALID, "reset_not_available_outside_reset_state")
        if candidate_action.action_id not in snapshot.available_actions:
            return PreflightResult(False, Validity.INVALID, "action_not_available", candidate_action.action_id)
        is_coord = candidate_action.action_id in snapshot.coordinate_action_ids or candidate_action.x is not None or candidate_action.y is not None
        if is_coord:
            if candidate_action.x is None or candidate_action.y is None:
                return PreflightResult(False, Validity.INVALID, "coordinate_payload_missing")
            if not (0 <= int(candidate_action.x) <= 63 and 0 <= int(candidate_action.y) <= 63 and int(candidate_action.x) < snapshot.width and int(candidate_action.y) < snapshot.height):
                return PreflightResult(False, Validity.INVALID, "coordinate_out_of_bounds")
            if config.require_coordinate_candidate_id and candidate_action.coordinate_candidate_id is None:
                return PreflightResult(False, Validity.INVALID, "coordinate_candidate_id_required")
            valid_candidates = {c.candidate_id for c in snapshot.coordinate_targets}
            if candidate_action.coordinate_candidate_id is not None and candidate_action.coordinate_candidate_id not in valid_candidates:
                return PreflightResult(False, Validity.INVALID, "coordinate_candidate_unknown")
            if is_coordinate_research_source(candidate_action.source):
                if memory.coordinate_candidate_clicked_this_attempt(
                    candidate_action.action_id,
                    candidate_action.coordinate_candidate_id,
                    candidate_action.x,
                    candidate_action.y,
                ):
                    return PreflightResult(False, Validity.INVALID, "coordinate_research_candidate_already_clicked_this_attempt")
                if memory.coordinate_probe_count(snapshot.level_index) >= config.max_coordinate_probes_per_level and not candidate_action.allow_exhaustion_revisit:
                    return PreflightResult(False, Validity.INVALID, "coordinate_probe_budget_exceeded")
            if (
                memory.is_coordinate_no_effect_suppressed(candidate_action.action_id, candidate_action.coordinate_candidate_id, int(candidate_action.x), int(candidate_action.y), snapshot.grid_hash)
                and not candidate_action.allow_exhaustion_revisit
            ):
                return PreflightResult(False, Validity.INVALID, "coordinate_exact_no_effect_repeat_suppressed")
        if (
            memory.action_attempt_count(candidate_action.suppression_signature, snapshot.semantic_state_signature) >= config.max_same_state_action_repeats
            and not candidate_action.allow_exhaustion_revisit
        ):
            # Confirmed positive actions may be reused; failed/irrelevant exact repeats may not.
            last = next((ev for ev in reversed(memory.events) if ev.action and ev.action.get("id") == candidate_action.action_id), None)
            if last is not None and last.progress is not Progress.POSITIVE:
                return PreflightResult(False, Validity.INVALID, "same_candidate_repeat_suppressed")
        return PreflightResult(True, Validity.VALID, "ok")


class TransitionJudge:
    def evaluate(self, before: ARGALiteSnapshot, action: CandidateAction, after: ARGALiteSnapshot, pending: PendingAction, memory: "GameMemory", config: V8Config) -> Judgment:
        cells = _changed_cells(before.full_grid_hex_rows, after.full_grid_hex_rows)
        changed_count = len(cells)
        changed_bbox = _changed_bbox(cells)
        color_delta = _changed_color_delta(before.full_grid_hex_rows, after.full_grid_hex_rows, cells)
        raw_visual_changes = _raw_visual_change_summary(
            before.full_grid_hex_rows,
            after.full_grid_hex_rows,
            cells,
            max_groups=config.max_visual_change_groups_in_memory,
            max_locations=config.max_visual_change_locations_per_group,
        )
        score_delta = _score_delta(before, after)
        levels_completed_delta = int(after.levels_completed) - int(before.levels_completed)
        level_index_delta = int(after.level_index) - int(before.level_index)
        game_over_delta = bool(after.game_over and not before.game_over)
        terminal_delta = bool((after.terminal and not before.terminal) or levels_completed_delta > 0 or level_index_delta > 0)

        near_target = 0
        far_target = 0
        target_in_changed_bbox = False
        if action.x is not None and action.y is not None and cells:
            for x, y in cells:
                if abs(x - action.x) + abs(y - action.y) <= 2:
                    near_target += 1
                else:
                    far_target += 1
            if changed_bbox is not None:
                r0, c0, r1, c1 = changed_bbox
                target_in_changed_bbox = c0 <= int(action.x) <= c1 and r0 <= int(action.y) <= r1

        changed_center = None
        if cells:
            changed_center = (int(round(sum(x for x, _ in cells) / len(cells))), int(round(sum(y for _, y in cells) / len(cells))))

        attribution = _attribution(action, changed_count, near_target, far_target, target_in_changed_bbox, score_delta, terminal_delta, game_over_delta, config)
        contract = action.verification_contract
        error_before = contract.before_metric if contract else None
        error_after = None
        error_delta = None
        affected_relations: tuple[str, ...] = ()
        observed_information_gain = 0.0
        mechanic_result = MechanicResult.UNKNOWN
        semantic_judgment = SemanticJudgment.UNRESOLVED

        if action.action_id == "RESET":
            truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.NEUTRAL, "reset_transition"
        elif game_over_delta:
            truth, relevance, progress, reason = TriTruth.FALSE, Relevance.RELEVANT, Progress.NEGATIVE, "game_over_after_action"
            mechanic_result = MechanicResult.MISMATCH
            semantic_judgment = SemanticJudgment.FORBIDDEN
        elif terminal_delta or levels_completed_delta > 0 or (score_delta is not None and score_delta > 0):
            truth, relevance, progress, reason = TriTruth.TRUE, Relevance.RELEVANT, Progress.POSITIVE, "score_or_terminal_progress"
            mechanic_result = MechanicResult.MATCH
            semantic_judgment = SemanticJudgment.REQUIRED
        elif contract is None:
            truth, relevance, progress, reason = _unbound_judgment(attribution, changed_count)
        elif contract.kind is VerificationContractKind.RELATION_ERROR_DECREASE:
            error_after = relation_metric_after(contract, before, after)
            error_delta = None if error_before is None or error_after is None else float(error_before) - float(error_after)
            affected_relations = contract.target_relation_ids
            if error_delta is not None and error_delta > config.relation_error_epsilon:
                truth, relevance, progress, reason = TriTruth.TRUE, Relevance.RELEVANT, Progress.POSITIVE, "relation_error_decreased"
                outcome = "error_decreased"
                mechanic_result = MechanicResult.MATCH
            elif error_delta is not None and error_delta < -config.relation_error_epsilon:
                truth, relevance, progress, reason = TriTruth.FALSE, Relevance.RELEVANT, Progress.NEGATIVE, "relation_error_increased"
                outcome = "error_increased"
                mechanic_result = MechanicResult.MISMATCH
            elif error_delta is not None:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.IRRELEVANT, Progress.NEUTRAL, "relation_error_unchanged"
                outcome = "unchanged"
                mechanic_result = MechanicResult.MISMATCH
            else:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "relation_error_unavailable"
                outcome = "unchanged"
            observed_information_gain = memory.resolve_question(contract.question_id, contract.question_type, contract.target_signature, outcome)
        elif contract.kind is VerificationContractKind.OBJECT_DISPLACEMENT:
            displacement = object_displacement(contract, before, after)
            error_before, error_after = 0.0, displacement
            error_delta = displacement
            if displacement is not None and displacement > config.relation_error_epsilon:
                truth, relevance, progress, reason = TriTruth.TRUE, Relevance.RELEVANT, Progress.POSITIVE, "target_object_displaced"
                outcome = "moved"
                mechanic_result = MechanicResult.MATCH
            elif displacement is not None:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.IRRELEVANT, Progress.NEUTRAL, "target_object_not_displaced"
                outcome = "not_moved"
                mechanic_result = MechanicResult.MISMATCH
            else:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "target_object_tracking_unavailable"
                outcome = "ambiguous"
                mechanic_result = MechanicResult.MISMATCH
            observed_information_gain = memory.resolve_question(contract.question_id, contract.question_type, contract.target_signature, outcome)
        elif contract.kind is VerificationContractKind.LOCAL_TARGET_CHANGE:
            inside, outside = changed_cells_in_regions(cells, target_regions(contract, before, after))
            overlap_fraction = inside / max(1, changed_count)
            if inside > 0 and overlap_fraction >= config.local_target_overlap_min_fraction:
                truth, relevance, progress, reason = TriTruth.TRUE, Relevance.RELEVANT, Progress.POSITIVE, "target_local_change_observed"
                outcome = "effect"
                mechanic_result = MechanicResult.MATCH
            elif changed_count == 0:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.IRRELEVANT, Progress.NEUTRAL, "target_local_no_effect"
                outcome = "no_effect"
                mechanic_result = MechanicResult.MISMATCH
            else:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "change_outside_target"
                outcome = "negative_effect" if outside else "no_effect"
            observed_information_gain = memory.resolve_question(contract.question_id, contract.question_type, contract.target_signature, outcome)
        elif contract.kind is VerificationContractKind.ACTION_SURFACE_CHANGE:
            changed = before.available_actions != after.available_actions
            if changed:
                truth, relevance, progress, reason = TriTruth.TRUE, Relevance.RELEVANT, Progress.POSITIVE, "action_surface_changed"
                outcome = "changed"
                mechanic_result = MechanicResult.MATCH
            else:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.IRRELEVANT, Progress.NEUTRAL, "action_surface_unchanged"
                outcome = "unchanged"
                mechanic_result = MechanicResult.MISMATCH
            observed_information_gain = memory.resolve_question(contract.question_id, contract.question_type, contract.target_signature, outcome)
        elif contract.kind is VerificationContractKind.SCORE_OR_TERMINAL:
            truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.IRRELEVANT, Progress.NEUTRAL, "no_score_or_terminal_progress"
            observed_information_gain = memory.resolve_question(contract.question_id, contract.question_type, contract.target_signature, "no_progress")
        elif contract.kind is VerificationContractKind.NO_OP_TEST:
            if attribution is Attribution.NO_VISIBLE_CHANGE:
                truth, relevance, progress, reason = TriTruth.TRUE, Relevance.IRRELEVANT, Progress.NEUTRAL, "no_op_confirmed"
                outcome = "no_effect"
                mechanic_result = MechanicResult.MATCH
            elif attribution is Attribution.ACTION_LINKED:
                truth, relevance, progress, reason = TriTruth.FALSE, Relevance.RELEVANT, Progress.NEUTRAL, "no_op_contradicted_by_action_effect"
                outcome = "effect"
                mechanic_result = MechanicResult.MISMATCH
            elif attribution is Attribution.PASSIVE_POSSIBLE:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "no_op_passive_possible_change"
                outcome = "negative_effect"
            else:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "no_op_mixed_or_uncertain_change"
                outcome = "negative_effect"
            observed_information_gain = memory.resolve_question(contract.question_id, contract.question_type, contract.target_signature, outcome)
        else:
            # ACTION_EFFECT_DISCOVERY: typed evidence resolves an action-effect question,
            # but a generic visual change still does not prove a semantic goal.
            if attribution is Attribution.NO_VISIBLE_CHANGE:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.RELEVANT, Progress.NEUTRAL, "typed_no_effect_observed"
                outcome = "no_effect"
                mechanic_result = MechanicResult.MATCH
            elif attribution is Attribution.ACTION_LINKED:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.RELEVANT, Progress.NEUTRAL, "typed_action_effect_observed"
                outcome = "effect"
                mechanic_result = MechanicResult.MATCH
            elif attribution is Attribution.PASSIVE_POSSIBLE:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "passive_possible_change"
                outcome = "negative_effect"
            else:
                truth, relevance, progress, reason = TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "mixed_or_uncertain_change"
                outcome = "negative_effect"
            observed_information_gain = memory.resolve_question(contract.question_id, contract.question_type, contract.target_signature, outcome)
            if observed_information_gain < config.information_gain_min_threshold and attribution is Attribution.NO_VISIBLE_CHANGE:
                relevance = Relevance.IRRELEVANT
                reason = "repeated_no_effect_irrelevant"

        if (
            contract is not None
            and contract.semantic_binding_id
            and not game_over_delta
            and not terminal_delta
        ):
            goal_progress, semantic_judgment, goal_before, goal_after, goal_delta = transition_goal_evaluation(
                contract,
                before,
                after,
            )
            if goal_progress is not Progress.UNKNOWN:
                progress = goal_progress
                error_before, error_after, error_delta = goal_before, goal_after, goal_delta
            elif contract.goal_operator is not None and contract.goal_operator.value == "CHANGE_ACTION_SURFACE":
                if before.available_actions != after.available_actions:
                    progress = Progress.POSITIVE
                    semantic_judgment = SemanticJudgment.REQUIRED
                else:
                    progress = Progress.NEUTRAL
                    semantic_judgment = SemanticJudgment.IRRELEVANT
            else:
                progress = Progress.NEUTRAL

        affected_objects = _affected_objects(before, after, cells)
        object_deltas = _object_deltas(before, after, cells, affected_objects)
        observed_delta: dict[str, Any] = {
            "changed_cell_count": changed_count,
            "changed_cells_xy": _changed_cell_records(
                before.full_grid_hex_rows,
                after.full_grid_hex_rows,
                cells,
                limit=config.max_changed_cells_per_action_diff,
            ),
            "changed_cell_runs": _changed_cell_runs(
                before.full_grid_hex_rows,
                after.full_grid_hex_rows,
                cells,
                limit=config.max_changed_cells_per_action_diff,
            ),
            "changed_center_xy": changed_center,
            "changed_bbox_rc": changed_bbox,
            "changed_color_delta": color_delta,
            "raw_visual_changes": raw_visual_changes,
            "near_target_change_count": near_target,
            "far_target_change_count": far_target,
            "target_in_changed_bbox": target_in_changed_bbox,
            "level_index_before": before.level_index,
            "level_index": after.level_index,
            "level_index_delta": level_index_delta,
            "levels_completed_before": before.levels_completed,
            "levels_completed_after": after.levels_completed,
            "levels_completed_delta": levels_completed_delta,
            "win_levels": after.win_levels,
            "state_before": before.state_name,
            "state_after": after.state_name,
            "game_over_delta": game_over_delta,
            "terminal_delta": terminal_delta,
            "step_index": after.step_index,
            "grid_hash_before": before.grid_hash,
            "grid_hash_after": after.grid_hash,
            "available_actions_before": list(before.available_actions),
            "available_actions_after": list(after.available_actions),
            "planning_action_ids_before": list(before.planning_action_ids or before.available_actions),
            "planning_action_ids_after": list(after.planning_action_ids or after.available_actions),
            "undo_action_ids_before": list(before.undo_action_ids),
            "undo_action_ids_after": list(after.undo_action_ids),
            "possible_actions_before": list(before.possible_actions or before.available_actions),
            "possible_actions_after": list(after.possible_actions or after.available_actions),
            "action_surface_added": sorted(set(after.available_actions) - set(before.available_actions)),
            "action_surface_removed": sorted(set(before.available_actions) - set(after.available_actions)),
            "planning_action_surface_added": sorted(set(after.planning_action_ids or after.available_actions) - set(before.planning_action_ids or before.available_actions)),
            "planning_action_surface_removed": sorted(set(before.planning_action_ids or before.available_actions) - set(after.planning_action_ids or after.available_actions)),
            "information_gain_observed": observed_information_gain,
            "error_before": error_before,
            "error_after": error_after,
            "error_delta": error_delta,
            "object_deltas": object_deltas,
        }
        if action.x is not None and action.y is not None:
            observed_delta["coordinate_xy"] = [int(action.x), int(action.y)]
            observed_delta["coordinate_cell_before"] = _grid_char(before.full_grid_hex_rows, int(action.x), int(action.y))
            observed_delta["coordinate_cell_after"] = _grid_char(after.full_grid_hex_rows, int(action.x), int(action.y))
        if contract is not None:
            observed_delta["target_signature"] = contract.target_signature
            observed_delta["target_object_ids"] = list(contract.target_object_ids)
            observed_delta["target_relation_ids"] = list(contract.target_relation_ids)
            observed_delta["target_coordinate_candidate_id"] = contract.target_coordinate_candidate_id
            observed_delta["target_objects_before"] = _object_summaries(before, contract.target_object_ids)
            observed_delta["target_objects_after"] = _object_summaries(after, contract.target_object_ids)
        return Judgment(
            truth=truth,
            relevance=relevance,
            validity=Validity.VALID,
            progress=progress,
            attribution=attribution,
            reason_code=reason,
            observed_delta=observed_delta,
            affected_objects=affected_objects,
            affected_relations=affected_relations,
            score_delta=score_delta,
            terminal_delta=terminal_delta,
            action=action,
            hypothesis_id=pending.hypothesis_id,
            before_hash=before.grid_hash,
            after_hash=after.grid_hash,
            contract_kind=contract.kind if contract else None,
            error_before=error_before,
            error_after=error_after,
            error_delta=error_delta,
            observed_information_gain=observed_information_gain,
            question_id=contract.question_id if contract else None,
            mechanic_result=mechanic_result,
            semantic_judgment=semantic_judgment,
            semantic_binding_id=contract.semantic_binding_id if contract else None,
        )


def _changed_cells(before_rows: tuple[str, ...], after_rows: tuple[str, ...]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    max_h = max(len(before_rows), len(after_rows))
    for y in range(max_h):
        br = before_rows[y] if y < len(before_rows) else ""
        ar = after_rows[y] if y < len(after_rows) else ""
        max_w = max(len(br), len(ar))
        for x in range(max_w):
            b = br[x] if x < len(br) else None
            a = ar[x] if x < len(ar) else None
            if b != a:
                cells.append((x, y))
    return cells


def _changed_cell_records(
    before_rows: tuple[str, ...],
    after_rows: tuple[str, ...],
    cells: list[tuple[int, int]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for x, y in cells[: max(0, int(limit))]:
        records.append({
            "xy": [int(x), int(y)],
            "before": _grid_char(before_rows, x, y),
            "after": _grid_char(after_rows, x, y),
        })
    return records


def _changed_cell_runs(
    before_rows: tuple[str, ...],
    after_rows: tuple[str, ...],
    cells: list[tuple[int, int]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Encode exact changed pixels as consecutive runs within each grid row."""
    runs: list[dict[str, Any]] = []
    ordered = sorted(cells, key=lambda item: (item[1], item[0]))
    start = end = row = None
    before_values: list[str] = []
    after_values: list[str] = []

    def flush() -> None:
        if row is None or start is None or end is None or len(runs) >= max(0, int(limit)):
            return
        runs.append({
            "y": int(row),
            "x_range_inclusive": [int(start), int(end)],
            "before": "".join(before_values),
            "after": "".join(after_values),
        })

    for x, y in ordered:
        if row is None or y != row or end is None or x != end + 1:
            flush()
            if len(runs) >= max(0, int(limit)):
                break
            row, start, end = y, x, x
            before_values = [_grid_char(before_rows, x, y)]
            after_values = [_grid_char(after_rows, x, y)]
            continue
        end = x
        before_values.append(_grid_char(before_rows, x, y))
        after_values.append(_grid_char(after_rows, x, y))
    flush()
    return runs


def _changed_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (min(ys), min(xs), max(ys), max(xs))


def _changed_color_delta(before_rows: tuple[str, ...], after_rows: tuple[str, ...], cells: list[tuple[int, int]]) -> dict[str, Any]:
    if not cells:
        return {"before_histogram": {}, "after_histogram": {}, "transitions": {}}
    before_hist: Counter[str] = Counter()
    after_hist: Counter[str] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    for x, y in cells:
        b = _grid_char(before_rows, x, y)
        a = _grid_char(after_rows, x, y)
        before_hist[b] += 1
        after_hist[a] += 1
        transitions[(b, a)] += 1
    return {
        "before_histogram": dict(sorted(before_hist.items())),
        "after_histogram": dict(sorted(after_hist.items())),
        "transitions": {f"{b}->{a}": n for (b, a), n in sorted(transitions.items())},
    }


def _raw_visual_change_summary(
    before_rows: tuple[str, ...],
    after_rows: tuple[str, ...],
    cells: list[tuple[int, int]],
    *,
    max_groups: int,
    max_locations: int,
) -> dict[str, Any]:
    """Compact exact cell evidence without assigning object or goal semantics."""
    if not cells:
        return {
            "coordinate_order": "x=column,y=row",
            "changed_cell_count": 0,
            "changed_cell_transition_groups": [],
            "isolated_center_cell_transition_groups": [],
            "local_3x3_transition_groups": [],
            "reciprocal_isolated_interior_transition_pairs": [],
            "repeated_isolated_interior_change": False,
        }

    height = max(len(before_rows), len(after_rows))
    width = max(
        max((len(row) for row in before_rows), default=0),
        max((len(row) for row in after_rows), default=0),
    )
    changed = set(cells)
    transition_positions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    isolated_positions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    patch_positions: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, int], ...]], list[tuple[int, int]]] = {}

    for x, y in cells:
        before_value = _grid_char(before_rows, x, y)
        after_value = _grid_char(after_rows, x, y)
        transition_positions.setdefault((before_value, after_value), []).append((x, y))
        neighboring_changes = [
            (nx, ny)
            for ny in range(y - 1, y + 2)
            for nx in range(x - 1, x + 2)
            if (nx, ny) != (x, y) and (nx, ny) in changed
        ]
        if not neighboring_changes:
            isolated_positions.setdefault((before_value, after_value), []).append((x, y))
        before_patch = tuple(
            "".join(_grid_char(before_rows, px, py) for px in range(x - 1, x + 2))
            for py in range(y - 1, y + 2)
        )
        after_patch = tuple(
            "".join(_grid_char(after_rows, px, py) for px in range(x - 1, x + 2))
            for py in range(y - 1, y + 2)
        )
        changed_offsets = tuple(sorted(
            (nx - x, ny - y)
            for ny in range(y - 1, y + 2)
            for nx in range(x - 1, x + 2)
            if (nx, ny) in changed
        ))
        patch_positions.setdefault((before_patch, after_patch, changed_offsets), []).append((x, y))

    transition_groups = _cell_transition_groups(
        transition_positions,
        width,
        height,
        max_groups=max_groups,
        max_locations=max_locations,
    )
    isolated_groups = _cell_transition_groups(
        isolated_positions,
        width,
        height,
        max_groups=max_groups,
        max_locations=max_locations,
    )
    local_groups = []
    ranked_patches = sorted(
        patch_positions.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[:max_groups]
    for (before_patch, after_patch, changed_offsets), positions in ranked_patches:
        edge_count, interior_count = _edge_and_interior_counts(positions, width, height)
        local_groups.append({
            "before_3x3_rows": list(before_patch),
            "after_3x3_rows": list(after_patch),
            "changed_offsets_xy": [[dx, dy] for dx, dy in changed_offsets],
            "occurrence_count": len(positions),
            "interior_location_count": interior_count,
            "edge_location_count": edge_count,
            "locations_bbox_xyxy": _locations_bbox_xyxy(positions),
            "locations_xy": [[x, y] for x, y in sorted(positions, key=lambda pos: (pos[1], pos[0]))[:max_locations]],
            "locations_truncated": len(positions) > max_locations,
        })
    repeated_interior = any(int(group["interior_location_count"]) >= 2 for group in isolated_groups)
    reciprocal_pairs = _reciprocal_isolated_pairs(isolated_groups)
    return {
        "coordinate_order": "x=column,y=row",
        "changed_cell_count": len(cells),
        "changed_cell_transition_groups": transition_groups,
        "isolated_center_cell_transition_groups": isolated_groups,
        "local_3x3_transition_groups": local_groups,
        "reciprocal_isolated_interior_transition_pairs": reciprocal_pairs,
        "repeated_isolated_interior_change": repeated_interior,
        "interpretation_status": "RAW_SYNCHRONOUS_VISUAL_EVIDENCE_ONLY",
    }


def _cell_transition_groups(
    grouped: dict[tuple[str, str], list[tuple[int, int]]],
    width: int,
    height: int,
    *,
    max_groups: int,
    max_locations: int,
) -> list[dict[str, Any]]:
    out = []
    ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))[:max_groups]
    for (before_value, after_value), positions in ranked:
        edge_count, interior_count = _edge_and_interior_counts(positions, width, height)
        ordered = sorted(positions, key=lambda pos: (pos[1], pos[0]))
        out.append({
            "before_value": before_value,
            "after_value": after_value,
            "occurrence_count": len(positions),
            "interior_location_count": interior_count,
            "edge_location_count": edge_count,
            "locations_bbox_xyxy": _locations_bbox_xyxy(positions),
            "locations_xy": [[x, y] for x, y in ordered[:max_locations]],
            "locations_truncated": len(positions) > max_locations,
        })
    return out


def _reciprocal_isolated_pairs(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_transition = {
        (str(group.get("before_value")), str(group.get("after_value"))): group
        for group in groups
        if int(group.get("interior_location_count") or 0) > 0
    }
    out = []
    seen: set[tuple[str, str]] = set()
    for (before_value, after_value), forward in sorted(by_transition.items()):
        if before_value == after_value or (before_value, after_value) in seen:
            continue
        reverse = by_transition.get((after_value, before_value))
        if reverse is None:
            continue
        seen.add((before_value, after_value))
        seen.add((after_value, before_value))
        out.append({
            "value_pair": [before_value, after_value],
            "forward_interior_count": int(forward.get("interior_location_count") or 0),
            "reverse_interior_count": int(reverse.get("interior_location_count") or 0),
            "forward_locations_bbox_xyxy": forward.get("locations_bbox_xyxy"),
            "reverse_locations_bbox_xyxy": reverse.get("locations_bbox_xyxy"),
            "interpretation_status": "SYNCHRONIZED_RECIPROCAL_VALUE_CHANGE_FACT_ONLY",
        })
    return out


def _locations_bbox_xyxy(positions: list[tuple[int, int]]) -> list[int] | None:
    if not positions:
        return None
    xs = [x for x, _ in positions]
    ys = [y for _, y in positions]
    return [min(xs), min(ys), max(xs), max(ys)]


def _edge_and_interior_counts(positions: list[tuple[int, int]], width: int, height: int) -> tuple[int, int]:
    edge = sum(1 for x, y in positions if x in {0, width - 1} or y in {0, height - 1})
    return edge, len(positions) - edge


def _grid_char(rows: tuple[str, ...], x: int, y: int) -> str:
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return rows[y][x]
    return "~"


def _object_summaries(snapshot: ARGALiteSnapshot, object_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id = {o.object_id: o for o in snapshot.objects}
    out: list[dict[str, Any]] = []
    for object_id in object_ids:
        obj = by_id.get(object_id)
        if obj is None:
            continue
        out.append({
            "object_id": obj.object_id,
            "stable_hash": obj.stable_hash,
            "shape_signature": obj.shape_signature,
            "topology_signature": obj.topology_signature,
            "colors": list(obj.colors),
            "area": obj.area,
            "bbox_rc": list(obj.bbox_rc),
            "centroid_rc": list(obj.centroid_rc),
            "tags": list(obj.tags),
            "border_touching": list(obj.border_touching),
        })
    return out


def _score_delta(before: ARGALiteSnapshot, after: ARGALiteSnapshot) -> float | None:
    if before.score is None or after.score is None:
        return None
    return float(after.score) - float(before.score)


def _attribution(action: CandidateAction, changed_count: int, near_target: int, far_target: int, target_in_changed_bbox: bool, score_delta: float | None, terminal_delta: bool, game_over_delta: bool, config: V8Config) -> Attribution:
    if changed_count == 0 and not score_delta and not terminal_delta and not game_over_delta:
        return Attribution.NO_VISIBLE_CHANGE
    if terminal_delta or game_over_delta:
        return Attribution.ACTION_LINKED
    if action.x is None or action.y is None:
        return Attribution.ACTION_LINKED
    if target_in_changed_bbox:
        return Attribution.ACTION_LINKED
    if near_target > 0 and far_target == 0:
        return Attribution.ACTION_LINKED
    if near_target > 0 and far_target > 0:
        far_ratio = far_target / max(1, changed_count)
        return Attribution.MIXED_OR_UNCERTAIN if far_ratio >= config.mixed_change_ratio_threshold else Attribution.ACTION_LINKED
    if config.passive_attribution_enabled and changed_count >= config.passive_change_threshold_cells:
        return Attribution.PASSIVE_POSSIBLE
    return Attribution.MIXED_OR_UNCERTAIN


def _unbound_judgment(attribution: Attribution, changed_count: int) -> tuple[TriTruth, Relevance, Progress, str]:
    if attribution is Attribution.NO_VISIBLE_CHANGE:
        return TriTruth.UNKNOWN, Relevance.IRRELEVANT, Progress.NEUTRAL, "unbound_no_visible_change"
    if changed_count > 0:
        return TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "unbound_visible_change_not_semantic_proof"
    return TriTruth.UNKNOWN, Relevance.UNDECIDED, Progress.UNKNOWN, "unbound_unclear"


def _affected_objects(before: ARGALiteSnapshot, after: ARGALiteSnapshot, cells: list[tuple[int, int]]) -> tuple[str, ...]:
    if not cells:
        return ()
    out: list[str] = []
    by_id = {obj.object_id: obj for obj in before.objects}
    by_id.update({obj.object_id: obj for obj in after.objects})
    for object_id in sorted(by_id):
        objects = [obj for obj in (next((o for o in before.objects if o.object_id == object_id), None), next((o for o in after.objects if o.object_id == object_id), None)) if obj is not None]
        if any(_cells_touch_object(cells, obj) for obj in objects):
            out.append(object_id)
    return tuple(out)


def _object_deltas(before: ARGALiteSnapshot, after: ARGALiteSnapshot, cells: list[tuple[int, int]], affected_object_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    before_map = {o.object_id: o for o in before.objects}
    after_map = {o.object_id: o for o in after.objects}
    candidate_ids = set(affected_object_ids)
    for object_id in set(before_map) & set(after_map):
        b = before_map[object_id]
        a = after_map[object_id]
        if b.bbox_rc != a.bbox_rc or b.centroid_rc != a.centroid_rc or b.colors != a.colors or b.area != a.area:
            candidate_ids.add(object_id)
    out: list[dict[str, Any]] = []
    for object_id in sorted(candidate_ids):
        b = before_map.get(object_id)
        a = after_map.get(object_id)
        if b is None and a is None:
            continue
        before_centroid = list(b.centroid_rc) if b is not None else None
        after_centroid = list(a.centroid_rc) if a is not None else None
        delta_rc = None
        if b is not None and a is not None:
            delta_rc = [
                round(float(a.centroid_rc[0]) - float(b.centroid_rc[0]), 6),
                round(float(a.centroid_rc[1]) - float(b.centroid_rc[1]), 6),
            ]
        changed_before = _cells_in_object(cells, b) if b is not None else 0
        changed_after = _cells_in_object(cells, a) if a is not None else 0
        payload = {
            "object_id": object_id,
            "lifecycle": "appeared" if b is None else ("disappeared" if a is None else "persisted"),
            "before_bbox_rc": list(b.bbox_rc) if b is not None else None,
            "after_bbox_rc": list(a.bbox_rc) if a is not None else None,
            "before_centroid_rc": before_centroid,
            "after_centroid_rc": after_centroid,
            "delta_centroid_rc": delta_rc,
            "motion_direction": _motion_direction(delta_rc),
            "before_colors": list(b.colors) if b is not None else None,
            "after_colors": list(a.colors) if a is not None else None,
            "before_area": int(b.area) if b is not None else None,
            "after_area": int(a.area) if a is not None else None,
            "before_shape_signature": b.shape_signature if b is not None else None,
            "after_shape_signature": a.shape_signature if a is not None else None,
            "shape_changed": bool(b is not None and a is not None and b.shape_signature != a.shape_signature),
            "palette_changed": bool(b is not None and a is not None and b.colors != a.colors),
            "area_delta": (int(a.area) - int(b.area)) if b is not None and a is not None else None,
            "changed_cells_in_before_bbox": changed_before,
            "changed_cells_in_after_bbox": changed_after,
            "tags": list((a or b).tags),
        }
        out.append({key: value for key, value in payload.items() if value not in (None, [], {})})
    out.sort(key=lambda item: (-(int(item.get("changed_cells_in_before_bbox", 0)) + int(item.get("changed_cells_in_after_bbox", 0))), str(item.get("object_id", ""))))
    return out[:12]


def _cells_touch_object(cells: list[tuple[int, int]], obj: Any) -> bool:
    return _cells_in_object(cells, obj) > 0


def _cells_in_object(cells: list[tuple[int, int]], obj: Any) -> int:
    if obj is None:
        return 0
    r0, c0, r1, c1 = obj.bbox_rc
    return sum(1 for x, y in cells if c0 <= x <= c1 and r0 <= y <= r1)


def _motion_direction(delta_rc: list[float] | None) -> str:
    if not delta_rc or len(delta_rc) != 2:
        return "unknown"
    dr, dc = float(delta_rc[0]), float(delta_rc[1])
    eps = 1e-6
    parts: list[str] = []
    if dr < -eps:
        parts.append("up")
    elif dr > eps:
        parts.append("down")
    if dc < -eps:
        parts.append("left")
    elif dc > eps:
        parts.append("right")
    return "+".join(parts) if parts else "stationary"
