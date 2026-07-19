from __future__ import annotations

from math import hypot
from typing import Any, Iterable

from .observe import stable_hash
from .types import (
    ARGALiteSnapshot,
    BindingStatus,
    EvidenceAuthority,
    EvidenceStatus,
    GoalMetricSpec,
    GoalOperator,
    Judgment,
    MechanicResult,
    MetricDirection,
    Progress,
    SemanticBindingResult,
    SemanticJudgment,
    SemanticObjective,
    TrajectoryEvaluation,
    VerificationContract,
)


_OBJECT_REQUIRED_KINDS = {
    "match_or_overlap",
    "relative_arrangement",
    "containment",
    "connection",
    "pattern_or_state",
    "select_or_activate",
}


def bind_semantic_objective(
    objective: SemanticObjective,
    snapshot: ARGALiteSnapshot,
    hypothesis_id: str,
) -> SemanticBindingResult:
    """Bind one Qwen objective to current tracked entities and one cheap metric."""
    objects = {obj.object_id: obj for obj in snapshot.objects}
    relations = {rel.relation_id: rel for rel in snapshot.relations}
    source_ids = _dedupe(objective.source_object_ids)
    reference_ids = _dedupe(objective.reference_object_ids)
    relation_ids = _dedupe(objective.relation_ids)
    unknown_objects = [item for item in (*source_ids, *reference_ids) if item not in objects]
    unknown_relations = [item for item in relation_ids if item not in relations]

    inferred_objects: list[str] = []
    inferred_relations: list[str] = []
    if not unknown_objects and not unknown_relations:
        relation_endpoints: list[str] = []
        for relation_id in relation_ids:
            relation = relations[relation_id]
            relation_endpoints.extend((relation.a, relation.b))
        if source_ids and not reference_ids:
            reference_ids = tuple(
                item for item in _dedupe(relation_endpoints) if item not in source_ids
            )
            inferred_objects.extend(reference_ids)
        elif reference_ids and not source_ids:
            source_ids = tuple(
                item for item in _dedupe(relation_endpoints) if item not in reference_ids
            )
            inferred_objects.extend(source_ids)
        elif not source_ids and not reference_ids and len(_dedupe(relation_endpoints)) >= 2:
            endpoints = _dedupe(relation_endpoints)
            source_ids = (endpoints[0],)
            reference_ids = tuple(endpoints[1:])
            inferred_objects.extend(endpoints)

    operator = _goal_operator(objective, relation_ids, relations)
    metric_spec = _metric_spec(objective, operator, source_ids, reference_ids)
    status = BindingStatus.GROUNDED
    reason_code = "semantic_objective_grounded"
    if unknown_objects or unknown_relations:
        status = BindingStatus.REJECTED
        reason_code = "semantic_objective_contains_unknown_ids"
        metric_spec = None
    elif objective.kind in _OBJECT_REQUIRED_KINDS and not source_ids:
        status = BindingStatus.REJECTED
        reason_code = "semantic_objective_missing_source_objects"
        metric_spec = None
    elif objective.kind in _OBJECT_REQUIRED_KINDS and not reference_ids and operator not in {
        GoalOperator.ACTIVATE,
        GoalOperator.COMPLETE_PATTERN,
        GoalOperator.MATCH_STATE,
    }:
        status = BindingStatus.PARTIAL
        reason_code = "semantic_objective_missing_reference_objects"
    elif metric_spec is None and operator not in {
        GoalOperator.CHANGE_ACTION_SURFACE,
        GoalOperator.ACTIVATE,
        GoalOperator.PROBE_AFFORDANCE,
        GoalOperator.PROBE_RELATION,
        GoalOperator.OTHER,
    }:
        status = BindingStatus.PARTIAL
        reason_code = "semantic_objective_has_no_supported_metric"

    binding_id = stable_hash(
        (
            hypothesis_id,
            objective.kind,
            source_ids,
            reference_ids,
            relation_ids,
            operator.value,
            snapshot.semantic_state_signature,
        ),
        "bind_",
    )
    provisional = SemanticBindingResult(
        binding_id=binding_id,
        hypothesis_id=hypothesis_id,
        status=status,
        objective=objective,
        goal_operator=operator,
        source_object_ids=source_ids,
        reference_object_ids=reference_ids,
        relation_ids=relation_ids,
        inferred_object_ids=_dedupe(inferred_objects),
        inferred_relation_ids=_dedupe(inferred_relations),
        metric_spec=metric_spec,
        baseline_value=None,
        reason_code=reason_code,
        evidence_refs=tuple(f"relation:{item}" for item in relation_ids),
        state_signature=snapshot.semantic_state_signature,
        action_surface_signature=stable_hash(tuple(snapshot.available_actions), "surface_"),
        game_id=snapshot.game_id,
        level_index=snapshot.level_index,
    )
    baseline = compute_binding_metric(provisional, snapshot)
    return SemanticBindingResult(
        binding_id=provisional.binding_id,
        hypothesis_id=provisional.hypothesis_id,
        status=provisional.status,
        objective=provisional.objective,
        goal_operator=provisional.goal_operator,
        source_object_ids=provisional.source_object_ids,
        reference_object_ids=provisional.reference_object_ids,
        relation_ids=provisional.relation_ids,
        inferred_object_ids=provisional.inferred_object_ids,
        inferred_relation_ids=provisional.inferred_relation_ids,
        metric_spec=provisional.metric_spec,
        baseline_value=baseline,
        reason_code=provisional.reason_code,
        evidence_refs=provisional.evidence_refs,
        state_signature=provisional.state_signature,
        action_surface_signature=provisional.action_surface_signature,
        game_id=provisional.game_id,
        level_index=provisional.level_index,
    )


def compute_binding_metric(
    binding: SemanticBindingResult,
    snapshot: ARGALiteSnapshot,
) -> float | None:
    spec = binding.metric_spec
    if spec is None:
        return None
    return _compute_metric(
        spec.name,
        binding.source_object_ids,
        binding.reference_object_ids,
        snapshot,
    )


def compute_contract_goal_metric(
    contract: VerificationContract,
    snapshot: ARGALiteSnapshot,
) -> float | None:
    if not contract.semantic_metric_name or not contract.source_object_ids:
        return None
    return _compute_metric(
        contract.semantic_metric_name,
        contract.source_object_ids,
        contract.reference_object_ids,
        snapshot,
    )


def metric_improvement(
    direction: MetricDirection,
    before: float,
    after: float,
    target_value: float | None = None,
) -> float:
    if direction is MetricDirection.MINIMIZE:
        return float(before) - float(after)
    if direction is MetricDirection.MAXIMIZE:
        return float(after) - float(before)
    if direction is MetricDirection.TARGET:
        target = float(target_value or 0.0)
        return abs(float(before) - target) - abs(float(after) - target)
    return float(after) - float(before)


def transition_goal_evaluation(
    contract: VerificationContract,
    before: ARGALiteSnapshot,
    after: ARGALiteSnapshot,
) -> tuple[Progress, SemanticJudgment, float | None, float | None, float | None]:
    if contract.metric_direction is None:
        return Progress.UNKNOWN, SemanticJudgment.UNRESOLVED, None, None, None
    before_value = compute_contract_goal_metric(contract, before)
    after_value = compute_contract_goal_metric(contract, after)
    if before_value is None or after_value is None:
        return Progress.UNKNOWN, SemanticJudgment.UNRESOLVED, before_value, after_value, None
    delta = metric_improvement(
        contract.metric_direction,
        before_value,
        after_value,
        contract.metric_target_value,
    )
    epsilon = max(0.0, float(contract.metric_epsilon))
    if delta > epsilon:
        return Progress.POSITIVE, SemanticJudgment.REQUIRED, before_value, after_value, delta
    if delta < -epsilon:
        return Progress.NEGATIVE, SemanticJudgment.FORBIDDEN, before_value, after_value, delta
    return Progress.NEUTRAL, SemanticJudgment.IRRELEVANT, before_value, after_value, delta


def evaluate_trajectory(
    binding: SemanticBindingResult | None,
    start: ARGALiteSnapshot,
    end: ARGALiteSnapshot,
    judgments: Iterable[Judgment],
    executed_action_ids: Iterable[str],
) -> TrajectoryEvaluation:
    items = tuple(judgments)
    actions = tuple(executed_action_ids)
    mechanic_result = _aggregate_mechanics(items)
    first_divergence = next(
        (
            index
            for index, item in enumerate(items)
            if item.mechanic_result is MechanicResult.MISMATCH
        ),
        None,
    )

    terminal_progress = any(
        item.terminal_delta
        or int(item.observed_delta.get("levels_completed_delta") or 0) > 0
        or int(item.observed_delta.get("level_index_delta") or 0) > 0
        for item in items
    )
    game_over = any(bool(item.observed_delta.get("game_over_delta")) for item in items)
    error_before = None
    error_after = None
    error_delta = None
    progress = Progress.UNKNOWN
    semantic = SemanticJudgment.UNRESOLVED
    reason = "trajectory_goal_unresolved"

    if terminal_progress:
        progress = Progress.POSITIVE
        semantic = SemanticJudgment.REQUIRED
        reason = "trajectory_completed_level"
    elif game_over:
        progress = Progress.NEGATIVE
        semantic = SemanticJudgment.FORBIDDEN
        reason = "trajectory_caused_game_over"
    elif binding is not None and binding.metric_spec is not None:
        error_before = compute_binding_metric(binding, start)
        error_after = compute_binding_metric(binding, end)
        if error_before is not None and error_after is not None:
            error_delta = metric_improvement(
                binding.metric_spec.direction,
                error_before,
                error_after,
                binding.metric_spec.target_value,
            )
            epsilon = max(0.0, float(binding.metric_spec.epsilon))
            if error_delta > epsilon:
                progress = Progress.POSITIVE
                semantic = SemanticJudgment.REQUIRED
                reason = "trajectory_improved_bound_goal_metric"
            elif error_delta < -epsilon:
                progress = Progress.NEGATIVE
                semantic = SemanticJudgment.FORBIDDEN
                reason = "trajectory_worsened_bound_goal_metric"
            else:
                progress = Progress.NEUTRAL
                semantic = SemanticJudgment.IRRELEVANT
                reason = "trajectory_left_bound_goal_metric_unchanged"
    elif mechanic_result is MechanicResult.MISMATCH:
        progress = Progress.NEGATIVE
        semantic = SemanticJudgment.FORBIDDEN
        reason = "trajectory_mechanics_diverged"

    binding_id = binding.binding_id if binding is not None else None
    evaluation_id = stable_hash(
        (
            binding_id,
            start.semantic_state_signature,
            end.semantic_state_signature,
            actions,
            reason,
        ),
        "traj_",
    )
    return TrajectoryEvaluation(
        evaluation_id=evaluation_id,
        hypothesis_id=binding.hypothesis_id if binding is not None else (items[-1].hypothesis_id if items else ""),
        binding_id=binding_id,
        level_index=start.level_index,
        executed_action_ids=actions,
        mechanic_result=mechanic_result,
        goal_progress=progress,
        semantic_judgment=semantic,
        reason_code=reason,
        error_before=error_before,
        error_after=error_after,
        error_delta=error_delta,
        first_divergence_step=first_divergence,
        source_object_ids=binding.source_object_ids if binding is not None else (),
        reference_object_ids=binding.reference_object_ids if binding is not None else (),
        relation_ids=binding.relation_ids if binding is not None else (),
        evidence_refs=tuple(dict.fromkeys(
            item.action.verification_contract.contract_id
            for item in items
            if item.action.verification_contract is not None
        )),
        start_state_signature=start.semantic_state_signature,
        end_state_signature=end.semantic_state_signature,
    )


def derive_invariant_observations(judgment: Judgment) -> list[dict[str, Any]]:
    """Derive bounded reverse-DSL facts from one official transition."""
    delta = judgment.observed_delta
    if (
        int(delta.get("levels_completed_delta") or 0) != 0
        or int(delta.get("level_index_delta") or 0) != 0
        or bool(delta.get("terminal_delta"))
        or bool(delta.get("game_over_delta"))
    ):
        # The after-frame belongs to another level or terminal state. Cross-level
        # object/surface deltas are not causal mechanics of the final action.
        return []
    action_id = judgment.action.action_id
    surface_signature = stable_hash(tuple(delta.get("available_actions_before") or ()), "surface_")
    observations: list[dict[str, Any]] = []
    moved: list[tuple[str, float, float]] = []

    for item in delta.get("object_deltas") or []:
        if not isinstance(item, dict) or str(item.get("lifecycle") or "persisted") != "persisted":
            continue
        object_id = str(item.get("object_id") or "")
        if not object_id:
            continue
        vector = item.get("delta_centroid_rc")
        if isinstance(vector, (list, tuple)) and len(vector) == 2:
            try:
                dr, dc = float(vector[0]), float(vector[1])
            except (TypeError, ValueError):
                dr = dc = 0.0
            if abs(dr) > 1e-9 or abs(dc) > 1e-9:
                moved.append((object_id, dc, dr))
                observations.append(_invariant(
                    "TRANSLATES_BY",
                    action_id,
                    (object_id,),
                    {"delta_xy": [_round(dc), _round(dr)]},
                    surface_signature,
                    judgment,
                ))
        if item.get("shape_changed") is False and (
            item.get("palette_changed") or object_id in {entry[0] for entry in moved}
        ):
            observations.append(_invariant(
                "PRESERVES_SHAPE",
                action_id,
                (object_id,),
                {},
                surface_signature,
                judgment,
            ))
        if item.get("palette_changed"):
            observations.append(_invariant(
                "CHANGES_COLOR",
                action_id,
                (object_id,),
                {
                    "before_colors": item.get("before_colors") or [],
                    "after_colors": item.get("after_colors") or [],
                },
                surface_signature,
                judgment,
            ))

    for index, first in enumerate(moved):
        for second in moved[index + 1 :]:
            if _vector_equal(first[1:], second[1:]):
                observations.append(_invariant(
                    "CO_MOVES_WITH",
                    action_id,
                    (first[0], second[0]),
                    {"delta_xy": [_round(first[1]), _round(first[2])]},
                    surface_signature,
                    judgment,
                ))
            elif _opposite_axis(first[1:], second[1:]):
                observations.append(_invariant(
                    "MOVES_OPPOSITE_ON_AXIS",
                    action_id,
                    (first[0], second[0]),
                    {
                        "first_delta_xy": [_round(first[1]), _round(first[2])],
                        "second_delta_xy": [_round(second[1]), _round(second[2])],
                    },
                    surface_signature,
                    judgment,
                ))

    added = tuple(delta.get("action_surface_added") or ())
    removed = tuple(delta.get("action_surface_removed") or ())
    if added or removed:
        observations.append(_invariant(
            "CHANGES_ACTION_SURFACE",
            action_id,
            (),
            {"added": list(added), "removed": list(removed)},
            surface_signature,
            judgment,
        ))
    if int(delta.get("changed_cell_count") or 0) == 0:
        observations.append(_invariant(
            "NO_VISIBLE_EFFECT_IN_STATE",
            action_id,
            (),
            {},
            surface_signature,
            judgment,
        ))
    return observations


def _goal_operator(
    objective: SemanticObjective,
    relation_ids: tuple[str, ...],
    relations: dict[str, Any],
) -> GoalOperator:
    kind = objective.kind
    relation_types = {relations[item].relation_type for item in relation_ids if item in relations}
    if kind == "match_or_overlap":
        return GoalOperator.OVERLAP
    if kind == "relative_arrangement":
        return GoalOperator.ALIGN if relation_types.intersection({"aligned_row", "aligned_col"}) else GoalOperator.MOVE_TOWARD
    if kind == "containment":
        return GoalOperator.CONTAIN
    if kind == "connection":
        return GoalOperator.EXTEND_LINE if "line_continuation" in relation_types else GoalOperator.CONNECT
    if kind == "pattern_or_state":
        return GoalOperator.COMPLETE_PATTERN
    if kind == "select_or_activate":
        return GoalOperator.ACTIVATE
    if kind == "surface_change":
        return GoalOperator.CHANGE_ACTION_SURFACE
    return GoalOperator.OTHER


def _metric_spec(
    objective: SemanticObjective,
    operator: GoalOperator,
    sources: tuple[str, ...],
    references: tuple[str, ...],
) -> GoalMetricSpec | None:
    if not sources or not references:
        return None
    if operator in {GoalOperator.OVERLAP, GoalOperator.MOVE_TOWARD, GoalOperator.ALIGN}:
        return GoalMetricSpec("centroid_distance", MetricDirection.MINIMIZE, epsilon=1e-6)
    if operator in {GoalOperator.CONNECT, GoalOperator.BRIDGE_GAP, GoalOperator.EXTEND_LINE}:
        return GoalMetricSpec("gap_distance", MetricDirection.MINIMIZE, epsilon=1e-6)
    if operator is GoalOperator.CONTAIN:
        return GoalMetricSpec("containment_outside_distance", MetricDirection.MINIMIZE, epsilon=1e-6)
    if operator in {GoalOperator.COMPLETE_PATTERN, GoalOperator.MATCH_STATE, GoalOperator.MATCH_GEOMETRY}:
        return GoalMetricSpec("palette_shape_mismatch", MetricDirection.MINIMIZE, epsilon=1e-6)
    return None


def _compute_metric(
    metric_name: str,
    source_ids: tuple[str, ...],
    reference_ids: tuple[str, ...],
    snapshot: ARGALiteSnapshot,
) -> float | None:
    objects = {obj.object_id: obj for obj in snapshot.objects}
    sources = [objects[item] for item in source_ids if item in objects]
    references = [objects[item] for item in reference_ids if item in objects]
    if not sources or not references:
        return None

    def pair_value(source: Any, reference: Any) -> float:
        if metric_name == "centroid_distance":
            return hypot(
                source.centroid_rc[0] - reference.centroid_rc[0],
                source.centroid_rc[1] - reference.centroid_rc[1],
            )
        if metric_name in {"gap_distance", "bbox_gap", "line_endpoint_distance"}:
            return _bbox_gap(source.bbox_rc, reference.bbox_rc)
        if metric_name == "containment_outside_distance":
            return _containment_outside_distance(reference.bbox_rc, source.bbox_rc)
        if metric_name == "palette_shape_mismatch":
            shape_penalty = 0.0 if source.shape_signature == reference.shape_signature else 1.0
            return shape_penalty + _histogram_distance(source.color_histogram, reference.color_histogram)
        return float("inf")

    values = [min(pair_value(source, reference) for reference in references) for source in sources]
    finite = [item for item in values if item != float("inf")]
    return sum(finite) / len(finite) if finite else None


def _aggregate_mechanics(judgments: tuple[Judgment, ...]) -> MechanicResult:
    if any(item.mechanic_result is MechanicResult.MISMATCH for item in judgments):
        return MechanicResult.MISMATCH
    if any(item.mechanic_result is MechanicResult.MATCH for item in judgments):
        return MechanicResult.MATCH
    return MechanicResult.UNKNOWN


def _invariant(
    predicate: str,
    action_id: str,
    subjects: tuple[str, ...],
    parameters: dict[str, Any],
    surface_signature: str,
    judgment: Judgment,
) -> dict[str, Any]:
    base_key = stable_hash((predicate, action_id, subjects, surface_signature), "invbase_")
    observation_key = stable_hash((base_key, parameters), "invobs_")
    return {
        "base_key": base_key,
        "observation_key": observation_key,
        "predicate": predicate,
        "action_id": action_id,
        "subject_object_ids": list(subjects),
        "parameters": parameters,
        "control_context": surface_signature,
        "authority": EvidenceAuthority.OFFICIAL_OBSERVATION.value,
        "status": EvidenceStatus.OBSERVED_ONCE.value,
        "evidence_refs": [judgment.before_hash, judgment.after_hash],
    }


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return hypot(dr, dc)


def _containment_outside_distance(
    container: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> float:
    cr0, cc0, cr1, cc1 = container
    ir0, ic0, ir1, ic1 = inner
    return float(
        max(0, cr0 - ir0)
        + max(0, cc0 - ic0)
        + max(0, ir1 - cr1)
        + max(0, ic1 - cc1)
    )


def _histogram_distance(first: dict[int, int], second: dict[int, int]) -> float:
    first_total = max(1, sum(first.values()))
    second_total = max(1, sum(second.values()))
    colors = set(first).union(second)
    return 0.5 * sum(
        abs(first.get(color, 0) / first_total - second.get(color, 0) / second_total)
        for color in colors
    )


def _vector_equal(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return abs(first[0] - second[0]) <= 1e-6 and abs(first[1] - second[1]) <= 1e-6


def _opposite_axis(first: tuple[float, float], second: tuple[float, float]) -> bool:
    horizontal = abs(first[1]) <= 1e-6 and abs(second[1]) <= 1e-6 and abs(first[0] + second[0]) <= 1e-6
    vertical = abs(first[0]) <= 1e-6 and abs(second[0]) <= 1e-6 and abs(first[1] + second[1]) <= 1e-6
    return (horizontal or vertical) and (abs(first[0]) + abs(first[1]) > 1e-6)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def _round(value: float) -> float:
    return round(float(value), 4)
