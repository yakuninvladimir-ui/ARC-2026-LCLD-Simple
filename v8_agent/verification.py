from __future__ import annotations

from math import hypot
from typing import Iterable

from .observe import stable_hash
from .relations import relation_by_id
from .types import (
    ARGALiteSnapshot,
    SemanticQuestionType,
    TestStep,
    VerificationContract,
    VerificationContractKind,
)


class VerificationBinder:
    """Ground one bounded test step into a fixed verifier contract.

    This intentionally is not a general DSL compiler. It performs ID checks,
    relation endpoint recovery, contract selection, and before-metric capture.
    """

    def bind(self, step: TestStep, snapshot: ARGALiteSnapshot, hypothesis_id: str | None = None) -> VerificationContract | None:
        objects = {o.object_id: o for o in snapshot.objects}
        relations = {r.relation_id: r for r in snapshot.relations}
        target_object_ids: tuple[str, ...] = ()
        target_relation_ids: tuple[str, ...] = ()
        region = None
        metric_name = None
        before_metric = None
        target_signature = None
        text = f"{step.kind} {step.expected_observation or ''}".lower()
        expected_type = _expected_type_from_text(text)
        step_object_ids = tuple(dict.fromkeys([*getattr(step, "target_object_ids", ()), *([step.target_object_id] if step.target_object_id else [])]))
        step_relation_ids = tuple(dict.fromkeys([*getattr(step, "target_relation_ids", ()), *([step.target_relation_id] if step.target_relation_id else [])]))
        prefer_object_contract = expected_type == "object_move" and bool(step_object_ids)

        if step_relation_ids and not prefer_object_contract:
            relation = relations.get(step_relation_ids[0])
            if relation is None:
                return None
            target_relation_ids = (relation.relation_id,)
            target_object_ids = (relation.a, relation.b)
            metric_name = relation.metric_name
            before_metric = relation.metric_value
            target_signature = relation.relation_signature or relation.relation_id
        elif step_object_ids:
            resolved = tuple(object_id for object_id in step_object_ids if object_id in objects)
            if not resolved:
                return None
            target_object_ids = resolved
            first = objects[resolved[0]]
            region = first.bbox_rc
            target_signature = stable_hash(tuple(objects[object_id].stable_hash for object_id in resolved), "targetobjs_")

        kind = _select_contract_kind(step, has_relation=bool(target_relation_ids), has_object=bool(target_object_ids))
        candidate = None
        if step.coordinate_candidate_id:
            candidate = next((c for c in snapshot.coordinate_targets if c.candidate_id == step.coordinate_candidate_id), None)
            if candidate is None:
                return None
            if not target_object_ids and candidate.object_id:
                obj = objects.get(candidate.object_id)
                if obj is not None:
                    target_object_ids = (obj.object_id,)
                    region = obj.bbox_rc
                    target_signature = candidate.target_signature or obj.stable_hash
            if not target_relation_ids and candidate.relation_id:
                relation = relations.get(candidate.relation_id)
                if relation is not None:
                    target_relation_ids = (relation.relation_id,)
                    target_object_ids = (relation.a, relation.b)
                    metric_name = relation.metric_name
                    before_metric = relation.metric_value
                    target_signature = candidate.target_signature or relation.relation_signature

        question_type = _question_type(kind)
        question_id = stable_hash((question_type.value, step.action_id, target_signature or "global"), "q_")
        contract_id = stable_hash((
            hypothesis_id,
            step.kind,
            step.action_id,
            target_object_ids,
            target_relation_ids,
            step.coordinate_candidate_id,
            kind.value,
            metric_name,
        ), "vc_")
        return VerificationContract(
            contract_id=contract_id,
            kind=kind,
            target_object_ids=target_object_ids,
            target_relation_ids=target_relation_ids,
            target_coordinate_candidate_id=step.coordinate_candidate_id,
            target_region_rc=region,
            metric_name=metric_name,
            before_metric=before_metric,
            expected_effect=step.expected_observation,
            question_id=question_id,
            question_type=question_type,
            target_signature=target_signature or f"action:{step.action_id}",
        )


def relation_metric_after(contract: VerificationContract, before: ARGALiteSnapshot, after: ARGALiteSnapshot) -> float | None:
    if not contract.target_relation_ids:
        return None
    relation_id = contract.target_relation_ids[0]
    existing = relation_by_id(after, relation_id)
    if existing is not None:
        return existing.metric_value
    before_relation = relation_by_id(before, relation_id)
    if before_relation is None:
        return None
    objects = {o.object_id: o for o in after.objects}
    a = objects.get(before_relation.a)
    b = objects.get(before_relation.b)
    if a is None or b is None:
        return None
    metric_name = contract.metric_name or before_relation.metric_name
    if metric_name == "centroid_distance":
        return hypot(a.centroid_rc[0] - b.centroid_rc[0], a.centroid_rc[1] - b.centroid_rc[1])
    if metric_name in {"gap_distance", "bbox_gap", "line_endpoint_distance"}:
        return _bbox_gap(a.bbox_rc, b.bbox_rc)
    if metric_name == "delta_col":
        return abs(a.centroid_rc[1] - b.centroid_rc[1])
    if metric_name == "delta_row":
        return abs(a.centroid_rc[0] - b.centroid_rc[0])
    if metric_name == "containment_outside_distance":
        return _containment_outside_distance(a.bbox_rc, b.bbox_rc)
    return None


def object_displacement(contract: VerificationContract, before: ARGALiteSnapshot, after: ARGALiteSnapshot) -> float | None:
    if not contract.target_object_ids:
        return None
    before_map = {o.object_id: o for o in before.objects}
    after_map = {o.object_id: o for o in after.objects}
    displacements: list[float] = []
    for obj_id in contract.target_object_ids:
        b = before_map.get(obj_id)
        a = after_map.get(obj_id)
        if b is None or a is None:
            continue
        displacements.append(hypot(a.centroid_rc[0] - b.centroid_rc[0], a.centroid_rc[1] - b.centroid_rc[1]))
    if not displacements:
        return None
    return max(displacements)


def target_regions(contract: VerificationContract, before: ARGALiteSnapshot, after: ARGALiteSnapshot) -> tuple[tuple[int, int, int, int], ...]:
    regions: list[tuple[int, int, int, int]] = []
    if contract.target_region_rc is not None:
        regions.append(contract.target_region_rc)
    for snapshot in (before, after):
        by_id = {o.object_id: o for o in snapshot.objects}
        for object_id in contract.target_object_ids:
            obj = by_id.get(object_id)
            if obj is not None and obj.bbox_rc not in regions:
                regions.append(obj.bbox_rc)
    return tuple(regions)


def changed_cells_in_regions(cells: Iterable[tuple[int, int]], regions: Iterable[tuple[int, int, int, int]]) -> tuple[int, int]:
    regions_tuple = tuple(regions)
    inside = 0
    outside = 0
    for x, y in cells:
        if any(r0 <= y <= r1 and c0 <= x <= c1 for r0, c0, r1, c1 in regions_tuple):
            inside += 1
        else:
            outside += 1
    return inside, outside


def _select_contract_kind(step: TestStep, *, has_relation: bool, has_object: bool) -> VerificationContractKind:
    explicit = str(step.contract_kind or "").upper()
    try:
        return VerificationContractKind(explicit)
    except ValueError:
        pass
    text = f"{step.kind} {step.expected_observation or ''}".lower()
    expected_type = _expected_type_from_text(text)
    if expected_type == "target_change":
        return VerificationContractKind.LOCAL_TARGET_CHANGE
    if expected_type == "object_move":
        return VerificationContractKind.OBJECT_DISPLACEMENT
    if expected_type == "relation_improvement":
        return VerificationContractKind.RELATION_ERROR_DECREASE
    if expected_type == "action_surface_change":
        return VerificationContractKind.ACTION_SURFACE_CHANGE
    if expected_type == "score_or_terminal":
        return VerificationContractKind.SCORE_OR_TERMINAL
    if has_relation:
        return VerificationContractKind.RELATION_ERROR_DECREASE
    if "score" in text or "terminal" in text or "level" in text or "win" in text:
        return VerificationContractKind.SCORE_OR_TERMINAL
    if "available action" in text or "action surface" in text or "unlock" in text or "toggle" in text:
        return VerificationContractKind.ACTION_SURFACE_CHANGE
    if has_object and any(token in text for token in ("move", "displace", "shift", "translate", "controll")):
        return VerificationContractKind.OBJECT_DISPLACEMENT
    if has_object:
        return VerificationContractKind.LOCAL_TARGET_CHANGE
    if any(token in text for token in ("no-op", "noop", "no effect", "does nothing")):
        return VerificationContractKind.NO_OP_TEST
    return VerificationContractKind.ACTION_EFFECT_DISCOVERY


def _expected_type_from_text(text: str) -> str | None:
    marker = "expected_type="
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1]
    token = tail.split(";", 1)[0].strip().lower()
    return token or None


def _question_type(kind: VerificationContractKind) -> SemanticQuestionType:
    if kind is VerificationContractKind.RELATION_ERROR_DECREASE:
        return SemanticQuestionType.RELATION_RELEVANCE
    if kind is VerificationContractKind.OBJECT_DISPLACEMENT:
        return SemanticQuestionType.CONTROLLABILITY
    if kind is VerificationContractKind.LOCAL_TARGET_CHANGE:
        return SemanticQuestionType.AFFORDANCE
    if kind is VerificationContractKind.ACTION_SURFACE_CHANGE:
        return SemanticQuestionType.ACTION_SURFACE_CHANGE
    if kind is VerificationContractKind.SCORE_OR_TERMINAL:
        return SemanticQuestionType.TERMINAL_PROGRESS
    return SemanticQuestionType.ACTION_EFFECT


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return hypot(dr, dc)


def _containment_outside_distance(container: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> float:
    cr0, cc0, cr1, cc1 = container
    ir0, ic0, ir1, ic1 = inner
    return float(max(0, cr0 - ir0) + max(0, cc0 - ic0) + max(0, ir1 - cr1) + max(0, ic1 - cc1))
