"""Canonical planning identity for one dual-view cycle.

PlanningSet is the single vocabulary shared by:
  - Qwen visual packet (PNG / annotated labels / object_layer ids)
  - offline verifier_packet (hex + planning_objects)
  - HypothesisItem targets / Contour B repair

GameSession owns the cached instance; builders are pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PlanningSet:
    snapshot_id: str
    grid_hash: str
    full_grid_hex_rows: tuple[str, ...]
    object_ids: frozenset[str]
    relation_ids: frozenset[str]
    allowed_action_ids: frozenset[str]
    allowed_coordinate_candidate_ids: frozenset[str]
    object_real_to_alias: Mapping[str, str] = field(default_factory=dict)
    object_alias_to_real: Mapping[str, str] = field(default_factory=dict)
    objects: tuple[dict[str, Any], ...] = ()
    relations: tuple[dict[str, Any], ...] = ()
    coordinate_candidates: tuple[dict[str, Any], ...] = ()
    component_graph: dict[str, Any] | None = None

    def contains_object(self, object_id: str | None) -> bool:
        if not object_id:
            return False
        if object_id in self.object_ids:
            return True
        real = self.object_alias_to_real.get(str(object_id))
        return bool(real and real in self.object_ids)

    def contains_action(self, action_id: str | None) -> bool:
        return bool(action_id) and str(action_id) in self.allowed_action_ids

    def contains_coordinate_candidate(self, candidate_id: str | None) -> bool:
        if candidate_id is None or candidate_id == "":
            return True
        if not self.allowed_coordinate_candidate_ids:
            return True
        return str(candidate_id) in self.allowed_coordinate_candidate_ids


def build_planning_set_from_snapshot(snapshot: Any, *, aliases: Mapping[str, Mapping[str, str]] | None = None) -> PlanningSet:
    """Build PlanningSet directly from ARGALiteSnapshot (pre-packet path)."""
    objects = tuple(
        {
            "id": str(getattr(obj, "object_id", None) or getattr(obj, "id", "")),
            "bbox_xyxy": list(getattr(obj, "bbox_xyxy", None) or []),
            "centroid_xy": list(getattr(obj, "centroid_xy", None) or getattr(obj, "centroid_rc", ())[::-1] or []),
        }
        for obj in (getattr(snapshot, "objects", ()) or ())
        if str(getattr(obj, "object_id", None) or getattr(obj, "id", "") or "")
    )
    object_ids = frozenset(item["id"] for item in objects if item["id"])
    relation_ids = frozenset(
        str(getattr(rel, "relation_id", None) or getattr(rel, "id", "") or "")
        for rel in (getattr(snapshot, "relations", ()) or ())
        if str(getattr(rel, "relation_id", None) or getattr(rel, "id", "") or "")
    )
    candidates = tuple(
        {
            "id": str(getattr(c, "candidate_id", None) or getattr(c, "id", "") or ""),
        }
        for c in (getattr(snapshot, "coordinate_targets", ()) or ())
        if str(getattr(c, "candidate_id", None) or getattr(c, "id", "") or "")
    )
    candidate_ids = frozenset(item["id"] for item in candidates if item["id"])
    allowed_actions = frozenset(str(a) for a in (getattr(snapshot, "available_actions", ()) or ()))
    alias_maps = aliases or {}
    real_to_alias = dict(alias_maps.get("object_real_to_alias") or {})
    alias_to_real = dict(alias_maps.get("object_alias_to_real") or {})
    if not real_to_alias and objects:
        for index, item in enumerate(objects):
            alias = f"obj{index}"
            real_to_alias[item["id"]] = alias
            alias_to_real[alias] = item["id"]
    hex_rows = tuple(str(row) for row in (getattr(snapshot, "full_grid_hex_rows", ()) or ()))
    return PlanningSet(
        snapshot_id=str(getattr(snapshot, "snapshot_id", "") or ""),
        grid_hash=str(getattr(snapshot, "grid_hash", "") or ""),
        full_grid_hex_rows=hex_rows,
        object_ids=object_ids,
        relation_ids=relation_ids,
        allowed_action_ids=allowed_actions,
        allowed_coordinate_candidate_ids=candidate_ids,
        object_real_to_alias=real_to_alias,
        object_alias_to_real=alias_to_real,
        objects=objects,
        relations=tuple(
            {"id": rid}
            for rid in sorted(relation_ids)
        ),
        coordinate_candidates=candidates,
        component_graph=getattr(snapshot, "component_graph", None),
    )


def build_planning_set_from_packet(packet: Mapping[str, Any], snapshot: Any) -> PlanningSet:
    """Rebuild PlanningSet from a Qwen packet so dual-view ids stay aligned."""
    object_layer = packet.get("object_layer") if isinstance(packet.get("object_layer"), dict) else {}
    objects_raw = object_layer.get("objects") if isinstance(object_layer.get("objects"), list) else []
    objects = tuple(item for item in objects_raw if isinstance(item, dict) and item.get("id"))
    object_ids = frozenset(str(item["id"]) for item in objects)
    relations_raw = object_layer.get("relations") if isinstance(object_layer.get("relations"), list) else []
    relation_ids = frozenset(
        str(item.get("id"))
        for item in relations_raw
        if isinstance(item, dict) and item.get("id")
    )
    action_space = packet.get("action_space") if isinstance(packet.get("action_space"), dict) else {}
    candidates_raw = action_space.get("coordinate_candidates") if isinstance(action_space.get("coordinate_candidates"), list) else []
    candidates = tuple(item for item in candidates_raw if isinstance(item, dict) and item.get("id"))
    candidate_ids = frozenset(str(item["id"]) for item in candidates)
    constraints = {}
    verifier = packet.get("verifier_packet") if isinstance(packet.get("verifier_packet"), dict) else {}
    if isinstance(verifier.get("execution_constraints"), dict):
        constraints = verifier["execution_constraints"]
    allowed_from_packet = constraints.get("allowed_action_ids")
    if isinstance(allowed_from_packet, (list, tuple, set)):
        allowed_actions = frozenset(str(a) for a in allowed_from_packet)
    else:
        allowed_actions = frozenset(str(a) for a in (getattr(snapshot, "available_actions", ()) or ()))
    allowed_cands = constraints.get("allowed_coordinate_candidate_ids")
    if isinstance(allowed_cands, (list, tuple, set)) and allowed_cands:
        candidate_ids = frozenset(str(c) for c in allowed_cands)
    aliases = {}
    # Prefer explicit dual-view alias maps if present on packet memory/helpers.
    state = packet.get("state") if isinstance(packet.get("state"), dict) else {}
    grid_hash = str(state.get("grid_hash") or getattr(snapshot, "grid_hash", "") or "")
    hex_rows = tuple(str(row) for row in (verifier.get("full_grid_hex_rows") or getattr(snapshot, "full_grid_hex_rows", ()) or ()))
    real_to_alias: dict[str, str] = {}
    alias_to_real: dict[str, str] = {}
    for index, item in enumerate(objects):
        oid = str(item["id"])
        alias = str(item.get("alias") or f"obj{index}")
        real_to_alias[oid] = alias
        alias_to_real[alias] = oid
    return PlanningSet(
        snapshot_id=str(getattr(snapshot, "snapshot_id", "") or ""),
        grid_hash=grid_hash,
        full_grid_hex_rows=hex_rows,
        object_ids=object_ids,
        relation_ids=relation_ids,
        allowed_action_ids=allowed_actions,
        allowed_coordinate_candidate_ids=candidate_ids,
        object_real_to_alias=real_to_alias,
        object_alias_to_real=alias_to_real,
        objects=objects,
        relations=tuple(item for item in relations_raw if isinstance(item, dict)),
        coordinate_candidates=candidates,
        component_graph=object_layer.get("component_graph") if isinstance(object_layer.get("component_graph"), dict) else None,
    )


def assert_dual_view_identity(packet: Mapping[str, Any], planning_set: PlanningSet) -> list[str]:
    """Return list of identity violations (empty means OK). Does not raise."""
    violations: list[str] = []
    verifier = packet.get("verifier_packet") if isinstance(packet.get("verifier_packet"), dict) else {}
    v_objects = verifier.get("planning_objects") if isinstance(verifier.get("planning_objects"), list) else []
    v_ids = {str(item.get("id")) for item in v_objects if isinstance(item, dict) and item.get("id")}
    if v_ids and v_ids != set(planning_set.object_ids):
        violations.append(
            f"planning_object_id_mismatch packet={sorted(planning_set.object_ids)} verifier={sorted(v_ids)}"
        )
    state = packet.get("state") if isinstance(packet.get("state"), dict) else {}
    packet_hash = state.get("grid_hash")
    if packet_hash and planning_set.grid_hash and str(packet_hash) != planning_set.grid_hash:
        violations.append(f"grid_hash_mismatch packet={packet_hash} planning_set={planning_set.grid_hash}")
    v_state = verifier.get("state") if isinstance(verifier.get("state"), dict) else {}
    v_hash = v_state.get("grid_hash")
    if v_hash and planning_set.grid_hash and str(v_hash) != planning_set.grid_hash:
        violations.append(f"verifier_grid_hash_mismatch verifier={v_hash} planning_set={planning_set.grid_hash}")
    return violations
