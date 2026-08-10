from __future__ import annotations

from typing import Any

from .memory import is_action_research_source, is_coordinate_research_source
from .observe import stable_hash


def build_verifier_packet(
    *,
    snapshot: Any,
    memory: Any,
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    aliases: dict[str, dict[str, str]],
    component_graph: dict[str, Any] | None,
    allowed_action_ids: list[str],
    role: str,
) -> dict[str, Any]:
    """Deterministic dual-view packet for offline trajectory verification.

    Shares the same planning-object set and aliases as the Qwen observation packet.
    Component graph is geometry evidence only and is never a planning ID source.
    """
    object_aliases = aliases.get("object_real_to_alias", {})
    planning_objects = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("id") or "")
        planning_objects.append({
            "id": object_id,
            "internal_id": _reverse_alias(object_aliases, object_id),
            "role": "PLANNING_OBJECT",
            "bbox_xyxy": item.get("bbox_xyxy"),
            "centroid_xy": item.get("centroid_xy"),
            "area": item.get("area"),
            "palette_histogram": item.get("palette_histogram"),
            "shape_signature": item.get("shape_signature"),
            "geometry_class": item.get("geometry_class"),
            "edge_contacts": item.get("edge_contacts") or [],
        })

    geometry_components = []
    if isinstance(component_graph, dict):
        for component in component_graph.get("components") or []:
            if not isinstance(component, dict):
                continue
            geometry_components.append({
                "id": component.get("id"),
                "role": "GEOMETRY_EVIDENCE_ONLY",
                "color": component.get("color"),
                "area": component.get("area"),
                "bbox_xyxy": component.get("bbox_xyxy"),
                "object_refs": list(component.get("object_refs") or []),
                "shape_hash": component.get("shape_hash"),
            })

    return {
        "schema_version": "v9.0.verifier_packet",
        "dual_view_contract": {
            "planning_ids": "tracked_objects_only",
            "geometry_evidence": "component_graph_same_color_4connected",
            "shared_with_qwen": (
                "object aliases, bbox_xyxy, full_grid_hex_rows, probe summaries, "
                "allowed_action_ids, coordinate candidates"
            ),
            "not_planning_ids": ("component_ids", "geometry-only regions without object_refs"),
        },
        "state": {
            "game_id": getattr(snapshot, "game_id", None),
            "level_index": getattr(snapshot, "level_index", None),
            "step_index": getattr(snapshot, "step_index", None),
            "grid_hash": getattr(snapshot, "grid_hash", None),
            "semantic_state_signature": getattr(snapshot, "semantic_state_signature", None),
            "available_actions": list(getattr(snapshot, "available_actions", ()) or ()),
            "planning_action_ids": list(
                getattr(snapshot, "planning_action_ids", None)
                or getattr(snapshot, "available_actions", ())
                or ()
            ),
            "coordinate_action_ids": list(getattr(snapshot, "coordinate_action_ids", ()) or ()),
            "undo_action_ids": list(getattr(snapshot, "undo_action_ids", ()) or ()),
            "role": role,
        },
        "full_grid_hex_rows": list(getattr(snapshot, "full_grid_hex_rows", ()) or ()),
        "grid_shape_hw": [
            int(getattr(snapshot, "height", 0) or 0),
            int(getattr(snapshot, "width", 0) or 0),
        ],
        "coordinate_order": "x=column,y=row; origin=top_left",
        "planning_objects": planning_objects,
        "relations": [
            {
                "id": item.get("id"),
                "type": item.get("type") or item.get("relation_type"),
                "object_ids": item.get("object_ids")
                or [item.get("source_object_id"), item.get("target_object_id")],
                "metric_name": item.get("metric_name"),
                "metric_value": item.get("metric_value"),
            }
            for item in relations
            if isinstance(item, dict)
        ],
        "coordinate_candidates": [
            {
                "id": item.get("id"),
                "object_id": item.get("object_id"),
                "relation_id": item.get("relation_id"),
                "location_xy": item.get("location_xy"),
                "cell_value": item.get("cell_value"),
            }
            for item in candidates
            if isinstance(item, dict)
        ],
        "geometry_evidence": {
            "component_graph_included": component_graph is not None,
            "components": geometry_components,
            "note": (
                "Components partition every cell into same-color 4-connected regions. "
                "They are geometric evidence. Trajectory planning may reference only "
                "planning_objects IDs and linked object_refs, never bare component IDs."
            ),
        },
        "action_probe_summaries": _probe_summaries(memory, snapshot),
        "execution_constraints": {
            "allowed_action_ids": sorted(str(v) for v in allowed_action_ids),
            "allowed_object_ids": sorted(str(item["id"]) for item in planning_objects if item.get("id")),
            "allowed_relation_ids": sorted(
                str(item.get("id")) for item in relations if isinstance(item, dict) and item.get("id")
            ),
            "allowed_coordinate_candidate_ids": sorted(
                str(item.get("id")) for item in candidates if isinstance(item, dict) and item.get("id")
            ),
        },
        "packet_fingerprint": stable_hash(
            (
                getattr(snapshot, "grid_hash", None),
                [item.get("id") for item in planning_objects],
                [item.get("id") for item in relations if isinstance(item, dict)],
                sorted(str(v) for v in allowed_action_ids),
            ),
            "vfy_",
        ),
    }


def _reverse_alias(object_aliases: dict[str, str], alias: str) -> str | None:
    for real, value in object_aliases.items():
        if value == alias:
            return real
    return None


def _probe_summaries(memory: Any, snapshot: Any) -> list[dict[str, Any]]:
    level_index = int(getattr(snapshot, "level_index", -1))
    out: list[dict[str, Any]] = []
    for record in getattr(memory, "action_memory_records", []) or []:
        if not isinstance(record, dict):
            continue
        record_level = record.get("level_index_before")
        if record_level is None:
            record_level = record.get("level_index", -1)
        try:
            if int(record_level) != level_index:
                continue
        except (TypeError, ValueError):
            continue
        source = str(record.get("source") or "")
        if not (is_action_research_source(source) or is_coordinate_research_source(source)):
            # Keep non-research effects too: offline verifier needs known transitions.
            pass
        item = {
            "step_index": record.get("step_index"),
            "action_id": record.get("action_id"),
            "source": source,
            "coordinate_candidate_id": record.get("coordinate_candidate_id"),
            "coordinate_xy": record.get("coordinate_xy"),
            "grid_hash_before": record.get("grid_hash_before"),
            "grid_hash_after": record.get("grid_hash_after"),
            "effect_outcome": record.get("effect_outcome"),
            "changed_cell_count": record.get("changed_cell_count"),
            "visible_effect_observed": bool(int(record.get("changed_cell_count") or 0) > 0),
            "action_surface_added": record.get("planning_action_surface_added")
            or record.get("action_surface_added")
            or [],
            "action_surface_removed": record.get("planning_action_surface_removed")
            or record.get("action_surface_removed")
            or [],
            "is_research_probe": is_action_research_source(source) or is_coordinate_research_source(source),
        }
        out.append({key: value for key, value in item.items() if value not in (None, [], {})})
    # Prefer recent evidence; keep a bounded window for the offline contour.
    return out[-64:]
