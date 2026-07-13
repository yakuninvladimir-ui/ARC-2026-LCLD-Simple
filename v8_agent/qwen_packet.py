from __future__ import annotations

from collections import deque
from typing import Any

from .component_graph import build_component_graph
from .config import V8Config
from .memory import ACTION_EFFECT_PROBE_IDS
from .observe import stable_hash
from .types import ARGALiteSnapshot, MemoryEvent, QwenRole, Relevance


class QwenPacketNotReady(RuntimeError):
    pass


class QwenPacketBuilder:
    def build_semantic_packet(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", bank: "HypothesisBank", role: QwenRole, config: V8Config) -> dict[str, Any]:
        if not qwen_planning_ready(snapshot, memory, role):
            raise QwenPacketNotReady("Qwen planning requires probed action mechanics for the current planning surface")
        return self._build_packet(snapshot, memory, bank, role, config)

    def build_coordinate_packet(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", bank: "HypothesisBank", config: V8Config) -> dict[str, Any]:
        if not snapshot.coordinate_action_ids or not snapshot.coordinate_targets:
            raise QwenPacketNotReady("coordinate planning requires a coordinate action and described candidates")
        return self._build_packet(snapshot, memory, bank, QwenRole.COORDINATE, config)

    def _build_packet(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", bank: "HypothesisBank", role: QwenRole, config: V8Config) -> dict[str, Any]:
        nested_component_view_ids = _nested_compact_component_view_ids(snapshot)
        raw_objects = _objects(snapshot, config, excluded_object_ids=nested_component_view_ids)
        allowed_action_ids = _allowed_action_ids(snapshot, role, memory)
        evidence_action_ids = list(allowed_action_ids)
        if role is QwenRole.COORDINATE:
            evidence_action_ids = list(dict.fromkeys([
                *_allowed_action_ids(snapshot, QwenRole.PRIMARY, memory),
                *snapshot.coordinate_action_ids,
            ]))
        all_object_ids = {item["id"] for item in raw_objects}
        all_relations = _relations(snapshot, all_object_ids, config)
        all_relation_ids = {item["id"] for item in all_relations}
        all_candidates = _coordinate_candidates(snapshot, all_object_ids, all_relation_ids, config)
        control_section = _control_model_section(memory, snapshot, evidence_action_ids, all_object_ids, config)
        raw_control_model = control_section["control_model"]
        raw_objects = _annotate_object_motion(raw_objects, raw_control_model)
        all_geometry_groups = _annotate_object_geometry(raw_objects)
        focus_ids = _focus_object_ids(raw_objects, all_geometry_groups, raw_control_model, all_candidates, role, config)
        focus_objects = [item for item in raw_objects if str(item.get("id")) in focus_ids]
        focus_relations = _focus_relations(all_relations, focus_ids, config)
        focus_relation_ids = {str(item.get("id")) for item in focus_relations}
        coordinate_execution_allowed = bool(set(allowed_action_ids) & set(snapshot.coordinate_action_ids))
        focus_candidates = [
            item for item in all_candidates
            if (item.get("object_id") is None or str(item.get("object_id")) in focus_ids)
            and (item.get("relation_id") is None or str(item.get("relation_id")) in focus_relation_ids)
        ] if role is QwenRole.COORDINATE or coordinate_execution_allowed else []
        focus_geometry_groups = _focus_geometry_groups(all_geometry_groups, focus_ids, config)
        aliases = build_model_aliases_from_records(raw_objects, all_relations, all_candidates, all_geometry_groups)
        objects = _alias_objects(focus_objects, aliases)
        _prune_aliased_object_geometry_refs(objects)
        relations = _alias_relations(focus_relations, aliases)
        candidates = _alias_candidates(focus_candidates, aliases)
        geometry_groups = _alias_geometry_groups(focus_geometry_groups, aliases)
        control_model = _alias_control_model(_focus_control_model(raw_control_model, focus_ids), aliases)
        control_groups = _control_groups(objects, control_model)
        surface_transitions = _alias_recent_evidence(
            _action_surface_transitions(memory, snapshot, config, focus_ids),
            aliases,
        )
        control_state_candidates = _control_state_transition_candidates(objects, control_groups, surface_transitions)
        _annotate_control_context(control_model, control_groups, control_state_candidates)
        priority_facts = _priority_facts(objects, geometry_groups, control_model)
        repeated_groups = _repeated_object_groups(objects, geometry_groups)
        component_graph = _component_graph_for_packet(snapshot, focus_ids, aliases, config)
        actions = _actions(snapshot, role, allowed_action_ids)
        candidate_ids = {item["id"] for item in candidates}
        internal_candidate_ids = {str(item["id"]) for item in focus_candidates}
        packet = {
            "schema_version": "v8.7.semantic_observation",
            "state": _state(snapshot),
            "scene": {
                "object_segmentation": {
                    "planning_object_contract": "Compact multicolor connected regions are canonical planning objects. Strictly nested same-color views are component detail, not peer gameplay objects.",
                    "suppressed_nested_component_view_count": len(nested_component_view_ids),
                    "suppressed_views_retained_in_component_graph": bool(config.include_component_graph_in_qwen_packet),
                },
                "priority_facts_not_goals": priority_facts,
                "control_groups": control_groups,
                "control_state_transition_candidates": control_state_candidates,
                "repeated_object_groups": repeated_groups,
                **({"component_graph": component_graph} if component_graph is not None else {}),
                "objects": objects,
                "exact_geometry_groups": geometry_groups,
                "relations": relations,
                "regions": _regions(snapshot),
                "coordinate_candidates": candidates,
            },
            "current_frame": _grid(snapshot, config),
            "action_model": control_model,
            "action_surface": {
                "actions": actions,
                "availability_semantics_status": "OBSERVED_LEGALITY_ONLY_NOT_INTRINSIC_ACTION_MEANING",
                "current_available_action_ids": list(snapshot.available_actions),
                "current_planning_action_ids": list(snapshot.planning_action_ids or ()),
                "undo_action_ids": list(snapshot.undo_action_ids or ()),
                "possible_action_ids": list(snapshot.possible_actions or snapshot.available_actions),
                "observed_transitions": surface_transitions,
            },
            "memory": {
                "recent_transition": _alias_recent_transition(_recent_transition(memory, focus_ids, focus_relation_ids), aliases),
                "recent_evidence": _alias_recent_evidence(_recent_evidence(memory, bank, snapshot, focus_ids, focus_relation_ids, internal_candidate_ids, config), aliases),
                "level_experience": _alias_recent_evidence(_level_experience(memory, snapshot, config), aliases),
                "level_attempts": _alias_recent_evidence(_level_attempt_context(memory, snapshot, config), aliases),
            },
            "execution_constraints": {
                "max_plan_steps": int(config.max_qwen_trajectory_steps),
                "allowed_action_ids": sorted(allowed_action_ids),
                "allowed_object_ids": sorted(item["id"] for item in objects),
                "allowed_relation_ids": sorted(item["id"] for item in relations),
                "allowed_coordinate_candidate_ids": sorted(candidate_ids),
                "raw_coordinates_allowed": bool(config.allow_qwen_raw_coordinates),
            },
        }
        _validate_packet_references(packet)
        return packet


def qwen_planning_ready(snapshot: ARGALiteSnapshot, memory: "GameMemory", role: QwenRole = QwenRole.PRIMARY) -> bool:
    if role is QwenRole.COORDINATE:
        return bool(snapshot.coordinate_action_ids and snapshot.coordinate_targets)
    required_actions = tuple(
        action_id
        for action_id in _planning_action_ids(snapshot)
        if action_id in ACTION_EFFECT_PROBE_IDS and action_id not in set(snapshot.coordinate_action_ids)
    )
    if not required_actions:
        return False
    known = {
        record.action_id
        for record in getattr(memory, "action_effects", {}).values()
        if getattr(record, "confidence", 0.0) >= 0.45
    }
    return all(action_id in known for action_id in required_actions)


def _state(snapshot: ARGALiteSnapshot) -> dict[str, Any]:
    return {
        "game_id": snapshot.game_id,
        "level_index": snapshot.level_index,
        "step_index": snapshot.step_index,
        "state_name": snapshot.state_name,
        "terminal": snapshot.terminal,
        "game_over": snapshot.game_over,
        "full_reset": snapshot.full_reset,
        "levels_completed": snapshot.levels_completed,
        "win_levels": snapshot.win_levels,
        "score": snapshot.score,
        "semantic_state_signature": snapshot.semantic_state_signature,
    }


def _component_graph_for_packet(
    snapshot: ARGALiteSnapshot,
    focus_ids: set[str],
    aliases: dict[str, dict[str, str]],
    config: V8Config,
) -> dict[str, Any] | None:
    if not config.include_component_graph_in_qwen_packet:
        return None
    object_aliases = aliases.get("object_real_to_alias", {})
    linked_records = [
        (object_aliases[str(record.object_id)], record)
        for record in snapshot.objects
        if str(record.object_id) in focus_ids and str(record.object_id) in object_aliases
    ]
    graph = build_component_graph(
        snapshot.full_grid_hex_rows,
        object_records=linked_records,
        max_components=config.max_components_in_packet,
        max_shape_runs=config.max_component_shape_runs,
        max_boundary_corners=config.max_component_boundary_corners,
    )
    return _compact_component_graph_for_qwen(graph)


def _compact_component_graph_for_qwen(graph: dict[str, Any]) -> dict[str, Any]:
    graph["version"] = "component_graph_v1_packet_compact"
    graph["packet_projection"] = (
        "Exact component topology and shape hashes are retained. Redundant geometry fields are omitted; "
        "use SCENE.objects masks for tracked objects and CURRENT_FRAME for visual detail."
    )
    graph.pop("color_shape_hash_semantics", None)
    graph.pop("boundary_coordinate_space", None)
    for component in graph.get("components") or []:
        if not isinstance(component, dict):
            continue
        for key in (
            "color_shape_hash",
            "frame_signature",
            "centroid_xy",
            "fill_ratio",
            "children",
            "normalized_shape_runs",
            "outer_boundary_corner_count",
            "outer_boundary_corners_xy",
        ):
            component.pop(key, None)
    return graph


def _grid(snapshot: ARGALiteSnapshot, config: V8Config) -> dict[str, Any]:
    histogram = { _hex_symbol(k): int(v) for k, v in snapshot.palette_histogram.items() }
    bg_id = None
    bg_conf = None
    if snapshot.palette_histogram:
        total = max(1, sum(int(v) for v in snapshot.palette_histogram.values()))
        bg_value, bg_count = max(snapshot.palette_histogram.items(), key=lambda item: int(item[1]))
        bg_id = _hex_symbol(bg_value)
        bg_conf = round(float(bg_count) / total, 4)
    out: dict[str, Any] = {
        "shape_hw": [snapshot.height, snapshot.width],
        "coordinate_order": "x=column,y=row",
        "origin": "top_left",
        "row_order": "top_to_bottom",
        "column_order": "left_to_right",
        "encoding": "categorical_hex",
        "symbols_are_categorical": True,
        "object_mask_encoding": {
            "palette_symbols": "0-F",
            "outside_object_mask": ".",
        },
        "palette_ids_seen": [_hex_symbol(v) for v in snapshot.palette_ids_seen],
        "palette_histogram": histogram,
        "background_id": bg_id,
        "background_confidence": bg_conf,
        "grid_source": "official_observation_grid",
    }
    if config.include_full_grid_in_qwen_packet:
        out["hex_rows"] = list(snapshot.full_grid_hex_rows)
    return {key: value for key, value in out.items() if value is not None}


def _regions(snapshot: ARGALiteSnapshot) -> dict[str, Any]:
    regions: list[dict[str, Any]] = []
    for obj in snapshot.objects:
        bbox = getattr(obj, "bbox_rc", None)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if _is_full_grid_container(obj, snapshot):
            continue
        r0, c0, r1, c1 = [int(v) for v in bbox]
        bbox_area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
        is_large = bbox_area >= 0.18 * max(1, int(snapshot.height) * int(snapshot.width))
        if not is_large and not _is_structural_strip(obj, snapshot):
            continue
        regions.append({
            "id": f"region_{len(regions)}",
            "bbox_xyxy": [c0, r0, c1, r1],
            "palette_histogram": {_hex_symbol(k): int(v) for k, v in getattr(obj, "color_histogram", {}).items()},
            "geometry_class": _geometry_class_from_bbox((r0, c0, r1, c1), int(getattr(obj, "area", 0) or 0), snapshot.height, snapshot.width),
            "edge_contacts": list(getattr(obj, "border_touching", ()) or ()),
            "semantic_role": "UNKNOWN",
        })
    return {
        "contract": "Large and edge-touching geometry is retained as neutral context. Geometry alone does not identify UI, obstacle, target, divider, or counter semantics.",
        "large_or_edge_regions": regions[:24],
    }


def _dedupe_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in regions:
        key = (item.get("id"), item.get("type"))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = dict(item)
            continue
        bbox_a = existing.get("bbox_xyxy") or []
        bbox_b = item.get("bbox_xyxy") or []
        if len(bbox_a) == 4 and len(bbox_b) == 4:
            existing["bbox_xyxy"] = [
                min(int(bbox_a[0]), int(bbox_b[0])),
                min(int(bbox_a[1]), int(bbox_b[1])),
                max(int(bbox_a[2]), int(bbox_b[2])),
                max(int(bbox_a[3]), int(bbox_b[3])),
            ]
        hist = dict(existing.get("palette_histogram") or {})
        for color, count in (item.get("palette_histogram") or {}).items():
            hist[str(color)] = int(hist.get(str(color), 0)) + int(count)
        if hist:
            existing["palette_histogram"] = hist
    for item in by_key.values():
        out.append(item)
    return out


def _actions(snapshot: ARGALiteSnapshot, role: QwenRole, allowed_action_ids: list[str]) -> list[dict[str, Any]]:
    allowed = set(allowed_action_ids)
    coordinate = set(snapshot.coordinate_action_ids)
    undo = set(snapshot.undo_action_ids)
    out = []
    ordered = list(dict.fromkeys([*snapshot.available_actions, *allowed_action_ids]))
    for action_id in ordered:
        is_coordinate = action_id in coordinate
        item: dict[str, Any] = {
            "id": action_id,
            "kind": "coordinate" if is_coordinate else ("undo" if action_id in undo else "simple"),
            "available_now": action_id in set(snapshot.available_actions),
            "planning_allowed": action_id in allowed,
            "undo": action_id in undo,
        }
        if is_coordinate:
            item["payload"] = {"x": "int", "y": "int"}
        out.append(item)
    return out


def build_model_aliases(snapshot: ARGALiteSnapshot, config: V8Config) -> dict[str, dict[str, str]]:
    # The reverse map must be built from the same canonical planning-object view
    # as the packet. Otherwise suppressing a nested component shifts every later
    # oN/rN/cN alias between Qwen input and executor output translation.
    objects = _objects(
        snapshot,
        config,
        excluded_object_ids=_nested_compact_component_view_ids(snapshot),
    )
    object_ids = {item["id"] for item in objects}
    relations = _relations(snapshot, object_ids, config)
    relation_ids = {item["id"] for item in relations}
    candidates = _coordinate_candidates(snapshot, object_ids, relation_ids, config)
    geometry_groups = _annotate_object_geometry([dict(item) for item in objects])
    return build_model_aliases_from_records(objects, relations, candidates, geometry_groups)


def build_model_aliases_from_records(objects: list[dict[str, Any]], relations: list[dict[str, Any]], candidates: list[dict[str, Any]], geometry_groups: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    object_aliases = {f"o{idx}": str(item["id"]) for idx, item in enumerate(objects)}
    relation_aliases = {f"r{idx}": str(item["id"]) for idx, item in enumerate(relations)}
    candidate_aliases = {f"c{idx}": str(item["id"]) for idx, item in enumerate(candidates)}
    geometry_aliases = {f"g{idx}": str(item["id"]) for idx, item in enumerate(geometry_groups)}
    return {
        "object_aliases": object_aliases,
        "relation_aliases": relation_aliases,
        "candidate_aliases": candidate_aliases,
        "geometry_aliases": geometry_aliases,
        "object_real_to_alias": {real: alias for alias, real in object_aliases.items()},
        "relation_real_to_alias": {real: alias for alias, real in relation_aliases.items()},
        "candidate_real_to_alias": {real: alias for alias, real in candidate_aliases.items()},
        "geometry_real_to_alias": {real: alias for alias, real in geometry_aliases.items()},
    }


def translate_model_ids_to_internal(output: dict[str, Any], snapshot: ARGALiteSnapshot, config: V8Config) -> dict[str, Any]:
    aliases = build_model_aliases(snapshot, config)
    out = _deepcopy_jsonable(output)
    _translate_ids_in_place(out, aliases, to_internal=True)
    return out


def _alias_objects(objects: list[dict[str, Any]], aliases: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    obj_map = aliases["object_real_to_alias"]
    geom_map = aliases["geometry_real_to_alias"]
    for item in objects:
        copy = _deepcopy_jsonable(item)
        copy["id"] = obj_map.get(str(item.get("id")), str(item.get("id")))
        copy.pop("track_id", None)
        geometry = copy.get("shape_geometry")
        if isinstance(geometry, dict):
            gid = geometry.get("exact_geometry_group_id")
            if gid is not None:
                geometry["exact_geometry_group_id"] = geom_map.get(str(gid), str(gid))
            same = geometry.get("same_exact_geometry_object_ids")
            if isinstance(same, list):
                geometry["same_exact_geometry_object_ids"] = [obj_map.get(str(v), str(v)) for v in same]
        for key in ("moved_under_action_ids", "stationary_under_probed_actions"):
            if key in copy and isinstance(copy[key], list):
                copy[key] = [str(v) for v in copy[key]]
        out.append(copy)
    return out


def _prune_aliased_object_geometry_refs(objects: list[dict[str, Any]]) -> None:
    retained_ids = {str(item.get("id")) for item in objects}
    for item in objects:
        geometry = item.get("shape_geometry") if isinstance(item, dict) else None
        if not isinstance(geometry, dict):
            continue
        geometry["same_exact_geometry_object_ids"] = [
            str(object_id)
            for object_id in geometry.get("same_exact_geometry_object_ids") or []
            if str(object_id) in retained_ids
        ]


def _alias_geometry_groups(groups: list[dict[str, Any]], aliases: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    obj_map = aliases["object_real_to_alias"]
    geom_map = aliases["geometry_real_to_alias"]
    for group in groups:
        copy = _deepcopy_jsonable(group)
        copy["id"] = geom_map.get(str(group.get("id")), str(group.get("id")))
        copy["object_ids"] = [obj_map.get(str(v), str(v)) for v in group.get("object_ids", [])]
        out.append(copy)
    return out


def _alias_relations(relations: list[dict[str, Any]], aliases: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    obj_map = aliases["object_real_to_alias"]
    rel_map = aliases["relation_real_to_alias"]
    for item in relations:
        source = obj_map.get(str(item.get("source_object_id")), str(item.get("source_object_id")))
        target = obj_map.get(str(item.get("target_object_id")), str(item.get("target_object_id")))
        relation_type = str(item.get("type"))
        if relation_type == "same_shape":
            relation_type = "same_exact_geometry"
        copy = {
            "id": rel_map.get(str(item.get("id")), str(item.get("id"))),
            "type": relation_type,
            "object_ids": [source, target],
            "measurements": _deepcopy_jsonable(item.get("measurements") or {}),
            "parser_confidence": item.get("parser_confidence"),
            "salience_score": item.get("salience_score"),
        }
        if relation_type not in {"relative_position"}:
            copy["satisfied"] = item.get("satisfied", True)
        out.append({key: value for key, value in copy.items() if value is not None})
    return out


def _alias_candidates(candidates: list[dict[str, Any]], aliases: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    obj_map = aliases["object_real_to_alias"]
    rel_map = aliases["relation_real_to_alias"]
    cand_map = aliases["candidate_real_to_alias"]
    for item in candidates:
        copy = _deepcopy_jsonable(item)
        copy["id"] = cand_map.get(str(item.get("id")), str(item.get("id")))
        if copy.get("object_id") is not None:
            copy["object_id"] = obj_map.get(str(copy["object_id"]), str(copy["object_id"]))
        if copy.get("relation_id") is not None:
            copy["relation_id"] = rel_map.get(str(copy["relation_id"]), str(copy["relation_id"]))
        if isinstance(copy.get("reason"), str):
            copy["reason"] = _replace_known_ids_in_text(copy["reason"], aliases)
        out.append(copy)
    return out


def _alias_control_model(control_model: dict[str, Any], aliases: dict[str, dict[str, str]]) -> dict[str, Any]:
    out = _deepcopy_jsonable(control_model)
    obj_map = aliases["object_real_to_alias"]
    out["delta_xy_contract"] = {
        "format": "[dx,dy]",
        "positive_dx": "right",
        "negative_dx": "left",
        "positive_dy": "down",
        "negative_dy": "up",
    }
    out["motion_invariants"] = {
        "translation_preserves_shape": True,
        "translation_preserves_orientation": True,
        "translation_preserves_palette": True,
        "translation_preserves_geometry_group": True,
    }
    actions = out.get("actions")
    if isinstance(actions, dict):
        for action in actions.values():
            if not isinstance(action, dict):
                continue
            motions = action.get("motions_xy") if isinstance(action.get("motions_xy"), dict) else {}
            aliased_motions: dict[str, Any] = {}
            effects = []
            for effect in action.get("effects") or []:
                if not isinstance(effect, dict):
                    continue
                copy = _deepcopy_jsonable(effect)
                if copy.get("object_id") is not None:
                    copy["object_id"] = obj_map.get(str(copy["object_id"]), str(copy["object_id"]))
                effects.append(copy)
            for real_id, delta_xy in motions.items():
                alias = obj_map.get(str(real_id), str(real_id))
                aliased_motions[alias] = delta_xy
                if not any(effect.get("kind") == "translation" and effect.get("object_id") == alias for effect in effects):
                    effects.append({
                        "kind": "translation",
                        "object_id": alias,
                        "delta_xy": delta_xy,
                        "direction": _direction_from_delta_xy(delta_xy),
                    })
            action["motions_xy"] = aliased_motions
            action["effects"] = effects
    return out


def _focus_control_model(control_model: dict[str, Any], focus_ids: set[str]) -> dict[str, Any]:
    out = _deepcopy_jsonable(control_model)
    actions = out.get("actions")
    if not isinstance(actions, dict):
        return out
    for action in actions.values():
        if not isinstance(action, dict):
            continue
        motions = action.get("motions_xy") if isinstance(action.get("motions_xy"), dict) else {}
        omitted_motion_ids = [str(object_id) for object_id in motions if str(object_id) not in focus_ids]
        action["motions_xy"] = {str(object_id): delta for object_id, delta in motions.items() if str(object_id) in focus_ids}
        effects = [effect for effect in action.get("effects") or [] if isinstance(effect, dict)]
        omitted_effects = [effect for effect in effects if effect.get("object_id") is not None and str(effect.get("object_id")) not in focus_ids]
        action["effects"] = [effect for effect in effects if effect.get("object_id") is None or str(effect.get("object_id")) in focus_ids]
        if omitted_motion_ids or omitted_effects:
            action["omitted_simultaneous_object_effect_count"] = len(set(omitted_motion_ids)) + len(omitted_effects)
            action["omitted_effect_contract"] = "additional observed object effects exist outside the compact focus set"
    return out


def _semantic_digest(objects: list[dict[str, Any]], relations: list[dict[str, Any]], geometry_groups: list[dict[str, Any]], control_model: dict[str, Any]) -> dict[str, Any]:
    digest_objects = []
    for item in objects:
        geometry = item.get("shape_geometry") if isinstance(item.get("shape_geometry"), dict) else {}
        moved = item.get("moved_under_action_ids") or []
        stationary = item.get("stationary_under_probed_actions") or []
        if moved and stationary:
            mobility = "PARTIALLY_MOVES_UNDER_OBSERVED_ACTIONS"
        elif moved:
            mobility = "MOVES_UNDER_OBSERVED_ACTIONS"
        elif stationary:
            mobility = "STATIONARY_UNDER_OBSERVED_ACTIONS"
        else:
            mobility = "MOBILITY_UNKNOWN"
        digest_objects.append({
            "id": item.get("id"),
            "mobility_class": mobility,
            "geometry_group": geometry.get("exact_geometry_group_id"),
            "orientation": geometry.get("orientation_label"),
            "palette_ids": sorted(str(k) for k in (item.get("palette_histogram") or {}).keys()),
        })
    scene_patterns = []
    for group in geometry_groups:
        ids = group.get("object_ids") or []
        if len(ids) >= 2:
            scene_patterns.append({"type": "SAME_EXACT_GEOMETRY", "object_ids": ids})
    for relation in relations:
        item = {"type": str(relation.get("type", "")).upper(), "relation_id": relation.get("id"), "object_ids": relation.get("object_ids", [])}
        measurements = relation.get("measurements")
        if measurements:
            item["measurements"] = measurements
        scene_patterns.append(item)
    motion_patterns = []
    actions = control_model.get("actions") if isinstance(control_model.get("actions"), dict) else {}
    for action_id, action in actions.items():
        effects = action.get("effects") if isinstance(action, dict) else []
        if not isinstance(effects, list):
            continue
        motion_patterns.append({
            "action_id": action_id,
            "pattern": _motion_pattern_label(effects),
            "effects": effects,
        })
    return {
        "objects": digest_objects,
        "scene_patterns": scene_patterns,
        "motion_patterns": motion_patterns,
    }


def _alias_recent_transition(transition: dict[str, Any], aliases: dict[str, dict[str, str]]) -> dict[str, Any]:
    out = _deepcopy_jsonable(transition)
    _translate_ids_in_place(out, aliases, to_internal=False)
    if "relation_changes_status" in out:
        out["relation_predicate_changes_status"] = "NOT_COMPUTED" if out.get("relation_changes") else "NO_PREDICATE_CHANGE"
        out["relation_measurement_changes_status"] = "NOT_COMPUTED"
        out.pop("relation_changes_status", None)
    return out


def _alias_recent_evidence(evidence: dict[str, Any], aliases: dict[str, dict[str, str]]) -> dict[str, Any]:
    out = _deepcopy_jsonable(evidence)
    _translate_ids_in_place(out, aliases, to_internal=False)
    return out


def _translate_ids_in_place(value: Any, aliases: dict[str, dict[str, str]], *, to_internal: bool) -> None:
    object_map = aliases["object_aliases"] if to_internal else aliases["object_real_to_alias"]
    relation_map = aliases["relation_aliases"] if to_internal else aliases["relation_real_to_alias"]
    candidate_map = aliases["candidate_aliases"] if to_internal else aliases["candidate_real_to_alias"]
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key in {"object_id", "source_object_id", "target_object_id", "linked_object_id"} and child is not None:
                value[key] = object_map.get(str(child), str(child))
            elif key in {
                "objects", "object_ids", "source_objects", "reference_objects",
                "target_object_ids", "tracked_object_ids", "affected_object_ids",
                "changed_object_ids", "supporting_fact_ids", "target_ids",
            } and isinstance(child, list):
                value[key] = [_translate_mixed_id(v, object_map, relation_map, candidate_map) for v in child]
            elif key in {"relation_id", "target_relation_id"} and child is not None:
                value[key] = relation_map.get(str(child), str(child))
            elif key in {"relations", "relation_ids", "target_relation_ids", "tracked_relation_ids"} and isinstance(child, list):
                value[key] = [relation_map.get(str(v), str(v)) for v in child]
            elif key in {"coordinate_candidate_id", "candidate_id"} and child is not None:
                value[key] = candidate_map.get(str(child), str(child))
            elif key in {"coordinate_candidate_ids", "allowed_coordinate_candidate_ids"} and isinstance(child, list):
                value[key] = [candidate_map.get(str(v), str(v)) for v in child]
            else:
                _translate_ids_in_place(child, aliases, to_internal=to_internal)
    elif isinstance(value, list):
        for child in value:
            _translate_ids_in_place(child, aliases, to_internal=to_internal)


def _translate_mixed_id(value: Any, object_map: dict[str, str], relation_map: dict[str, str], candidate_map: dict[str, str]) -> str:
    text = str(value)
    if text in object_map:
        return object_map[text]
    if text in relation_map:
        return relation_map[text]
    if text in candidate_map:
        return candidate_map[text]
    return text


def _replace_known_ids_in_text(text: str, aliases: dict[str, dict[str, str]]) -> str:
    out = text
    for mapping_name in ("object_real_to_alias", "relation_real_to_alias", "candidate_real_to_alias", "geometry_real_to_alias"):
        for real, alias in aliases[mapping_name].items():
            out = out.replace(real, alias)
    return out


def _deepcopy_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deepcopy_jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_deepcopy_jsonable(child) for child in value]
    if isinstance(value, tuple):
        return [_deepcopy_jsonable(child) for child in value]
    return value


def _direction_from_delta_xy(delta_xy: Any) -> str:
    if not isinstance(delta_xy, (list, tuple)) or len(delta_xy) < 2:
        return "unknown"
    try:
        dx = float(delta_xy[0])
        dy = float(delta_xy[1])
    except Exception:
        return "unknown"
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "stationary"
    parts = []
    if dy < -1e-6:
        parts.append("up")
    elif dy > 1e-6:
        parts.append("down")
    if dx < -1e-6:
        parts.append("left")
    elif dx > 1e-6:
        parts.append("right")
    return "_".join(parts) if parts else "unknown"


def _motion_pattern_label(effects: list[dict[str, Any]]) -> str:
    moving = [effect for effect in effects if effect.get("direction") not in {None, "stationary", "unknown"}]
    if len(moving) < 2:
        return "SINGLE_OBJECT_TRANSLATION" if moving else "NO_OBJECT_TRANSLATION"
    dx_values = []
    dy_values = []
    for effect in moving:
        delta = effect.get("delta_xy")
        if isinstance(delta, list) and len(delta) >= 2:
            dx_values.append(float(delta[0]))
            dy_values.append(float(delta[1]))
    if dx_values and all(abs(dy) < 1e-6 for dy in dy_values) and len({1 if dx > 0 else -1 if dx < 0 else 0 for dx in dx_values}) > 1:
        return "MULTI_OBJECT_HORIZONTAL_OPPOSITE_DIRECTIONS"
    if dy_values and all(abs(dx) < 1e-6 for dx in dx_values) and len({1 if dy > 0 else -1 if dy < 0 else 0 for dy in dy_values}) == 1:
        return "MULTI_OBJECT_SHARED_VERTICAL_TRANSLATION"
    return "MULTI_OBJECT_TRANSLATION"


def _objects(
    snapshot: ARGALiteSnapshot,
    config: V8Config,
    *,
    excluded_object_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    out = []
    excluded = excluded_object_ids or set()
    selected = [obj for obj in snapshot.objects if obj.object_id not in excluded]
    for idx, obj in enumerate(selected[: config.max_objects_in_packet]):
        if _is_full_grid_container(obj, snapshot):
            continue
        bbox_area = (obj.bbox_rc[2] - obj.bbox_rc[0] + 1) * (obj.bbox_rc[3] - obj.bbox_rc[1] + 1)
        item = {
            "id": obj.object_id,
            "source_type": "tracked_connected_region",
            "track_id": obj.track_id or obj.object_id,
            "bbox_xyxy": _bbox_xyxy(obj.bbox_rc),
            "centroid_xy": _centroid_xy(obj.centroid_rc),
            "area": obj.area,
            "palette_histogram": {_hex_symbol(k): int(v) for k, v in obj.color_histogram.items()},
            "shape_profile": _shape_profile(obj),
            "shape_signature": obj.shape_signature,
            "geometry_class": _geometry_class_from_bbox(obj.bbox_rc, obj.area, snapshot.height, snapshot.width),
            "edge_contacts": list(getattr(obj, "border_touching", ()) or ()),
            "holes": int(getattr(obj, "holes", 0) or 0),
        }
        if config.include_object_local_masks and idx < config.max_object_masks_in_packet and bbox_area <= 1024:
            item["local_hex_rows"] = list(obj.local_mask_hex_rows)
        elif config.include_object_local_masks:
            item["local_mask_omitted"] = "large_region; inspect current_frame and shape_profile"
        out.append(item)
    return out


def _nested_compact_component_view_ids(snapshot: ARGALiteSnapshot) -> set[str]:
    """Hide duplicate monochrome views of compact multicolor planning objects."""
    records = list(snapshot.objects)
    occupied = {obj.object_id: _record_occupied_cells(obj) for obj in records}
    compact_multicolor_parents = [
        obj
        for obj in records
        if len(obj.colors) > 1
        and (obj.bbox_rc[2] - obj.bbox_rc[0] + 1) <= 16
        and (obj.bbox_rc[3] - obj.bbox_rc[1] + 1) <= 16
    ]
    hidden: set[str] = set()
    for child in records:
        if len(child.colors) != 1:
            continue
        child_cells = occupied.get(child.object_id, frozenset())
        if not child_cells:
            continue
        for parent in compact_multicolor_parents:
            if parent.object_id == child.object_id or child.area >= parent.area:
                continue
            parent_cells = occupied.get(parent.object_id, frozenset())
            if child_cells < parent_cells:
                hidden.add(child.object_id)
                break
    return hidden


def _record_occupied_cells(obj: Any) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = obj.bbox_rc
    return frozenset(
        (r0 + local_r, c0 + local_c)
        for local_r, row in enumerate(obj.local_mask_hex_rows)
        for local_c, value in enumerate(row)
        if value != "."
    )


def _geometry_class_from_bbox(bbox: Any, area: int, frame_height: int, frame_width: int) -> str:
    r0, c0, r1, c1 = [int(v) for v in bbox]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    fill = float(area) / max(1, height * width)
    if height == 1 and width == 1:
        return "point"
    if height == 1 or width == 1:
        return "line"
    if (width <= 3 and height >= 0.5 * frame_height) or (height <= 3 and width >= 0.5 * frame_width):
        return "long_strip"
    if height <= 16 and width <= 16:
        return "compact_sparse_shape" if fill < 0.6 else "compact_shape"
    if fill < 0.45:
        return "large_sparse_region"
    return "large_region"


def _annotate_object_motion(objects: list[dict[str, Any]], control_model: dict[str, Any]) -> list[dict[str, Any]]:
    actions = control_model.get("actions") if isinstance(control_model, dict) else {}
    if not isinstance(actions, dict):
        return objects
    probed_action_ids = sorted(str(action_id) for action_id in actions)
    moved_by_object: dict[str, list[str]] = {}
    changed_by_object: dict[str, list[str]] = {}
    for action_id, action_model in actions.items():
        if not isinstance(action_model, dict):
            continue
        motions = action_model.get("motions_xy")
        if not isinstance(motions, dict):
            continue
        for object_id, delta_xy in motions.items():
            if _nonzero_delta(delta_xy):
                moved_by_object.setdefault(str(object_id), []).append(str(action_id))
        for effect in action_model.get("effects") or []:
            if isinstance(effect, dict) and effect.get("object_id") is not None:
                changed_by_object.setdefault(str(effect["object_id"]), []).append(str(action_id))
    out: list[dict[str, Any]] = []
    for item in objects:
        copy = dict(item)
        moved = sorted(dict.fromkeys(moved_by_object.get(str(copy.get("id")), [])))
        changed = sorted(dict.fromkeys(changed_by_object.get(str(copy.get("id")), [])))
        copy["moved_under_action_ids"] = moved
        copy["changed_under_action_ids"] = changed
        copy["unchanged_under_probed_actions"] = [action_id for action_id in probed_action_ids if action_id not in set(changed)]
        out.append(copy)
    return out


def _annotate_object_geometry(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in objects:
        rows = _binary_mask_rows(item.get("local_hex_rows") or [])
        shape_signature = str(item.get("shape_signature") or "")
        if not rows and not shape_signature:
            continue
        group_id = stable_hash(("exact_mask", tuple(rows)) if rows else ("shape_signature", shape_signature), "geom_")
        profile = dict(item.get("shape_profile") or {})
        geometry = {
            "exact_geometry_group_id": group_id,
            "occupied_mask_rows": rows or None,
            "orientation_label": _shape_orientation_label(rows) if rows else str(item.get("geometry_class") or "unknown"),
            "bbox_hw": profile.get("bbox_hw"),
            "row_occupancy": profile.get("row_occupancy"),
            "col_occupancy": profile.get("col_occupancy"),
        }
        item["shape_geometry"] = {key: value for key, value in geometry.items() if value not in (None, [], {})}
        groups.setdefault(group_id, []).append(item)

    out = []
    for group_id, members in sorted(groups.items(), key=lambda entry: (-len(entry[1]), entry[0])):
        object_ids = sorted(str(member.get("id")) for member in members)
        for member in members:
            member["shape_geometry"]["same_exact_geometry_object_ids"] = object_ids
        sample = members[0].get("shape_geometry") or {}
        out.append({
            "id": group_id,
            "object_ids": object_ids,
            "orientation_label": sample.get("orientation_label"),
            "occupied_mask_rows": sample.get("occupied_mask_rows"),
        } | ({"shape_signature": str(members[0].get("shape_signature"))} if not sample.get("occupied_mask_rows") else {}))
    return out


def _focus_object_ids(
    objects: list[dict[str, Any]],
    geometry_groups: list[dict[str, Any]],
    control_model: dict[str, Any],
    coordinate_candidates: list[dict[str, Any]],
    role: QwenRole,
    config: V8Config,
) -> set[str]:
    limit = config.max_coordinate_objects_in_packet if role is QwenRole.COORDINATE else config.max_semantic_objects_in_packet
    object_by_id = {str(item.get("id")): item for item in objects}
    motion_ids: set[str] = set()
    changed_ids: set[str] = set()
    actions = control_model.get("actions") if isinstance(control_model, dict) else {}
    if isinstance(actions, dict):
        for action in actions.values():
            if not isinstance(action, dict):
                continue
            for object_id, delta in (action.get("motions_xy") or {}).items():
                if _nonzero_delta(delta):
                    motion_ids.add(str(object_id))
            for effect in action.get("effects") or []:
                if not isinstance(effect, dict) or not effect.get("object_id"):
                    continue
                changed_ids.add(str(effect["object_id"]))
    candidate_ids = {
        str(item.get("object_id"))
        for item in coordinate_candidates
        if item.get("object_id") is not None
    }
    group_by_object: dict[str, list[str]] = {}
    group_sizes: dict[str, int] = {}
    for group in geometry_groups:
        members = [str(value) for value in group.get("object_ids") or []]
        group_id = str(group.get("id"))
        group_sizes[group_id] = len(members)
        for object_id in members:
            group_by_object.setdefault(object_id, []).append(group_id)
    exact_mates: set[str] = set()
    for group in geometry_groups:
        members = {str(value) for value in group.get("object_ids") or []}
        if members & motion_ids:
            exact_mates.update(members)

    def score(item: dict[str, Any]) -> tuple[float, str]:
        object_id = str(item.get("id"))
        profile = item.get("shape_profile") if isinstance(item.get("shape_profile"), dict) else {}
        bbox_hw = profile.get("bbox_hw") or [0, 0]
        compact = len(bbox_hw) == 2 and int(bbox_hw[0]) <= 16 and int(bbox_hw[1]) <= 16
        repeated = max((group_sizes.get(group_id, 1) for group_id in group_by_object.get(object_id, ())), default=1)
        value = 0.0
        if object_id in motion_ids:
            value += 10000.0
        if object_id in changed_ids:
            value += 8000.0
        if object_id in exact_mates:
            value += 7000.0
        if role is QwenRole.COORDINATE and object_id in candidate_ids:
            value += 6000.0
        if repeated > 1:
            value += 300.0 + min(100.0, repeated * 5.0)
        if compact:
            value += 120.0
        if len(item.get("palette_histogram") or {}) > 1:
            value += 80.0
        value += min(60.0, float(item.get("area") or 0) ** 0.5)
        if str(item.get("geometry_class") or "").startswith("large") and object_id not in changed_ids:
            value -= 40.0
        return value, object_id

    ranked = sorted(objects, key=lambda item: (-score(item)[0], score(item)[1]))
    return {str(item.get("id")) for item in ranked[: max(1, int(limit))] if str(item.get("id")) in object_by_id}


def _focus_relations(relations: list[dict[str, Any]], focus_ids: set[str], config: V8Config) -> list[dict[str, Any]]:
    priorities = {
        "same_shape": 9,
        "contains": 8,
        "frame_contains": 8,
        "near": 7,
        "separated_by_gap": 7,
        "line_continuation": 7,
        "same_color": 6,
        "same_row": 5,
        "same_column": 5,
        "relative_position": 2,
    }
    kept = [
        item for item in relations
        if str(item.get("source_object_id")) in focus_ids and str(item.get("target_object_id")) in focus_ids
    ]
    kept.sort(key=lambda item: (-priorities.get(str(item.get("type")), 4), -float(item.get("salience_score") or 0.0), str(item.get("id"))))
    return kept[: max(1, int(config.max_semantic_relations_in_packet))]


def _focus_geometry_groups(groups: list[dict[str, Any]], focus_ids: set[str], config: V8Config) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        members = [str(value) for value in group.get("object_ids") or [] if str(value) in focus_ids]
        if not members:
            continue
        copy = _deepcopy_jsonable(group)
        copy["object_ids"] = members
        out.append(copy)
    out.sort(key=lambda item: (-len(item.get("object_ids") or []), str(item.get("id"))))
    return out[: max(1, int(config.max_semantic_groups_in_packet))]


def _priority_facts(objects: list[dict[str, Any]], geometry_groups: list[dict[str, Any]], control_model: dict[str, Any]) -> list[dict[str, Any]]:
    object_by_id = {str(item.get("id")): item for item in objects}
    facts: list[dict[str, Any]] = []

    def mobility(object_id: str) -> str:
        item = object_by_id.get(object_id) or {}
        moved = item.get("moved_under_action_ids") or []
        changed = item.get("changed_under_action_ids") or []
        if moved:
            return "TRANSLATION_OBSERVED"
        if changed:
            return "NON_TRANSLATION_CHANGE_OBSERVED"
        return "NO_CHANGE_OBSERVED_UNDER_CURRENT_PROBES"

    for group in geometry_groups:
        members = [str(value) for value in group.get("object_ids") or []]
        if len(members) < 2:
            continue
        movers = [object_id for object_id in members if mobility(object_id) == "TRANSLATION_OBSERVED"]
        non_movers = [object_id for object_id in members if object_id not in movers]
        if movers and non_movers:
            for source in movers[:3]:
                reference = non_movers[0]
                source_position = (object_by_id.get(source) or {}).get("centroid_xy")
                reference_position = (object_by_id.get(reference) or {}).get("centroid_xy")
                displacement = _point_delta_xy(source_position, reference_position)
                source_effects = []
                for action_id, action in (control_model.get("actions") or {}).items():
                    effects = action.get("effects") if isinstance(action, dict) else []
                    translation = next((
                        effect for effect in (effects or [])
                        if isinstance(effect, dict)
                        and effect.get("kind") == "translation"
                        and str(effect.get("object_id")) == source
                    ), None)
                    if translation is None:
                        continue
                    source_effects.append({
                        "action_id": str(action_id),
                        "source_delta_xy": translation.get("delta_xy"),
                        "available_now": bool(action.get("available_now")),
                        "planning_available_now": bool(action.get("planning_available_now")),
                        "surface_before_matches_current": bool(action.get("surface_before_matches_current")),
                        "observed_control_group_ids": list(action.get("observed_control_group_ids") or []),
                        "current_control_context_status": action.get("current_control_context_status") or "UNKNOWN",
                        "simultaneously_translated_object_ids": [
                            str(effect.get("object_id")) for effect in (effects or [])
                            if isinstance(effect, dict) and effect.get("kind") == "translation"
                        ],
                    })
                facts.append({
                    "id": f"f{len(facts)}",
                    "type": "MOVABLE_REFERENCE_EXACT_GEOMETRY_CORRESPONDENCE",
                    "object_ids": [source, reference],
                    "movable_object_id": source,
                    "reference_object_id": reference,
                    "source_to_reference_delta_xy": displacement,
                    "observed_source_translation_by_action": source_effects,
                    "current_inferred_control_group_id": control_model.get("current_inferred_control_group_id"),
                    "supported_control_group_switch_action_ids": [
                        str(action_id)
                        for action_id, action in (control_model.get("actions") or {}).items()
                        if isinstance(action, dict)
                        and action.get("current_control_context_status") == "SUPPORTED_CONTROL_GROUP_SWITCH_TRIGGER"
                    ],
                    "geometry_group_id": group.get("id"),
                    "evidence": "same exact occupied mask and orientation; translation observed for first object and not observed for second on current probes",
                    "goal_status": "CANDIDATE_CORRESPONDENCE_NOT_A_PRESELECTED_GOAL",
                })
        else:
            facts.append({
                "id": f"f{len(facts)}",
                "type": "SAME_EXACT_GEOMETRY_GROUP",
                "object_ids": members[:8],
                "geometry_group_id": group.get("id"),
                "evidence": "same exact occupied mask and orientation",
                "goal_status": "FACT_NOT_A_PRESELECTED_GOAL",
            })
    return facts[:24]


def _control_groups(objects: list[dict[str, Any]], control_model: dict[str, Any]) -> list[dict[str, Any]]:
    object_ids = {str(item.get("id")) for item in objects}
    coupled_by_members: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    actions = control_model.get("actions") if isinstance(control_model, dict) else {}
    if not isinstance(actions, dict):
        return []
    for action_id, action in actions.items():
        if not isinstance(action, dict):
            continue
        effects = [
            effect for effect in action.get("effects") or []
            if isinstance(effect, dict)
            and effect.get("kind") == "translation"
            and str(effect.get("object_id")) in object_ids
        ]
        members = tuple(sorted({str(effect.get("object_id")) for effect in effects}))
        if len(members) < 2:
            continue
        coupled_by_members.setdefault(members, []).append({
            "action_id": str(action_id),
            "observation_step_index": action.get("observation_step_index"),
            "surface_before": action.get("surface_before") or [],
            "surface_after": action.get("surface_after") or [],
        })
    out = []
    for members, observations in sorted(coupled_by_members.items(), key=lambda item: (-len(item[0]), item[0])):
        observations.sort(key=lambda item: (item.get("observation_step_index") is None, item.get("observation_step_index") or 0, item["action_id"]))
        out.append({
            "id": f"cg{len(out)}",
            "object_ids": list(members),
            "observed_action_ids": [item["action_id"] for item in observations],
            "chronological_observations": observations,
            "interpretation": "these objects translated simultaneously in each listed observed action context; directions and legal surfaces are factual, while controller identity and goal semantics remain unknown",
        })
    return out[:12]


def _control_state_transition_candidates(
    objects: list[dict[str, Any]],
    control_groups: list[dict[str, Any]],
    surface_transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for transition in surface_transitions:
        if not isinstance(transition, dict) or transition.get("surface_changed") is not True:
            continue
        visual = transition.get("simultaneous_raw_visual_changes")
        if not isinstance(visual, dict):
            continue
        reciprocal_pairs = visual.get("reciprocal_isolated_interior_transition_pairs") or []
        if not reciprocal_pairs:
            continue
        before_surface = _surface_key(transition.get("available_before"))
        after_surface = _surface_key(transition.get("available_after"))
        before_groups = [
            (group, _group_surface_score(group, before_surface))
            for group in control_groups
            if _group_surface_score(group, before_surface) > 0
        ]
        after_groups = [
            (group, _group_surface_score(group, after_surface))
            for group in control_groups
            if _group_surface_score(group, after_surface) > 0
        ]
        best: tuple[int, dict[str, Any]] | None = None
        for before_group, before_surface_score in before_groups:
            before_members = set(str(value) for value in before_group.get("object_ids") or [])
            for after_group, after_surface_score in after_groups:
                if before_group.get("id") == after_group.get("id"):
                    continue
                after_members = set(str(value) for value in after_group.get("object_ids") or [])
                shared = sorted(before_members & after_members)
                before_only = sorted(before_members - after_members)
                after_only = sorted(after_members - before_members)
                if not shared or not before_only or not after_only:
                    continue
                for reciprocal in reciprocal_pairs:
                    if not isinstance(reciprocal, dict):
                        continue
                    value_pair = [str(value) for value in reciprocal.get("value_pair") or []]
                    if len(value_pair) != 2:
                        continue
                    forward_bbox = reciprocal.get("forward_locations_bbox_xyxy")
                    reverse_bbox = reciprocal.get("reverse_locations_bbox_xyxy")
                    forward_objects = set(_objects_containing_change_bbox(objects, forward_bbox))
                    reverse_objects = set(_objects_containing_change_bbox(objects, reverse_bbox))
                    mapping = None
                    if forward_objects & set(before_only) and reverse_objects & set(after_only):
                        mapping = (
                            value_pair,
                            forward_bbox,
                            sorted(forward_objects & set(before_only)),
                            [value_pair[1], value_pair[0]],
                            reverse_bbox,
                            sorted(reverse_objects & set(after_only)),
                        )
                    elif reverse_objects & set(before_only) and forward_objects & set(after_only):
                        mapping = (
                            [value_pair[1], value_pair[0]],
                            reverse_bbox,
                            sorted(reverse_objects & set(before_only)),
                            value_pair,
                            forward_bbox,
                            sorted(forward_objects & set(after_only)),
                        )
                    if mapping is None:
                        continue
                    before_values, before_bbox, before_marker_objects, after_values, after_bbox, after_marker_objects = mapping
                    candidate = {
                        "id": f"cs{len(candidates)}",
                        "type": "SUPPORTED_CONTROL_GROUP_SWITCH",
                        "trigger_action_id": transition.get("trigger_action_id"),
                        "transition_step_index": transition.get("step_index"),
                        "before_control_group_id": before_group.get("id"),
                        "after_control_group_id": after_group.get("id"),
                        "shared_translated_object_ids": shared,
                        "before_only_translated_object_ids": before_only,
                        "after_only_translated_object_ids": after_only,
                        "before_group_observed_action_ids": list(before_group.get("observed_action_ids") or []),
                        "after_group_observed_action_ids": list(after_group.get("observed_action_ids") or []),
                        "before_action_surface": list(transition.get("available_before") or []),
                        "after_action_surface": list(transition.get("available_after") or []),
                        "before_surface_match_score": before_surface_score,
                        "after_surface_match_score": after_surface_score,
                        "marker_evidence": [
                            {
                                "control_group_id": before_group.get("id"),
                                "object_ids": before_marker_objects,
                                "center_value_transition": f"{before_values[0]}->{before_values[1]}",
                                "locations_bbox_xyxy": before_bbox,
                            },
                            {
                                "control_group_id": after_group.get("id"),
                                "object_ids": after_marker_objects,
                                "center_value_transition": f"{after_values[0]}->{after_values[1]}",
                                "locations_bbox_xyxy": after_bbox,
                            },
                        ],
                        "inference_status": "SUPPORTED_BY_RECIPROCAL_MARKERS_SURFACE_CHANGE_AND_COUPLED_MOTION",
                        "effect_scope_contract": "motion vectors observed in another control group may be planned only after the switch trigger; the executor verifies the resulting group before continuing",
                        "goal_status": "MECHANICS_INFERENCE_NOT_A_PRESELECTED_GOAL",
                    }
                    score = before_surface_score + after_surface_score + len(shared) * 10 + len(before_marker_objects) + len(after_marker_objects)
                    if best is None or score > best[0]:
                        best = (score, candidate)
        if best is not None:
            candidate = best[1]
            candidate["id"] = f"cs{len(candidates)}"
            candidates.append(candidate)
    if candidates:
        latest = max(candidates, key=lambda item: _safe_step_index(item.get("transition_step_index")))
        latest["current_inferred_control_group_id"] = latest.get("after_control_group_id")
        latest["current_context_status"] = "LATEST_OBSERVED_MARKER_SWITCH_POINTS_TO_AFTER_GROUP"
    return candidates[:8]


def _annotate_control_context(
    control_model: dict[str, Any],
    control_groups: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    actions = control_model.get("actions") if isinstance(control_model, dict) else None
    if not isinstance(actions, dict):
        return
    action_groups: dict[str, list[str]] = {}
    for group in control_groups:
        group_id = str(group.get("id"))
        for action_id in group.get("observed_action_ids") or []:
            action_groups.setdefault(str(action_id), []).append(group_id)
    for action_id, action in actions.items():
        if isinstance(action, dict):
            action["observed_control_group_ids"] = sorted(set(action_groups.get(str(action_id), [])))
    current = next((item for item in reversed(candidates) if item.get("current_inferred_control_group_id")), None)
    if current is None:
        control_model["control_state_inference_status"] = "UNKNOWN"
        return
    before_group = str(current.get("before_control_group_id"))
    after_group = str(current.get("after_control_group_id"))
    trigger = str(current.get("trigger_action_id"))
    control_model["control_state_inference_status"] = "SUPPORTED_CONTROL_GROUP_SWITCH"
    control_model["current_inferred_control_group_id"] = after_group
    control_model["control_state_transition_candidate_id"] = current.get("id")
    for action_id, action in actions.items():
        if not isinstance(action, dict):
            continue
        groups = set(str(value) for value in action.get("observed_control_group_ids") or [])
        if str(action_id) == trigger:
            action["current_control_context_status"] = "SUPPORTED_CONTROL_GROUP_SWITCH_TRIGGER"
        elif after_group in groups and before_group not in groups:
            action["current_control_context_status"] = "OBSERVED_IN_CURRENT_CONTROL_GROUP"
        elif before_group in groups and after_group not in groups:
            action["current_control_context_status"] = "OBSERVED_IN_NONCURRENT_CONTROL_GROUP"
        elif before_group in groups and after_group in groups:
            action["current_control_context_status"] = "OBSERVED_IN_BOTH_CONTROL_GROUPS"
        else:
            action["current_control_context_status"] = "NO_CONTROL_GROUP_ASSIGNMENT"


def _group_surface_score(group: dict[str, Any], surface: tuple[str, ...]) -> int:
    if not surface:
        return 0
    for observation in group.get("chronological_observations") or []:
        if not isinstance(observation, dict):
            continue
        if _surface_key(observation.get("surface_before")) == surface:
            return 100
        if _surface_key(observation.get("surface_after")) == surface:
            return 100
    observed_actions = set(str(value) for value in group.get("observed_action_ids") or [])
    if observed_actions and observed_actions.issubset(set(surface)):
        return 20
    return 0


def _objects_containing_change_bbox(objects: list[dict[str, Any]], change_bbox: Any) -> list[str]:
    if not isinstance(change_bbox, (list, tuple)) or len(change_bbox) != 4:
        return []
    try:
        x0, y0, x1, y1 = (int(value) for value in change_bbox)
    except (TypeError, ValueError):
        return []
    matches = []
    for item in objects:
        bbox = item.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            bx0, by0, bx1, by1 = (int(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if bx0 <= x0 <= x1 <= bx1 and by0 <= y0 <= y1 <= by1:
            slack = (bx1 - bx0 + 1) * (by1 - by0 + 1) - (x1 - x0 + 1) * (y1 - y0 + 1)
            matches.append((slack, str(item.get("id"))))
    matches.sort()
    return [object_id for _, object_id in matches]


def _surface_key(values: Any) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in values or []))


def _safe_step_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _repeated_object_groups(objects: list[dict[str, Any]], geometry_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item.get("id")): item for item in objects}
    out = []
    for group in geometry_groups:
        members = [str(value) for value in group.get("object_ids") or []]
        if len(members) < 2:
            continue
        out.append({
            "geometry_group_id": group.get("id"),
            "object_ids": members,
            "orientation": group.get("orientation_label"),
            "positions_xy": {object_id: (by_id.get(object_id) or {}).get("centroid_xy") for object_id in members},
            "palettes": {object_id: sorted((by_id.get(object_id) or {}).get("palette_histogram", {}).keys()) for object_id in members},
        })
    return out[:24]


def _binary_mask_rows(local_rows: list[str] | tuple[str, ...]) -> list[str]:
    rows = [str(row) for row in local_rows if str(row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [
        "".join("#" if x < len(row) and row[x] != "." else "." for x in range(width))
        for row in rows
    ]


def _point_delta_xy(source: Any, target: Any) -> list[float] | None:
    if not isinstance(source, (list, tuple)) or not isinstance(target, (list, tuple)) or len(source) < 2 or len(target) < 2:
        return None
    try:
        return [round(float(target[0]) - float(source[0]), 3), round(float(target[1]) - float(source[1]), 3)]
    except (TypeError, ValueError):
        return None


def _shape_orientation_label(rows: list[str]) -> str:
    if not rows:
        return "unknown"
    height = len(rows)
    width = max(len(row) for row in rows)
    if height <= 0 or width <= 0:
        return "unknown"
    normalized = [row.ljust(width, ".") for row in rows]
    row_counts = [row.count("#") for row in normalized]
    col_counts = [sum(1 for row in normalized if row[x] == "#") for x in range(width)]
    filled = sum(row_counts)
    if filled == height * width:
        if height == width:
            return "solid_square"
        if height == 1 or width == 1:
            return "solid_line"
        return "solid_rectangle"

    full_rows = [idx for idx, count in enumerate(row_counts) if count == width]
    full_cols = [idx for idx, count in enumerate(col_counts) if count == height]
    row_edge = _contiguous_edge_label(full_rows, height, "top", "bottom")
    col_edge = _contiguous_edge_label(full_cols, width, "left", "right")
    if row_edge and col_edge:
        return f"{row_edge}_{col_edge}_corner_L"
    if row_edge:
        return f"{row_edge}_bar_plus_partial_columns"
    if col_edge:
        return f"{col_edge}_bar_plus_partial_rows"
    return "irregular_mask"


def _contiguous_edge_label(indices: list[int], size: int, start_label: str, end_label: str) -> str | None:
    if not indices or size <= 0:
        return None
    ordered = sorted(indices)
    if ordered == list(range(ordered[-1] + 1)):
        return start_label
    if ordered == list(range(ordered[0], size)):
        return end_label
    return None


def _scene_entities(snapshot: ARGALiteSnapshot, objects: list[dict[str, Any]], config: V8Config) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    targetable = {obj["id"]: obj for obj in objects}
    for idx, tracked in enumerate(snapshot.objects[: max(0, config.max_objects_in_packet)]):
        obj = targetable.get(tracked.object_id)
        if obj is not None:
            entities.append({
                "id": f"scene_{obj['id']}",
                "source_type": "tracked_region",
                "linked_object_id": obj["id"],
                "bbox_xyxy": obj["bbox_xyxy"],
                "centroid_xy": obj["centroid_xy"],
                "area": obj["area"],
                "palette_histogram": obj["palette_histogram"],
                "shape_profile": obj.get("shape_profile", {}),
                "local_hex_rows": obj.get("local_hex_rows", []),
            })
            continue
        entity = {
            "id": f"scene_{tracked.object_id}",
            "source_type": "tracked_region",
            "linked_object_id": None,
            "bbox_xyxy": _bbox_xyxy(tracked.bbox_rc),
            "centroid_xy": _centroid_xy(tracked.centroid_rc),
            "area": tracked.area,
            "palette_histogram": {_hex_symbol(k): int(v) for k, v in tracked.color_histogram.items()},
            "shape_profile": _shape_profile(tracked),
            "local_hex_rows": [],
        }
        if config.include_object_local_masks and idx < config.max_object_masks_in_packet and not _is_full_grid_container(tracked, snapshot):
            entity["local_hex_rows"] = list(tracked.local_mask_hex_rows)
        entities.append(entity)
    entities.extend(_color_component_entities(snapshot, limit=max(0, config.max_objects_in_packet)))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in sorted(entities, key=lambda entry: (-int(entry.get("area") or 0), str(entry.get("id")))):
        key = (tuple(item.get("bbox_xyxy") or ()), tuple(sorted((item.get("palette_histogram") or {}).items())), item.get("source_type"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[: max(config.max_objects_in_packet, len(objects))]


def _relations(snapshot: ARGALiteSnapshot, allowed_object_ids: set[str], config: V8Config) -> list[dict[str, Any]]:
    object_by_id = {obj.object_id: obj for obj in snapshot.objects if obj.object_id in allowed_object_ids}
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for relation in snapshot.relations:
        if len(out) >= config.max_relations_in_packet:
            break
        if relation.a not in allowed_object_ids or relation.b not in allowed_object_ids:
            continue
        item = _relation_packet(relation, object_by_id)
        key = _relation_canonical_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _relation_packet(relation: Any, object_by_id: dict[str, Any]) -> dict[str, Any]:
    source = object_by_id.get(relation.a)
    target = object_by_id.get(relation.b)
    delta_xy = _object_translation_xy(source, target)
    metric_value = None if relation.metric_value is None else round(float(relation.metric_value), 3)
    measurements: dict[str, Any] = {}
    relation_type = str(relation.relation_type)
    satisfied = True
    if relation_type in {"same_shape", "translated_shape"}:
        relation_type = "same_shape"
        if delta_xy is not None:
            measurements["translation_xy"] = delta_xy
        if metric_value is not None:
            measurements["centroid_distance"] = metric_value
    elif relation_type == "aligned_row":
        relation_type = "same_row"
        measurements["row_difference"] = 0.0
        if metric_value is not None:
            measurements["horizontal_gap"] = metric_value
    elif relation_type == "aligned_col":
        relation_type = "same_column"
        measurements["column_difference"] = 0.0
        if metric_value is not None:
            measurements["vertical_gap"] = metric_value
    elif relation_type in {"left_of", "right_of", "above", "below"}:
        relation_type = "relative_position"
        if delta_xy is not None:
            measurements["delta_xy"] = delta_xy
            qualitative = []
            if delta_xy[0] > 0:
                qualitative.append("left_of")
            elif delta_xy[0] < 0:
                qualitative.append("right_of")
            if delta_xy[1] > 0:
                qualitative.append("above")
            elif delta_xy[1] < 0:
                qualitative.append("below")
            measurements["qualitative"] = qualitative
        elif metric_value is not None and relation.metric_name:
            measurements[str(relation.metric_name)] = metric_value
    else:
        if delta_xy is not None:
            measurements["delta_xy"] = delta_xy
        if metric_value is not None and relation.metric_name:
            measurements[str(relation.metric_name)] = metric_value
    return {
        "id": relation.relation_id,
        "type": relation_type,
        "source_object_id": relation.a,
        "target_object_id": relation.b,
        "satisfied": satisfied,
        "measurements": measurements,
        "parser_confidence": round(float(relation.confidence), 4),
        "salience_score": round(float(relation.salience_score), 4),
    }


def _relation_canonical_key(item: dict[str, Any]) -> tuple[Any, ...]:
    relation_type = item.get("type")
    source = item.get("source_object_id")
    target = item.get("target_object_id")
    pair = tuple(sorted(str(v) for v in (source, target)))
    if relation_type in {"same_shape", "same_row", "same_column", "relative_position"}:
        return (relation_type, pair)
    return (relation_type, source, target)


def _object_translation_xy(source: Any, target: Any) -> list[float] | None:
    if source is None or target is None:
        return None
    try:
        sr, sc = source.centroid_rc
        tr, tc = target.centroid_rc
        return [round(float(tc) - float(sc), 3), round(float(tr) - float(sr), 3)]
    except Exception:
        return None


def _color_component_entities(snapshot: ARGALiteSnapshot, *, limit: int) -> list[dict[str, Any]]:
    rows = list(snapshot.full_grid_hex_rows)
    if not rows:
        return []
    height = len(rows)
    width = len(rows[0])
    bg = _hex_symbol(max(snapshot.palette_histogram.items(), key=lambda item: int(item[1]))[0]) if snapshot.palette_histogram else None
    seen: set[tuple[int, int]] = set()
    entities: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            color = rows[y][x]
            if color == bg or (x, y) in seen:
                continue
            q: deque[tuple[int, int]] = deque([(x, y)])
            seen.add((x, y))
            cells: list[tuple[int, int]] = []
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in seen:
                        continue
                    if rows[ny][nx] != color:
                        continue
                    seen.add((nx, ny))
                    q.append((nx, ny))
            if len(cells) < 4:
                continue
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            local_rows = []
            cell_set = set(cells)
            for yy in range(y0, y1 + 1):
                local_rows.append("".join(rows[yy][xx] if (xx, yy) in cell_set else "." for xx in range(x0, x1 + 1)))
            entity_id = stable_hash(("color_component", color, x0, y0, x1, y1, len(cells), tuple(local_rows)), "scene_")
            entities.append({
                "id": entity_id,
                "source_type": "same_color_component",
                "color": color,
                "bbox_xyxy": [x0, y0, x1, y1],
                "centroid_xy": [round(sum(xs) / len(cells), 3), round(sum(ys) / len(cells), 3)],
                "area": len(cells),
                "palette_histogram": {color: len(cells)},
                "local_hex_rows": local_rows,
            })
    entities.sort(key=lambda item: (-int(item["area"]), item["bbox_xyxy"], item["id"]))
    return entities[:limit]


def _coordinate_candidates(snapshot: ARGALiteSnapshot, allowed_object_ids: set[str], allowed_relation_ids: set[str], config: V8Config) -> list[dict[str, Any]]:
    out = []
    for candidate in snapshot.coordinate_targets:
        if len(out) >= config.max_coordinate_candidates_in_packet:
            break
        if candidate.object_id is not None and candidate.object_id not in allowed_object_ids:
            continue
        if candidate.relation_id is not None and candidate.relation_id not in allowed_relation_ids:
            continue
        out.append({
            "id": candidate.candidate_id,
            "object_id": candidate.object_id,
            "relation_id": candidate.relation_id,
            "location_xy": [int(candidate.x), int(candidate.y)],
            "cell_value": _grid_value_at(snapshot, candidate.x, candidate.y),
            "location_role": candidate.source,
            "target_class": _target_class(candidate, snapshot),
            "salience": round(float(candidate.salience_score), 4),
            "reason": candidate.reason,
            "region_signature": candidate.region_signature,
            "target_signature": candidate.target_signature,
            "raw_xy_hidden_from_model": True,
        })
    return out


def _grid_value_at(snapshot: ARGALiteSnapshot, x: int, y: int) -> str | None:
    rows = snapshot.full_grid_hex_rows
    if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
        return str(rows[y][x])
    return None


def _control_model_section(memory: "GameMemory", snapshot: ARGALiteSnapshot, allowed_action_ids: list[str], target_object_ids: set[str], config: V8Config) -> dict[str, Any]:
    action_effect_table = _action_effect_table(memory, snapshot, allowed_action_ids, target_object_ids, config)
    control_model = _control_model(action_effect_table, allowed_action_ids, snapshot)
    return {
        "control_model": control_model,
    }


def _control_model(action_effect_table: list[dict[str, Any]], allowed_action_ids: list[str], snapshot: ARGALiteSnapshot) -> dict[str, Any]:
    actions: dict[str, Any] = {}
    step_sizes: list[float] = []
    observed_count = 0
    current_available = set(snapshot.available_actions)
    current_planning_surface = set(_planning_action_ids(snapshot))
    for row in action_effect_table:
        action_id = str(row.get("action_id"))
        motions: dict[str, list[float]] = {}
        for motion in row.get("object_motion") or []:
            object_id = str(motion.get("object_id"))
            delta_xy = motion.get("delta_centroid_xy")
            if not isinstance(delta_xy, list):
                continue
            motions[object_id] = delta_xy
            for value in delta_xy:
                try:
                    amount = abs(float(value))
                except Exception:
                    continue
                if amount > 1e-6:
                    step_sizes.append(amount)
        observed = row.get("observed") is True
        if observed:
            observed_count += 1
        surface_before = row.get("planning_action_ids_before") or row.get("available_actions_before") or []
        surface_after = row.get("planning_action_ids_after") or row.get("available_actions_after") or []
        actions[action_id] = {
            "observed": observed,
            "available_now": action_id in current_available,
            "planning_available_now": action_id in current_planning_surface,
            "observation_step_index": row.get("step_index"),
            "observation_level": row.get("level_index"),
            "observation_state_signature": row.get("state_signature"),
            "surface_before_matches_current": bool(surface_before) and set(surface_before) == current_planning_surface,
            "motions_xy": motions,
            "effects": _typed_action_effects(row),
            "changed_cell_count": row.get("changed_cell_count"),
            "changed_bbox_xyxy": _bbox_xyxy_from_rc(row.get("changed_bbox_rc")),
            "surface_before": surface_before,
            "surface_after": surface_after,
            "surface_added": row.get("planning_action_surface_added") or row.get("action_surface_added") or [],
            "surface_removed": row.get("planning_action_surface_removed") or row.get("action_surface_removed") or [],
            "terminal_or_level_progress": bool(row.get("terminal_or_level_progress")),
        }
        visual_changes = _visual_changes_for_packet(
            row.get("raw_visual_changes"),
            changed_cell_count=row.get("changed_cell_count"),
            surface_changed=bool(actions[action_id]["surface_added"] or actions[action_id]["surface_removed"]),
        )
        if visual_changes:
            actions[action_id]["simultaneous_raw_visual_change_summary"] = _visual_change_digest(visual_changes)
            actions[action_id]["detailed_visual_evidence"] = f"ACTION_SURFACE.observed_transitions step_index={row.get('step_index')}"
            actions[action_id]["visual_surface_causality_status"] = "COINCIDENT_OBSERVATION_CAUSAL_SEMANTICS_UNKNOWN"
    if not allowed_action_ids:
        status = "UNKNOWN"
    elif observed_count == len(allowed_action_ids):
        status = "KNOWN"
    elif observed_count > 0:
        status = "PARTIALLY_KNOWN"
    else:
        status = "UNKNOWN"
    return {
        "control_mapping_status": status,
        "known_scope": "state-scoped observations across encountered action surfaces",
        "coordinate_order": "x=column,y=row",
        "repeatability_status": "PROVISIONALLY_CONFIRMED" if status in {"KNOWN", "PARTIALLY_KNOWN"} else "UNKNOWN",
        "collision_behavior_status": "UNKNOWN",
        "boundary_behavior_status": "UNKNOWN",
        "action_mapping_state_invariant": "UNKNOWN",
        "effect_scope_contract": "Each action entry is the latest observed effect and is scoped by observation_state_signature plus surface_before. Revalidate after any state or action-surface transition.",
        "translation_step_size": _common_step_size(step_sizes),
        "actions": actions,
        "undo_action_ids": list(snapshot.undo_action_ids),
    }


def _typed_action_effects(row: dict[str, Any]) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    moving_ids: set[str] = set()
    for motion in row.get("object_motion") or []:
        if not isinstance(motion, dict) or motion.get("object_id") is None:
            continue
        object_id = str(motion["object_id"])
        moving_ids.add(object_id)
        effects.append({
            "kind": "translation",
            "object_id": object_id,
            "delta_xy": motion.get("delta_centroid_xy"),
            "direction": _direction_from_delta_xy(motion.get("delta_centroid_xy")),
        })
        extra = _nontranslation_effect(motion)
        if extra is not None:
            effects.append(extra)
    for change in row.get("stationary_or_local_changes") or []:
        if not isinstance(change, dict) or change.get("object_id") is None:
            continue
        object_id = str(change["object_id"])
        extra = _nontranslation_effect(change)
        effects.append(extra or {"kind": "local_visual_change", "object_id": object_id})
    for side_effect in row.get("side_effects") or []:
        if isinstance(side_effect, dict):
            effects.append({"kind": "unbound_region_change", **_deepcopy_jsonable(side_effect)})
    return effects


def _nontranslation_effect(change: dict[str, Any]) -> dict[str, Any] | None:
    object_id = change.get("object_id")
    if object_id is None:
        return None
    lifecycle = str(change.get("lifecycle") or "persisted")
    before_bbox = change.get("before_bbox_rc")
    after_bbox = change.get("after_bbox_rc")
    before_colors = change.get("before_colors") or []
    after_colors = change.get("after_colors") or []
    area_delta = change.get("area_delta")
    if lifecycle == "disappeared" or (before_bbox and not after_bbox):
        kind = "disappearance_or_merge"
    elif lifecycle == "appeared" or (after_bbox and not before_bbox):
        kind = "appearance_or_split"
    elif change.get("shape_changed") or area_delta not in (None, 0, 0.0):
        kind = "shape_area_or_visibility_change"
    elif change.get("palette_changed") or (before_colors != after_colors and before_colors and after_colors):
        kind = "palette_or_state_change"
    else:
        return None
    return {key: value for key, value in {
        "kind": kind,
        "object_id": str(object_id),
        "before_palette_ids": before_colors,
        "after_palette_ids": after_colors,
        "area_delta": area_delta,
    }.items() if value not in (None, [], {})}


def _common_step_size(values: list[float]) -> float | None:
    nonzero = sorted({round(v, 6) for v in values if v > 1e-6})
    if not nonzero:
        return None
    if len(nonzero) == 1:
        return nonzero[0]
    return None


def _is_successful_mechanics_probe(event: MemoryEvent) -> bool:
    return (
        event.reason_code == "typed_action_effect_observed"
        and event.relevance is Relevance.RELEVANT
        and event.contract_kind == "ACTION_EFFECT_DISCOVERY"
    )


def _recent_transition(memory: "GameMemory", allowed_object_ids: set[str], allowed_relation_ids: set[str]) -> dict[str, Any]:
    if not getattr(memory, "events", None):
        return {}
    event: MemoryEvent = memory.events[-1]
    action = event.action or {}
    reasoning = action.get("reasoning", {}) if isinstance(action, dict) else {}
    related_action_memory = None
    for record in reversed(getattr(memory, "action_memory_records", [])):
        if record.get("event_id") == event.event_id:
            related_action_memory = record
            break
    changed_relations = [rid for rid in (related_action_memory or {}).get("affected_relation_ids", []) if rid in allowed_relation_ids]
    return {
        "action_id": action.get("action_id") or action.get("id"),
        "coordinate_candidate_id": reasoning.get("coordinate_candidate_id"),
        "object_changes": _object_change_summaries(related_action_memory, allowed_object_ids),
        "relation_changes_status": "NOT_COMPUTED" if changed_relations else "NO_TRACKED_RELATION_CHANGE",
        "relation_changes": [{"relation_id": relation_id} for relation_id in changed_relations] or [],
        "action_surface_changed": bool((related_action_memory or {}).get("action_surface_added") or (related_action_memory or {}).get("action_surface_removed")),
        "score_delta": related_action_memory.get("score_delta") if related_action_memory else None,
        "terminal_progress": bool((related_action_memory or {}).get("terminal_delta") or (related_action_memory or {}).get("levels_completed_delta")),
    }


def _action_effect_table(memory: "GameMemory", snapshot: ARGALiteSnapshot, allowed_action_ids: list[str], target_object_ids: set[str], config: V8Config) -> list[dict[str, Any]]:
    records = [
        record
        for record in getattr(memory, "action_memory_records", [])[-config.max_action_memory_records_in_packet :]
        if str(record.get("action_id")) in set(allowed_action_ids)
        and _record_matches_snapshot_level(record, snapshot)
    ]
    latest_by_action: dict[str, dict[str, Any]] = {}
    for record in records:
        latest_by_action[str(record.get("action_id"))] = record
    table = []
    for action_id in allowed_action_ids:
        record = latest_by_action.get(action_id)
        if record is None:
            table.append({
                "action_id": action_id,
                "observed": False,
                "changes": "unknown",
                "moves_objects": "unknown",
                "object_motion": [],
                "side_effects": [],
            })
            continue
        object_deltas = record.get("object_deltas") or []
        motion = []
        stationary_or_local = []
        for item in object_deltas:
            object_id = item.get("object_id")
            if object_id not in target_object_ids:
                continue
            delta = item.get("delta_centroid_rc")
            direction = item.get("motion_direction")
            moved = _is_rigid_translation(item)
            compact = {
                "object_id": object_id,
                "delta_centroid_rc": delta,
                "delta_centroid_xy": _delta_rc_to_xy(delta),
                "motion_direction": direction,
                "lifecycle": item.get("lifecycle"),
                "before_bbox_rc": item.get("before_bbox_rc"),
                "after_bbox_rc": item.get("after_bbox_rc"),
                "before_colors": item.get("before_colors"),
                "after_colors": item.get("after_colors"),
                "before_shape_signature": item.get("before_shape_signature"),
                "after_shape_signature": item.get("after_shape_signature"),
                "shape_changed": item.get("shape_changed"),
                "palette_changed": item.get("palette_changed"),
                "area_delta": item.get("area_delta"),
            }
            compact = {key: value for key, value in compact.items() if value not in (None, [], {})}
            if moved:
                motion.append(compact)
            else:
                stationary_or_local.append(compact)
        side_effects = _side_effects(record, snapshot)
        table.append({
            "action_id": action_id,
            "observed": True,
            "step_index": record.get("step_index"),
            "level_index": record.get("level_index"),
            "state_signature": record.get("state_signature") or record.get("grid_hash_before"),
            "changed_cell_count": record.get("changed_cell_count"),
            "changed_bbox_rc": record.get("changed_bbox_rc"),
            "changed_color_delta": record.get("changed_color_delta"),
            "changes": bool(record.get("changed_cell_count") or object_deltas or side_effects),
            "moves_objects": bool(motion),
            "object_motion": motion,
            "stationary_or_local_changes": stationary_or_local,
            "side_effects": side_effects,
            "action_surface_added": record.get("planning_action_surface_added") or record.get("action_surface_added") or [],
            "action_surface_removed": record.get("planning_action_surface_removed") or record.get("action_surface_removed") or [],
            "available_actions_before": record.get("available_actions_before") or [],
            "available_actions_after": record.get("available_actions_after") or [],
            "planning_action_ids_before": record.get("planning_action_ids_before") or [],
            "planning_action_ids_after": record.get("planning_action_ids_after") or [],
            "raw_visual_changes": record.get("raw_visual_changes") or {},
            "terminal_or_level_progress": bool(record.get("terminal_delta") or record.get("levels_completed_delta")),
        })
    return table


def _record_matches_snapshot_level(record: dict[str, Any], snapshot: ARGALiteSnapshot) -> bool:
    level_value = record.get("level_index_before")
    if level_value is None:
        level_value = record.get("level_index")
    if level_value is None:
        return True
    try:
        return int(level_value) == int(getattr(snapshot, "level_index", 0))
    except (TypeError, ValueError):
        return False


def _is_rigid_translation(change: dict[str, Any]) -> bool:
    delta = change.get("delta_centroid_rc")
    if not _nonzero_delta(delta) or str(change.get("lifecycle") or "persisted") != "persisted":
        return False
    if change.get("shape_changed") is True or change.get("palette_changed") is True:
        return False
    if change.get("area_delta") not in (None, 0, 0.0):
        return False
    before_shape = change.get("before_shape_signature")
    after_shape = change.get("after_shape_signature")
    if before_shape and after_shape and before_shape != after_shape:
        return False
    before_bbox = change.get("before_bbox_rc")
    after_bbox = change.get("after_bbox_rc")
    if isinstance(before_bbox, (list, tuple)) and isinstance(after_bbox, (list, tuple)) and len(before_bbox) == 4 and len(after_bbox) == 4:
        before_hw = (int(before_bbox[2]) - int(before_bbox[0]), int(before_bbox[3]) - int(before_bbox[1]))
        after_hw = (int(after_bbox[2]) - int(after_bbox[0]), int(after_bbox[3]) - int(after_bbox[1]))
        if before_hw != after_hw:
            return False
    return True


def _nonzero_delta(delta: Any) -> bool:
    if not isinstance(delta, (list, tuple)):
        return False
    for value in delta:
        try:
            if abs(float(value)) > 1e-6:
                return True
        except Exception:
            return False
    return False


def _is_full_grid_container(obj: Any, snapshot: ARGALiteSnapshot) -> bool:
    bbox = getattr(obj, "bbox_rc", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    return int(bbox[0]) <= 0 and int(bbox[1]) <= 0 and int(bbox[2]) >= int(snapshot.height) - 1 and int(bbox[3]) >= int(snapshot.width) - 1


def _is_structural_strip(obj: Any, snapshot: ARGALiteSnapshot) -> bool:
    bbox = getattr(obj, "bbox_rc", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    r0, c0, r1, c1 = [int(v) for v in bbox]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = int(getattr(obj, "area", 0) or 0)
    snapshot_h = max(1, int(getattr(snapshot, "height", 0) or 1))
    snapshot_w = max(1, int(getattr(snapshot, "width", 0) or 1))
    if width <= 3 and height >= 0.70 * snapshot_h:
        return True
    if height <= 3 and width >= 0.70 * snapshot_w:
        return True
    if (r0 == 0 or c0 == 0 or r1 >= snapshot_h - 1 or c1 >= snapshot_w - 1) and area >= 0.70 * max(height, width):
        return True
    return False


def _shape_profile(obj: Any) -> dict[str, Any]:
    rows = list(getattr(obj, "local_mask_hex_rows", ()) or ())
    if not rows:
        return {}
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    row_counts = [sum(1 for ch in row if ch != ".") for row in rows]
    col_counts = []
    for x in range(width):
        col_counts.append(sum(1 for row in rows if x < len(row) and row[x] != "."))
    filled = sum(row_counts)
    return {
        "bbox_hw": [height, width],
        "fill_ratio": round(filled / max(1, height * width), 4),
        "row_occupancy": row_counts,
        "col_occupancy": col_counts,
    }


def _delta_rc_to_xy(delta: Any) -> list[float] | None:
    if not isinstance(delta, (list, tuple)) or len(delta) < 2:
        return None
    try:
        return [round(float(delta[1]), 3), round(float(delta[0]), 3)]
    except Exception:
        return None


def _side_effects(record: dict[str, Any], snapshot: ARGALiteSnapshot) -> list[dict[str, Any]]:
    out = []
    bbox = record.get("changed_bbox_rc")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        r0, c0, r1, c1 = [int(v) for v in bbox]
        if c1 >= max(0, snapshot.width - 2):
            out.append({
                "type": "edge_region_change",
                "region": "right_edge",
                "semantic_role": "UNKNOWN",
            })
        if r1 >= max(0, snapshot.height - 2):
            out.append({
                "type": "edge_region_change",
                "region": "bottom_edge",
                "semantic_role": "UNKNOWN",
            })
    added = record.get("planning_action_surface_added") or record.get("action_surface_added") or []
    removed = record.get("planning_action_surface_removed") or record.get("action_surface_removed") or []
    if added or removed:
        out.append({"type": "action_surface_change", "added": added, "removed": removed})
    return out


def _recent_evidence(
    memory: "GameMemory",
    bank: "HypothesisBank",
    snapshot: ARGALiteSnapshot,
    allowed_object_ids: set[str],
    allowed_relation_ids: set[str],
    allowed_candidate_ids: set[str],
    config: V8Config,
) -> dict[str, Any]:
    successful = []
    failed = []
    mechanics_probe_summary = []
    mechanics_probe_actions: set[str] = set()
    seen: set[tuple[Any, ...]] = set()
    current_event_ids = {
        str(record.get("event_id"))
        for record in getattr(memory, "action_memory_records", [])
        if _record_matches_snapshot_level(record, snapshot) and record.get("event_id")
    }
    for event in getattr(memory, "events", [])[-config.max_recent_transitions_in_packet :]:
        if current_event_ids and event.event_id not in current_event_ids:
            continue
        action = event.action or {}
        reasoning = action.get("reasoning", {}) if isinstance(action, dict) else {}
        candidate_id = reasoning.get("coordinate_candidate_id")
        if candidate_id and candidate_id not in allowed_candidate_ids:
            candidate_id = None
        signature = (event.before_hash, action.get("action_id") or action.get("id"), candidate_id, event.reason_code)
        if signature in seen:
            continue
        seen.add(signature)
        target_ids = _event_target_ids(action, allowed_object_ids, allowed_relation_ids)
        item = {
            "state_signature": event.before_hash,
            "action_id": action.get("action_id") or action.get("id"),
            "candidate_id": candidate_id,
            "outcome": event.reason_code,
            "target_ids": target_ids,
        }
        if _is_successful_mechanics_probe(event):
            action_id = str(action.get("action_id") or action.get("id") or "")
            if action_id and action_id not in mechanics_probe_actions:
                mechanics_probe_actions.add(action_id)
                mechanics_probe_summary.append({
                    "action_id": action_id,
                    "outcome": "CONFIRMED_MOTION_OR_EFFECT",
                })
            continue
        if event.progress.value == "POSITIVE":
            successful.append(item)
        else:
            failed.append({
                **item,
                "judgment": event.relevance.value,
                "reason": event.summary,
                "retry_allowed": False,
            })
    for rejection in getattr(bank, "invalid_rejections", [])[-config.max_memory_notes_in_packet :]:
        failed.append({
            "state_signature": snapshot.semantic_state_signature,
            "action_id": rejection.get("action_id"),
            "candidate_id": rejection.get("coordinate_candidate_id"),
            "judgment": "INVALID",
            "reason": str(rejection.get("reason") or "invalid_qwen_candidate")[:180],
            "retry_allowed": False,
            "target_ids": [],
        })
    return {
        "mechanics_probe_summary": mechanics_probe_summary[-config.max_memory_notes_in_packet :],
        "successful_steps": successful[-config.max_memory_notes_in_packet :],
        "failed_or_irrelevant_steps": failed[-config.max_memory_notes_in_packet :],
        "open_questions": _open_questions(memory, config),
    }


def _action_surface_transitions(memory: "GameMemory", snapshot: ARGALiteSnapshot, config: V8Config, allowed_object_ids: set[str] | None = None) -> list[dict[str, Any]]:
    out = []
    current_surface = set(_planning_action_ids(snapshot))
    for record in getattr(memory, "action_surface_memory_records", []):
        level_value = record.get("level_index_before")
        if level_value is None:
            level_value = record.get("level_index", -1)
        if int(level_value) != int(snapshot.level_index):
            continue
        before = [str(value) for value in record.get("planning_action_ids_before") or record.get("available_actions_before") or []]
        after = [str(value) for value in record.get("planning_action_ids_after") or record.get("available_actions_after") or []]
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        if not added and not removed:
            continue
        before_matches_current = set(before) == current_surface
        after_matches_current = set(after) == current_surface
        if before_matches_current:
            current_surface_position = "OBSERVED_BEFORE"
        elif after_matches_current:
            current_surface_position = "OBSERVED_AFTER"
        else:
            current_surface_position = "DIFFERENT_SURFACE"
        affected = [
            str(object_id) for object_id in record.get("affected_object_ids") or []
            if allowed_object_ids is None or str(object_id) in allowed_object_ids
        ]
        rigid_translations = []
        for change in record.get("object_deltas") or []:
            object_id = str(change.get("object_id"))
            if allowed_object_ids is not None and object_id not in allowed_object_ids:
                continue
            if not _is_rigid_translation(change):
                continue
            delta_xy = _delta_rc_to_xy(change.get("delta_centroid_rc"))
            rigid_translations.append({
                "object_id": object_id,
                "delta_xy": delta_xy,
                "direction": _direction_from_delta_xy(delta_xy),
            })
        visual_changes = _visual_changes_for_packet(
            record.get("raw_visual_changes"),
            changed_cell_count=record.get("changed_cell_count"),
            surface_changed=bool(added or removed),
        )
        transition = {
            "step_index": record.get("step_index"),
            "state_signature_before": record.get("state_signature"),
            "trigger_action_id": record.get("action_id"),
            "available_before": before,
            "available_after": after,
            "added": added,
            "removed": removed,
            "surface_changed": bool(added or removed),
            "forward_source_matches_current": before_matches_current,
            "current_surface_position": current_surface_position,
            "replay_effect_status": (
                "FORWARD_TRANSITION_OBSERVED_FROM_CURRENT_SURFACE"
                if before_matches_current
                else "NOT_APPLICABLE_TO_CURRENT_SURFACE_EFFECT_UNKNOWN_OR_REVERSIBLE"
            ),
            "transition_class": "SCENE_AND_ACTION_SURFACE_CHANGED" if (added or removed) and record.get("changed_cell_count") else "NO_ACTION_SURFACE_CHANGE",
            "goal_status": "OBSERVED_TRANSITION_NOT_A_PRESELECTED_GOAL",
            "changed_cell_count": record.get("changed_cell_count"),
            "changed_bbox_xyxy": _bbox_xyxy_from_rc(record.get("changed_bbox_rc")),
            "affected_object_ids": affected,
            "simultaneous_rigid_translations": rigid_translations,
        }
        if visual_changes:
            transition["simultaneous_raw_visual_changes"] = visual_changes
            transition["visual_and_surface_changed_same_transition"] = bool(added or removed)
            transition["causal_semantics_status"] = "UNKNOWN_OBSERVED_COINCIDENCE_ONLY"
        out.append(transition)
    return out[-config.max_action_memory_records_in_packet :]


def _visual_changes_for_packet(raw: Any, *, changed_cell_count: Any, surface_changed: bool) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    repeated = raw.get("repeated_isolated_interior_change") is True
    try:
        compact_surface_change = surface_changed and int(changed_cell_count or 0) <= 96
    except Exception:
        compact_surface_change = False
    if not repeated and not compact_surface_change:
        return None
    return _deepcopy_jsonable(raw)


def _visual_change_digest(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "changed_cell_count": raw.get("changed_cell_count"),
        "repeated_isolated_interior_change": raw.get("repeated_isolated_interior_change") is True,
        "reciprocal_isolated_interior_transition_pairs": _deepcopy_jsonable(raw.get("reciprocal_isolated_interior_transition_pairs") or []),
        "local_3x3_transition_group_count": len(raw.get("local_3x3_transition_groups") or []),
    }


def _level_experience(memory: "GameMemory", snapshot: ARGALiteSnapshot, config: V8Config) -> dict[str, Any]:
    by_level: dict[int, list[dict[str, Any]]] = {}
    for record in getattr(memory, "action_memory_records", [])[-config.max_action_memory_records_in_packet * 2 :]:
        level = int(record.get("level_index_before", record.get("level_index", 0)) or 0)
        by_level.setdefault(level, []).append(record)
    completed = []
    current = []
    for level, records in sorted(by_level.items()):
        terminal_records = [
            record for record in records
            if record.get("terminal_delta") or record.get("levels_completed_delta") or record.get("level_index_delta")
        ]
        item = {
            "level_index": level,
            "action_runs": _action_runs(records, config.max_qwen_trajectory_steps),
            "terminal_or_level_progress_observed": bool(terminal_records),
            "control_or_affordance_change_actions": list(dict.fromkeys(
                str(record.get("action_id")) for record in records
                if record not in terminal_records
                and (record.get("planning_action_surface_added") or record.get("planning_action_surface_removed"))
            )),
        }
        terminal_record = terminal_records[-1] if terminal_records else None
        if terminal_record is not None:
            item["terminal_transition"] = {
                "action_id": terminal_record.get("action_id"),
                "source": terminal_record.get("source"),
                "hypothesis_id": terminal_record.get("hypothesis_id"),
                "semantic_hypothesis": terminal_record.get("hypothesis_claim"),
            }
        if level < int(snapshot.level_index) or item["terminal_or_level_progress_observed"]:
            completed.append(item)
        elif level == int(snapshot.level_index):
            current.append(item)
    return {
        "scope": "same_game_only; cleared when game_id changes",
        "completed_level_summaries": completed[-4:],
        "current_level_action_history": current[-1:] if current else [],
    }


def _level_attempt_context(memory: "GameMemory", snapshot: ARGALiteSnapshot, config: V8Config) -> dict[str, Any]:
    records = [
        _deepcopy_jsonable(record)
        for record in getattr(memory, "level_attempt_records", [])
        if int(record.get("level_index", -1)) == int(snapshot.level_index)
    ]
    return {
        "current_attempt_index": int(getattr(memory, "current_attempt_index", lambda _level: 0)(snapshot.level_index)),
        "policy": "one primary semantic call per attempt; when latest-frame coordinate research is required, one coordinate call precedes primary; RESET starts a new attempt with retained game memory",
        "previous_failed_attempts": records[-max(1, int(config.max_memory_notes_in_packet)) :],
    }


def _action_runs(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for record in records[-max(1, int(limit)) :]:
        action_id = str(record.get("action_id") or "")
        if not action_id:
            continue
        source = str(record.get("source") or "unknown")
        if runs and runs[-1]["action_id"] == action_id and runs[-1]["source"] == source:
            runs[-1]["count"] += 1
        else:
            runs.append({"action_id": action_id, "count": 1, "source": source})
    return runs


def _validate_packet_references(packet: dict[str, Any]) -> None:
    scene = packet["scene"]
    constraints = packet["execution_constraints"]
    object_ids = {item["id"] for item in scene["objects"]}
    relation_ids = {item["id"] for item in scene["relations"]}
    candidate_ids = {item["id"] for item in scene["coordinate_candidates"]}
    action_ids = {item["id"] for item in packet["action_surface"]["actions"]}
    assert set(constraints["allowed_object_ids"]) == object_ids
    assert set(constraints["allowed_relation_ids"]) == relation_ids
    assert set(constraints["allowed_coordinate_candidate_ids"]) == candidate_ids
    assert set(constraints["allowed_action_ids"]).issubset(action_ids)
    component_graph = scene.get("component_graph")
    if isinstance(component_graph, dict):
        component_ids = {
            str(item.get("id"))
            for item in component_graph.get("components", [])
            if isinstance(item, dict) and item.get("id") is not None
        }
        for component in component_graph.get("components", []):
            if not isinstance(component, dict):
                continue
            assert set(component.get("object_refs") or []).issubset(object_ids)
            parent = component.get("parent")
            if parent is not None:
                assert str(parent) in component_ids
            assert set(component.get("children") or []).issubset(component_ids)
        for edge in component_graph.get("adjacency", []):
            assert str(edge.get("a")) in component_ids
            assert str(edge.get("b")) in component_ids
        for group in component_graph.get("same_shape_groups", []):
            assert set(group.get("component_ids") or []).issubset(component_ids)
        for candidate in component_graph.get("background_candidates_not_facts", []):
            assert str(candidate.get("component_id")) in component_ids
    for obj in scene["objects"]:
        geometry = obj.get("shape_geometry") if isinstance(obj, dict) else None
        if isinstance(geometry, dict):
            assert set(geometry.get("same_exact_geometry_object_ids") or []).issubset(object_ids)
    for group in scene.get("exact_geometry_groups", []):
        assert set(group.get("object_ids") or []).issubset(object_ids)
    control_group_ids = {str(group.get("id")) for group in scene.get("control_groups", [])}
    for candidate in scene.get("control_state_transition_candidates", []):
        for key in (
            "shared_translated_object_ids",
            "before_only_translated_object_ids",
            "after_only_translated_object_ids",
        ):
            assert set(candidate.get(key) or []).issubset(object_ids)
        assert str(candidate.get("before_control_group_id")) in control_group_ids
        assert str(candidate.get("after_control_group_id")) in control_group_ids
        for marker in candidate.get("marker_evidence") or []:
            assert set(marker.get("object_ids") or []).issubset(object_ids)
    for relation in scene["relations"]:
        if "object_ids" in relation:
            assert set(relation["object_ids"]).issubset(object_ids)
        else:
            assert relation["source_object_id"] in object_ids
            assert relation["target_object_id"] in object_ids
    for candidate in scene["coordinate_candidates"]:
        if candidate.get("object_id") is not None:
            assert candidate["object_id"] in object_ids
        if candidate.get("relation_id") is not None:
            assert candidate["relation_id"] in relation_ids


def _planning_action_ids(snapshot: ARGALiteSnapshot) -> list[str]:
    if snapshot.planning_action_ids:
        return list(snapshot.planning_action_ids)
    undo = set(snapshot.undo_action_ids)
    return [action_id for action_id in snapshot.available_actions if action_id not in undo]


def _allowed_action_ids(snapshot: ARGALiteSnapshot, role: QwenRole, memory: "GameMemory") -> list[str]:
    if role is QwenRole.COORDINATE:
        return list(snapshot.coordinate_action_ids)
    known = {
        record.action_id
        for record in getattr(memory, "action_effects", {}).values()
        if getattr(record, "confidence", 0.0) >= 0.45
    }
    observed_surface: set[str] = set(_planning_action_ids(snapshot))
    for record in getattr(memory, "action_surface_memory_records", []):
        if int(record.get("level_index", -1)) != int(snapshot.level_index):
            continue
        for key in ("planning_action_ids_before", "planning_action_ids_after"):
            observed_surface.update(str(value) for value in record.get(key) or [])
    undo = set(snapshot.undo_action_ids)
    allowed = [
        action_id for action_id in sorted(observed_surface)
        if action_id in known and action_id not in undo
    ]
    if role is QwenRole.RESERVE:
        allowed.extend(action_id for action_id in snapshot.undo_action_ids if action_id in snapshot.available_actions)
    return list(dict.fromkeys(allowed))


def _bbox_xyxy(bbox_rc: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox_rc
    return [c0, r0, c1, r1]


def _centroid_xy(centroid_rc: tuple[float, float]) -> list[float]:
    return [round(float(centroid_rc[1]), 3), round(float(centroid_rc[0]), 3)]


def _hex_symbol(value: Any) -> str:
    try:
        ivalue = int(value)
    except Exception:
        return str(value)
    if 0 <= ivalue <= 15:
        return "0123456789ABCDEF"[ivalue]
    return str(ivalue)


def _target_class(candidate: Any, snapshot: ARGALiteSnapshot) -> str:
    if candidate.object_id:
        obj = next((item for item in snapshot.objects if item.object_id == candidate.object_id), None)
        if obj is not None:
            height = obj.bbox_rc[2] - obj.bbox_rc[0] + 1
            width = obj.bbox_rc[3] - obj.bbox_rc[1] + 1
            if len(obj.colors) > 1:
                return "multicolor_connected_object"
            if height <= 5 and width <= 5:
                return "compact_connected_object"
            return "connected_object"
    if candidate.relation_id:
        return "relation"
    return "region"


def _effect_target_class(target_signature: str | None) -> str:
    if not target_signature:
        return "action_surface"
    if target_signature.startswith("rel_"):
        return "relation"
    return "target"


def _effect_scope(outcome: str) -> str:
    if outcome in {"moved", "not_moved"}:
        return "object_local"
    if outcome in {"error_decreased", "unchanged", "error_increased"}:
        return "relation_local"
    if outcome in {"progress", "no_progress", "negative"}:
        return "global"
    return "target_local"


def _effect_text(outcome: str) -> str:
    mapping = {
        "effect": "produces verifier-observed change",
        "no_effect": "no verifier-observed semantic effect",
        "not_moved": "target object did not move",
        "unchanged": "target relation did not improve",
        "no_progress": "no score or terminal progress",
        "negative_effect": "negative or invalid transition effect",
    }
    return mapping.get(str(outcome), str(outcome))


def _mechanism_text(action_id: str, outcome: str) -> str:
    if action_id in ACTION_EFFECT_PROBE_IDS:
        return f"simple action with verifier-classified outcome: {outcome}"
    return f"available action with verifier-classified outcome: {outcome}"


def _object_change_summaries(record: dict[str, Any] | None, allowed_object_ids: set[str]) -> list[dict[str, Any]]:
    out = []
    if not record:
        return out
    for item in record.get("object_deltas") or []:
        object_id = item.get("object_id")
        if object_id not in allowed_object_ids:
            continue
        before_bbox = _bbox_xyxy_from_rc(item.get("before_bbox_rc"))
        after_bbox = _bbox_xyxy_from_rc(item.get("after_bbox_rc"))
        before_centroid = _centroid_xy_from_rc(item.get("before_centroid_rc"))
        after_centroid = _centroid_xy_from_rc(item.get("after_centroid_rc"))
        out.append({
            "object_id": object_id,
            "before": {key: value for key, value in {
                "bbox_xyxy": before_bbox,
                "centroid_xy": before_centroid,
            }.items() if value is not None},
            "after": {key: value for key, value in {
                "bbox_xyxy": after_bbox,
                "centroid_xy": after_centroid,
            }.items() if value is not None},
            "delta_xy": _delta_rc_to_xy(item.get("delta_centroid_rc")),
        })
    return out[:8]


def _bbox_xyxy_from_rc(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        r0, c0, r1, c1 = [int(v) for v in bbox]
    except Exception:
        return None
    return [c0, r0, c1, r1]


def _centroid_xy_from_rc(centroid: Any) -> list[float] | None:
    if not isinstance(centroid, (list, tuple)) or len(centroid) < 2:
        return None
    try:
        return [round(float(centroid[1]), 3), round(float(centroid[0]), 3)]
    except Exception:
        return None


def _event_target_ids(action: dict[str, Any], allowed_object_ids: set[str], allowed_relation_ids: set[str]) -> list[str]:
    reasoning = action.get("reasoning", {}) if isinstance(action, dict) else {}
    contract = reasoning.get("verification_contract")
    ids: list[str] = []
    if contract in allowed_object_ids or contract in allowed_relation_ids:
        ids.append(str(contract))
    return ids


def _open_questions(memory: "GameMemory", config: V8Config) -> list[dict[str, Any]]:
    out = []
    for question in getattr(memory, "semantic_questions", {}).values():
        if question.resolved_outcome is not None:
            continue
        out.append({
            "question_id": question.question_id,
            "question_type": question.question_type.value,
            "target_signature": question.target_signature,
            "observations": dict(question.observations),
        })
    return out[: config.max_memory_notes_in_packet]
