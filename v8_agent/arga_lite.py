from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from math import log1p

from .config import V8Config
from .observe import HEX, encode_grid_hex_rows, grid_hash, palette_histogram, stable_hash
from .relations import build_relations
from .types import ARGALiteSnapshot, CoordinateTargetCandidate, ObjectRecord, WorldState


def _bg_color(grid: tuple[tuple[int, ...], ...]) -> int:
    h, w = len(grid), len(grid[0])
    border: list[int] = []
    border.extend(grid[0])
    if h > 1:
        border.extend(grid[-1])
    for r in range(1, h - 1):
        border.append(grid[r][0])
        if w > 1:
            border.append(grid[r][-1])
    border_hist = Counter(border)
    global_hist = Counter(v for row in grid for v in row)
    border_color, border_count = border_hist.most_common(1)[0]
    if border_count / max(1, len(border)) >= 0.45:
        return int(border_color)
    return int(global_hist.most_common(1)[0][0])


class ARGALiteBuilder:
    def build(self, state: WorldState, memory: "GameMemory", config: V8Config) -> ARGALiteSnapshot:
        height = len(state.grid)
        width = len(state.grid[0])
        hist = palette_histogram(state.grid)
        full_rows = encode_grid_hex_rows(state.grid)
        raw_objects = self._extract_objects(state.grid, config)
        objects = memory.assign_object_tracks(state.level_index, state.step_index, raw_objects) if config.preserve_object_tracks else raw_objects
        relations = build_relations(objects, config.max_relations_in_packet)
        coordinate_ids = tuple(a for a in state.available_actions if _is_coordinate_action(a, state.raw))
        coordinate_slots = _official_coordinate_slots(state.raw, width, height)
        targets = generate_coordinate_targets(
            state.grid,
            objects,
            relations,
            memory,
            config,
            official_slots=coordinate_slots,
        ) if coordinate_ids else ()
        gh = grid_hash(state.grid)
        object_hash = stable_hash([(o.object_id, o.stable_hash, o.bbox_rc) for o in objects], "o_")
        relation_hash = stable_hash([(r.relation_id, r.metric_value) for r in relations], "rel_")
        semantic = stable_hash((object_hash, relation_hash, state.available_actions, state.state_name), "sem_")
        return ARGALiteSnapshot(
            snapshot_id=stable_hash((state.game_id, state.level_index, state.step_index, gh, semantic), "snap_"),
            game_id=state.game_id,
            level_index=state.level_index,
            step_index=state.step_index,
            height=height,
            width=width,
            palette_ids_seen=tuple(sorted(hist)),
            palette_histogram=hist,
            coordinate_order="x=col,y=row",
            full_grid_hex_rows=full_rows,
            objects=objects,
            relations=relations,
            coordinate_targets=targets,
            available_actions=state.available_actions,
            coordinate_action_ids=coordinate_ids,
            grid_hash=gh,
            object_hash=object_hash,
            relation_hash=relation_hash,
            semantic_state_signature=semantic,
            score=state.score,
            terminal=state.terminal,
            state_name=state.state_name,
            levels_completed=state.levels_completed,
            win_levels=state.win_levels,
            game_over=state.game_over,
            full_reset=state.full_reset,
            planning_action_ids=state.planning_action_ids,
            undo_action_ids=state.undo_action_ids,
            possible_actions=state.possible_actions,
        )

    def _extract_objects(self, grid: tuple[tuple[int, ...], ...], config: V8Config) -> tuple[ObjectRecord, ...]:
        h, w = len(grid), len(grid[0])
        bg = _bg_color(grid)
        seen: set[tuple[int, int]] = set()
        objects: list[ObjectRecord] = []
        for r in range(h):
            for c in range(w):
                if (r, c) in seen or grid[r][c] == bg:
                    continue
                cells = _component(grid, r, c, bg, seen, merge_colors=config.merge_multicolor_components)
                objects.append(_object_from_cells(grid, cells, h, w))
        if config.merge_multicolor_components:
            existing_bboxes = {
                obj.bbox_rc
                for obj in objects
                if not _spans_grid_like(obj, h, w) and obj.area >= 4
            }
            for obj in _same_color_recovery_objects(grid, bg, h, w):
                if obj.bbox_rc in existing_bboxes:
                    continue
                objects.append(obj)
                existing_bboxes.add(obj.bbox_rc)
        return _balanced_object_selection(objects, config.max_objects_in_packet, h, w)


def _component(
    grid: tuple[tuple[int, ...], ...],
    sr: int,
    sc: int,
    background: int,
    seen: set[tuple[int, int]],
    *,
    merge_colors: bool,
) -> list[tuple[int, int]]:
    h, w = len(grid), len(grid[0])
    start_color = grid[sr][sc]
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    seen.add((sr, sc))
    cells: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                continue
            value = grid[nr][nc]
            if value == background:
                continue
            if not merge_colors and value != start_color:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return cells


def _same_color_recovery_objects(grid: tuple[tuple[int, ...], ...], bg: int, height: int, width: int) -> list[ObjectRecord]:
    seen: set[tuple[int, int]] = set()
    out: list[ObjectRecord] = []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen or grid[r][c] == bg:
                continue
            cells = _component(grid, r, c, bg, seen, merge_colors=False)
            if len(cells) < 4:
                continue
            out.append(_object_from_cells(grid, cells, height, width))
    return out


def _object_from_cells(grid: tuple[tuple[int, ...], ...], cells: list[tuple[int, int]], height: int, width: int) -> ObjectRecord:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    cell_set = set(cells)
    hist: Counter[int] = Counter(int(grid[r][c]) for r, c in cells)
    local_rows: list[str] = []
    bool_rows: list[str] = []
    for r in range(r0, r1 + 1):
        local_chars: list[str] = []
        bool_chars: list[str] = []
        for c in range(c0, c1 + 1):
            if (r, c) in cell_set:
                local_chars.append(HEX[int(grid[r][c])])
                bool_chars.append("1")
            else:
                local_chars.append(".")
                bool_chars.append("0")
        local_rows.append("".join(local_chars))
        bool_rows.append("".join(bool_chars))
    holes = _count_holes(bool_rows)
    shape_signature = stable_hash((r1 - r0 + 1, c1 - c0 + 1, bool_rows), "shape_")
    topology_signature = stable_hash((shape_signature, holes, len(hist)), "topo_")
    intrinsic = stable_hash((tuple(sorted(hist.items())), shape_signature, holes), "intrinsic_")
    frame_stable = stable_hash((r0, c0, r1, c1, intrinsic), "frameobj_")
    frame_object_id = "obj_" + frame_stable.split("_", 1)[-1][:10]
    border: list[str] = []
    if r0 == 0: border.append("top")
    if r1 == height - 1: border.append("bottom")
    if c0 == 0: border.append("left")
    if c1 == width - 1: border.append("right")
    sym: list[str] = []
    if bool_rows == list(reversed(bool_rows)):
        sym.append("horizontal")
    if all(row == row[::-1] for row in bool_rows):
        sym.append("vertical")
    area = len(cells)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    tags: list[str] = []
    if area == 1:
        tags.append("singleton")
    if border:
        tags.append("border_touching")
    if holes > 0:
        tags.append("hollow")
    if len(hist) > 1:
        tags.append("multicolor_cluster")
    if r0 == r1 or c0 == c1:
        tags.append("line_like")
    perimeter_min = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 3 and area >= perimeter_min and area < bbox_area:
        tags.append("frame_like")
    if bbox_area > 0 and area / bbox_area < 0.45:
        tags.append("sparse")
    salience = float(area) + (4.0 if area == 1 else 0.0) + len(border) * 0.2 + holes * 1.5 + (1.0 if len(hist) > 1 else 0.0)
    return ObjectRecord(
        object_id=frame_object_id,
        bbox_rc=(r0, c0, r1, c1),
        centroid_rc=(sum(rows) / area, sum(cols) / area),
        area=area,
        colors=tuple(sorted(hist)),
        color_histogram=dict(sorted(hist.items())),
        shape_signature=shape_signature,
        local_mask_hex_rows=tuple(local_rows),
        holes=holes,
        symmetry_hints=tuple(sym),
        border_touching=tuple(border),
        tags=tuple(tags),
        stable_hash=intrinsic,
        salience_score=salience,
        track_id="",
        frame_object_id=frame_object_id,
        topology_signature=topology_signature,
    )


def _spans_grid_like(obj: ObjectRecord, height: int, width: int) -> bool:
    r0, c0, r1, c1 = obj.bbox_rc
    return r0 <= 0 and c0 <= 0 and r1 >= height - 1 and c1 >= width - 1


def _balanced_object_selection(
    objects: list[ObjectRecord],
    limit: int,
    height: int,
    width: int,
) -> tuple[ObjectRecord, ...]:
    """Keep large context, repeated motifs, rare shapes, and spatial coverage."""
    if limit <= 0 or not objects:
        return ()
    if len(objects) <= limit:
        return tuple(sorted(objects, key=lambda o: (-o.salience_score, o.bbox_rc, o.object_id)))

    shape_counts = Counter(obj.shape_signature for obj in objects)
    color_counts = Counter(color for obj in objects for color in obj.colors)
    selected: dict[str, ObjectRecord] = {}

    def add(obj: ObjectRecord) -> None:
        if len(selected) < limit:
            selected.setdefault(obj.object_id, obj)

    # Preserve a bounded amount of global context without allowing it to consume
    # the entire object budget.
    for obj in sorted(objects, key=lambda o: (-o.area, o.bbox_rc, o.object_id))[: min(8, limit)]:
        add(obj)

    # Repeated exact masks are common clue/target vocabularies. Keep several
    # members from every large group before generic salience ranking.
    by_shape: dict[str, list[ObjectRecord]] = {}
    for obj in objects:
        by_shape.setdefault(obj.shape_signature, []).append(obj)
    repeated = sorted(by_shape.values(), key=lambda group: (-len(group), -max(item.area for item in group), group[0].shape_signature))
    for group in repeated:
        if len(group) < 2:
            continue
        for obj in sorted(group, key=lambda o: (o.bbox_rc, o.object_id))[:3]:
            add(obj)
        if len(selected) >= limit:
            break

    # Guarantee coarse spatial coverage so a dense panel in one corner does not
    # hide isolated objects elsewhere in the frame.
    bins: dict[tuple[int, int], list[ObjectRecord]] = {}
    for obj in objects:
        br = min(3, max(0, int(4 * obj.centroid_rc[0] / max(1, height))))
        bc = min(3, max(0, int(4 * obj.centroid_rc[1] / max(1, width))))
        bins.setdefault((br, bc), []).append(obj)
    for key in sorted(bins):
        candidate = max(bins[key], key=lambda o: (_object_diversity_score(o, shape_counts, color_counts), -o.area, o.object_id))
        add(candidate)

    ranked = sorted(
        objects,
        key=lambda o: (-_object_diversity_score(o, shape_counts, color_counts), o.bbox_rc, o.object_id),
    )
    for obj in ranked:
        add(obj)
        if len(selected) >= limit:
            break
    return tuple(sorted(selected.values(), key=lambda o: (-o.salience_score, o.bbox_rc, o.object_id)))


def _object_diversity_score(obj: ObjectRecord, shape_counts: Counter[str], color_counts: Counter[int]) -> float:
    bbox_height = obj.bbox_rc[2] - obj.bbox_rc[0] + 1
    bbox_width = obj.bbox_rc[3] - obj.bbox_rc[1] + 1
    compact = bbox_height <= 16 and bbox_width <= 16
    rare_shape = 10.0 / max(1, shape_counts[obj.shape_signature])
    rare_color = sum(4.0 / max(1, color_counts[color]) for color in obj.colors)
    repeated_shape = min(8.0, float(shape_counts[obj.shape_signature])) if shape_counts[obj.shape_signature] > 1 else 0.0
    return (
        log1p(max(1, obj.area))
        + rare_shape
        + rare_color
        + repeated_shape
        + (4.0 if compact else 0.0)
        + (3.0 if len(obj.colors) > 1 else 0.0)
        + (2.0 if obj.holes > 0 else 0.0)
    )


def _count_holes(bool_rows: list[str]) -> int:
    if not bool_rows:
        return 0
    h, w = len(bool_rows), len(bool_rows[0])
    seen: set[tuple[int, int]] = set()
    holes = 0
    for r in range(h):
        for c in range(w):
            if bool_rows[r][c] != "0" or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr in {0, h - 1} or cc in {0, w - 1}:
                    touches_border = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool_rows[nr][nc] == "0" and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _is_coordinate_action(action_id: str, raw: dict | None = None) -> bool:
    meta = (raw or {}).get("metadata", {}) if isinstance(raw, dict) else {}
    explicit = meta.get("coordinate_action_ids") or meta.get("coordinate_actions")
    if explicit and str(action_id) in {str(x) for x in explicit}:
        return True
    return str(action_id).upper() in {"ACTION6", "CLICK", "TAP", "POINTER", "SELECT"}


def generate_coordinate_targets(
    grid: tuple[tuple[int, ...], ...],
    objects: tuple[ObjectRecord, ...],
    relations: tuple,
    memory: "GameMemory",
    config: V8Config,
    *,
    official_slots: tuple[tuple[int, int], ...] | None = None,
) -> tuple[CoordinateTargetCandidate, ...]:
    h, w = len(grid), len(grid[0])
    seen: set[tuple[int, int, str | None, str | None]] = set()
    candidates: list[CoordinateTargetCandidate] = []

    def add(x: int, y: int, source: str, obj: ObjectRecord | None, rel_id: str | None, reason: str, salience: float) -> None:
        if not (0 <= x <= 63 and 0 <= y <= 63 and 0 <= x < w and 0 <= y < h):
            return
        object_id = obj.object_id if obj else None
        key = (x, y, object_id, rel_id)
        if key in seen:
            return
        seen.add(key)
        if source == "official_coordinate_slot":
            target_signature = stable_hash((source, x, y), "target_")
        else:
            target_signature = stable_hash((source, object_id, rel_id), "target_") if object_id or rel_id else stable_hash((source, x, y), "target_")
        region_sig = stable_hash((source, object_id, rel_id, x, y), "reg_")
        # Candidate identity follows the stable object/relation signature, not its current coordinates.
        cid = stable_hash((source, object_id, rel_id, target_signature), "ct_")
        candidates.append(CoordinateTargetCandidate(cid, x, y, source, object_id, rel_id, region_sig, reason, salience, target_signature))

    if official_slots is not None:
        background = _bg_color(grid)
        for x, y in official_slots:
            obj = _object_at_slot(objects, x, y)
            value = int(grid[y][x])
            local_colors = {
                int(grid[yy][xx])
                for yy in range(max(0, y - 1), min(h, y + 2))
                for xx in range(max(0, x - 1), min(w, x + 2))
            }
            salience = 1.0 + (1.5 if value != background else 0.0) + min(1.5, 0.3 * len(local_colors))
            if obj is not None:
                salience += 1.0 + min(2.0, log1p(max(1, obj.area)) / 3.0)
            reason = f"official legal coordinate slot [{x},{y}]; categorical cell value {HEX[value] if 0 <= value < len(HEX) else value}"
            if obj is not None:
                reason += f"; intersects {obj.object_id}"
            add(x, y, "official_coordinate_slot", obj, None, reason, salience)
        candidates.sort(key=lambda c: (-c.salience_score, c.y, c.x, c.candidate_id))
        return tuple(candidates[: config.max_coordinate_candidates_in_packet])

    for obj in objects:
        r0, c0, r1, c1 = obj.bbox_rc
        cy, cx = int(round(obj.centroid_rc[0])), int(round(obj.centroid_rc[1]))
        compact_bonus = 2.0 if (r1 - r0 + 1) <= 16 and (c1 - c0 + 1) <= 16 else 0.0
        salience = log1p(max(1, obj.area)) + compact_bonus + (1.0 if len(obj.colors) > 1 else 0.0)
        occupied = _occupied_cell_near(obj, cx, cy)
        if occupied is not None:
            ox, oy = occupied
            add(ox, oy, "occupied_near_centroid", obj, None, f"occupied cell nearest centroid of {obj.object_id}", salience + 0.5)
        else:
            add(cx, cy, "centroid", obj, None, f"centroid of {obj.object_id}", salience)

    by_id = {o.object_id: o for o in objects}
    for rel in relations:
        if rel.relation_type not in {"unique_symbol_pair", "separated_by_gap", "line_continuation", "button_like_structure", "frame_contains"}:
            continue
        a, b = by_id.get(rel.a), by_id.get(rel.b)
        if a is None or b is None:
            continue
        x = int(round((a.centroid_rc[1] + b.centroid_rc[1]) / 2))
        y = int(round((a.centroid_rc[0] + b.centroid_rc[0]) / 2))
        add(x, y, "relation_hotspot", None, rel.relation_id, f"hotspot for {rel.relation_type} {rel.relation_id}", rel.salience_score + 1.0)

    changed = memory.last_changed_region_center()
    if changed is not None:
        add(changed[0], changed[1], "last_changed_region", None, None, "center of last changed region", 1.2)
    empty = _largest_empty_region_center(grid)
    if empty is not None:
        x, y, size = empty
        add(x, y, "empty_region_center", None, None, f"center of empty region size {size}", 0.5 + min(size, 20) / 20)
    add(w // 2, h // 2, "grid_center", None, None, "grid center", 0.25)
    candidates.sort(key=lambda c: (-c.salience_score, c.candidate_id, c.x, c.y))
    return tuple(candidates[: config.max_coordinate_candidates_in_packet])


def _official_coordinate_slots(raw: dict | None, width: int, height: int) -> tuple[tuple[int, int], ...] | None:
    metadata = (raw or {}).get("metadata", {}) if isinstance(raw, dict) else {}
    if not isinstance(metadata, dict) or "coordinate_slots" not in metadata:
        return None
    out: list[tuple[int, int]] = []
    for value in metadata.get("coordinate_slots") or []:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        try:
            row, col = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            continue
        x, y = col, row
        if 0 <= x < width and 0 <= y < height:
            out.append((x, y))
    slots = tuple(dict.fromkeys(out))
    return slots or None


def _object_at_slot(objects: tuple[ObjectRecord, ...], x: int, y: int) -> ObjectRecord | None:
    containing = [
        obj for obj in objects
        if obj.bbox_rc[0] <= y <= obj.bbox_rc[2] and obj.bbox_rc[1] <= x <= obj.bbox_rc[3]
    ]
    if not containing:
        return None

    def key(obj: ObjectRecord) -> tuple[int, int, float, str]:
        r0, c0, _, _ = obj.bbox_rc
        local_y, local_x = y - r0, x - c0
        occupied = (
            0 <= local_y < len(obj.local_mask_hex_rows)
            and 0 <= local_x < len(obj.local_mask_hex_rows[local_y])
            and obj.local_mask_hex_rows[local_y][local_x] != "."
        )
        return (0 if occupied else 1, obj.area, -float(obj.salience_score), obj.object_id)

    return min(containing, key=key)


def _occupied_cell_near(obj: ObjectRecord, center_x: int, center_y: int) -> tuple[int, int] | None:
    r0, c0, _, _ = obj.bbox_rc
    cells: list[tuple[int, int]] = []
    for local_y, row in enumerate(obj.local_mask_hex_rows):
        for local_x, value in enumerate(row):
            if value != ".":
                cells.append((c0 + local_x, r0 + local_y))
    if not cells:
        return None
    return min(cells, key=lambda cell: (abs(cell[0] - center_x) + abs(cell[1] - center_y), cell[1], cell[0]))


def _largest_empty_region_center(grid: tuple[tuple[int, ...], ...]) -> tuple[int, int, int] | None:
    bg = _bg_color(grid)
    h, w = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    best: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            region: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == bg and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            if len(region) > len(best):
                best = region
    if not best:
        return None
    return int(round(sum(c for _, c in best) / len(best))), int(round(sum(r for r, _ in best) / len(best))), len(best)
