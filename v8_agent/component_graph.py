from __future__ import annotations

from collections import Counter, deque
from math import log1p
from typing import Any, Iterable

from .observe import stable_hash


_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR_INDEX = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


def build_component_graph(
    hex_rows: Iterable[str],
    *,
    object_records: Iterable[tuple[str, Any]] = (),
    max_components: int = 96,
    max_shape_runs: int = 32,
    max_boundary_corners: int = 32,
) -> dict[str, Any]:
    """Build a compact all-color topological graph for one observed frame.

    Components partition every cell into equal-symbol 4-connected regions. They are
    geometric evidence, not replacements for the tracked multicolor object model.
    """
    rows = tuple(str(row) for row in hex_rows)
    if not rows:
        return _empty_graph()
    width = len(rows[0])
    if width <= 0 or any(len(row) != width for row in rows):
        raise ValueError("component graph requires a non-empty rectangular frame")

    height = len(rows)
    components, owner = _segment(rows)
    shared_edges = _shared_edges(owner)
    adjacency = _adjacency_sets(len(components), shared_edges)
    parents, children = _topological_nesting(components, adjacency)
    object_cells = _object_cell_index(object_records)

    for component in components:
        refs: set[str] = set()
        for cell in component["cells"]:
            refs.update(object_cells.get(cell, ()))
        component["object_refs"] = sorted(refs)

    selected_indices = _select_components(components, max_components, height, width)
    selected = set(selected_indices)
    component_ids = {index: f"c{index}" for index in selected_indices}
    nodes = []
    for index in selected_indices:
        component = components[index]
        parent = parents[index]
        selected_children = [component_ids[child] for child in children[index] if child in selected]
        node: dict[str, Any] = {
            "id": component_ids[index],
            "color": component["color"],
            "area": component["area"],
            "bbox_xyxy": component["bbox_xyxy"],
            "centroid_xy": component["centroid_xy"],
            "fill_ratio": component["fill_ratio"],
            "shape_hash": component["shape_hash"],
            "color_shape_hash": component["color_shape_hash"],
            "frame_signature": component["frame_signature"],
            "hole_count": component["hole_count"],
            "edge_contacts": component["edge_contacts"],
            "object_refs": component["object_refs"],
            "parent": component_ids.get(parent),
            "children": selected_children,
        }
        if parent is not None and parent not in selected:
            node["parent_omitted"] = True
        omitted_children = len(children[index]) - len(selected_children)
        if omitted_children:
            node["omitted_child_count"] = omitted_children

        shape_runs = component["shape_runs"]
        if len(shape_runs) <= max(0, int(max_shape_runs)):
            node["normalized_shape_runs"] = shape_runs

        corners = _outer_boundary_corners(component["cells"])
        corner_count = len(corners)
        node["outer_boundary_corner_count"] = corner_count
        if corner_count <= max(0, int(max_boundary_corners)):
            node["outer_boundary_corners_xy"] = corners
        elif "normalized_shape_runs" not in node:
            node["geometry_detail_omitted"] = "complexity_limit; use hashes, topology, object refs, and current_frame"
        nodes.append(node)

    edges = [
        {
            "a": component_ids[a],
            "b": component_ids[b],
            "shared_edge_cells": count,
        }
        for (a, b), count in sorted(shared_edges.items())
        if a in selected and b in selected
    ]
    omitted_incident_edges = sum(
        1
        for a, b in shared_edges
        if (a in selected) != (b in selected)
    )
    same_shape_groups = _same_shape_groups(components, selected_indices, component_ids)
    background_candidates = _background_candidates(components, selected_indices, component_ids, height, width)

    return {
        "version": "component_graph_v1",
        "contract": (
            "Every frame cell belongs to exactly one same-color 4-connected component. "
            "A component is geometric evidence, not automatically one gameplay object; "
            "a multicolor object may reference several components. Component IDs are "
            "observation-only: response trajectories may use only linked object_refs."
        ),
        "connectivity": 4,
        "covers_all_frame_cells": True,
        "shape_hash_semantics": "translation_invariant_and_color_independent",
        "color_shape_hash_semantics": "translation_invariant_and_color_sensitive",
        "boundary_coordinate_space": "grid_vertex_xy; x=column,y=row",
        "component_id_scope": "current_frame_reading_order; use hashes and object_refs across frames",
        "frame_shape_hw": [height, width],
        "complete": len(selected_indices) == len(components),
        "component_count_total": len(components),
        "component_count_included": len(selected_indices),
        "omitted_component_count": len(components) - len(selected_indices),
        "omitted_incident_adjacency_count": omitted_incident_edges,
        "background_candidates_not_facts": background_candidates,
        "same_shape_groups": same_shape_groups,
        "components": nodes,
        "adjacency": edges,
    }


def _empty_graph() -> dict[str, Any]:
    return {
        "version": "component_graph_v1",
        "connectivity": 4,
        "covers_all_frame_cells": True,
        "frame_shape_hw": [0, 0],
        "complete": True,
        "component_count_total": 0,
        "component_count_included": 0,
        "components": [],
        "adjacency": [],
        "same_shape_groups": [],
        "background_candidates_not_facts": [],
    }


def _segment(rows: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[list[int]]]:
    height = len(rows)
    width = len(rows[0])
    owner = [[-1] * width for _ in range(height)]
    components: list[dict[str, Any]] = []
    for start_y in range(height):
        for start_x in range(width):
            if owner[start_y][start_x] >= 0:
                continue
            color = rows[start_y][start_x]
            index = len(components)
            queue = deque([(start_y, start_x)])
            owner[start_y][start_x] = index
            cells: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for dy, dx in _ORTH:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    if owner[ny][nx] >= 0 or rows[ny][nx] != color:
                        continue
                    owner[ny][nx] = index
                    queue.append((ny, nx))
            components.append(_component_record(index, color, cells, height, width))
    return components, owner


def _component_record(index: int, color: str, cells: list[tuple[int, int]], height: int, width: int) -> dict[str, Any]:
    cells.sort()
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    y0, y1 = min(ys), max(ys)
    x0, x1 = min(xs), max(xs)
    shape_runs = _normalized_shape_runs(cells, y0, x0)
    bbox_height = y1 - y0 + 1
    bbox_width = x1 - x0 + 1
    shape_key = (bbox_height, bbox_width, tuple(shape_runs))
    shape_hash = stable_hash(shape_key, "cshape_")
    color_shape_hash = stable_hash((color, shape_key), "ccshape_")
    edge_contacts = []
    if y0 == 0:
        edge_contacts.append("top")
    if y1 == height - 1:
        edge_contacts.append("bottom")
    if x0 == 0:
        edge_contacts.append("left")
    if x1 == width - 1:
        edge_contacts.append("right")
    area = len(cells)
    return {
        "index": index,
        "color": color,
        "cells": tuple(cells),
        "area": area,
        "bbox_xyxy": [x0, y0, x1, y1],
        "centroid_xy": [round(sum(xs) / area, 3), round(sum(ys) / area, 3)],
        "fill_ratio": round(area / max(1, bbox_height * bbox_width), 4),
        "shape_runs": shape_runs,
        "shape_hash": shape_hash,
        "color_shape_hash": color_shape_hash,
        "frame_signature": stable_hash((color_shape_hash, x0, y0, x1, y1), "cmp_"),
        "hole_count": _count_holes(set(cells), y0, x0, y1, x1),
        "edge_contacts": edge_contacts,
        "object_refs": [],
    }


def _normalized_shape_runs(cells: list[tuple[int, int]], y0: int, x0: int) -> list[str]:
    by_row: dict[int, list[int]] = {}
    for y, x in cells:
        by_row.setdefault(y - y0, []).append(x - x0)
    encoded = []
    for local_y in sorted(by_row):
        values = sorted(by_row[local_y])
        ranges: list[str] = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = value
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        encoded.append(f"{local_y}:{','.join(ranges)}")
    return encoded


def _count_holes(cells: set[tuple[int, int]], y0: int, x0: int, y1: int, x1: int) -> int:
    seen: set[tuple[int, int]] = set()
    holes = 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (y, x) in cells or (y, x) in seen:
                continue
            queue = deque([(y, x)])
            seen.add((y, x))
            touches_bbox = False
            while queue:
                cy, cx = queue.popleft()
                if cy in {y0, y1} or cx in {x0, x1}:
                    touches_bbox = True
                for dy, dx in _ORTH:
                    ny, nx = cy + dy, cx + dx
                    if not (y0 <= ny <= y1 and x0 <= nx <= x1):
                        continue
                    if (ny, nx) in cells or (ny, nx) in seen:
                        continue
                    seen.add((ny, nx))
                    queue.append((ny, nx))
            if not touches_bbox:
                holes += 1
    return holes


def _shared_edges(owner: list[list[int]]) -> Counter[tuple[int, int]]:
    height = len(owner)
    width = len(owner[0]) if height else 0
    counts: Counter[tuple[int, int]] = Counter()
    for y in range(height):
        for x in range(width):
            current = owner[y][x]
            if y + 1 < height and owner[y + 1][x] != current:
                other = owner[y + 1][x]
                counts[(min(current, other), max(current, other))] += 1
            if x + 1 < width and owner[y][x + 1] != current:
                other = owner[y][x + 1]
                counts[(min(current, other), max(current, other))] += 1
    return counts


def _adjacency_sets(count: int, shared_edges: Counter[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(count)]
    for a, b in shared_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _topological_nesting(components: list[dict[str, Any]], adjacency: list[set[int]]) -> tuple[list[int | None], list[list[int]]]:
    count = len(components)
    edge_nodes = {
        index
        for index, component in enumerate(components)
        if component["edge_contacts"]
    }
    enclosers = [set() for _ in range(count)]
    candidates = [
        index
        for index, component in enumerate(components)
        if component["area"] >= 4
        and component["bbox_xyxy"][2] - component["bbox_xyxy"][0] + 1 >= 3
        and component["bbox_xyxy"][3] - component["bbox_xyxy"][1] + 1 >= 3
    ]
    all_nodes = set(range(count))
    for blocked in candidates:
        reachable = set(edge_nodes)
        reachable.discard(blocked)
        queue = deque(sorted(reachable))
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor == blocked or neighbor in reachable:
                    continue
                reachable.add(neighbor)
                queue.append(neighbor)
        for enclosed in all_nodes - reachable - {blocked}:
            enclosers[enclosed].add(blocked)

    parents: list[int | None] = [None] * count
    children = [[] for _ in range(count)]
    for index, values in enumerate(enclosers):
        if not values:
            continue
        parent = max(values, key=lambda candidate: (len(enclosers[candidate]), -components[candidate]["area"], -candidate))
        parents[index] = parent
        children[parent].append(index)
    for values in children:
        values.sort()
    return parents, children


def _object_cell_index(object_records: Iterable[tuple[str, Any]]) -> dict[tuple[int, int], set[str]]:
    index: dict[tuple[int, int], set[str]] = {}
    for alias, record in object_records:
        bbox = getattr(record, "bbox_rc", None)
        rows = getattr(record, "local_mask_hex_rows", None)
        if not alias or not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not rows:
            continue
        y0, x0 = int(bbox[0]), int(bbox[1])
        for local_y, row in enumerate(rows):
            for local_x, value in enumerate(str(row)):
                if value == ".":
                    continue
                index.setdefault((y0 + local_y, x0 + local_x), set()).add(str(alias))
    return index


def _select_components(components: list[dict[str, Any]], limit: int, height: int, width: int) -> list[int]:
    limit = max(1, int(limit))
    if len(components) <= limit:
        return list(range(len(components)))

    shape_counts = Counter(component["shape_hash"] for component in components)
    color_counts = Counter(component["color"] for component in components)
    frame_area = max(1, height * width)

    def score(component: dict[str, Any]) -> tuple[float, int]:
        repeated = shape_counts[component["shape_hash"]]
        value = (
            1000.0 * bool(component["object_refs"])
            + 80.0 * min(4, repeated - 1)
            + 100.0 * component["area"] / frame_area
            + 20.0 / max(1, color_counts[component["color"]])
            + 4.0 * bool(component["edge_contacts"])
            + log1p(component["area"])
        )
        return value, -component["index"]

    selected: dict[int, None] = {}

    def add(component: dict[str, Any]) -> None:
        if len(selected) < limit:
            selected.setdefault(component["index"], None)

    for component in sorted(components, key=lambda item: (-bool(item["object_refs"]), -len(item["object_refs"]), -item["area"], item["index"])):
        if component["object_refs"]:
            add(component)
    for component in sorted(components, key=lambda item: (-item["area"], item["index"]))[:8]:
        add(component)
    for color in sorted(color_counts):
        add(max((item for item in components if item["color"] == color), key=lambda item: (item["area"], -item["index"])))
    repeated_groups: dict[str, list[dict[str, Any]]] = {}
    for component in components:
        repeated_groups.setdefault(component["shape_hash"], []).append(component)
    for group in sorted(repeated_groups.values(), key=lambda values: (-len(values), -max(item["area"] for item in values), values[0]["shape_hash"])):
        if len(group) < 2:
            continue
        for component in sorted(group, key=lambda item: item["index"])[:3]:
            add(component)
        if len(selected) >= limit:
            break
    for component in sorted(components, key=lambda item: (-score(item)[0], -score(item)[1])):
        add(component)
        if len(selected) >= limit:
            break
    return sorted(selected)


def _same_shape_groups(components: list[dict[str, Any]], selected_indices: list[int], component_ids: dict[int, str]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for index in selected_indices:
        groups.setdefault(components[index]["shape_hash"], []).append(index)
    out = []
    for shape_hash, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(indices) < 2:
            continue
        colors = [components[index]["color"] for index in indices]
        out.append({
            "shape_hash": shape_hash,
            "component_ids": [component_ids[index] for index in indices],
            "colors": colors,
            "same_geometry_different_colors": len(set(colors)) > 1,
        })
    return out[:32]


def _background_candidates(
    components: list[dict[str, Any]],
    selected_indices: list[int],
    component_ids: dict[int, str],
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    frame_area = max(1, height * width)
    ranked = sorted(
        selected_indices,
        key=lambda index: (-components[index]["area"], -len(components[index]["edge_contacts"]), index),
    )
    return [
        {
            "component_id": component_ids[index],
            "color": components[index]["color"],
            "area_fraction": round(components[index]["area"] / frame_area, 4),
            "edge_contacts": components[index]["edge_contacts"],
            "status": "CANDIDATE_NOT_FACT",
        }
        for index in ranked[:3]
    ]


def _outer_boundary_corners(cells: tuple[tuple[int, int], ...]) -> list[list[int]]:
    cell_set = set(cells)
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for y, x in cells:
        if (y - 1, x) not in cell_set:
            edges.add(((x, y), (x + 1, y)))
        if (y, x + 1) not in cell_set:
            edges.add(((x + 1, y), (x + 1, y + 1)))
        if (y + 1, x) not in cell_set:
            edges.add(((x + 1, y + 1), (x, y + 1)))
        if (y, x - 1) not in cell_set:
            edges.add(((x, y + 1), (x, y)))
    if not edges:
        return []

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
    unvisited = set(edges)
    loops: list[list[tuple[int, int]]] = []
    while unvisited:
        first = min(unvisited)
        start, end = first
        loop = [start]
        current = first
        for _ in range(len(edges) + 1):
            if current not in unvisited:
                break
            unvisited.remove(current)
            _, end = current
            loop.append(end)
            if end == start:
                break
            candidates = [
                (end, candidate_end)
                for candidate_end in outgoing.get(end, ())
                if (end, candidate_end) in unvisited
            ]
            if not candidates:
                break
            current = min(candidates, key=lambda edge: _turn_rank(loop[-2], edge[0], edge[1]))
        if len(loop) >= 4 and loop[-1] == loop[0]:
            loops.append(loop[:-1])
    if not loops:
        return []
    outer = max(loops, key=lambda loop: abs(_signed_area(loop)))
    corners = _remove_collinear_vertices(outer)
    return [[int(x), int(y)] for x, y in corners]


def _turn_rank(previous: tuple[int, int], current: tuple[int, int], following: tuple[int, int]) -> tuple[int, int, int]:
    incoming = (current[0] - previous[0], current[1] - previous[1])
    outgoing = (following[0] - current[0], following[1] - current[1])
    incoming_index = _DIR_INDEX.get(incoming, 0)
    outgoing_index = _DIR_INDEX.get(outgoing, 0)
    turn = (outgoing_index - incoming_index) % 4
    priority = {1: 0, 0: 1, 3: 2, 2: 3}.get(turn, 4)
    return priority, following[1], following[0]


def _remove_collinear_vertices(loop: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(loop) <= 3:
        return loop
    corners = []
    for index, current in enumerate(loop):
        previous = loop[index - 1]
        following = loop[(index + 1) % len(loop)]
        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (following[0] - current[0], following[1] - current[1])
        if incoming != outgoing:
            corners.append(current)
    return corners


def _signed_area(loop: list[tuple[int, int]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(loop):
        x2, y2 = loop[(index + 1) % len(loop)]
        area += x1 * y2 - x2 * y1
    return area / 2.0
