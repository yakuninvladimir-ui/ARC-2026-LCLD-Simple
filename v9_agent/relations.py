from __future__ import annotations

from math import hypot

from .observe import stable_hash
from .types import ObjectRecord, RelationRecord


def build_relations(objects: tuple[ObjectRecord, ...], max_relations: int) -> tuple[RelationRecord, ...]:
    records: list[RelationRecord] = []

    def add(
        relation_type: str,
        a: ObjectRecord,
        b: ObjectRecord,
        metric_name: str | None = None,
        metric_value: float | None = None,
        confidence: float = 1.0,
        salience: float = 1.0,
    ) -> None:
        signature = stable_hash((relation_type, a.object_id, b.object_id), "relsig_")
        records.append(RelationRecord(
            relation_id="rel_" + signature.split("_", 1)[-1][:12],
            relation_type=relation_type,
            a=a.object_id,
            b=b.object_id,
            metric_name=metric_name,
            metric_value=None if metric_value is None else float(metric_value),
            confidence=float(confidence),
            salience_score=float(salience),
            relation_signature=signature,
        ))

    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            center_distance = hypot(a.centroid_rc[0] - b.centroid_rc[0], a.centroid_rc[1] - b.centroid_rc[1])
            gap = _bbox_gap(a.bbox_rc, b.bbox_rc)
            if set(a.colors).intersection(b.colors):
                add("same_color", a, b, confidence=1.0, salience=1.1)
            if a.shape_signature == b.shape_signature:
                add("same_shape", a, b, "centroid_distance", center_distance, 1.0, 1.5)
                add("translated_shape", a, b, "centroid_distance", center_distance, 0.95, 1.6)
            elif a.area == b.area:
                mirror_axis = _mirror_match(a, b)
                if mirror_axis is not None:
                    add("mirror_candidate", a, b, f"mirror_axis_{mirror_axis}", 0.0, 0.9, 1.65)
                rotation = _rotation_match(a, b)
                if rotation is not None:
                    add("rotation_candidate", a, b, f"rotation_degrees_{rotation}", 0.0, 0.9, 1.6)
            if abs(a.centroid_rc[0] - b.centroid_rc[0]) <= 0.5:
                add("aligned_row", a, b, "delta_col", abs(a.centroid_rc[1] - b.centroid_rc[1]), 0.9, 1.0)
            if abs(a.centroid_rc[1] - b.centroid_rc[1]) <= 0.5:
                add("aligned_col", a, b, "delta_row", abs(a.centroid_rc[0] - b.centroid_rc[0]), 0.9, 1.0)
            if gap <= 1.0:
                add("near", a, b, "bbox_gap", gap, 0.9, 0.8)
            elif gap <= max(8.0, 0.5 * center_distance):
                add("separated_by_gap", a, b, "gap_distance", gap, 0.85, 1.3)
            if _contains(a.bbox_rc, b.bbox_rc):
                add("contains", a, b, "containment_outside_distance", 0.0, 1.0, 1.4)
                if "frame_like" in a.tags or a.holes > 0:
                    add("frame_contains", a, b, "containment_outside_distance", 0.0, 1.0, 1.8)
            elif _contains(b.bbox_rc, a.bbox_rc):
                add("contains", b, a, "containment_outside_distance", 0.0, 1.0, 1.4)
                if "frame_like" in b.tags or b.holes > 0:
                    add("frame_contains", b, a, "containment_outside_distance", 0.0, 1.0, 1.8)
            if _line_like(a) and _line_like(b):
                add("line_continuation", a, b, "line_endpoint_distance", gap, 0.75, 1.5)
            if _button_like(a):
                add("button_like_structure", a, b, "centroid_distance", center_distance, 0.55, 1.2)
            if _button_like(b):
                add("button_like_structure", b, a, "centroid_distance", center_distance, 0.55, 1.2)
            if a.centroid_rc[1] < b.centroid_rc[1]:
                add("left_of", a, b, "delta_col", b.centroid_rc[1] - a.centroid_rc[1], 0.7, 0.4)
                add("right_of", b, a, "delta_col", b.centroid_rc[1] - a.centroid_rc[1], 0.7, 0.4)
            if a.centroid_rc[0] < b.centroid_rc[0]:
                add("above", a, b, "delta_row", b.centroid_rc[0] - a.centroid_rc[0], 0.7, 0.4)
                add("below", b, a, "delta_row", b.centroid_rc[0] - a.centroid_rc[0], 0.7, 0.4)

    shape_groups: dict[tuple[str, tuple[int, ...]], list[ObjectRecord]] = {}
    for obj in objects:
        shape_groups.setdefault((obj.shape_signature, obj.colors), []).append(obj)
    for group in shape_groups.values():
        if len(group) == 2:
            add("unique_symbol_pair", group[0], group[1], "centroid_distance", hypot(
                group[0].centroid_rc[0] - group[1].centroid_rc[0],
                group[0].centroid_rc[1] - group[1].centroid_rc[1],
            ), 0.85, 2.0)

    repeated_shape_groups: dict[str, list[ObjectRecord]] = {}
    for obj in objects:
        repeated_shape_groups.setdefault(obj.shape_signature, []).append(obj)
    for group in repeated_shape_groups.values():
        if len(group) < 3:
            continue
        ordered = sorted(group, key=lambda item: item.object_id)
        for first, second in zip(ordered, ordered[1:]):
            add("repeated_pattern", first, second, "pattern_support", float(len(group)), 0.9, 1.7)

    # Global top-K after all pairs are considered. Metric values never participate in identity.
    dedup: dict[str, RelationRecord] = {}
    for record in records:
        previous = dedup.get(record.relation_id)
        if previous is None or record.salience_score > previous.salience_score:
            dedup[record.relation_id] = record
    return tuple(sorted(dedup.values(), key=lambda r: (-r.salience_score, r.relation_type, r.a, r.b))[:max_relations])


def relation_error(snapshot: object, relation_id: str) -> float | None:
    for relation in getattr(snapshot, "relations", ()):
        if relation.relation_id == relation_id:
            return relation.metric_value
    return None


def relation_by_id(snapshot: object, relation_id: str) -> RelationRecord | None:
    return next((r for r in getattr(snapshot, "relations", ()) if r.relation_id == relation_id), None)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return hypot(dr, dc)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 and oc0 <= ic0 and or1 >= ir1 and oc1 >= ic1 and outer != inner


def _line_like(obj: ObjectRecord) -> bool:
    r0, c0, r1, c1 = obj.bbox_rc
    return obj.area >= 2 and (r0 == r1 or c0 == c1 or "line_like" in obj.tags)


def _button_like(obj: ObjectRecord) -> bool:
    r0, c0, r1, c1 = obj.bbox_rc
    return obj.area <= 9 and (r1 - r0) <= 2 and (c1 - c0) <= 2 and "border_touching" not in obj.tags


def _occupancy_mask(obj: ObjectRecord) -> tuple[str, ...]:
    return tuple("".join("0" if value == "." else "1" for value in row) for row in obj.local_mask_hex_rows)


def _mirror_match(first: ObjectRecord, second: ObjectRecord) -> str | None:
    source = _occupancy_mask(first)
    target = _occupancy_mask(second)
    if not source or not target:
        return None
    if tuple(row[::-1] for row in source) == target:
        return "vertical"
    if tuple(reversed(source)) == target:
        return "horizontal"
    return None


def _rotation_match(first: ObjectRecord, second: ObjectRecord) -> int | None:
    source = _occupancy_mask(first)
    target = _occupancy_mask(second)
    if not source or not target:
        return None
    current = source
    for degrees in (90, 180, 270):
        current = _rotate_clockwise(current)
        if current == target:
            return degrees
    return None


def _rotate_clockwise(mask: tuple[str, ...]) -> tuple[str, ...]:
    if not mask:
        return ()
    width = len(mask[0])
    if any(len(row) != width for row in mask):
        return ()
    return tuple("".join(mask[row][column] for row in range(len(mask) - 1, -1, -1)) for column in range(width))
