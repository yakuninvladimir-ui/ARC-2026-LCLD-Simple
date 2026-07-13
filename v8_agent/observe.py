from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

HEX = "0123456789ABCDEF"


def stable_hash(payload: Any, prefix: str = "") -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return prefix + hashlib.sha256(raw).hexdigest()[:16]


def normalize_grid(raw_grid: Any) -> tuple[tuple[int, ...], ...]:
    if raw_grid is None:
        raise ValueError("observation has no grid/frame")
    raw_grid = _collapse_frame_axes(raw_grid)
    if hasattr(raw_grid, "tolist"):
        raw_grid = raw_grid.tolist()
    rows = list(raw_grid)
    if not rows:
        raise ValueError("grid is empty")
    out: list[tuple[int, ...]] = []
    width: int | None = None
    for row in rows:
        if hasattr(row, "tolist"):
            row = row.tolist()
        vals = tuple(int(v) for v in row)
        if width is None:
            width = len(vals)
            if width <= 0:
                raise ValueError("grid row is empty")
        if len(vals) != width:
            raise ValueError("grid rows have inconsistent widths")
        if len(out) >= 64 or width > 64:
            raise ValueError(f"grid exceeds 64x64 contract: h>{len(out)}, w={width}")
        out.append(vals)
    return tuple(out)


def _collapse_frame_axes(raw_grid: Any) -> Any:
    ndim = getattr(raw_grid, "ndim", None)
    shape = getattr(raw_grid, "shape", None)
    if ndim is not None and shape is not None:
        try:
            ndim_i = int(ndim)
        except Exception:
            ndim_i = 0
        if ndim_i == 3:
            if int(shape[2]) <= 4:
                return raw_grid[:, :, 0]
            return raw_grid[-1]
        if ndim_i == 4:
            latest = raw_grid[-1]
            latest_ndim = int(getattr(latest, "ndim", 0) or 0)
            latest_shape = getattr(latest, "shape", ())
            if latest_ndim == 3 and len(latest_shape) >= 3 and int(latest_shape[2]) <= 4:
                return latest[:, :, 0]
            if latest_ndim == 3:
                return latest[-1]
            return latest
    return raw_grid


def encode_grid_hex_rows(grid: tuple[tuple[int, ...], ...]) -> tuple[str, ...]:
    rows: list[str] = []
    for row in grid:
        chars: list[str] = []
        for value in row:
            ivalue = int(value)
            if not 0 <= ivalue <= 15:
                raise ValueError(f"palette id outside 0..15: {ivalue}")
            chars.append(HEX[ivalue])
        rows.append("".join(chars))
    return tuple(rows)


def palette_histogram(grid: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for row in grid:
        for v in row:
            hist[int(v)] = hist.get(int(v), 0) + 1
    return dict(sorted(hist.items()))


def grid_hash(grid: tuple[tuple[int, ...], ...]) -> str:
    return stable_hash(encode_grid_hex_rows(grid), "g_")
