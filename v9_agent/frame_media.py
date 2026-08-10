from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from typing import Iterable, Sequence


# ARC-AGI-3 logical palette from arc_agi.rendering. Keeping the table local
# makes the competition payload independent of the optional rendering package.
ARC_AGI_3_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),
    (204, 204, 204),
    (153, 153, 153),
    (102, 102, 102),
    (51, 51, 51),
    (0, 0, 0),
    (229, 58, 163),
    (255, 123, 204),
    (249, 60, 49),
    (30, 147, 255),
    (136, 216, 241),
    (255, 220, 0),
    (255, 133, 27),
    (146, 18, 49),
    (79, 204, 48),
    (163, 86, 214),
)

# High-contrast overlay colors cycled by planning-object index.
_ANNOTATION_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 64, 64),
    (64, 220, 64),
    (64, 140, 255),
    (255, 200, 32),
    (255, 96, 220),
    (32, 230, 230),
    (255, 140, 64),
    (180, 120, 255),
)

# Compact 5x7 bitmap font for labels (bits row-major, MSB left).
_FONT_5X7: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("01110", "10001", "00001", "00110", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "b": ("10000", "10000", "11110", "10001", "10001", "10001", "11110"),
    "c": ("00000", "00000", "01110", "10001", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01111", "10001", "10001", "10001", "01111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "00000", "01111", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "j": ("00010", "00000", "00110", "00010", "00010", "10010", "01100"),
    "k": ("10000", "10000", "10010", "10100", "11000", "10100", "10010"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10001", "10001"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "p": ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    "q": ("00000", "00000", "01111", "10001", "01111", "00001", "00001"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    "t": ("01000", "01000", "11100", "01000", "01000", "01001", "00110"),
    "u": ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "00000", "10001", "10001", "10001", "01010", "00100"),
    "w": ("00000", "00000", "10001", "10001", "10101", "10101", "01010"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    "y": ("00000", "00000", "10001", "10001", "01111", "00001", "01110"),
    "z": ("00000", "00000", "11111", "00010", "00100", "01000", "11111"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def current_frame_png(hex_rows: Iterable[str], *, cell_scale: int = 8) -> dict[str, object]:
    rows = tuple(str(row).upper() for row in hex_rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("current frame PNG requires a non-empty rectangular grid")
    if any(char not in "0123456789ABCDEF" for row in rows for char in row):
        raise ValueError("current frame PNG accepts palette symbols 0-F only")

    scale = max(1, int(cell_scale))
    source_height = len(rows)
    source_width = len(rows[0])
    png_bytes = _encode_png(rows, scale)
    seen = sorted(set("".join(rows)))
    palette = {
        symbol: "#%02X%02X%02X" % ARC_AGI_3_RGB[int(symbol, 16)]
        for symbol in seen
    }
    return {
        "attachment_id": "current_frame_png",
        "media_type": "image/png",
        "encoding": "base64",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
        "sha256": hashlib.sha256(png_bytes).hexdigest(),
        "grid_shape_hw": [source_height, source_width],
        "image_shape_hw": [source_height * scale, source_width * scale],
        "cell_scale": scale,
        "coordinate_order": "x=column,y=row; origin=top_left",
        "palette_id_to_rgb": palette,
    }


def annotated_frame_png(
    hex_rows: Iterable[str],
    annotations: Sequence[dict[str, object]],
    *,
    cell_scale: int = 8,
) -> dict[str, object]:
    """Render the grid PNG with planning-object bboxes and short alias labels.

    ``annotations`` items use packet-facing fields:
      - label: short alias such as ``obj0`` (required for text)
      - bbox_xyxy: [x0, y0, x1, y1] inclusive grid coordinates
    """
    rows = tuple(str(row).upper() for row in hex_rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("annotated frame PNG requires a non-empty rectangular grid")
    if any(char not in "0123456789ABCDEF" for row in rows for char in row):
        raise ValueError("annotated frame PNG accepts palette symbols 0-F only")

    scale = max(1, int(cell_scale))
    source_height = len(rows)
    source_width = len(rows[0])
    image = _rasterize_grid(rows, scale)
    drawn: list[dict[str, object]] = []
    for index, item in enumerate(annotations):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if x1 < x0 or y1 < y0:
            continue
        if x1 < 0 or y1 < 0 or x0 >= source_width or y0 >= source_height:
            continue
        x0 = max(0, min(source_width - 1, x0))
        y0 = max(0, min(source_height - 1, y0))
        x1 = max(0, min(source_width - 1, x1))
        y1 = max(0, min(source_height - 1, y1))
        color = _ANNOTATION_COLORS[index % len(_ANNOTATION_COLORS)]
        label = str(item.get("label") or item.get("id") or f"obj{index}")
        px0, py0 = x0 * scale, y0 * scale
        px1, py1 = (x1 + 1) * scale - 1, (y1 + 1) * scale - 1
        _draw_rect(image, px0, py0, px1, py1, color, thickness=max(1, scale // 4))
        _draw_label(image, label, px0 + 1, py0 + 1, color, scale=max(1, scale // 8))
        drawn.append({
            "label": label,
            "bbox_xyxy": [x0, y0, x1, y1],
            "overlay_rgb": list(color),
        })

    png_bytes = _encode_rgb_image(image)
    seen = sorted(set("".join(rows)))
    palette = {
        symbol: "#%02X%02X%02X" % ARC_AGI_3_RGB[int(symbol, 16)]
        for symbol in seen
    }
    return {
        "attachment_id": "annotated_frame_png",
        "media_type": "image/png",
        "encoding": "base64",
        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
        "sha256": hashlib.sha256(png_bytes).hexdigest(),
        "grid_shape_hw": [source_height, source_width],
        "image_shape_hw": [source_height * scale, source_width * scale],
        "cell_scale": scale,
        "coordinate_order": "x=column,y=row; origin=top_left",
        "palette_id_to_rgb": palette,
        "annotation_contract": (
            "Bounding boxes and labels mark TRACKED PLANNING OBJECTS only. "
            "Labels match object_layer.objects[].id aliases (obj0, obj1, ...). "
            "Component-graph geometry is not drawn here."
        ),
        "annotations": drawn,
    }


def _rasterize_grid(rows: tuple[str, ...], scale: int) -> list[list[tuple[int, int, int]]]:
    height = len(rows) * scale
    width = len(rows[0]) * scale
    image: list[list[tuple[int, int, int]]] = [
        [(0, 0, 0) for _ in range(width)] for _ in range(height)
    ]
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            color = ARC_AGI_3_RGB[int(symbol, 16)]
            y0, x0 = y * scale, x * scale
            for py in range(y0, y0 + scale):
                line = image[py]
                for px in range(x0, x0 + scale):
                    line[px] = color
    return image


def _draw_rect(
    image: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    height = len(image)
    width = len(image[0]) if height else 0
    if width <= 0 or height <= 0:
        return
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width - 1, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height - 1, y1))
    t = max(1, int(thickness))
    for y in range(y0, y1 + 1):
        row = image[y]
        for x in range(x0, x1 + 1):
            on_border = (
                y < y0 + t
                or y > y1 - t
                or x < x0 + t
                or x > x1 - t
            )
            if on_border:
                row[x] = color


def _draw_label(
    image: list[list[tuple[int, int, int]]],
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: int,
) -> None:
    height = len(image)
    width = len(image[0]) if height else 0
    if width <= 0 or height <= 0:
        return
    glyph_w, glyph_h = 5, 7
    spacing = 1
    pixel = max(1, int(scale))
    cleaned = "".join(ch if ch in _FONT_5X7 else "?" for ch in str(text).lower())
    if not cleaned:
        return
    box_w = (len(cleaned) * (glyph_w + spacing) - spacing) * pixel + 2
    box_h = glyph_h * pixel + 2
    bx0 = max(0, min(width - 1, x))
    by0 = max(0, min(height - 1, y))
    bx1 = max(0, min(width - 1, bx0 + box_w))
    by1 = max(0, min(height - 1, by0 + box_h))
    # Dark label plate for contrast.
    for py in range(by0, by1 + 1):
        row = image[py]
        for px in range(bx0, bx1 + 1):
            row[px] = (16, 16, 16)
    cursor_x = bx0 + 1
    cursor_y = by0 + 1
    for ch in cleaned:
        pattern = _FONT_5X7.get(ch) or _FONT_5X7[" "]
        for row_index, bits in enumerate(pattern):
            for col_index, bit in enumerate(bits):
                if bit != "1":
                    continue
                for dy in range(pixel):
                    py = cursor_y + row_index * pixel + dy
                    if not (0 <= py < height):
                        continue
                    row = image[py]
                    for dx in range(pixel):
                        px = cursor_x + col_index * pixel + dx
                        if 0 <= px < width:
                            row[px] = color
        cursor_x += (glyph_w + spacing) * pixel
        if cursor_x >= width:
            break


def _encode_png(rows: tuple[str, ...], scale: int) -> bytes:
    image = _rasterize_grid(rows, scale)
    return _encode_rgb_image(image)


def _encode_rgb_image(image: list[list[tuple[int, int, int]]]) -> bytes:
    height = len(image)
    width = len(image[0]) if height else 0
    if height <= 0 or width <= 0:
        raise ValueError("PNG image is empty")
    scanlines: list[bytes] = []
    for row in image:
        scanline = b"\x00" + b"".join(bytes(pixel) for pixel in row)
        scanlines.append(scanline)
    compressed = zlib.compress(b"".join(scanlines), level=9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def _chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)
