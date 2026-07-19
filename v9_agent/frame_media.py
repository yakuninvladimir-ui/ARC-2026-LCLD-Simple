from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from typing import Iterable


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


def _encode_png(rows: tuple[str, ...], scale: int) -> bytes:
    width = len(rows[0]) * scale
    height = len(rows) * scale
    scanlines: list[bytes] = []
    for row in rows:
        expanded = b"".join(bytes(ARC_AGI_3_RGB[int(symbol, 16)]) * scale for symbol in row)
        scanline = b"\x00" + expanded
        scanlines.extend([scanline] * scale)
    compressed = zlib.compress(b"".join(scanlines), level=9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")


def _chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)
