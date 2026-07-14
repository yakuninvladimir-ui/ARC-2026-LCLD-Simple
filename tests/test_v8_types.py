from v8_agent.observe import encode_grid_hex_rows
from v8_agent.types import TriTruth, Relevance, Validity, Progress, Attribution, QwenRole


def test_enums_values():
    assert TriTruth.TRUE.value == "TRUE"
    assert Relevance.IRRELEVANT.value == "IRRELEVANT"
    assert Validity.UNCHECKED.value == "UNCHECKED"
    assert Progress.UNKNOWN.value == "UNKNOWN"
    assert Attribution.NO_VISIBLE_CHANGE.value == "NO_VISIBLE_CHANGE"
    assert QwenRole.COORDINATE.value == "coordinate"


def test_grid_hex_rows_1x1_and_palette_ids():
    assert encode_grid_hex_rows(((10,),)) == ("A",)
    row = tuple(range(16))
    assert encode_grid_hex_rows((row,)) == ("0123456789ABCDEF",)


def test_grid_hex_rows_64x64():
    grid = tuple(tuple((r + c) % 16 for c in range(64)) for r in range(64))
    rows = encode_grid_hex_rows(grid)
    assert len(rows) == 64
    assert all(len(row) == 64 for row in rows)


def test_grid_hex_rejects_invalid_palette():
    try:
        encode_grid_hex_rows(((16,),))
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid palette id to raise")
