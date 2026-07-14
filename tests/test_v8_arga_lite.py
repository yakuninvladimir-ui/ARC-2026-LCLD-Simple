from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config
from v8_agent.game_adapter import GameAdapter
from v8_agent.memory import GameMemory


def test_arga_preserves_full_grid_and_candidates_in_bounds():
    grid = [[0 for _ in range(8)] for _ in range(8)]
    grid[1][1] = 2
    grid[1][2] = 2
    grid[6][6] = 3
    obs = {"grid": grid, "metadata": {"game_id": "g", "available_actions": ["ACTION1", "ACTION6"], "frame_index": 0}}
    state = GameAdapter().to_world_state(obs)
    snap = ARGALiteBuilder().build(state, GameMemory(), V8Config())
    assert snap.full_grid_hex_rows[1] == "02200000"
    assert snap.objects
    assert snap.coordinate_action_ids == ("ACTION6",)
    assert snap.coordinate_targets
    assert all(0 <= c.x < snap.width and 0 <= c.y < snap.height for c in snap.coordinate_targets)


def test_arga_preserves_64x64_grid():
    grid = [[(r + c) % 16 for c in range(64)] for r in range(64)]
    obs = {"grid": grid, "metadata": {"game_id": "g", "available_actions": ["ACTION1"], "frame_index": 0}}
    state = GameAdapter().to_world_state(obs)
    snap = ARGALiteBuilder().build(state, GameMemory(), V8Config())
    assert len(snap.full_grid_hex_rows) == 64
    assert all(len(row) == 64 for row in snap.full_grid_hex_rows)


def test_adapter_preserves_explicitly_empty_current_action_surface():
    state = GameAdapter().to_world_state({
        "grid": [[0, 0], [0, 1]],
        "last_frame_available_actions": [],
        "possible_actions": ["ACTION1", "ACTION6"],
        "metadata": {"game_id": "g"},
    })

    assert state.available_actions == ()
    assert state.planning_action_ids == ()
    assert state.possible_actions == ("ACTION1", "ACTION6")


def test_coordinate_targets_use_official_slots_when_supplied():
    grid = [[0 for _ in range(8)] for _ in range(8)]
    grid[2][1] = 3
    obs = {
        "grid": grid,
        "metadata": {
            "game_id": "g",
            "available_actions": ["ACTION6"],
            "coordinate_action_ids": ["ACTION6"],
            "coordinate_slots": [[1, 2], [6, 6]],
        },
    }
    state = GameAdapter().to_world_state(obs)
    snap = ARGALiteBuilder().build(state, GameMemory(), V8Config())

    assert {(target.x, target.y) for target in snap.coordinate_targets} == {(2, 1), (6, 6)}
    assert all(target.source == "official_coordinate_slot" for target in snap.coordinate_targets)
