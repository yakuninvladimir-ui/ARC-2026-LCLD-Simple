from __future__ import annotations

from typing import Any, Mapping

from .observe import grid_hash, normalize_grid
from .types import WorldState

_FINAL_STATES = {"WIN", "WON", "VICTORY", "DONE", "TERMINAL"}
_GAME_OVER_STATES = {"GAME_OVER"}


def action_name(action: Any) -> str:
    if hasattr(action, "name"):
        return str(getattr(action, "name")).split(".")[-1].upper()
    value = getattr(action, "value", None)
    if isinstance(value, str):
        return action_name(value)
    if isinstance(value, int):
        return action_name(value)
    if isinstance(action, int):
        if action == 0:
            return "RESET"
        if 1 <= action <= 7:
            return f"ACTION{action}"
    text = str(action).split(".")[-1].strip().upper()
    if text.isdigit():
        return action_name(int(text))
    if text in {"RESTART", "RESET_LEVEL", "RESET_GAME"}:
        return "RESET"
    return text


def _action_values(raw: Any) -> tuple[str, ...]:
    if raw in (None, ""):
        return ()
    if isinstance(raw, (str, bytes)):
        values = (raw,)
    else:
        try:
            values = tuple(raw)
        except TypeError:
            values = (raw,)
    return tuple(dict.fromkeys(action_name(a) for a in values if a not in (None, "")))


def _current_action_values(raw_observation: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("current_available_actions", "last_frame_available_actions", "latest_frame_available_actions", "runtime_available_actions"):
        present, value = _explicit_field(raw_observation, metadata, key)
        if present and value is not None:
            return _action_values(value)
    present, value = _explicit_field(raw_observation, metadata, "available_actions")
    if present and value is not None:
        return _action_values(value)
    return tuple(f"ACTION{i}" for i in range(1, 7))


def _possible_action_values(raw_observation: Mapping[str, Any], metadata: Mapping[str, Any], current_actions: tuple[str, ...]) -> tuple[str, ...]:
    present, value = _explicit_field(raw_observation, metadata, "possible_actions")
    if present and value is not None:
        return _action_values(value)
    return current_actions


def _explicit_field(raw_observation: Mapping[str, Any], metadata: Mapping[str, Any], key: str) -> tuple[bool, Any]:
    if key in raw_observation:
        return True, raw_observation.get(key)
    if key in metadata:
        return True, metadata.get(key)
    return False, None


def _undo_action_values(current_actions: tuple[str, ...], possible_actions: tuple[str, ...], metadata: Mapping[str, Any]) -> tuple[str, ...]:
    undo_ids = set(_action_values(metadata.get("undo_action_ids") or metadata.get("undo_actions")))
    raw_undo = metadata.get("undo_action_id")
    if raw_undo:
        undo_ids.add(action_name(raw_undo))
    if metadata.get("undo_available") or "ACTION7" in current_actions or "ACTION7" in possible_actions:
        undo_ids.add("ACTION7")
    undo_ids.add("UNDO")
    return tuple(action_id for action_id in current_actions if action_id in undo_ids)


def _planning_action_values(current_actions: tuple[str, ...], undo_actions: tuple[str, ...]) -> tuple[str, ...]:
    undo = set(undo_actions)
    return tuple(action_id for action_id in current_actions if action_id not in undo)


class GameAdapter:
    def to_world_state(self, raw_observation: Mapping[str, Any]) -> WorldState:
        metadata = dict(raw_observation.get("metadata", {}) or {})
        grid_raw = raw_observation.get("grid", raw_observation.get("frame"))
        grid = normalize_grid(grid_raw)
        available = _current_action_values(raw_observation, metadata)
        possible = _possible_action_values(raw_observation, metadata, available)
        undo_actions = _undo_action_values(available, possible, metadata)
        planning_actions = _planning_action_values(available, undo_actions)
        game_id = str(metadata.get("game_id", raw_observation.get("game_id", "anonymous_game")) or "anonymous_game")
        levels_completed = _int_value(metadata.get("levels_completed", raw_observation.get("levels_completed", metadata.get("level_index", raw_observation.get("level_index", 0)))), 0)
        level_index = _int_value(metadata.get("level_index", raw_observation.get("level_index", levels_completed)), levels_completed)
        step_index = _int_value(metadata.get("step_index", metadata.get("frame_index", raw_observation.get("step_index", 0))), 0)
        score = _score_value(metadata.get("score", raw_observation.get("score")))
        state_text = _state_name(metadata.get("state", raw_observation.get("state", "")))
        win_levels_raw = metadata.get("win_levels", raw_observation.get("win_levels"))
        win_levels = _int_value(win_levels_raw, -1)
        if win_levels < 0:
            win_levels = None
        game_over = bool(metadata.get("game_over", False) or state_text in _GAME_OVER_STATES)
        win_or_done = bool(metadata.get("win", False) or state_text in _FINAL_STATES)
        all_levels_done = bool(win_levels is not None and win_levels > 0 and levels_completed >= win_levels)
        terminal = bool(metadata.get("terminal", False) or win_or_done or all_levels_done)
        # GAME_OVER is deliberately not terminal for the agent loop: it is a reset/replay signal.
        if game_over and not win_or_done and not all_levels_done:
            terminal = False
        raw = {
            "metadata": metadata,
            "action_surface": {
                "current_available_actions": list(available),
                "planning_action_ids": list(planning_actions),
                "undo_action_ids": list(undo_actions),
                "possible_actions": list(possible),
            },
            "observation_keys": sorted(map(str, raw_observation.keys())),
            "state_name": state_text,
            "levels_completed": levels_completed,
            "win_levels": win_levels,
            "game_over": game_over,
            "full_reset": bool(metadata.get("full_reset", raw_observation.get("full_reset", False))),
        }
        return WorldState(
            game_id=game_id,
            level_index=level_index,
            step_index=step_index,
            grid=grid,
            available_actions=available,
            score=score,
            terminal=terminal,
            raw=raw,
            state_hash=grid_hash(grid),
            state_name=state_text,
            levels_completed=levels_completed,
            win_levels=win_levels,
            game_over=game_over,
            full_reset=bool(metadata.get("full_reset", raw_observation.get("full_reset", False))),
            planning_action_ids=planning_actions,
            undo_action_ids=undo_actions,
            possible_actions=possible,
        )


def _state_name(state: Any) -> str:
    if hasattr(state, "name"):
        return str(getattr(state, "name")).split(".")[-1].upper()
    value = getattr(state, "value", None)
    if value is not None:
        return str(value).split(".")[-1].upper()
    return str(state or "").split(".")[-1].upper()


def _int_value(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _score_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
