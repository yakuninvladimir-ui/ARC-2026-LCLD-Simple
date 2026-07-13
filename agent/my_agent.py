"""ARC-AGI-3 V8.3 compact verified hypothesis agent.

Competition-facing wrapper.  The active runtime imports only `v8_agent/*`; it does
not import the old V7 pipeline/DSL/router stack.  Qwen paths are read from the
same runtime environment names used by the previous archive:

- ARC_QWEN_MODEL_PATH / ARC_LLM_MODEL_PATH
- ARC_QWEN_LLAMA_CLI_PATH
- ARC_LLM_ADVISOR_BACKEND or ARC_V8_QWEN_BACKEND
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from v8_agent import GameSession, config_from_mapping, default_config_dict

try:  # Official starter runtime.
    from agents.agent import Agent as _StarterAgent
except Exception:  # pragma: no cover - local package tests usually do not ship starter runtime.
    _StarterAgent = object

try:  # Official runtime action types.
    from arcengine import ActionInput as _RuntimeActionInput
    from arcengine import GameAction as _RuntimeGameAction
except Exception:  # pragma: no cover
    _RuntimeActionInput = None
    _RuntimeGameAction = None

_TERMINAL_STATES = {"WIN", "WON", "VICTORY", "DONE", "TERMINAL"}
_GAME_OVER_STATES = {"GAME_OVER"}
_NOT_STARTED_STATES = {"NOT_PLAYED", "NOT_STARTED"}


def default_config() -> dict[str, Any]:
    cfg = default_config_dict()
    # Compatibility aliases inherited from the V7.7 wrapper/notebook environment.
    cfg.update({
        "llm_advisor_backend": cfg["qwen_backend"],
        "qwen_context_tokens": cfg["qwen_context_tokens"],
        "qwen_minimum_acceptance_context_tokens": cfg["qwen_minimum_acceptance_context_tokens"],
        "qwen_max_input_tokens": cfg["qwen_max_input_tokens"],
        "qwen_max_output_tokens": cfg["qwen_max_output_tokens"],
        "llm_timeout_seconds": cfg["qwen_timeout_seconds"],
        "qwen_prompt_profile": "v8_3_verified_contract_json",
        "llm_reference_model_filename": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "max_qwen_calls_per_level": cfg["max_total_qwen_calls_per_level"],
        "max_qwen_primary_calls_per_level": cfg["max_primary_qwen_calls_per_level"],
        "max_qwen_replan_calls_per_level": cfg["max_reserve_qwen_calls_per_level"],
        "max_qwen_coordinate_calls_per_level": cfg["max_coordinate_qwen_calls_per_level"],
    })
    return cfg


class MyAgent(_StarterAgent):
    """Starter-compatible adapter around the V8.3 runtime."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, config: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        if args or kwargs:
            try:
                super().__init__(*args, **kwargs)  # type: ignore[misc]
            except TypeError:
                # Local tests may instantiate without the official starter constructor.
                pass
        self.config = default_config()
        if config:
            self.config.update(dict(config))
        self._session = GameSession(config_from_mapping(self.config))
        self.MAX_ACTIONS = int(self.config.get("max_actions_per_game", self.MAX_ACTIONS) or self.MAX_ACTIONS)

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        state_name = _state_name(getattr(latest_frame, "state", ""))
        if state_name in _TERMINAL_STATES:
            return True
        completed = getattr(latest_frame, "levels_completed", None)
        win_levels = getattr(latest_frame, "win_levels", None)
        try:
            return completed is not None and win_levels is not None and int(win_levels) > 0 and int(completed) >= int(win_levels)
        except (TypeError, ValueError):
            return False

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        # GAME_OVER is reset-capable in the official environment. GameSession
        # emits RESET and records the resulting official transition.
        obs = frame_to_observation(latest_frame, frame_index=max(0, len(frames) - 1))
        self._session.update_runtime_config(self.config)
        action_dict = self._session.act(obs)
        return _coerce_action_dict_to_game_action(action_dict, latest_frame)

    def observe_action_result(self, latest_frame: Any, *, frame_index: int = 0) -> bool:
        return self._session.observe_action_result(frame_to_observation(latest_frame, frame_index=frame_index))

    def harness_telemetry(self) -> dict[str, Any]:
        return self._session.harness_telemetry()



def agent_fn(obs: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Any:
    """Raw callable entrypoint: returns official ActionInput when arcengine is available, otherwise a plain dict."""
    merged = default_config()
    if config:
        merged.update(dict(config))
    if not hasattr(agent_fn, "_session"):
        agent_fn._session = GameSession(config_from_mapping(merged))  # type: ignore[attr-defined]
    else:
        agent_fn._session.update_runtime_config(merged)  # type: ignore[attr-defined]
    action_dict = agent_fn._session.act(dict(obs))  # type: ignore[attr-defined]
    return _to_runtime_action_input(action_dict)


def frame_to_observation(frame: Any, *, frame_index: int = 0) -> dict[str, Any]:
    grid = getattr(frame, "frame", None)
    metadata = {
        "available_actions": [_action_name(a) for a in getattr(frame, "available_actions", ()) or ()],
        "frame_index": frame_index,
        "full_reset": bool(getattr(frame, "full_reset", False)),
        "game_id": str(getattr(frame, "game_id", "") or "anonymous_game"),
        "guid": getattr(frame, "guid", None),
        "score": getattr(frame, "score", None),
        "state": _state_name(getattr(frame, "state", "")),
        "levels_completed": getattr(frame, "levels_completed", 0),
        "win_levels": getattr(frame, "win_levels", None),
    }
    return {"frame": grid, "grid": _grid_to_2d(grid), "metadata": metadata}


def _grid_to_2d(grid: Any) -> list[list[int]]:
    grid = _collapse_frame_axes(grid)
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    out = []
    for row in grid or []:
        if hasattr(row, "tolist"):
            row = row.tolist()
        out.append([int(v) for v in row])
    return out


def _collapse_frame_axes(grid: Any) -> Any:
    ndim = getattr(grid, "ndim", None)
    shape = getattr(grid, "shape", None)
    if ndim is None or shape is None:
        return grid
    try:
        ndim_i = int(ndim)
    except Exception:
        return grid
    if ndim_i == 3:
        if int(shape[2]) <= 4:
            return grid[:, :, 0]
        return grid[-1]
    if ndim_i == 4:
        latest = grid[-1]
        latest_ndim = int(getattr(latest, "ndim", 0) or 0)
        latest_shape = getattr(latest, "shape", ())
        if latest_ndim == 3 and len(latest_shape) >= 3 and int(latest_shape[2]) <= 4:
            return latest[:, :, 0]
        if latest_ndim == 3:
            return latest[-1]
        return latest
    return grid


def _to_runtime_action_input(action_dict: Mapping[str, Any]) -> Any:
    action_id = str(action_dict.get("action_id") or action_dict.get("id") or "ACTION1")
    data = dict(action_dict.get("data", {}) or {})
    reasoning = dict(action_dict.get("reasoning", {}) or {})
    if _RuntimeActionInput is not None and _RuntimeGameAction is not None:
        game_action = _runtime_game_action_by_name(action_id)
        if game_action is not None:
            return _RuntimeActionInput(id=game_action, data=data, reasoning=reasoning)
    return {"id": action_id, "action_id": action_id, "data": data, "reasoning": reasoning}


def _coerce_action_dict_to_game_action(action_dict: Mapping[str, Any], latest_frame: Any) -> Any:
    action_id = str(action_dict.get("action_id") or action_dict.get("id") or "ACTION1")
    data = dict(action_dict.get("data", {}) or {})
    reasoning = dict(action_dict.get("reasoning", {}) or {})
    action = _game_action_by_name(action_id, getattr(latest_frame, "available_actions", ()) or (), allow_unavailable=(action_id == "RESET"))
    if action is None:
        # Absolute fallback: choose first available simple action, else reset if possible.
        action = _game_action_by_name(_first_available_action_name(latest_frame), getattr(latest_frame, "available_actions", ()) or (), allow_unavailable=True)
    if hasattr(action, "set_data"):
        try:
            action.set_data(data)
        except Exception:
            try:
                action.set_data({})
            except Exception:
                pass
    try:
        setattr(action, "reasoning", reasoning)
    except Exception:
        pass
    return action


def _first_available_action_name(latest_frame: Any) -> str:
    for item in getattr(latest_frame, "available_actions", ()) or ():
        name = _action_name(item)
        if name != "RESET":
            return name
    return "RESET"


def _game_action_by_name(name: str, available_actions: Sequence[Any], *, allow_unavailable: bool = False) -> Any:
    normalized = _action_name(name)
    runtime_action = _runtime_game_action_by_name(normalized)
    for action in available_actions:
        if _action_name(action) == normalized:
            return runtime_action if runtime_action is not None else action
    if not available_actions and runtime_action is not None:
        return runtime_action
    if allow_unavailable and runtime_action is not None:
        return runtime_action
    if available_actions:
        return available_actions[0]
    return normalized


def _runtime_game_action_by_name(name: str) -> Any | None:
    if _RuntimeGameAction is None:
        return None
    normalized = _action_name(name)
    members = getattr(_RuntimeGameAction, "__members__", {}) or {}
    if normalized in members:
        return members[normalized]
    if hasattr(_RuntimeGameAction, normalized):
        return getattr(_RuntimeGameAction, normalized)
    if hasattr(_RuntimeGameAction, "from_name"):
        try:
            return _RuntimeGameAction.from_name(normalized)
        except Exception:
            pass
    if normalized.startswith("ACTION") and normalized.removeprefix("ACTION").isdigit() and hasattr(_RuntimeGameAction, "from_id"):
        try:
            return _RuntimeGameAction.from_id(int(normalized.removeprefix("ACTION")))
        except Exception:
            return None
    if normalized == "RESET" and hasattr(_RuntimeGameAction, "from_id"):
        try:
            return _RuntimeGameAction.from_id(0)
        except Exception:
            return None
    return None


def _state_name(state: Any) -> str:
    if hasattr(state, "name"):
        return str(getattr(state, "name")).split(".")[-1].upper()
    value = getattr(state, "value", None)
    if value is not None:
        return str(value).split(".")[-1].upper()
    return str(state or "").split(".")[-1].upper()


def _action_name(action: Any) -> str:
    if hasattr(action, "name"):
        return str(getattr(action, "name")).split(".")[-1].upper()
    value = getattr(action, "value", None)
    if isinstance(value, str):
        return _action_name(value)
    if isinstance(value, int):
        return _action_name(value)
    if isinstance(action, int):
        if action == 0:
            return "RESET"
        if 1 <= action <= 7:
            return f"ACTION{action}"
    text = str(action).split(".")[-1].strip().upper()
    if text.isdigit():
        return _action_name(int(text))
    if text in {"RESTART", "RESET_LEVEL", "RESET_GAME"}:
        return "RESET"
    return text
