"""Build the V9 Qwen3.6-27B FP8/vLLM ARC-AGI-3 Kaggle notebook.

Place this file in the project root and run:

    python build_notebook_v9.py

The notebook embeds the verified LCLD agent and keeps the Tufa-derived
competition lifecycle while using Tufa's Qwen FP8 snapshot and vLLM wheelhouse.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from textwrap import dedent
import zipfile


ACCELERATOR = "rtx6000"
_ACCELERATORS = {
    "rtx6000": {
        "name": "nvidiaRtx6000",
        "machine_shape": "NvidiaRtxPro6000",
        "gpu": True,
    },
}

ROOT = Path(__file__).resolve().parent
WORKING_WRAPPER_ROOT = ROOT / "notebook_wrapper"
NOTEBOOK_ROOT = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_ROOT / "arc-prize-2026-lcld-qwen-v9.ipynb"
KERNEL_METADATA_PATH = NOTEBOOK_ROOT / "kernel-metadata.json"
ASSET_METADATA_PATH = ROOT / "assets" / "dataset-metadata.json"
VLLM_WHEELHOUSE_DATASET_SOURCE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
QWEN_MODEL_DATASET_SOURCE = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
COMPETITION_SOURCE = "arc-prize-2026-arc-agi-3"

MARKER = "ARC_V9_SAFE_HARNESS_THINKING32K_STATIC_SCHEMA_SERIAL_GATEWAY"
MAX_NOTEBOOK_BYTES = 985_000

PAYLOAD_EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", "notebooks", "assets"}
PAYLOAD_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".old"}
PAYLOAD_EXCLUDED_RELATIVE_PATHS = {
    "build_notebook.py",
    "build_notebook_v9.py",
    "pyproject.toml",
}

_LAST_PAYLOAD_STATS: dict[str, object] = {}


def code_cell(source: str) -> dict:
    source = dedent(source)
    stripped = source.lstrip()
    if not stripped.startswith(("!", "%%")):
        compile(source, "<generated_notebook_cell>", "exec")
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _asset_dataset_source() -> str:
    if not ASSET_METADATA_PATH.exists():
        return VLLM_WHEELHOUSE_DATASET_SOURCE
    try:
        return str(json.loads(ASSET_METADATA_PATH.read_text(encoding="utf-8")).get("id") or VLLM_WHEELHOUSE_DATASET_SOURCE)
    except Exception:
        return VLLM_WHEELHOUSE_DATASET_SOURCE


def _generated_kaggle_agent_source() -> str:
    return dedent(
        r'''
        """V9 direct-Arcade compatibility shim for the generated Kaggle notebook."""

        from __future__ import annotations

        from typing import Any, Mapping

        from v9_agent import GameSession, config_from_mapping
        from submission import default_config, _state_name

        try:
            from arcengine import ActionInput, GameAction
        except Exception:  # local structural tests outside Kaggle
            ActionInput = None
            GameAction = None


        class ARC_AGI_Agent:
            def __init__(self, config: Mapping[str, Any] | None = None) -> None:
                cfg = default_config()
                if config:
                    cfg.update(dict(config))
                self.config = cfg
                self._session = GameSession(config_from_mapping(cfg))
                self._observed_transition_ingestions = 0
                self._observed_transition_duplicate_skips = 0
                self._current_attempt_index = 1
                self._attempt_history_count = 0
                self._termination_records: list[dict[str, Any]] = []

            def _update_config(self, config: Mapping[str, Any] | None = None) -> None:
                if not config:
                    return
                merged = dict(self.config)
                merged.update(dict(config))
                self.config = merged
                # Apply dynamic timeout/budget fields to the active session without
                # recreating it and losing task-local memory.
                self._session.update_runtime_config(merged)

            def _ingest_pending_transition(self, observation: Mapping[str, Any]) -> bool:
                # The competition loop calls observe_action_result explicitly after
                # every accepted gateway step. This fallback is retained for callers
                # that omit the callback, but it does not create duplicate ingestions.
                if getattr(self._session, "pending_action", None) is None:
                    return False
                committed = self._session.observe_action_result(dict(observation))
                if committed:
                    self._observed_transition_ingestions += 1
                return committed

            def act(self, observation: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Any:
                self._update_config(config)
                observation_dict = dict(observation)
                self._ingest_pending_transition(observation_dict)
                action = self._session.act(observation_dict)
                return _to_action_input(action)

            def reset_after_game_over(
                self,
                observation: Mapping[str, Any],
                config: Mapping[str, Any] | None = None,
            ) -> Any:
                """Return the single RESET required by the Tufa-style GAME_OVER loop.

                GameSession handles GAME_OVER before any Qwen role is selected, so this
                path consumes zero model calls while retaining the failed-attempt memory.
                """
                self._update_config(config)
                observation_dict = dict(observation)
                self._ingest_pending_transition(observation_dict)
                if _state_name(dict(observation_dict.get("metadata", {}) or {}).get("state", "")) != "GAME_OVER":
                    raise RuntimeError("reset_after_game_over requires a GAME_OVER observation")
                action = self._session.act(observation_dict)
                if _action_name(action.get("id", action.get("action_id", ""))) != "RESET":
                    raise RuntimeError("GAME_OVER reset path emitted a non-RESET action")
                return _to_action_input(action)

            def observe_action_result(self, after_observation: Mapping[str, Any] | None = None) -> bool:
                committed = self._session.observe_action_result(after_observation)
                if committed:
                    self._observed_transition_ingestions += 1
                else:
                    self._observed_transition_duplicate_skips += 1
                return committed

            def record_orchestration_termination(self, reason: str, metadata: Mapping[str, Any] | None = None) -> None:
                self._termination_records.append({"reason": str(reason), "metadata": dict(metadata or {})})

            def harness_telemetry(self) -> dict[str, Any]:
                session = self._session.harness_telemetry()
                telemetry = dict(session)
                attempt_indexes = [
                    int(value or 0)
                    for value in (session.get("level_attempt_index_by_level", {}) or {}).values()
                ]
                telemetry.update({
                    "current_attempt_index": max(attempt_indexes, default=0) + 1,
                    "attempt_history_count": len(session.get("level_attempt_records", []) or []),
                    "competition_level_reset_ingestions": int(session.get("game_over_reset_count", 0) or 0),
                    "observed_transition_ingestions": int(session.get("observed_transition_ingestions", self._observed_transition_ingestions) or 0),
                    "observed_transition_duplicate_skips": int(session.get("observed_transition_duplicate_skips", self._observed_transition_duplicate_skips) or 0),
                    "pending_official_transition": bool(session.get("pending_official_transition", False)),
                    "pending_transition_token": session.get("pending_transition_token"),
                    "last_committed_transition_token": session.get("last_committed_transition_token"),
                    "failed_memory_count": int(session.get("failed_memory_count", 0) or 0),
                    "irrelevant_memory_count": int(session.get("irrelevant_memory_count", 0) or 0),
                    "termination_records": list(self._termination_records),
                })
                return telemetry

            def _cleanup_old_session(self) -> None:
                # Release game-local state immediately when a worker finishes.
                # This method is called only after final telemetry is collected.
                session = getattr(self, "_session", None)
                self._session = None
                if session is None:
                    return
                for attr in (
                    "pending_action", "_latest_snapshot", "_last_action_selection",
                    "memory", "bank", "packet_builder", "arga_lite", "explorer",
                    "preflight_judge", "transition_judge", "policy", "qwen",
                ):
                    try:
                        setattr(session, attr, None)
                    except Exception:
                        pass


        def arcade_step_args(native_action: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
            if ActionInput is not None and isinstance(native_action, ActionInput):
                return native_action.id, dict(getattr(native_action, "data", {}) or {}), dict(getattr(native_action, "reasoning", {}) or {})
            if isinstance(native_action, Mapping):
                action_id = native_action.get("id", native_action.get("action_id", "ACTION1"))
                data = dict(native_action.get("data", {}) or {})
                reasoning = dict(native_action.get("reasoning", {}) or {})
                return _game_action_by_name(action_id), data, reasoning
            action_id = getattr(native_action, "id", native_action)
            data = dict(getattr(native_action, "data", {}) or {})
            reasoning = dict(getattr(native_action, "reasoning", {}) or {})
            return _game_action_by_name(action_id), data, reasoning


        def _to_action_input(action: Mapping[str, Any]) -> Any:
            action_id = action.get("id", action.get("action_id", "ACTION1"))
            data = dict(action.get("data", {}) or {})
            reasoning = dict(action.get("reasoning", {}) or {})
            game_action = _game_action_by_name(action_id)
            if ActionInput is not None:
                return ActionInput(id=game_action, data=data, reasoning=reasoning)
            return {"id": _action_name(game_action), "data": data, "reasoning": reasoning}


        def _game_action_by_name(value: Any) -> Any:
            if GameAction is None:
                return _action_name(value)
            name = _action_name(value)
            members = getattr(GameAction, "__members__", {}) or {}
            if name in members:
                return members[name]
            if hasattr(GameAction, name):
                return getattr(GameAction, name)
            if hasattr(GameAction, "from_name"):
                try:
                    return GameAction.from_name(name)
                except Exception:
                    pass
            if name.startswith("ACTION") and name.removeprefix("ACTION").isdigit() and hasattr(GameAction, "from_id"):
                try:
                    return GameAction.from_id(int(name.removeprefix("ACTION")))
                except Exception:
                    pass
            if name == "RESET" and hasattr(GameAction, "from_id"):
                try:
                    return GameAction.from_id(0)
                except Exception:
                    pass
            return name


        def _action_name(value: Any) -> str:
            if hasattr(value, "name"):
                return str(getattr(value, "name")).split(".")[-1].upper()
            raw = getattr(value, "value", value)
            if isinstance(raw, int):
                if raw == 0:
                    return "RESET"
                if 1 <= raw <= 7:
                    return f"ACTION{raw}"
            text = str(raw).split(".")[-1].strip().upper()
            if text.isdigit():
                return _action_name(int(text))
            return text or "ACTION1"
        '''
    ).strip() + "\n"


def _generated_submission_source() -> str:
    return dedent(
        r'''
        """V9 direct-Arcade submission helpers for the generated Kaggle notebook."""

        from __future__ import annotations

        import os
        from typing import Any

        from v9_agent.config import default_config_dict


        def default_config() -> dict[str, Any]:
            cfg = default_config_dict()
            thinking_enabled = os.environ.get("ARC_QWEN_ENABLE_THINKING", "true").lower() in {"1", "true", "yes", "on"}
            reasoning_budget_tokens = (
                int(os.environ.get("ARC_QWEN_REASONING_BUDGET_TOKENS", "32000"))
                if thinking_enabled
                else 0
            )
            cfg.update({
                "enable_qwen": True,
                "qwen_backend": os.environ.get("ARC_V8_QWEN_BACKEND", os.environ.get("ARC_LLM_ADVISOR_BACKEND", "vllm")),
                "qwen_model_path": os.environ.get("ARC_QWEN_MODEL_PATH") or os.environ.get("ARC_LLM_MODEL_PATH"),
                "qwen_llama_cli_path": os.environ.get("ARC_QWEN_LLAMA_CLI_PATH"),
                "qwen_llama_device": os.environ.get("ARC_QWEN_LLAMA_DEVICE"),
                "qwen_split_mode": os.environ.get("ARC_QWEN_SPLIT_MODE", os.environ.get("LLAMA_ARG_SPLIT_MODE", "")),
                "qwen_tensor_split": os.environ.get("ARC_QWEN_TENSOR_SPLIT", os.environ.get("LLAMA_ARG_TENSOR_SPLIT", "")),
                "qwen_gpu_layers": int(os.environ.get("ARC_QWEN_GPU_LAYERS", "999")),
                "qwen_timeout_seconds": int(os.environ.get("ARC_QWEN_TIMEOUT_SECONDS", "600")),
                "qwen_context_tokens": int(os.environ.get("ARC_QWEN_CONTEXT_TOKENS", "131072")),
                "qwen_minimum_acceptance_context_tokens": int(os.environ.get("ARC_QWEN_MINIMUM_ACCEPTANCE_CONTEXT_TOKENS", "65536")),
                "qwen_max_input_tokens": int(os.environ.get("ARC_QWEN_MAX_INPUT_TOKENS", "65536")),
                "qwen_max_output_tokens": int(os.environ.get("ARC_QWEN_MAX_OUTPUT_TOKENS", "49152")),
                "qwen_reserved_runtime_margin_tokens": int(os.environ.get("ARC_QWEN_RESERVED_RUNTIME_MARGIN_TOKENS", "8192")),
                "qwen_enable_thinking": thinking_enabled,
                "qwen_reasoning_mode": "on" if thinking_enabled else "off",
                "qwen_reasoning_budget_tokens": reasoning_budget_tokens,
                "qwen_temperature": float(os.environ.get("ARC_QWEN_TEMPERATURE", "0.6")),
                "qwen_top_k": int(os.environ.get("ARC_QWEN_TOP_K", "20")),
                "qwen_top_p": float(os.environ.get("ARC_QWEN_TOP_P", "0.95")),
                "qwen_min_p": float(os.environ.get("ARC_QWEN_MIN_P", "0.0")),
                "qwen_presence_penalty": float(os.environ.get("ARC_QWEN_PRESENCE_PENALTY", "0.0")),
                "qwen_repeat_penalty": float(os.environ.get("ARC_QWEN_REPEAT_PENALTY", "1.0")),
                "qwen_vllm_base_url": os.environ.get("ARC_QWEN_VLLM_BASE_URL", "http://127.0.0.1:1234/v1"),
                "qwen_vllm_api_key": os.environ.get("ARC_QWEN_VLLM_API_KEY", "EMPTY"),
                "qwen_vllm_model": os.environ.get("ARC_QWEN_VLLM_MODEL", "vrfai/Qwen3.6-27B-FP8"),
                "qwen_multimodal_enabled": os.environ.get("ARC_QWEN_MULTIMODAL_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
                "qwen_require_runtime": os.environ.get("LCLD_REQUIRE_QWEN_RUNTIME", "1").lower() in {"1", "true", "yes", "on"},
                "max_qwen_calls_per_game": int(os.environ.get("ARC_V8_MAX_QWEN_CALLS_PER_GAME", "20")),
                "max_primary_qwen_calls_per_level": int(os.environ.get("ARC_MAX_QWEN_PRIMARY_CALLS_PER_LEVEL", "1")),
                "max_reserve_qwen_calls_per_level": int(os.environ.get("ARC_MAX_QWEN_REPLAN_CALLS_PER_LEVEL", "0")),
                "max_coordinate_qwen_calls_per_level": int(os.environ.get("ARC_MAX_QWEN_COORDINATE_CALLS_PER_LEVEL", "1")),
                "max_total_qwen_calls_per_level": int(os.environ.get("ARC_MAX_TOTAL_QWEN_CALLS_PER_LEVEL", "2")),
                "max_actions_per_game": int(os.environ.get("LCLD_MAX_ACTIONS_PER_GAME", "200")),
                "max_level_attempts": int(os.environ.get("LCLD_MAX_LEVEL_ATTEMPTS", "0")),
                "max_actions_per_level": int(os.environ.get("LCLD_MAX_ACTIONS_PER_LEVEL", "200")),
                "game_wall_clock_limit_seconds": int(float(os.environ.get("LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS", "6000"))),
                "max_game_over_resets_per_game": 0,
                "max_game_over_resets_per_level": 0,
                "reset_on_game_over": os.environ.get("LCLD_RESET_ON_GAME_OVER", "1").lower() in {"1", "true", "yes", "on"},
                # Compatibility names used by the direct competition harness.
                "llm_advisor_backend": os.environ.get("ARC_LLM_ADVISOR_BACKEND", "vllm"),
                "llm_timeout_seconds": int(os.environ.get("ARC_QWEN_TIMEOUT_SECONDS", "600")),
                "qwen_prompt_profile": "v8_3_verified_contract_json",
                "qwen_trace_dir": None,
            })
            return cfg


        def frame_to_world_json(frame: Any) -> dict[str, Any]:
            grid = _frame_grid_to_2d(getattr(frame, "frame", None))
            return {
                "grid": grid,
                "available_actions": [_action_name(a) for a in getattr(frame, "available_actions", ()) or ()],
                "game_id": str(getattr(frame, "game_id", "") or "anonymous_game"),
                "guid": getattr(frame, "guid", None),
                "score": getattr(frame, "score", None),
                "state": _state_name(getattr(frame, "state", "")),
                "levels_completed": getattr(frame, "levels_completed", 0),
                "win_levels": getattr(frame, "win_levels", None),
                "full_reset": bool(getattr(frame, "full_reset", False)),
            }


        def _frame_grid_to_2d(grid: Any) -> list[list[int]]:
            grid = _collapse_frame_axes(grid)
            if hasattr(grid, "tolist"):
                grid = grid.tolist()
            if grid is None:
                return []
            out: list[list[int]] = []
            for row in grid:
                if hasattr(row, "tolist"):
                    row = row.tolist()
                out.append([int(v) for v in row])
            return out


        def _collapse_frame_axes(grid: Any) -> Any:
            """Return the final visible 2D ARC frame.

            ARC's ``FrameData.frame`` is a temporal list of 2D grids. It is
            not a 3D world or an RGB image. A bare 2D grid is also accepted
            because local adapters already expose that normalized form.
            """
            if grid is None:
                return []
            ndim = getattr(grid, "ndim", None)
            if ndim is not None:
                try:
                    while int(ndim) > 2:
                        if len(grid) == 0:
                            return []
                        grid = grid[-1]
                        ndim = getattr(grid, "ndim", 0)
                    return grid
                except (TypeError, ValueError):
                    return grid

            if not isinstance(grid, (list, tuple)) or not grid:
                return grid

            first = grid[0]
            first_ndim = getattr(first, "ndim", None)
            if first_ndim is not None:
                try:
                    if int(first_ndim) >= 2:
                        return grid[-1]
                except (TypeError, ValueError):
                    pass

            # A temporal Python representation has depth three:
            # frames -> rows -> scalar cells. A normalized grid has depth two.
            if isinstance(first, (list, tuple)) and first:
                first_row = first[0]
                if isinstance(first_row, (list, tuple)):
                    return grid[-1]
                first_row_ndim = getattr(first_row, "ndim", None)
                try:
                    if first_row_ndim is not None and int(first_row_ndim) >= 1:
                        return grid[-1]
                except (TypeError, ValueError):
                    pass
            return grid


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
            value = getattr(action, "value", action)
            if isinstance(value, int):
                if value == 0:
                    return "RESET"
                if 1 <= value <= 7:
                    return f"ACTION{value}"
            text = str(value).split(".")[-1].strip().upper()
            if text.isdigit():
                return _action_name(int(text))
            return text
        '''
    ).strip() + "\n"


def _source_payload() -> str:
    required = [
        ROOT / "v9_agent" / "__init__.py",
        ROOT / "v9_agent" / "session.py",
        ROOT / "v9_agent" / "llm.py",
        ROOT / "v9_agent" / "qwen_packet.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("V9 source tree is incomplete; missing: " + repr(missing))

    buffer = io.BytesIO()
    archived_files: list[str] = []
    excluded_files: list[str] = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_LZMA) as archive:
        for folder in (ROOT / "v9_agent",):
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(ROOT).as_posix()
                if any(part in PAYLOAD_EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
                    excluded_files.append(rel)
                    continue
                if path.suffix in PAYLOAD_EXCLUDED_SUFFIXES or rel in PAYLOAD_EXCLUDED_RELATIVE_PATHS:
                    excluded_files.append(rel)
                    continue
                archive.write(path, Path("Code") / path.relative_to(ROOT))
                archived_files.append(rel)
        archive.writestr("Code/kaggle_agent.py", _generated_kaggle_agent_source())
        archive.writestr("Code/submission.py", _generated_submission_source())
        archive.writestr("Code/lcld_competition_child.py", _competition_child_source())
        archived_files.extend([
            "kaggle_agent.py",
            "submission.py",
            "lcld_competition_child.py",
        ])

    payload_bytes = buffer.getvalue()
    encoded = base64.b64encode(payload_bytes).decode("ascii")
    _LAST_PAYLOAD_STATS.clear()
    _LAST_PAYLOAD_STATS.update({
        "archived_file_count": len(archived_files),
        "excluded_files": tuple(sorted(excluded_files)),
        "payload_base64_chars": len(encoded),
        "payload_zip_bytes": len(payload_bytes),
    })
    return encoded


def _preflight_script_source() -> str:
    source = dedent(
        """\
        import importlib
        import os
        import pathlib
        import sys

        phase = os.environ.get('LCLD_PREFLIGHT_PHASE', 'unknown')
        full_import_sweep = os.environ.get('LCLD_FULL_IMPORT_SWEEP', '').strip().lower() == 'true'
        print('=== LCLD DIRECT-AGENT STRUCTURAL PREFLIGHT START ===', phase, 'full_import_sweep=', full_import_sweep, flush=True)
        print('cwd=', os.getcwd(), flush=True)
        print('ARC_AGENT_CODE_DIR=', os.environ.get('ARC_AGENT_CODE_DIR'), flush=True)
        print('ARC_QWEN_MODEL_PATH=', os.environ.get('ARC_QWEN_MODEL_PATH'), flush=True)
        print('ARC_QWEN_LLAMA_CLI_PATH=', os.environ.get('ARC_QWEN_LLAMA_CLI_PATH'), flush=True)

        code_dir = pathlib.Path(os.environ['ARC_AGENT_CODE_DIR']).resolve()
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        print('sys.path.head=', sys.path[:8], flush=True)

        for rel in (
            'kaggle_agent.py', 'submission.py', 'lcld_competition_child.py',
            'v9_agent/session.py', 'v9_agent/llm.py', 'v9_agent/qwen_packet.py',
        ):
            path = code_dir / rel
            assert path.exists(), 'required payload file missing: ' + str(path)
            print('[OK] payload file', rel, str(path), flush=True)

        for name in ('kaggle_agent', 'submission', 'v9_agent', 'v9_agent.session', 'v9_agent.llm'):
            mod = importlib.import_module(name)
            mod_file_raw = getattr(mod, '__file__', None)
            if mod_file_raw:
                mod_file = pathlib.Path(mod_file_raw).resolve()
                assert mod_file.is_relative_to(code_dir), name + ' imported from wrong location: ' + str(mod_file)
                print('[OK] import', name, '->', mod_file, flush=True)
            else:
                print('[OK] import', name, '-> namespace/no-file', flush=True)

        if full_import_sweep:
            for path in sorted((code_dir / 'v9_agent').glob('*.py')):
                if path.name == '__init__.py':
                    continue
                name = 'v9_agent.' + path.stem
                mod = importlib.import_module(name)
                mod_file = pathlib.Path(str(getattr(mod, '__file__', ''))).resolve()
                assert mod_file.is_relative_to(code_dir), name + ' imported from wrong location: ' + str(mod_file)
                print('[OK] full_import', name, '->', mod_file, flush=True)

        from kaggle_agent import ARC_AGI_Agent, arcade_step_args
        from submission import default_config
        cfg = default_config()
        delegate = ARC_AGI_Agent(cfg)
        assert hasattr(delegate, 'act') and callable(delegate.act), 'ARC_AGI_Agent.act missing'
        assert hasattr(delegate, 'observe_action_result') and callable(delegate.observe_action_result), 'observe_action_result missing'
        assert hasattr(delegate, 'reset_after_game_over') and callable(delegate.reset_after_game_over), 'reset_after_game_over missing'
        assert hasattr(delegate, 'harness_telemetry') and callable(delegate.harness_telemetry), 'harness_telemetry missing'
        assert callable(arcade_step_args), 'arcade_step_args missing'
        print('delegate qwen_backend=', delegate.config.get('qwen_backend'), flush=True)
        print('delegate qwen_vllm_base_url=', delegate.config.get('qwen_vllm_base_url'), flush=True)
        print('delegate qwen_vllm_model=', delegate.config.get('qwen_vllm_model'), flush=True)
        assert delegate.config.get('qwen_backend') == 'vllm', repr(delegate.config)
        thinking_enabled = os.environ.get('ARC_QWEN_ENABLE_THINKING', 'true').lower() in {'1', 'true', 'yes', 'on'}
        reasoning_budget_tokens = int(os.environ.get('ARC_QWEN_REASONING_BUDGET_TOKENS', '32000')) if thinking_enabled else 0
        assert bool(delegate.config.get('qwen_enable_thinking')) == thinking_enabled, repr(delegate.config)
        assert delegate.config.get('qwen_reasoning_mode') == ('on' if thinking_enabled else 'off'), repr(delegate.config)
        assert int(delegate.config.get('qwen_reasoning_budget_tokens', -1)) == reasoning_budget_tokens, repr(delegate.config)
        assert delegate.config.get('qwen_vllm_model') == 'vrfai/Qwen3.6-27B-FP8', repr(delegate.config)
        assert delegate.config.get('qwen_multimodal_enabled') is True, repr(delegate.config)
        assert int(delegate.config.get('qwen_context_tokens', 0)) == 131072, repr(delegate.config)
        assert int(delegate.config.get('qwen_max_input_tokens', 0)) == 65536, repr(delegate.config)
        assert int(delegate.config.get('qwen_max_output_tokens', 0)) == 49152, repr(delegate.config)
        assert float(delegate.config.get('qwen_temperature', -1)) == 0.6, repr(delegate.config)
        assert float(delegate.config.get('qwen_top_p', -1)) == 0.95, repr(delegate.config)
        assert int(delegate.config.get('qwen_top_k', -1)) == 20, repr(delegate.config)
        assert float(delegate.config.get('qwen_presence_penalty', -1)) == 0.0, repr(delegate.config)
        print('=== LCLD DIRECT-AGENT STRUCTURAL PREFLIGHT OK ===', phase, flush=True)
        """
    )
    compile(source, "<arc_v83_structural_preflight_builder>", "exec")
    return source


def _working_wrapper_source(name: str) -> str:
    path = WORKING_WRAPPER_ROOT / name
    if not path.is_file():
        raise SystemExit("Known-working notebook wrapper source is missing: " + str(path))
    return path.read_text(encoding="utf-8")


def _replace_function_block(source: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = source.find(start_marker)
    if start < 0:
        raise SystemExit("Working wrapper start marker is missing: " + start_marker)
    end = source.find(end_marker, start)
    if end < 0:
        raise SystemExit("Working wrapper end marker is missing: " + end_marker)
    if source.find(start_marker, start + len(start_marker)) >= 0:
        raise SystemExit("Working wrapper start marker is ambiguous: " + start_marker)
    replacement_text = replacement.rstrip()
    leading_indent = start_marker[: len(start_marker) - len(start_marker.lstrip())]
    if leading_indent:
        replacement_text = "\n".join(
            (leading_indent + line) if line else ""
            for line in replacement_text.splitlines()
        )
    return source[:start] + replacement_text + "\n\n" + source[end:]


def _replace_exact_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one exact match, found {count}")
    return source.replace(old, new, 1)


def _adapt_working_phase_b_source(source: str) -> str:
    current_frame = dedent(
        """\
        def _current_frame(env):
            raw = getattr(env, 'observation_space', None)
            if callable(raw):
                raw = raw()
            if raw is None:
                observe = getattr(env, 'observe', None)
                if callable(observe):
                    raw = observe()
            if raw is None:
                raise RuntimeError(
                    'Arcade environment exposes no current observation; '
                    'the harness must not call env.reset() implicitly'
                )
            return _frame_data(raw)
        """
    )
    terminal_reason = dedent(
        """\
        def _terminal_reason(frame):
            state_name = _state(frame)
            # Tufa/TAAF semantics: engine GAME_OVER is recoverable and requires one
            # RESET. Only a successful/terminal completion ends the game loop.
            if state_name in {'WIN', 'WON', 'DONE', 'TERMINAL', 'VICTORY'}:
                return 'state:' + state_name
            completed = getattr(frame, 'levels_completed', None)
            win_levels = getattr(frame, 'win_levels', None)
            try:
                if completed is not None and int(win_levels or 0) > 0 and int(completed) >= int(win_levels):
                    return 'all_levels_completed'
            except (TypeError, ValueError):
                pass
            return ''

        def _agent_level_limit_reason(exc):
            reason = str(getattr(exc, 'reason_code', '') or '')
            if reason in {'level_attempt_limit_reached', 'level_action_limit_reached'}:
                return reason
            text = str(exc)
            for candidate in ('level_attempt_limit_reached', 'level_action_limit_reached'):
                if candidate in text:
                    return candidate
            return ''
        """
    )
    direct_config = dedent(
        """\
        def _direct_config():
            config = default_config()
            thinking_enabled = os.environ.get('ARC_QWEN_ENABLE_THINKING', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
            reasoning_budget_tokens = (
                int(os.environ.get('ARC_QWEN_REASONING_BUDGET_TOKENS', '32000'))
                if thinking_enabled
                else 0
            )
            config.update({
                'allow_in_memory_env': True,
                'environment_adapter': None,
                'external_action_effect_research': True,
                'action_effect_exploration_before_qwen': True,
                'qwen_context_tokens': int(os.environ.get('ARC_QWEN_CONTEXT_TOKENS', '131072')),
                'qwen_minimum_acceptance_context_tokens': 65536,
                'qwen_max_input_tokens': int(os.environ.get('ARC_QWEN_MAX_INPUT_TOKENS', '65536')),
                'qwen_max_output_tokens': int(os.environ.get('ARC_QWEN_MAX_OUTPUT_TOKENS', '49152')),
                'qwen_enable_thinking': thinking_enabled,
                'qwen_reasoning_mode': 'on' if thinking_enabled else 'off',
                'qwen_reasoning_budget_tokens': reasoning_budget_tokens,
                'qwen_temperature': 0.6,
                'qwen_top_p': 0.95,
                'qwen_top_k': 20,
                'qwen_presence_penalty': 0.0,
                'qwen_strict_required': True,
                'qwen_timeout_seconds': int(os.environ.get('ARC_QWEN_TIMEOUT_SECONDS', '600')),
                'llm_timeout_seconds': int(os.environ.get('ARC_LLM_TIMEOUT_SECONDS', '600')),
                'action_selection_timeout_s': 6000.0,
                'major_cycle_wall_clock_budget_seconds': 6000,
                'total_game_wall_clock_limit_seconds': 6000,
                'max_level_attempts': 0,
                'max_actions_per_game': 200,
                'max_actions_per_level': 200,
            })
            return config
        """
    )
    run_direct_game = dedent(
        """\
        def _run_direct_game(env, game_id, initial_frame, abort_event=None):
            config = _direct_config()
            delegate = ARC_AGI_Agent(config)
            game_wall_limit = max(0.0, float(os.getenv('LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '6000')))
            started = time.monotonic()
            accepted_actions = 0
            proposed_actions = 0
            rejected_actions = 0
            game_over_resets = 0
            frame_index = 0
            latest = _frame_data(initial_frame)
            stop_reason = ''
            last_engine_action = ''

            _trace(
                'direct_agent_init',
                game_id=game_id,
                initial_state=_state(latest),
                initial_guid=getattr(latest, 'guid', None),
                game_wall_limit_seconds=game_wall_limit,
                termination_bound='game_wall_clock_only',
                reset_policy='tufa_initial_observation_then_single_reset_after_game_over',
                initial_reset_required=False,
            )

            try:
                while True:
                    if abort_event is not None and abort_event.is_set():
                        stop_reason = 'parallel_abort'
                        break
                    stop_reason = _terminal_reason(latest)
                    if stop_reason:
                        break
                    elapsed = time.monotonic() - started
                    if game_wall_limit > 0 and elapsed >= game_wall_limit:
                        stop_reason = 'game_wall_clock_limit'
                        recorder = getattr(delegate, 'record_orchestration_termination', None)
                        if callable(recorder):
                            recorder(stop_reason, {
                                'elapsed_seconds': elapsed,
                                'limit_seconds': game_wall_limit,
                                'accepted_actions': accepted_actions,
                            })
                        break

                    state_name = _state(latest)
                    observation = _observation(latest, frame_index, game_id)
                    if state_name == 'GAME_OVER':
                        # Exact Tufa loop invariant: after GAME_OVER, execute one RESET
                        # before any next model/analyzer call. Never issue RESET twice.
                        if last_engine_action == 'RESET':
                            stop_reason = 'game_over_persisted_after_single_reset'
                            _trace(
                                'game_over_persisted_after_reset',
                                game_id=game_id,
                                accepted_action_count=accepted_actions,
                                guid=getattr(latest, 'guid', None),
                            )
                            break
                        try:
                            native_action = delegate.reset_after_game_over(observation, config)
                        except Exception as exc:
                            level_limit_reason = _agent_level_limit_reason(exc)
                            if not level_limit_reason:
                                raise
                            stop_reason = 'agent:' + level_limit_reason
                            _trace(
                                'agent_level_limit_terminal',
                                game_id=game_id,
                                reason=level_limit_reason,
                                state=state_name,
                                accepted_action_count=accepted_actions,
                            )
                            break
                        action_id, action_data, reasoning = arcade_step_args(native_action)
                        action_name = str(getattr(action_id, 'name', action_id)).split('.')[-1].upper()
                        if action_name != 'RESET':
                            raise RuntimeError(
                                'Tufa-style GAME_OVER path must emit RESET, got ' + action_name
                            )
                        reasoning = dict(reasoning or {})
                        reasoning.setdefault('source', 'tufa_game_over_auto_reset')
                        reasoning.setdefault('generated_tokens', 0)
                        game_over_resets += 1
                        _trace(
                            'game_over_auto_reset',
                            game_id=game_id,
                            reset_count=game_over_resets,
                            accepted_action_count=accepted_actions,
                            guid_before=getattr(latest, 'guid', None),
                        )
                    else:
                        try:
                            native_action = delegate.act(observation, config)
                        except Exception as exc:
                            level_limit_reason = _agent_level_limit_reason(exc)
                            if not level_limit_reason:
                                raise
                            stop_reason = 'agent:' + level_limit_reason
                            _trace(
                                'agent_level_limit_terminal',
                                game_id=game_id,
                                reason=level_limit_reason,
                                state=state_name,
                                accepted_action_count=accepted_actions,
                            )
                            break
                        action_id, action_data, reasoning = arcade_step_args(native_action)
                        action_name = str(getattr(action_id, 'name', action_id)).split('.')[-1].upper()

                    proposed_actions += 1
                    _trace(
                        'gateway_step_proposed',
                        game_id=game_id,
                        proposed_action_count=proposed_actions,
                        accepted_action_count=accepted_actions,
                        action=action_name,
                        data=action_data,
                        state_before=state_name,
                        guid_before=getattr(latest, 'guid', None),
                    )

                    try:
                        raw_next = env.step(action_id, data=action_data, reasoning=reasoning)
                        next_frame = _frame_data(raw_next)
                    except Exception as exc:
                        rejected_actions += 1
                        _trace(
                            'gateway_step_rejected',
                            game_id=game_id,
                            proposed_action_count=proposed_actions,
                            accepted_action_count=accepted_actions,
                            rejected_action_count=rejected_actions,
                            action=action_name,
                            exc_type=type(exc).__name__,
                            error=str(exc)[:2000],
                        )
                        raise

                    # Count only after the gateway returned a non-null, usable frame.
                    accepted_actions += 1
                    _record_gateway_action()
                    frame_index += 1
                    latest = next_frame
                    last_engine_action = action_name
                    transition_committed = delegate.observe_action_result(
                        _observation(latest, frame_index, game_id)
                    )
                    _trace(
                        'gateway_step_accepted',
                        game_id=game_id,
                        proposed_action_count=proposed_actions,
                        accepted_action_count=accepted_actions,
                        rejected_action_count=rejected_actions,
                        action=action_name,
                        state_after=_state(latest),
                        guid_after=getattr(latest, 'guid', None),
                        levels_completed=getattr(latest, 'levels_completed', 0),
                        transition_committed=bool(transition_committed),
                    )

                if not stop_reason:
                    stop_reason = 'loop_exit'
                telemetry = _compact_harness_telemetry(delegate.harness_telemetry())
                return {
                    'action_count': int(accepted_actions),
                    'proposed_action_count': int(proposed_actions),
                    'rejected_action_count': int(rejected_actions),
                    'game_over_reset_count': int(game_over_resets),
                    'levels_completed': int(getattr(latest, 'levels_completed', 0) or 0),
                    'final_state': _state(latest),
                    'final_guid': str(getattr(latest, 'guid', '') or ''),
                    'stop_reason': stop_reason,
                    'telemetry_summary': telemetry,
                }
            except Exception as exc:
                failure_metrics = {
                    'action_count': int(accepted_actions),
                    'proposed_action_count': int(proposed_actions),
                    'rejected_action_count': int(rejected_actions),
                    'game_over_reset_count': int(game_over_resets),
                    'levels_completed': int(getattr(latest, 'levels_completed', 0) or 0),
                    'final_state': _state(latest),
                    'final_guid': str(getattr(latest, 'guid', '') or ''),
                    'stop_reason': 'exception:' + type(exc).__name__,
                }
                raise DirectGameFailure(str(exc), metrics=failure_metrics) from exc
            finally:
                _cleanup_delegate(delegate)
        """
    )
    write_results = dedent(
        """\
        def _write_results(status, results, game_count):
            payload = {
                'marker': MARKER,
                'status': status,
                'created_at_utc': _utc_now(),
                'execution_path': 'isolated_child_parallel_games_initial_observation_then_GAME_OVER_RESET_then_ARC_AGI_Agent_act_to_env_step',
                'unconditional_initial_reset': False,
                'one_attempt_game_over_terminal': False,
                'game_over_reset_policy': 'single_reset_before_next_qwen_call',
                'game_concurrency': min(
                    max(1, int(os.environ.get('LCLD_GAME_CONCURRENCY', '4'))),
                    max(1, int(game_count)),
                ),
                'vllm_max_num_seqs': int(VLLM_MAX_NUM_SEQS),
                'qwen_timeout_seconds': int(os.environ.get('ARC_QWEN_TIMEOUT_SECONDS', '600')),
                'game_wall_clock_limit_seconds': int(float(
                    os.environ.get('LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '6000')
                )),
                'max_level_attempts': 0,
                'max_actions_per_game': 200,
                'max_actions_per_level': 200,
                'competition_reset_semantics': 'official_gateway_current_level_reset',
                'game_count': int(game_count),
                'attempted_games': sum(1 for item in results if item.get('status') != 'skipped_global_deadline'),
                'completed_games': sum(1 for item in results if item.get('status') == 'completed'),
                'failed_games': sum(1 for item in results if item.get('status') == 'failed'),
                'skipped_games': sum(1 for item in results if item.get('status') == 'skipped_global_deadline'),
                'total_actions': sum(int(item.get('action_count', 0) or 0) for item in results),
                'total_proposed_actions': sum(int(item.get('proposed_action_count', 0) or 0) for item in results),
                'total_rejected_actions': sum(int(item.get('rejected_action_count', 0) or 0) for item in results),
                'total_game_over_resets': sum(int(item.get('game_over_reset_count', 0) or 0) for item in results),
                'levels_completed_observed': sum(int(item.get('levels_completed', 0) or 0) for item in results),
                'results': results,
                'scorecard_owner': 'isolated_gameplay_child_scorecard',
                'explicit_scorecard_opened': bool(scorecard_id),
                'explicit_scorecard_closed': bool(scorecard_closed),
                'scorecard_close_attempted': bool(scorecard_close_attempted),
                'scorecard_close_disposition': scorecard_close_disposition,
                'scorecard_close_error': scorecard_close_error,
                'gateway_make_reset_observed': gateway_make_reset_event.is_set(),
                'gateway_make_reset_count': int(gateway_make_reset_count),
                'gateway_action_observed': accepted_gateway_action_event.is_set(),
                'gateway_accepted_action_count': int(accepted_gateway_action_count),
                'gateway_activity_observed': _gateway_activity_count() > 0,
                'gateway_activity_count': _gateway_activity_count(),
                'phase_b_parquet_created_by_notebook': False,
            }
            temporary = result_path.with_suffix('.json.tmp')
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\\n',
                encoding='utf-8',
            )
            temporary.replace(result_path)
            return payload

        def _close_shared_scorecard():
            global scorecard_closed, scorecard_close_attempted
            global scorecard_close_disposition, scorecard_close_error
            with scorecard_close_lock:
                if scorecard_close_attempted:
                    return None
                if arcade is None or not scorecard_id:
                    scorecard_close_disposition = 'not_open'
                    return None
                scorecard_close_attempted = True

            try:
                closed = arcade.close_scorecard(scorecard_id)
            except BaseException as exc:
                response = getattr(exc, 'response', None)
                status_code = getattr(response, 'status_code', None)
                scorecard_close_error = f'{type(exc).__name__}: {exc}'
                if status_code in {404, 409, 410}:
                    scorecard_closed = False
                    scorecard_close_disposition = 'missing_or_auto_closed'
                else:
                    scorecard_close_disposition = 'close_failed'
                _trace(
                    'competition_scorecard_close_error_absorbed',
                    scorecard_id=scorecard_id,
                    disposition=scorecard_close_disposition,
                    status_code=status_code,
                    error=scorecard_close_error[:2000],
                )
                print(
                    '[Phase B] scorecard close was not fatal:',
                    scorecard_close_disposition,
                    scorecard_close_error,
                    flush=True,
                )
                return None

            scorecard_closed = True
            scorecard_close_disposition = 'closed' if closed is not None else 'closed_no_payload'
            _trace(
                'competition_scorecard_closed',
                scorecard_id=scorecard_id,
                disposition=scorecard_close_disposition,
            )
            return closed
        """
    )

    source = _replace_function_block(source, "    def _current_frame(env):", "    def _state(frame):", current_frame)
    source = _replace_function_block(source, "    def _terminal_reason(frame):", "    def _observation(frame, frame_index, game_id):", terminal_reason)
    source = _replace_function_block(source, "    def _direct_config():", "    def _cleanup_delegate(delegate):", direct_config)
    source = _replace_function_block(
        source,
        "    def _run_direct_game(env, game_id, initial_frame, abort_event=None):",
        "    def _write_results(status, results, game_count):",
        run_direct_game,
    )
    source = _replace_function_block(
        source,
        "    def _write_results(status, results, game_count):",
        "    try:\n        (working_root / 'arc_phase_marker.txt').write_text(",
        write_results,
    )
    return source


def _competition_child_source() -> str:
    source = _adapt_working_phase_b_source(_working_wrapper_source("working_phase_b.py"))
    compile(source, "<lcld_competition_child>", "exec")
    return source


def _adapt_working_common_source(source: str, preflight_script: str) -> str:
    payload_check = dedent(
        """\
        def _assert_payload_structure():
            code_dir = pathlib.Path(os.environ.get('ARC_AGENT_CODE_DIR', '/tmp/arc_lcld_agent/Code')).resolve()
            required_payload = [
                code_dir / 'kaggle_agent.py',
                code_dir / 'submission.py',
                code_dir / 'lcld_competition_child.py',
                code_dir / 'v9_agent' / '__init__.py',
                code_dir / 'v9_agent' / 'session.py',
                code_dir / 'v9_agent' / 'llm.py',
                code_dir / 'v9_agent' / 'qwen_packet.py',
            ]
            missing_payload = [str(path) for path in required_payload if not path.exists()]
            if missing_payload:
                raise FileNotFoundError('Embedded V9 Code payload missing files: ' + repr(missing_payload))
            os.environ['ARC_AGENT_CODE_DIR'] = str(code_dir)
            log_ok('ARC_AGENT_CODE_DIR', os.environ['ARC_AGENT_CODE_DIR'])
            return code_dir
        """
    )
    structural_preflight = dedent(
        f"""\
        def structural_preflight(*, phase, full_import_sweep):
            preflight_code = {preflight_script!r}
            compile(preflight_code, f'<lcld_direct_structural_preflight_{{phase}}>', 'exec')
            preflight_path = working_root / f'lcld_direct_structural_preflight_{{phase}}.py'
            preflight_path.write_text(preflight_code, encoding='utf-8')
            env = dict(os.environ)
            existing_pythonpath = env.get('PYTHONPATH', '')
            code_dir = pathlib.Path(os.environ['ARC_AGENT_CODE_DIR']).resolve()
            py_entries = [str(code_dir)]
            if existing_pythonpath:
                py_entries.append(existing_pythonpath)
            env['PYTHONPATH'] = os.pathsep.join(py_entries)
            print('Preflight PYTHONPATH=', env['PYTHONPATH'], flush=True)
            env['LCLD_PREFLIGHT_PHASE'] = phase
            env['LCLD_FULL_IMPORT_SWEEP'] = 'true' if full_import_sweep else 'false'
            timeout = 420 if full_import_sweep else 180
            run_cmd([sys.executable, '-u', preflight_path], cwd=working_root, env=env, timeout=timeout, check=True)
            log_ok(f'direct_structural_preflight_{{phase}}', 'ok')
        """
    )
    source = _replace_function_block(
        source,
        "def _assert_payload_structure():",
        "def _configure_qwen_env",
        payload_check,
    )
    source = _replace_function_block(
        source,
        "def structural_preflight(*, phase, full_import_sweep):",
        "def write_diagnostics_manifest",
        structural_preflight,
    )
    return source


def _current_agent_unpack_source(payload: str) -> str:
    return dedent(
        f"""\
        import base64, io, os, pathlib, sys, zipfile

        payload = {payload!r}
        agent_root = pathlib.Path('/tmp/arc_lcld_agent')
        if agent_root.exists():
            import shutil
            shutil.rmtree(agent_root)
        agent_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
            archive.extractall(agent_root)
        code_dir = (agent_root / 'Code').resolve()
        os.environ['ARC_AGENT_CODE_DIR'] = str(code_dir)
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        required = [
            code_dir / 'kaggle_agent.py',
            code_dir / 'submission.py',
            code_dir / 'lcld_competition_child.py',
            code_dir / 'v9_agent' / '__init__.py',
            code_dir / 'v9_agent' / 'session.py',
            code_dir / 'v9_agent' / 'llm.py',
            code_dir / 'v9_agent' / 'qwen_packet.py',
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Embedded V9 payload is incomplete: ' + repr(missing))
        print('LCLD V9 source ready:', os.environ['ARC_AGENT_CODE_DIR'], flush=True)
        """
    )


def build() -> dict:
    payload = _source_payload()
    common_source = _working_wrapper_source("working_common_v9.py")
    return {
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaRtx6000",
                "isGpuEnabled": True,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.13",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(_working_wrapper_source("working_header.md")),
            code_cell(_working_wrapper_source("working_install.py")),
            code_cell(_current_agent_unpack_source(payload)),
            code_cell(common_source),
            code_cell(_working_wrapper_source("working_phase_b_parent.py")),
            code_cell(_working_wrapper_source("working_phase_a.py")),
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build()
    serialized = json.dumps(notebook, ensure_ascii=False, separators=(",", ":"))
    notebook_size = len(serialized.encode("utf-8"))
    if notebook_size > MAX_NOTEBOOK_BYTES:
        raise SystemExit(
            f"Generated notebook is {notebook_size} bytes, above the {MAX_NOTEBOOK_BYTES}-byte safety cap."
        )
    NOTEBOOK_PATH.write_text(serialized, encoding="utf-8")
    KERNEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if KERNEL_METADATA_PATH.exists():
        try:
            original_metadata = json.loads(KERNEL_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            original_metadata = {}
    else:
        original_metadata = {}
    metadata = {
        "id": "vladimiryakunin/arc-prize-2026-lcld-qwen-v9",
        "title": "ARC Prize 2026 - LCLD Qwen V9",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": _ACCELERATORS[ACCELERATOR]["gpu"],
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": _ACCELERATORS[ACCELERATOR]["machine_shape"],
        "keywords": [],
        "dataset_sources": [VLLM_WHEELHOUSE_DATASET_SOURCE, QWEN_MODEL_DATASET_SOURCE],
        "kernel_sources": [],
        "competition_sources": [COMPETITION_SOURCE],
        "model_sources": [],
    }
    if metadata != original_metadata:
        KERNEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {NOTEBOOK_PATH} "
        f"(accelerator={ACCELERATOR}, notebook_bytes={notebook_size}, "
        f"payload_zip_bytes={_LAST_PAYLOAD_STATS.get('payload_zip_bytes')}, "
        f"payload_files={_LAST_PAYLOAD_STATS.get('archived_file_count')}, "
        f"excluded={_LAST_PAYLOAD_STATS.get('excluded_files')})"
    )


if __name__ == "__main__":
    main()
