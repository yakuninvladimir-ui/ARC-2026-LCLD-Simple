"""Build the offline ARC-AGI-3 Kaggle notebook for the V8.3 agent package.

Place this file in the root of the unpacked V8.3 archive and run:

    python build_notebook.py

The builder embeds the local ``agent/`` and ``v8_agent/`` source tree into the
notebook payload.  It keeps the direct Arcade competition path: Phase B calls a
lightweight ``ARC_AGI_Agent.act()`` shim directly and submits the resulting
``ActionInput`` through ``env.step()``.  It does not route gameplay through
``MyAgent``.

Qwen diagnostics: Phase A runs one short 4k smoke probe on two T4 GPUs in
explicit layer-split mode.  The previous max-context/heavy probe remains
removed.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from textwrap import dedent
import zipfile


ACCELERATOR = "t4"
_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}

ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "arc-prize-2026-lcld-qwen.ipynb"
KERNEL_METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
ASSET_METADATA_PATH = ROOT / "assets" / "dataset-metadata.json"
RUNTIME_DATASET_SOURCE = "vladimiryakunin/arc-lcld-qwen35-runtime"
RUNTIME_DATASET_SLUG = "arc-lcld-qwen35-runtime"
COMPETITION_SOURCE = "arc-prize-2026-arc-agi-3"

MARKER = "ARC_V8_3_COMPACT_VERIFIED_QWEN_DIRECT_ARCADE"
PREFERRED_QWEN_MODEL_FILENAME = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MAX_NOTEBOOK_BYTES = 985_000

PAYLOAD_EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", "notebooks", "assets"}
PAYLOAD_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
PAYLOAD_EXCLUDED_RELATIVE_PATHS = {
    "build_notebook.py",
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
        return RUNTIME_DATASET_SOURCE
    try:
        return str(json.loads(ASSET_METADATA_PATH.read_text(encoding="utf-8")).get("id") or RUNTIME_DATASET_SOURCE)
    except Exception:
        return RUNTIME_DATASET_SOURCE


def _generated_kaggle_agent_source() -> str:
    return dedent(
        r'''
        """V8.3 direct-Arcade compatibility shim for the generated Kaggle notebook."""

        from __future__ import annotations

        from typing import Any, Mapping

        from v8_agent import GameSession, config_from_mapping
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

            def act(self, observation: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Any:
                if config:
                    merged = dict(self.config)
                    merged.update(dict(config))
                    self.config = merged
                    # Apply dynamic timeout/budget fields to the active session without
                    # recreating it and losing task-local memory.
                    self._session.update_runtime_config(merged)
                action = self._session.act(dict(observation))
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
                return {
                    "current_attempt_index": self._current_attempt_index,
                    "attempt_history_count": self._attempt_history_count,
                    "competition_level_reset_ingestions": int(session.get("game_over_reset_count", 0) or 0),
                    "observed_transition_ingestions": int(session.get("observed_transition_ingestions", self._observed_transition_ingestions) or 0),
                    "observed_transition_duplicate_skips": int(session.get("observed_transition_duplicate_skips", self._observed_transition_duplicate_skips) or 0),
                    "pending_official_transition": bool(session.get("pending_official_transition", False)),
                    "pending_transition_token": session.get("pending_transition_token"),
                    "last_committed_transition_token": session.get("last_committed_transition_token"),
                    "failed_memory_count": int(session.get("failed_memory_count", 0) or 0),
                    "irrelevant_memory_count": int(session.get("irrelevant_memory_count", 0) or 0),
                    "action_selection": {
                        "actions_emitted_from_degraded_path": 0,
                        "score_delta_from_degraded_actions": 0.0,
                        "levels_completed_after_degraded_action": 0,
                        "terminal_wins_after_degraded_action": 0,
                    },
                    "termination_records": list(self._termination_records),
                }

            def _cleanup_old_session(self) -> None:
                return None


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
        """V8.3 direct-Arcade submission helpers for the generated Kaggle notebook."""

        from __future__ import annotations

        import os
        from typing import Any

        from v8_agent.config import default_config_dict


        def default_config() -> dict[str, Any]:
            cfg = default_config_dict()
            cfg.update({
                "enable_qwen": True,
                "qwen_backend": os.environ.get("ARC_V8_QWEN_BACKEND", os.environ.get("ARC_LLM_ADVISOR_BACKEND", "llama_cli")),
                "qwen_model_path": os.environ.get("ARC_QWEN_MODEL_PATH") or os.environ.get("ARC_LLM_MODEL_PATH"),
                "qwen_llama_cli_path": os.environ.get("ARC_QWEN_LLAMA_CLI_PATH"),
                "qwen_llama_device": os.environ.get("ARC_QWEN_LLAMA_DEVICE"),
                "qwen_split_mode": os.environ.get("ARC_QWEN_SPLIT_MODE", os.environ.get("LLAMA_ARG_SPLIT_MODE", "")),
                "qwen_tensor_split": os.environ.get("ARC_QWEN_TENSOR_SPLIT", os.environ.get("LLAMA_ARG_TENSOR_SPLIT", "")),
                "qwen_gpu_layers": int(os.environ.get("ARC_QWEN_GPU_LAYERS", "999")),
                "qwen_timeout_seconds": int(os.environ.get("ARC_QWEN_TIMEOUT_SECONDS", "500")),
                "qwen_context_tokens": int(os.environ.get("ARC_QWEN_CONTEXT_TOKENS", "98304")),
                "qwen_minimum_acceptance_context_tokens": int(os.environ.get("ARC_QWEN_MINIMUM_ACCEPTANCE_CONTEXT_TOKENS", "65536")),
                "qwen_max_input_tokens": int(os.environ.get("ARC_QWEN_MAX_INPUT_TOKENS", "65536")),
                "qwen_max_output_tokens": int(os.environ.get("ARC_QWEN_MAX_OUTPUT_TOKENS", "4096")),
                "qwen_reserved_runtime_margin_tokens": int(os.environ.get("ARC_QWEN_RESERVED_RUNTIME_MARGIN_TOKENS", "8192")),
                "qwen_require_runtime": os.environ.get("LCLD_REQUIRE_QWEN_RUNTIME", "1").lower() in {"1", "true", "yes", "on"},
                "max_qwen_calls_per_game": int(os.environ.get("ARC_V8_MAX_QWEN_CALLS_PER_GAME", "20")),
                "max_primary_qwen_calls_per_level": int(os.environ.get("ARC_MAX_QWEN_PRIMARY_CALLS_PER_LEVEL", "1")),
                "max_reserve_qwen_calls_per_level": int(os.environ.get("ARC_MAX_QWEN_REPLAN_CALLS_PER_LEVEL", "1")),
                "max_coordinate_qwen_calls_per_level": int(os.environ.get("ARC_MAX_QWEN_COORDINATE_CALLS_PER_LEVEL", "1")),
                "max_total_qwen_calls_per_level": int(os.environ.get("ARC_MAX_TOTAL_QWEN_CALLS_PER_LEVEL", "3")),
                "max_actions_per_game": int(os.environ.get("LCLD_MAX_ACTIONS_PER_GAME", "250")),
                "game_wall_clock_limit_seconds": int(float(os.environ.get("LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS", "5000"))),
                "max_game_over_resets_per_game": 0,
                "max_game_over_resets_per_level": 0,
                "reset_on_game_over": os.environ.get("LCLD_RESET_ON_GAME_OVER", "1").lower() in {"1", "true", "yes", "on"},
                # Compatibility names used by the direct competition harness.
                "llm_advisor_backend": os.environ.get("ARC_LLM_ADVISOR_BACKEND", "llama_cli"),
                "llm_timeout_seconds": int(os.environ.get("ARC_QWEN_TIMEOUT_SECONDS", "500")),
                "qwen_prompt_profile": "v8_3_verified_contract_json",
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
            out: list[list[int]] = []
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
        ROOT / "agent" / "my_agent.py",
        ROOT / "v8_agent" / "__init__.py",
        ROOT / "v8_agent" / "session.py",
        ROOT / "v8_agent" / "llm.py",
        ROOT / "v8_agent" / "qwen_packet.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("V8.3 source tree is incomplete; missing: " + repr(missing))

    buffer = io.BytesIO()
    archived_files: list[str] = []
    excluded_files: list[str] = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_LZMA) as archive:
        for folder in (ROOT / "agent", ROOT / "v8_agent"):
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
        archived_files.extend(["kaggle_agent.py", "submission.py"])

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

        phase = os.environ.get('ARC_V83_PREFLIGHT_PHASE', 'unknown')
        print('=== ARC V8.3 DIRECT-AGENT STRUCTURAL PREFLIGHT START ===', phase, flush=True)
        print('cwd=', os.getcwd(), flush=True)
        print('ARC_AGENT_CODE_DIR=', os.environ.get('ARC_AGENT_CODE_DIR'), flush=True)
        print('ARC_QWEN_MODEL_PATH=', os.environ.get('ARC_QWEN_MODEL_PATH'), flush=True)
        print('ARC_QWEN_LLAMA_CLI_PATH=', os.environ.get('ARC_QWEN_LLAMA_CLI_PATH'), flush=True)

        code_dir = pathlib.Path(os.environ['ARC_AGENT_CODE_DIR']).resolve()
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        print('sys.path.head=', sys.path[:8], flush=True)

        for rel in (
            'kaggle_agent.py', 'submission.py', 'agent/my_agent.py',
            'v8_agent/session.py', 'v8_agent/llm.py', 'v8_agent/qwen_packet.py',
        ):
            path = code_dir / rel
            assert path.exists(), 'required payload file missing: ' + str(path)
            print('[OK] payload file', rel, str(path), flush=True)

        for name in ('kaggle_agent', 'submission', 'agent.my_agent', 'v8_agent', 'v8_agent.session', 'v8_agent.llm'):
            mod = importlib.import_module(name)
            mod_file_raw = getattr(mod, '__file__', None)
            if mod_file_raw:
                mod_file = pathlib.Path(mod_file_raw).resolve()
                assert mod_file.is_relative_to(code_dir), name + ' imported from wrong location: ' + str(mod_file)
                print('[OK] import', name, '->', mod_file, flush=True)
            else:
                print('[OK] import', name, '-> namespace/no-file', flush=True)

        from kaggle_agent import ARC_AGI_Agent, arcade_step_args
        from submission import default_config
        cfg = default_config()
        delegate = ARC_AGI_Agent(cfg)
        assert hasattr(delegate, 'act') and callable(delegate.act), 'ARC_AGI_Agent.act missing'
        assert hasattr(delegate, 'observe_action_result') and callable(delegate.observe_action_result), 'observe_action_result missing'
        assert hasattr(delegate, 'harness_telemetry') and callable(delegate.harness_telemetry), 'harness_telemetry missing'
        assert callable(arcade_step_args), 'arcade_step_args missing'
        print('delegate qwen_backend=', delegate.config.get('qwen_backend'), flush=True)
        print('delegate qwen_model_path=', delegate.config.get('qwen_model_path'), flush=True)
        print('delegate qwen_llama_cli_path=', delegate.config.get('qwen_llama_cli_path'), flush=True)
        print('=== ARC V8.3 DIRECT-AGENT STRUCTURAL PREFLIGHT OK ===', phase, flush=True)
        """
    )
    compile(source, "<arc_v83_structural_preflight_builder>", "exec")
    return source


def build() -> dict:
    payload = _source_payload()
    preflight_script = _preflight_script_source()

    install_cell = code_cell(
        """\
        import pathlib
        import subprocess
        import sys

        competition_root = pathlib.Path('/kaggle/input/competitions/__COMPETITION_SOURCE__')
        wheel_dir = competition_root / 'arc_agi_3_wheels'
        if not wheel_dir.is_dir():
            raise RuntimeError(
                'Required Kaggle competition source is not mounted: '
                + str(wheel_dir)
                + '. Publish with notebooks/kernel-metadata.json, not as a bare ipynb.'
            )
        subprocess.run(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '--no-index',
                '--find-links',
                str(wheel_dir),
                'arc-agi',
                'python-dotenv',
            ],
            check=True,
        )
        """.replace('__COMPETITION_SOURCE__', COMPETITION_SOURCE)
    )

    unpack_cell = code_cell(
        f"""\
        import base64, io, os, pathlib, sys, zipfile

        payload = {payload!r}
        agent_root = pathlib.Path('/tmp/arc_v8_3_agent')
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
            code_dir / 'agent' / 'my_agent.py',
            code_dir / 'v8_agent' / 'session.py',
            code_dir / 'v8_agent' / 'llm.py',
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Embedded V8.3 payload is incomplete: ' + repr(missing))
        print('V8.3 source ready:', os.environ['ARC_AGENT_CODE_DIR'], flush=True)
        """
    )

    common_definitions_cell = code_cell(
        """\
        import json
        import os
        import pathlib
        import shutil
        import socket
        import stat
        import subprocess
        import sys
        import tarfile
        import tempfile
        import time
        from datetime import datetime, timezone

        MARKER = '__MARKER__'
        PREFERRED_QWEN_MODEL_FILENAME = '__PREFERRED_QWEN_MODEL_FILENAME__'

        input_root = pathlib.Path('/kaggle/input')
        working_root = pathlib.Path('/kaggle/working')
        working_root.mkdir(parents=True, exist_ok=True)
        qwen_trace_root = working_root / 'arc_qwen_trace'
        qwen_trace_root.mkdir(parents=True, exist_ok=True)
        submission_path = working_root / 'submission.parquet'
        os.environ.setdefault('MPLBACKEND', 'agg')

        print('=== ARC V8.3 common definitions loaded:', MARKER, '===', flush=True)
        print('KAGGLE_IS_COMPETITION_RERUN =', os.getenv('KAGGLE_IS_COMPETITION_RERUN'), flush=True)
        print('Python =', sys.version.replace('\\n', ' '), flush=True)
        print('CWD =', os.getcwd(), flush=True)

        def _env_present(name):
            return bool(os.getenv(name))

        def _gateway_dns_hint():
            try:
                socket.gethostbyname('gateway')
                return True
            except OSError:
                return False

        RERUN_ENV_TRUE = _env_present('KAGGLE_IS_COMPETITION_RERUN')
        GATEWAY_DNS_HINT = _gateway_dns_hint()
        IS_PHASE_B_CANDIDATE = bool(RERUN_ENV_TRUE or GATEWAY_DNS_HINT)
        print('RERUN_ENV_TRUE =', RERUN_ENV_TRUE, flush=True)
        print('GATEWAY_DNS_HINT =', GATEWAY_DNS_HINT, flush=True)
        print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)

        def _utc_now():
            return datetime.now(timezone.utc).isoformat()

        def log_ok(key, value='ok'):
            print(f'[OK] {key} = {value}', flush=True)

        def run_cmd(cmd, *, cwd=None, env=None, timeout=60, check=True, tail=8000):
            cmd = [str(x) for x in cmd]
            print('\\n$ ' + ' '.join(cmd), flush=True)
            started = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            elapsed = time.time() - started
            print(f'returncode={result.returncode} elapsed={elapsed:.2f}s', flush=True)
            if result.stdout:
                print('--- stdout tail ---', flush=True)
                print(result.stdout[-tail:], flush=True)
            if result.stderr:
                print('--- stderr tail ---', flush=True)
                print(result.stderr[-tail:], flush=True)
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
            return result

        def find_one(pattern, desc):
            matches = sorted(input_root.rglob(pattern))
            print(f'{desc} candidates ({len(matches)}):', flush=True)
            for item in matches[:60]:
                print('  ', item, flush=True)
            return matches[0] if matches else None

        def _replace_with_link_or_copy(link_path, target_path):
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                os.symlink(target_path.name, link_path)
                print('repaired shared-library link:', link_path.name, '->', target_path.name, flush=True)
            except Exception as exc:
                shutil.copy2(target_path, link_path)
                print('repaired shared-library stub by copying:', link_path.name, 'from', target_path.name, 'symlink_error=', type(exc).__name__, flush=True)

        def repair_zero_length_shared_library_links(runtime_dir):
            runtime_dir = pathlib.Path(runtime_dir)
            libs = sorted(p for p in runtime_dir.iterdir() if p.is_file() and '.so' in p.name)
            nonzero = [p for p in libs if p.stat().st_size > 0]
            zero = [p for p in libs if p.stat().st_size == 0]
            print('shared-library inventory before repair:', {'nonzero': len(nonzero), 'zero': len(zero)}, flush=True)
            unresolved = []
            for stub in zero:
                name = stub.name
                prefix = name.split('.so', 1)[0] + '.so' if '.so' in name else name
                exact_chain_candidates = [p for p in nonzero if p.name.startswith(name + '.') and p.name != name]
                family_candidates = [p for p in nonzero if p.name.startswith(prefix + '.') and p.name != name]
                candidates = exact_chain_candidates or family_candidates
                if not candidates:
                    unresolved.append(stub.name)
                    continue
                target = sorted(candidates, key=lambda p: (len(p.name), p.name))[-1]
                _replace_with_link_or_copy(stub, target)
            if unresolved:
                raise FileNotFoundError('unresolved zero-byte shared-library stubs: ' + repr(unresolved))

        def materialize_llama_runtime(cli_path):
            cli_path = pathlib.Path(cli_path)
            src_dir = cli_path.parent
            dst_dir = working_root / 'llama-runtime'
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            print('Copying llama runtime directory:', src_dir, '->', dst_dir, flush=True)
            shutil.copytree(src_dir, dst_dir, symlinks=True)
            target_cli = dst_dir / cli_path.name
            if not target_cli.exists():
                raise FileNotFoundError(f'llama-cli was not copied to runtime dir: {target_cli}')
            target_cli.chmod(target_cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            repair_zero_length_shared_library_links(dst_dir)
            return target_cli, dst_dir

        def extract_llama_runtime_archive(archive_path):
            archive_path = pathlib.Path(archive_path)
            extract_dir = working_root / 'llama-runtime-archive'
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            extract_root = extract_dir.resolve()
            print('Extracting llama runtime archive:', archive_path, '->', extract_dir, flush=True)
            with tarfile.open(archive_path, 'r:*') as archive:
                members = archive.getmembers()
                for member in members:
                    target = (extract_dir / member.name).resolve()
                    if target != extract_root and extract_root not in target.parents:
                        raise RuntimeError('unsafe path in llama runtime archive: ' + member.name)
                archive.extractall(extract_dir, members=members)
            candidates = sorted(path for path in extract_dir.rglob('llama-cli') if path.is_file())
            if not candidates:
                raise FileNotFoundError('llama runtime archive did not contain llama-cli: ' + str(archive_path))
            return candidates[0]

        def assert_llama_cli_capabilities(cli_path, *, env):
            result = subprocess.run([str(cli_path), '--help'], capture_output=True, text=True, timeout=30, check=False, env=env)
            help_text = (result.stdout or '') + '\\n' + (result.stderr or '')
            if result.returncode != 0:
                raise RuntimeError('llama-cli --help could not start; output_tail=' + repr(help_text[-4000:]))
            required_flags = ('--reasoning', '--flash-attn', '--cache-type-k', '--cache-type-v', '--no-warmup', '--single-turn', '--simple-io', '--split-mode', '--tensor-split')
            missing = [flag for flag in required_flags if flag not in help_text]
            if missing:
                raise RuntimeError('llama-cli is incompatible; missing flags: ' + repr(missing))
            log_ok('llama-cli capability audit', 'required flags present')
            return help_text

        def _capture_phase_b_qwen_command(model_path, llama_cli):
            from submission import default_config
            from v8_agent import config_from_mapping
            from v8_agent.llm import QwenClient
            from v8_agent.types import QwenRole
            import v8_agent.llm as qwen_llm

            config_mapping = default_config()
            config_mapping.update({
                'enable_qwen': True,
                'qwen_backend': 'llama_cli',
                'qwen_model_path': str(model_path),
                'qwen_llama_cli_path': str(llama_cli),
                'qwen_require_runtime': True,
                'qwen_trace_dir': None,
            })
            previous_trace_dir = os.environ.pop('ARC_QWEN_TRACE_DIR', None)
            try:
                config = config_from_mapping(config_mapping)
            finally:
                if previous_trace_dir is not None:
                    os.environ['ARC_QWEN_TRACE_DIR'] = previous_trace_dir

            captured = []
            real_run = qwen_llm.subprocess.run

            def capture_run(cmd, **kwargs):
                captured.append([str(value) for value in cmd])
                return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')

            qwen_llm.subprocess.run = capture_run
            try:
                QwenClient()._call_llama_cli_once(
                    QwenRole.PRIMARY,
                    {
                        'schema_version': 'phase_b_command_preflight',
                        'execution_constraints': {'allowed_action_ids': ['ACTION1']},
                        'scene': {},
                    },
                    config,
                )
            finally:
                qwen_llm.subprocess.run = real_run

            if len(captured) != 1:
                raise RuntimeError('Phase-B Qwen command builder did not emit exactly one subprocess command: ' + repr(captured))
            command = captured[0]
            if not command or pathlib.Path(command[0]).resolve() != pathlib.Path(llama_cli).resolve():
                raise RuntimeError('Phase-B Qwen command selected an unexpected executable: ' + repr(command[:1]))
            return command

        def assert_phase_b_command_builder_contract():
            with tempfile.TemporaryDirectory(prefix='arc_v83_phase_b_command_') as temp_dir_raw:
                temp_dir = pathlib.Path(temp_dir_raw)
                fake_model = temp_dir / PREFERRED_QWEN_MODEL_FILENAME
                fake_model.write_bytes(b'command-construction-only')
                fake_cli = temp_dir / 'llama-cli'
                fake_cli.write_text('#!/bin/sh\\nexit 0\\n', encoding='utf-8')
                fake_cli.chmod(fake_cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

                env_values = _phase_b_qwen_env_values(fake_model, fake_cli)
                previous = {key: os.environ.get(key) for key in env_values}
                try:
                    os.environ.update(env_values)
                    command = _capture_phase_b_qwen_command(fake_model, fake_cli)
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value

            required_tokens = (
                '-m', '-n', '-c', '--temp', '--top-k', '--top-p', '--min-p',
                '--presence-penalty', '--repeat-penalty', '--seed',
                '--no-display-prompt', '--simple-io', '--reasoning',
                '--reasoning-budget', '--log-disable', '--single-turn',
                '--chat-template-kwargs', '--device', '--split-mode', '--tensor-split', '--spec-type',
                '--spec-draft-n-max', '-ngl', '-f',
            )
            missing = [token for token in required_tokens if token not in command]
            if missing:
                raise RuntimeError('actual Phase-B Qwen command is incomplete; missing tokens: ' + repr(missing))
            log_ok('Phase-B command builder contract', 'actual v8_agent.llm command captured without model execution')
            return command

        def assert_phase_b_qwen_command_compatible(model_path, llama_cli, *, env, help_text=None):
            command = _capture_phase_b_qwen_command(model_path, llama_cli)
            if help_text is None:
                result = subprocess.run([command[0], '--help'], capture_output=True, text=True, timeout=30, check=False, env=env)
                help_text = (result.stdout or '') + '\\n' + (result.stderr or '')
                if result.returncode != 0:
                    raise RuntimeError('Phase-B Qwen executable --help failed; output_tail=' + repr(help_text[-4000:]))
            command_flags = sorted({token for token in command[1:] if token.startswith('-')})
            missing = [flag for flag in command_flags if flag not in help_text]
            if missing:
                raise RuntimeError(
                    'actual Phase-B Qwen command is incompatible with this llama executable; '
                    'unsupported flags: ' + repr(missing) + '; command=' + repr(command)
                )
            log_ok('actual Phase-B Qwen command audit', 'all emitted flags are supported')
            return command

        def assert_llama_runtime_dependencies(cli_path, *, env):
            ldd_result = run_cmd(['ldd', str(cli_path)], env=env, timeout=30, check=True)
            ldd_text = (ldd_result.stdout or '') + '\\n' + (ldd_result.stderr or '')
            unresolved = [line.strip() for line in ldd_text.splitlines() if 'not found' in line.lower()]
            if unresolved:
                raise RuntimeError('ldd reports unresolved llama runtime dependencies: ' + repr(unresolved))
            log_ok('llama runtime ldd audit', 'no unresolved dependencies')

        def _phase_b_qwen_env_values(model_path, llama_cli):
            return {
                'ARC_QWEN_MODEL_PATH': str(model_path),
                'ARC_LLM_MODEL_PATH': str(model_path),
                'ARC_QWEN_LLAMA_CLI_PATH': str(llama_cli),
                'ARC_QWEN_LLAMA_DEVICE': 'CUDA0,CUDA1',
                'ARC_QWEN_SPLIT_MODE': 'layer',
                'ARC_QWEN_TENSOR_SPLIT': '1,1',
                'LLAMA_ARG_SPLIT_MODE': 'layer',
                'LLAMA_ARG_TENSOR_SPLIT': '1,1',
                'ARC_QWEN_GPU_LAYERS': '999',
                'ARC_V8_QWEN_BACKEND': 'llama_cli',
                'ARC_LLM_ADVISOR_BACKEND': 'llama_cli',
                'ARC_V8_ENABLE_QWEN': 'true',
                'ARC_QWEN_CONTEXT_TOKENS': '98304',
                'ARC_QWEN_MINIMUM_ACCEPTANCE_CONTEXT_TOKENS': '65536',
                'ARC_QWEN_MAX_INPUT_TOKENS': '65536',
                'ARC_QWEN_MAX_OUTPUT_TOKENS': '4096',
                'ARC_QWEN_RESERVED_RUNTIME_MARGIN_TOKENS': '8192',
                'ARC_QWEN_REASONING_MODE': 'off',
                'ARC_QWEN_REASONING_BUDGET_TOKENS': '0',
                'ARC_QWEN_TIMEOUT_SECONDS': '500',
                'ARC_QWEN_TEMPERATURE': '0.7',
                'ARC_QWEN_TOP_K': '20',
                'ARC_QWEN_TOP_P': '0.8',
                'ARC_QWEN_MIN_P': '0.0',
                'ARC_QWEN_PRESENCE_PENALTY': '1.5',
                'ARC_QWEN_REPEAT_PENALTY': '1.0',
                'ARC_QWEN_ENABLE_THINKING': '0',
                'ARC_QWEN_SPEC_TYPE': 'draft-mtp',
                'ARC_QWEN_SPEC_DRAFT_N_MAX': '2',
                'ARC_QWEN_SEED': '0',
                'ARC_QWEN_FLASH_ATTN': 'on',
                'ARC_QWEN_CACHE_TYPE_K': 'q8_0',
                'ARC_QWEN_CACHE_TYPE_V': 'q8_0',
                'ARC_QWEN_BATCH_SIZE': '2048',
                'ARC_QWEN_UBATCH_SIZE': '512',
                'ARC_QWEN_NO_WARMUP': 'true',
                'ARC_V8_MAX_QWEN_CALLS_PER_GAME': '20',
                'ARC_MAX_QWEN_PRIMARY_CALLS_PER_LEVEL': '1',
                'ARC_MAX_QWEN_COORDINATE_CALLS_PER_LEVEL': '1',
                'ARC_MAX_QWEN_REPLAN_CALLS_PER_LEVEL': '1',
                'ARC_MAX_TOTAL_QWEN_CALLS_PER_LEVEL': '3',
                'ARC_QWEN_TRACE_DIR': '/kaggle/working/arc_qwen_trace',
                'LCLD_REQUIRE_QWEN_RUNTIME': '1',
                'ARC_V8_REQUIRE_QWEN_RUNTIME': '1',
                'LCLD_MAX_ACTIONS_PER_GAME': '250',
                'LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS': os.environ.get('LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '5000'),
                'LCLD_MAX_GAME_OVER_RESETS_PER_GAME': '0',
                'LCLD_MAX_GAME_OVER_RESETS_PER_LEVEL': '0',
                'LCLD_RESET_ON_GAME_OVER': os.environ.get('LCLD_RESET_ON_GAME_OVER', '1'),
                'ENVIRONMENTS_DIR': '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files',
                'ARC_API_BASE': 'http://gateway:8001',
            }

        def _configure_qwen_env(model_path, llama_cli, llama_runtime_dir):
            os.environ.update(_phase_b_qwen_env_values(model_path, llama_cli))
            os.environ['LD_LIBRARY_PATH'] = str(llama_runtime_dir) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
            log_ok('ARC_QWEN_MODEL_PATH', os.environ['ARC_QWEN_MODEL_PATH'])
            log_ok('ARC_QWEN_LLAMA_CLI_PATH', os.environ['ARC_QWEN_LLAMA_CLI_PATH'])

        def setup_model_and_llama(*, phase, heavy_diagnostics, qwen_probe):
            model_path = find_one(PREFERRED_QWEN_MODEL_FILENAME, 'Qwen model')
            if model_path is None:
                raise FileNotFoundError(PREFERRED_QWEN_MODEL_FILENAME + ' not found in /kaggle/input')
            model_size = int(model_path.stat().st_size)
            if model_size < 20_000_000_000:
                raise RuntimeError(f'Qwen model file is unexpectedly small ({model_size} bytes): {model_path}')
            log_ok('Qwen model selected', f'{model_path} ({model_size} bytes)')

            llama_cli_path = find_one('llama-cli', 'prebuilt llama-cli')
            if llama_cli_path is None:
                archive_path = find_one('llama-cli-linux-kaggle-p100-cuda.tar.gz', 'prebuilt llama runtime archive') or find_one('llama.cpp-b9722.tar.gz', 'legacy prebuilt llama runtime archive')
                if archive_path is None:
                    raise FileNotFoundError('prebuilt llama-cli or llama runtime archive is required')
                llama_cli_path = extract_llama_runtime_archive(archive_path)
            log_ok('llama-cli source selected', llama_cli_path)

            llama_cli, llama_runtime_dir = materialize_llama_runtime(llama_cli_path)
            _configure_qwen_env(model_path, llama_cli, llama_runtime_dir)
            diagnostic_env = dict(os.environ)
            help_text = assert_llama_cli_capabilities(llama_cli, env=diagnostic_env)
            if str(phase).startswith('phase_b'):
                assert_llama_runtime_dependencies(llama_cli, env=diagnostic_env)
                phase_b_command = assert_phase_b_qwen_command_compatible(
                    model_path,
                    llama_cli,
                    env=diagnostic_env,
                    help_text=help_text,
                )
            else:
                phase_b_command = []

            if heavy_diagnostics:
                run_cmd(['bash', '-lc', 'uname -a; df -h /kaggle/working /kaggle/input || true; free -h || true'], env=diagnostic_env, timeout=30, check=True)
                run_cmd(['bash', '-lc', 'nvidia-smi'], env=diagnostic_env, timeout=60, check=True)
            else:
                log_ok(f'{phase} lightweight llama diagnostics', 'skipped heavy diagnostics')

            qwen_smoke_metrics = {}
            if qwen_probe:
                baseline_prompt = working_root / 'qwen_probe_baseline.txt'
                baseline_prompt.write_text('Return strict JSON only: {"ok": true}\\n', encoding='utf-8')
                cmd = [
                    str(llama_cli), '-m', str(model_path),
                    '-n', '64', '-c', '4096',
                    '--temp', os.environ['ARC_QWEN_TEMPERATURE'],
                    '--top-k', os.environ['ARC_QWEN_TOP_K'],
                    '--top-p', os.environ['ARC_QWEN_TOP_P'],
                    '--min-p', os.environ['ARC_QWEN_MIN_P'],
                    '--presence-penalty', os.environ['ARC_QWEN_PRESENCE_PENALTY'],
                    '--repeat-penalty', os.environ['ARC_QWEN_REPEAT_PENALTY'],
                    '--seed', os.environ['ARC_QWEN_SEED'],
                    '--no-display-prompt', '--simple-io', '--single-turn',
                    '--reasoning', 'off',
                    '--flash-attn', os.environ['ARC_QWEN_FLASH_ATTN'],
                    '--cache-type-k', os.environ['ARC_QWEN_CACHE_TYPE_K'],
                    '--cache-type-v', os.environ['ARC_QWEN_CACHE_TYPE_V'],
                    '--batch-size', os.environ['ARC_QWEN_BATCH_SIZE'],
                    '--ubatch-size', os.environ['ARC_QWEN_UBATCH_SIZE'],
                    '--no-warmup', '--device', os.environ['ARC_QWEN_LLAMA_DEVICE'],
                    '--split-mode', os.environ['ARC_QWEN_SPLIT_MODE'],
                    '--tensor-split', os.environ['ARC_QWEN_TENSOR_SPLIT'],
                    '-ngl', os.environ['ARC_QWEN_GPU_LAYERS'], '-f', str(baseline_prompt),
                ]
                if os.environ.get('ARC_QWEN_SPEC_TYPE'):
                    insert_at = cmd.index('-ngl')
                    cmd[insert_at:insert_at] = ['--spec-type', os.environ['ARC_QWEN_SPEC_TYPE']]
                    if os.environ.get('ARC_QWEN_SPEC_DRAFT_N_MAX'):
                        cmd[insert_at + 2:insert_at + 2] = ['--spec-draft-n-max', os.environ['ARC_QWEN_SPEC_DRAFT_N_MAX']]
                print('=== QWEN LIGHT SMOKE START ===', flush=True)
                started = time.monotonic()
                result = subprocess.run(cmd, env=diagnostic_env, text=True, capture_output=True, timeout=500, check=False)
                elapsed = time.monotonic() - started
                combined = (result.stdout or '') + '\\n' + (result.stderr or '')
                raw_path = working_root / 'qwen_probe_baseline_4k_full.log'
                raw_path.write_text(combined, encoding='utf-8', errors='replace')
                print('--- stdout tail ---\\n' + (result.stdout or '')[-12000:], flush=True)
                print('--- stderr tail ---\\n' + (result.stderr or '')[-16000:], flush=True)
                qwen_smoke_metrics = {
                    'baseline_4k': {
                        'status': 'ok' if result.returncode == 0 else 'nonzero_exit',
                        'returncode': int(result.returncode),
                        'elapsed_seconds': round(elapsed, 6),
                        'full_log': str(raw_path),
                        'thinking_marker': '[Start thinking]' in combined or '<think>' in combined,
                    }
                }
                print('QWEN_LIGHT_SMOKE_SUMMARY=' + json.dumps(qwen_smoke_metrics, sort_keys=True), flush=True)
                if result.returncode != 0 or qwen_smoke_metrics['baseline_4k']['thinking_marker'] or '"ok"' not in (result.stdout or ''):
                    raise RuntimeError('baseline Qwen smoke failed: ' + repr(qwen_smoke_metrics['baseline_4k']))
                log_ok('Qwen light smoke', 'baseline_4k completed')
            else:
                log_ok('Qwen light smoke', 'skipped')

            return {
                'model_path': str(model_path),
                'llama_cli': str(llama_cli),
                'llama_runtime_dir': str(llama_runtime_dir),
                'qwen_smoke': qwen_smoke_metrics,
                'phase_b_command': phase_b_command,
            }

        def setup_arcade_client_env():
            env_path = working_root / '.env'
            env_path.write_text(
                'SCHEME=http\\n'
                'HOST=gateway\\n'
                'PORT=8001\\n'
                'ARC_API_KEY=test-key-123\\n'
                'ARC_API_BASE=http://gateway:8001\\n'
                'ARC_BASE_URL=http://gateway:8001/\\n'
                'OPERATION_MODE=online\\n'
                'ENVIRONMENTS_DIR=/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files\\n'
                'RECORDINGS_DIR=/kaggle/working/server_recording\\n'
                'LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS=5000\\n'
                'LCLD_MAX_GAME_OVER_RESETS_PER_GAME=0\\n'
                'LCLD_MAX_GAME_OVER_RESETS_PER_LEVEL=0\\n'
                'LCLD_RESET_ON_GAME_OVER=1\\n',
                encoding='utf-8',
            )
            log_ok('direct Arcade client env', env_path)
            return env_path

        def structural_preflight(*, phase):
            preflight_code = __PREFLIGHT_SCRIPT_SOURCE__
            compile(preflight_code, f'<arc_v83_structural_preflight_{phase}>', 'exec')
            preflight_path = working_root / f'arc_v83_structural_preflight_{phase}.py'
            preflight_path.write_text(preflight_code, encoding='utf-8')
            env = dict(os.environ)
            code_dir = pathlib.Path(os.environ['ARC_AGENT_CODE_DIR']).resolve()
            env['PYTHONPATH'] = str(code_dir) + (os.pathsep + env.get('PYTHONPATH', '') if env.get('PYTHONPATH') else '')
            env['ARC_V83_PREFLIGHT_PHASE'] = phase
            run_cmd([sys.executable, '-u', preflight_path], cwd=working_root, env=env, timeout=180, check=True)
            log_ok(f'direct_structural_preflight_{phase}', 'ok')

        def setup_runtime(*, phase, heavy_diagnostics, qwen_probe):
            print(f'=== ARC V8.3 {phase} runtime setup START ===', flush=True)
            code_dir = pathlib.Path(os.environ.get('ARC_AGENT_CODE_DIR', '/tmp/arc_v8_3_agent/Code')).resolve()
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))
            runtime_info = setup_model_and_llama(phase=phase, heavy_diagnostics=heavy_diagnostics, qwen_probe=qwen_probe)
            arcade_env_path = setup_arcade_client_env()
            structural_preflight(phase=phase)
            manifest = {
                'marker': MARKER,
                'phase': phase,
                'created_at_utc': _utc_now(),
                'arc_agent_code_dir': os.environ.get('ARC_AGENT_CODE_DIR'),
                'runtime_info': runtime_info,
                'qwen_probe': qwen_probe,
                'heavy_diagnostics': heavy_diagnostics,
                'max_context_qwen_probe_removed': True,
            }
            path = working_root / f'arc_v83_{phase}_diagnostics.json'
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8')
            print(path.read_text(encoding='utf-8'), flush=True)
            print(f'=== ARC V8.3 {phase} runtime setup OK ===', flush=True)
            return arcade_env_path, manifest

        def gateway_handshake_or_die():
            print('=== ARC V8.3 Phase B gateway handshake START ===', flush=True)
            run_cmd(['bash', '-lc', 'date -u; curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games'], cwd=working_root, env=dict(os.environ), timeout=6000, check=True)
            print('=== ARC V8.3 Phase B gateway handshake OK ===', flush=True)
        """.replace('__MARKER__', MARKER)
        .replace('__PREFERRED_QWEN_MODEL_FILENAME__', PREFERRED_QWEN_MODEL_FILENAME)
        .replace('__PREFLIGHT_SCRIPT_SOURCE__', repr(preflight_script))
    )

    phase_a_cell = code_cell(
        """\
        print('=== ARC V8.3 Phase A two-T4 Qwen smoke gate ===', flush=True)
        print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)

        if not IS_PHASE_B_CANDIDATE:
            import pandas as pd
            if submission_path.exists():
                raise RuntimeError('NON-RERUN CONTAMINATION: submission.parquet exists before Phase-A smoke')

            phase_a_model_candidates = sorted(input_root.rglob(PREFERRED_QWEN_MODEL_FILENAME))
            if not phase_a_model_candidates:
                raise RuntimeError(
                    'Required Kaggle runtime dataset is not mounted: '
                    'vladimiryakunin/arc-lcld-qwen35-runtime. Expected '
                    + PREFERRED_QWEN_MODEL_FILENAME
                    + ' below /kaggle/input. Publish with notebooks/kernel-metadata.json, not as a bare ipynb.'
                )

            gpu_inventory = run_cmd(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader'],
                cwd=working_root,
                env=dict(os.environ),
                timeout=60,
                check=True,
            )
            gpu_lines = [line.strip() for line in (gpu_inventory.stdout or '').splitlines() if line.strip()]
            if len(gpu_lines) != 2 or any('T4' not in line.upper() for line in gpu_lines):
                raise RuntimeError('Phase-A smoke requires exactly two T4 GPUs; inventory=' + repr(gpu_lines))

            phase_b_command = assert_phase_b_command_builder_contract()
            arcade_env_path, manifest = setup_runtime(
                phase='phase_a_qwen_smoke',
                heavy_diagnostics=False,
                qwen_probe=True,
            )

            result = {
                'marker': MARKER,
                'phase': 'PHASE_A_TWO_T4_QWEN_SMOKE_OK',
                'created_at_utc': _utc_now(),
                'gateway_contacted': False,
                'gpu_inventory': gpu_lines,
                'qwen_runtime_setup': True,
                'qwen_smoke_probe': True,
                'qwen_smoke': manifest['runtime_info']['qwen_smoke'],
                'phase_b_command_builder_checked': True,
                'phase_b_command': phase_b_command,
                'max_context_qwen_probe_removed': True,
            }
            path = working_root / 'arc_v83_phase_a_smoke_result.json'
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8')
            print(path.read_text(encoding='utf-8'), flush=True)

            submission = pd.DataFrame(
                data=[['1_0', '1', True, 1]],
                columns=['row_id', 'game_id', 'end_of_game', 'score'],
            )
            submission.to_parquet(submission_path, index=False)
            read_back = pd.read_parquet(submission_path)
            if read_back.empty or str(read_back.iloc[0]['row_id']) != '1_0' or int(read_back.iloc[0]['score']) != 1:
                raise RuntimeError('Phase-A dummy parquet write/read failed')
            print('=== ARC V8.3 PHASE A TWO-T4 QWEN SMOKE OK ===', flush=True)
        else:
            print('Phase-A Qwen smoke skipped: this is a Phase-B competition rerun.', flush=True)
        """
    )

    phase_b_cell = code_cell(
        """\
        print('=== ARC V8.3 Phase B direct Arcade default-scorecard gate ===', flush=True)
        print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)

        if IS_PHASE_B_CANDIDATE:
            import gc
            import traceback
            from dotenv import load_dotenv

            trace_path = working_root / 'arc_v83_direct_agent_trace.log'
            result_path = working_root / 'arc_v83_kaggle_default_scorecard_results.json'
            failure_path = working_root / 'arc_v83_phase_b_failure.json'
            for stale_path in (submission_path, trace_path, result_path, failure_path):
                if stale_path.exists():
                    print('Removing stale Phase-B artifact:', stale_path, stale_path.stat().st_size, 'bytes', flush=True)
                    stale_path.unlink()

            def _trace(event, **fields):
                payload = {'time_utc': _utc_now(), 'event': event, **fields}
                with trace_path.open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False, default=str) + '\\n')

            def _game_id(env_info):
                value = getattr(env_info, 'game_id', None)
                return str(value if value is not None else env_info)

            def _frame_data(raw):
                if raw is None:
                    raise ValueError('gateway returned None frame data')
                try:
                    from arcengine import FrameData
                    if isinstance(raw, FrameData):
                        return raw
                except Exception:
                    pass
                if getattr(raw, 'frame', None) is None:
                    raise TypeError('gateway frame has no frame attribute: ' + type(raw).__name__)
                return raw

            def _current_frame(env):
                raw = getattr(env, 'observation_space', None)
                if callable(raw):
                    raw = raw()
                if raw is None:
                    observe = getattr(env, 'observe', None)
                    if callable(observe):
                        raw = observe()
                if raw is None:
                    reset = getattr(env, 'reset', None)
                    if callable(reset):
                        raw = reset()
                return _frame_data(raw)

            def _terminal_reason(frame):
                state_name = _state_name(getattr(frame, 'state', ''))
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

            def _observation(frame, frame_index, game_id):
                world_json = frame_to_world_json(frame)
                state_name = _state_name(getattr(frame, 'state', ''))
                metadata = {
                    'available_actions': list(world_json.get('available_actions', [])),
                    'frame_index': int(frame_index),
                    'full_reset': bool(world_json.get('full_reset', False)),
                    'game_id': world_json.get('game_id') or game_id,
                    'guid': world_json.get('guid'),
                    'official_runtime_input': True,
                    'score': world_json.get('score'),
                    'state': world_json.get('state'),
                    'game_over': state_name == 'GAME_OVER',
                    'win': state_name in {'WIN', 'WON', 'VICTORY'},
                    'levels_completed': getattr(frame, 'levels_completed', 0),
                    'win_levels': getattr(frame, 'win_levels', None),
                }
                return {'frame': world_json['grid'], 'grid': _frame_grid_to_2d(world_json['grid']), 'metadata': metadata}

            class DirectGameFailure(RuntimeError):
                def __init__(self, message, *, metrics):
                    super().__init__(message)
                    self.metrics = dict(metrics)

            def _run_direct_game(env, game_id):
                from arcengine import GameAction
                base_config = default_config()
                delegate = ARC_AGI_Agent(base_config)
                max_actions = max(1, int(os.getenv('LCLD_MAX_ACTIONS_PER_GAME', '250')))
                game_wall_limit = min(5000.0, max(1.0, float(os.getenv('LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '5000'))))
                qwen_call_limit = max(1, min(500, int(os.getenv('ARC_QWEN_TIMEOUT_SECONDS', '500'))))
                started = time.monotonic()
                accepted_actions = 0
                proposed_actions = 0
                rejected_actions = 0
                game_over_reset_count = 0
                level_game_over_reset_count = 0
                frame_index = 0
                latest = _current_frame(env)
                observed_levels_completed = int(getattr(latest, 'levels_completed', 0) or 0)
                stop_reason = ''
                termination_cause = ''

                _trace('direct_agent_init', game_id=game_id, initial_state=_state_name(getattr(latest, 'state', '')), max_actions=max_actions, game_wall_limit_seconds=game_wall_limit, reset_on_game_over=True)
                try:
                    while accepted_actions < max_actions:
                        now = time.monotonic()
                        if now - started >= game_wall_limit:
                            termination_cause = 'GAME_WALL_CLOCK_TIMEOUT'
                            stop_reason = 'game_wall_clock_timeout'
                            delegate.record_orchestration_termination(termination_cause, metadata={'reset_suppressed': True, 'terminal_by_timeout': True})
                            _trace('orchestration_timeout_terminal', game_id=game_id, cause=termination_cause, elapsed_game_seconds=round(now - started, 6), reset_suppressed=True)
                            break
                        success_reason = _terminal_reason(latest)
                        if success_reason:
                            termination_cause = 'ENVIRONMENT_SUCCESS'
                            stop_reason = success_reason
                            break

                        state_name = _state_name(getattr(latest, 'state', ''))
                        if state_name == 'GAME_OVER':
                            game_over_reset_count += 1
                            level_game_over_reset_count += 1
                            _trace('game_over_replay_requested', game_id=game_id, game_over_reset_count=game_over_reset_count, level_game_over_reset_count=level_game_over_reset_count)
                        is_initial_reset = state_name in {'NOT_PLAYED', 'NOT_STARTED'} and accepted_actions == 0
                        if state_name == 'NOT_STARTED' and not is_initial_reset:
                            termination_cause = 'ENVIRONMENT_NOT_STARTED_AFTER_INITIAL_RESET'
                            stop_reason = 'environment_not_started_after_initial_reset'
                            break

                        if is_initial_reset:
                            action_id = GameAction.RESET
                            action_data = {}
                            reasoning = {'agent': 'arc_v8_3', 'source': 'direct_start_reset', 'reset_reason': 'initial_start'}
                            action_was_agent_emitted = False
                        else:
                            remaining_game = max(0.0, game_wall_limit - (now - started))
                            dynamic_qwen_timeout = max(1, min(qwen_call_limit, int(remaining_game)))
                            step_config = dict(base_config)
                            step_config.update({'qwen_timeout_seconds': dynamic_qwen_timeout, 'llm_timeout_seconds': dynamic_qwen_timeout})
                            native_action = delegate.act(_observation(latest, frame_index, game_id), step_config)
                            action_id, action_data, reasoning = arcade_step_args(native_action)
                            action_was_agent_emitted = True

                        proposed_actions += 1
                        _trace('gateway_step_proposed', game_id=game_id, proposed_action_count=proposed_actions, accepted_action_count=accepted_actions, action=str(getattr(action_id, 'name', action_id)), data=action_data, state_before=state_name)
                        try:
                            raw_next = env.step(action_id, data=action_data, reasoning=reasoning)
                            next_frame = _frame_data(raw_next)
                        except Exception as exc:
                            rejected_actions += 1
                            _trace('gateway_step_rejected', game_id=game_id, proposed_action_count=proposed_actions, accepted_action_count=accepted_actions, rejected_action_count=rejected_actions, action=str(getattr(action_id, 'name', action_id)), exc_type=type(exc).__name__, error=str(exc)[:2000])
                            raise

                        accepted_actions += 1
                        frame_index += 1
                        latest = next_frame
                        current_completed = int(getattr(latest, 'levels_completed', 0) or 0)
                        if current_completed > observed_levels_completed:
                            observed_levels_completed = current_completed
                            level_game_over_reset_count = 0
                            _trace('level_progress_observed', game_id=game_id, levels_completed=current_completed, accepted_action_count=accepted_actions)
                        if action_was_agent_emitted:
                            if not delegate.observe_action_result(_observation(latest, frame_index, game_id)):
                                raise RuntimeError('accepted agent action did not ingest an official transition')
                            _trace('official_transition_ingested', game_id=game_id, accepted_action_count=accepted_actions, action=str(getattr(action_id, 'name', action_id)), state_after=_state_name(getattr(latest, 'state', '')))
                        _trace('gateway_step_accepted', game_id=game_id, proposed_action_count=proposed_actions, accepted_action_count=accepted_actions, rejected_action_count=rejected_actions, action=str(getattr(action_id, 'name', action_id)), state_after=_state_name(getattr(latest, 'state', '')), levels_completed=getattr(latest, 'levels_completed', 0))

                    if not stop_reason:
                        termination_cause = 'MAX_ACTIONS' if accepted_actions >= max_actions else 'LOOP_EXIT'
                        stop_reason = 'max_actions' if accepted_actions >= max_actions else 'loop_exit'
                        delegate.record_orchestration_termination(termination_cause, metadata={'reset_suppressed': True})
                    telemetry = delegate.harness_telemetry()
                    if bool(telemetry.get('pending_official_transition', False)):
                        raise RuntimeError('Phase-B loop exited with an unconsumed official transition')
                    return {
                        'action_count': int(accepted_actions),
                        'proposed_action_count': int(proposed_actions),
                        'rejected_action_count': int(rejected_actions),
                        'levels_completed': int(getattr(latest, 'levels_completed', 0) or 0),
                        'final_state': _state_name(getattr(latest, 'state', '')),
                        'final_guid': str(getattr(latest, 'guid', '') or ''),
                        'stop_reason': stop_reason,
                        'termination_cause': termination_cause,
                        'game_over_reset_count': int(game_over_reset_count),
                        'level_game_over_reset_count': int(level_game_over_reset_count),
                        'observed_transition_ingestions': int(telemetry.get('observed_transition_ingestions', 0) or 0),
                    }
                except Exception as exc:
                    raise DirectGameFailure(str(exc), metrics={
                        'action_count': int(accepted_actions),
                        'proposed_action_count': int(proposed_actions),
                        'rejected_action_count': int(rejected_actions),
                        'levels_completed': int(getattr(latest, 'levels_completed', 0) or 0),
                        'final_state': _state_name(getattr(latest, 'state', '')),
                        'final_guid': str(getattr(latest, 'guid', '') or ''),
                        'stop_reason': 'exception:' + type(exc).__name__,
                        'termination_cause': 'EXCEPTION',
                        'game_over_reset_count': int(game_over_reset_count),
                        'level_game_over_reset_count': int(level_game_over_reset_count),
                    }) from exc
                finally:
                    cleanup = getattr(delegate, '_cleanup_old_session', None)
                    if callable(cleanup):
                        cleanup()
                    gc.collect()

            def _write_results(status, results, game_count):
                payload = {
                    'marker': MARKER,
                    'status': status,
                    'created_at_utc': _utc_now(),
                    'execution_path': 'direct_ARC_AGI_Agent_act_to_env_step',
                    'game_count': int(game_count),
                    'attempted_games': len(results),
                    'completed_games': sum(1 for item in results if item.get('status') == 'completed'),
                    'failed_games': sum(1 for item in results if item.get('status') == 'failed'),
                    'total_actions': sum(int(item.get('action_count', 0) or 0) for item in results),
                    'total_proposed_actions': sum(int(item.get('proposed_action_count', 0) or 0) for item in results),
                    'total_rejected_actions': sum(int(item.get('rejected_action_count', 0) or 0) for item in results),
                    'levels_completed_observed': sum(int(item.get('levels_completed', 0) or 0) for item in results),
                    'results': results,
                    'scorecard_owner': 'kaggle_gateway_default',
                    'explicit_scorecard_opened': False,
                    'explicit_scorecard_closed': False,
                    'phase_b_parquet_created_by_notebook': False,
                }
                temporary = result_path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8')
                temporary.replace(result_path)
                return payload

            try:
                gateway_handshake_or_die()
                arcade_env_path, manifest = setup_runtime(phase='phase_b_direct_arcade', heavy_diagnostics=False, qwen_probe=False)
                (working_root / 'server_recording').mkdir(parents=True, exist_ok=True)
                load_dotenv(dotenv_path=arcade_env_path, override=True)
                code_dir = pathlib.Path(os.environ['ARC_AGENT_CODE_DIR']).resolve()
                if str(code_dir) not in sys.path:
                    sys.path.insert(0, str(code_dir))

                import arc_agi
                from kaggle_agent import ARC_AGI_Agent, arcade_step_args
                from submission import default_config, frame_to_world_json, _frame_grid_to_2d, _state_name

                arcade = arc_agi.Arcade()
                env_infos = list(arcade.available_environments)
                if not env_infos:
                    raise RuntimeError('Kaggle gateway returned no available environments')
                print('[Phase B] direct ARC_AGI_Agent scoring:', len(env_infos), 'environments; Arcade.make(game_id) uses Kaggle default scorecard', flush=True)

                results = []
                for index, env_info in enumerate(env_infos):
                    game_id = _game_id(env_info)
                    started = time.monotonic()
                    status = 'completed'
                    error_type = ''
                    error_text = ''
                    metrics = {'action_count': 0, 'proposed_action_count': 0, 'rejected_action_count': 0, 'levels_completed': 0, 'final_state': '', 'final_guid': '', 'stop_reason': ''}
                    print(f'[Phase B] starting game {index + 1}/{len(env_infos)}: {game_id}', flush=True)
                    try:
                        env = arcade.make(game_id)
                        if env is None:
                            raise RuntimeError('Arcade.make returned None for ' + game_id)
                        metrics = _run_direct_game(env, game_id)
                    except Exception as exc:
                        status = 'failed'
                        error_type = type(exc).__name__
                        error_text = str(exc)
                        failure_metrics = getattr(exc, 'metrics', None)
                        if isinstance(failure_metrics, dict):
                            metrics.update(failure_metrics)
                        print(f'[Phase B] game {game_id} failed after {time.monotonic() - started:.1f}s: {error_type}: {error_text}', flush=True)
                        traceback.print_exc()
                    result = {'game_id': game_id, 'status': status, **metrics, 'elapsed_seconds': round(time.monotonic() - started, 3), 'error_type': error_type, 'error': error_text[:2000]}
                    results.append(result)
                    progress = _write_results('in_progress', results, len(env_infos))
                    print('[Phase B] game result:', result, 'aggregate_accepted_actions=', progress['total_actions'], flush=True)

                final_payload = _write_results('games_attempted_kaggle_scorecard_open', results, len(env_infos))
                if final_payload['attempted_games'] <= 0:
                    raise RuntimeError('no Kaggle environments were attempted')
                if final_payload['total_actions'] <= 0:
                    raise RuntimeError('all games failed before any gateway-accepted action')
                if not trace_path.exists():
                    raise FileNotFoundError('direct-agent trace was not created')
                trace_text = trace_path.read_text(encoding='utf-8', errors='replace')
                for required_marker in ('direct_agent_init', 'gateway_step_proposed', 'gateway_step_accepted'):
                    if required_marker not in trace_text:
                        raise RuntimeError('required direct-agent trace marker is absent: ' + required_marker)
                print('=== ARC V8.3 PHASE B DIRECT GAMEPLAY COMPLETE; KAGGLE OWNS SCORECARD/SUBMISSION ===', flush=True)
                print(json.dumps(final_payload, indent=2)[:8000], flush=True)
            except BaseException as exc:
                if submission_path.exists():
                    try:
                        submission_path.unlink()
                        print('Deleted Phase-B parquet after fatal setup/orchestration failure:', submission_path, flush=True)
                    except OSError:
                        pass
                failure_path.write_text(json.dumps({'marker': MARKER, 'phase': 'PHASE_B_FATAL_FAILURE', 'error_type': type(exc).__name__, 'error': str(exc)}, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8')
                raise
        else:
            print('Phase B skipped: neither the rerun flag nor the gateway DNS hint is present.', flush=True)
        """
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(f"Unknown ACCELERATOR={ACCELERATOR!r}; pick one of {sorted(_ACCELERATORS)}")
    accel = _ACCELERATORS[ACCELERATOR]
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "file_extension": ".py"},
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 - V8.3 Simplified Qwen Direct Arcade\n\n"
                "The notebook embeds the V8.3 source payload and uses direct Arcade Phase-B gameplay. "
                "Phase A validates a two-T4 layer split with one short 4k Qwen smoke; max-context probes remain removed."
            ),
            install_cell,
            unpack_cell,
            common_definitions_cell,
            phase_b_cell,
            phase_a_cell,
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
        "id": "vladimiryakunin/arc-prize-2026-lcld-qwen",
        "title": "ARC Prize 2026 - LCLD Qwen",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": _ACCELERATORS[ACCELERATOR]["gpu"],
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [_asset_dataset_source()],
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
