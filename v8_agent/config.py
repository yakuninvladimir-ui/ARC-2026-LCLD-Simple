from __future__ import annotations

from dataclasses import dataclass, fields, replace
import os
from typing import Any


@dataclass(frozen=True, slots=True)
class V8Config:
    # LLM proposer. V8.3 keeps the public class name for package compatibility.
    enable_qwen: bool = True
    qwen_backend: str = "fake"  # disabled | fake | ollama | vllm | qwen_local | llama_cli
    qwen_model_path: str | None = None
    qwen_llama_cli_path: str | None = None
    qwen_ollama_base_url: str = "http://127.0.0.1:11434"
    qwen_ollama_model: str = "qwen_local_3_5"
    qwen_ollama_keep_alive: str = "-1"
    qwen_vllm_base_url: str = "http://127.0.0.1:1234/v1"
    qwen_vllm_api_key: str = "EMPTY"
    qwen_vllm_model: str = "vrfai/Qwen3.6-27B-FP8"
    qwen_llama_device: str | None = None
    qwen_split_mode: str = ""
    qwen_tensor_split: str = ""
    qwen_gpu_layers: int = 999
    qwen_temperature: float = 0.0
    qwen_top_k: int = 20
    qwen_top_p: float = 0.95
    qwen_min_p: float = 0.0
    qwen_presence_penalty: float = 1.5
    qwen_repeat_penalty: float = 1.0
    qwen_seed: int = 0
    qwen_timeout_seconds: int = 500
    qwen_context_tokens: int = 98304
    qwen_minimum_acceptance_context_tokens: int = 65536
    qwen_max_input_tokens: int = 65536
    qwen_max_output_tokens: int = 12288
    qwen_reserved_runtime_margin_tokens: int = 8192
    qwen_enable_thinking: bool = False
    qwen_reasoning_mode: str = "off"
    qwen_reasoning_budget_tokens: int = 0
    qwen_spec_type: str = ""
    qwen_spec_draft_n_max: int = 0
    qwen_require_runtime: bool = False
    qwen_trace_dir: str | None = None
    qwen_empty_output_retry_enabled: bool = False
    prompt_compaction_strategy: str = "normal"  # normal | aggressive | minimal
    prompt_compression_trigger_ratio: float = 0.82
    prompt_tail_priority_enabled: bool = True

    max_qwen_calls_per_game: int = 20
    max_primary_qwen_calls_per_level: int = 1
    max_reserve_qwen_calls_per_level: int = 0
    max_coordinate_qwen_calls_per_level: int = 1
    max_total_qwen_calls_per_level: int = 2
    min_steps_between_qwen_calls: int = 3
    qwen_stall_threshold: int = 3

    include_full_grid_in_qwen_packet: bool = True
    include_object_local_masks: bool = True
    include_component_graph_in_qwen_packet: bool = True
    max_components_in_packet: int = 96
    max_component_shape_runs: int = 32
    max_component_boundary_corners: int = 32
    max_objects_in_packet: int = 64
    max_object_masks_in_packet: int = 32
    max_relations_in_packet: int = 192
    max_semantic_objects_in_packet: int = 24
    max_coordinate_objects_in_packet: int = 48
    max_semantic_relations_in_packet: int = 24
    max_semantic_groups_in_packet: int = 24
    max_recent_transitions_in_packet: int = 12
    max_memory_notes_in_packet: int = 20
    max_active_hypotheses_in_packet: int = 30

    max_test_plan_length_default: int = 3
    max_test_plan_length_confirmed_rule: int = 5
    max_qwen_trajectory_steps: int = 50
    max_simple_action_probes_per_level: int = 10
    max_action_memory_records_in_packet: int = 40
    max_visual_change_groups_in_memory: int = 12
    max_visual_change_locations_per_group: int = 16
    max_clue_patterns_in_packet: int = 64
    max_coordinate_candidates_in_packet: int = 96
    max_coordinate_probes_per_level: int = 24
    max_coordinate_probe_repeats_per_signature: int = 1
    max_same_state_action_repeats: int = 1
    reject_unchanged_failed_trajectories: bool = True

    allow_qwen_raw_coordinates: bool = False
    require_coordinate_candidate_id: bool = True
    require_json_only: bool = True
    reject_hallucinated_ids: bool = True

    passive_attribution_enabled: bool = True
    passive_change_threshold_cells: int = 3
    mixed_change_ratio_threshold: float = 0.50
    relation_error_epsilon: float = 1e-6
    local_target_overlap_min_fraction: float = 0.10
    information_gain_min_threshold: float = 0.03

    # Competition orchestration. GAME_OVER is reset-capable; timeout belongs to
    # the outer harness and must not be converted into RESET.
    max_actions_per_game: int = 200
    max_actions_per_level: int = 0
    max_level_attempts: int = 4
    game_wall_clock_limit_seconds: int = 6000
    reset_on_game_over: bool = True
    max_game_over_resets_per_game: int = 0  # 0 = unlimited; diagnostics only
    max_game_over_resets_per_level: int = 0  # 0 = unlimited; diagnostics only
    default_coordinate_action_id: str = "ACTION6"

    merge_multicolor_components: bool = True
    preserve_object_tracks: bool = True


def _bool_env(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce(value: Any, target: type, current: Any) -> Any:
    if value is None:
        return current
    if target is bool:
        return _bool_env(str(value), bool(current))
    if target is int:
        return int(float(value))
    if target is float:
        return float(value)
    if target is str:
        return str(value)
    return value


def config_from_mapping(mapping: dict[str, Any] | None = None) -> V8Config:
    cfg = V8Config()
    data = dict(mapping or {})

    aliases = {
        "llm_advisor_backend": "qwen_backend",
        "llm_timeout_seconds": "qwen_timeout_seconds",
        "qwen_strict_required": "qwen_require_runtime",
        "real_qwen_required": "qwen_require_runtime",
        "qwen_context_tokens": "qwen_context_tokens",
        "max_qwen_calls_per_level": "max_total_qwen_calls_per_level",
        "max_qwen_primary_calls_per_level": "max_primary_qwen_calls_per_level",
        "max_qwen_replan_calls_per_level": "max_reserve_qwen_calls_per_level",
        "max_qwen_coordinate_calls_per_level": "max_coordinate_qwen_calls_per_level",
    }
    for old, new in aliases.items():
        if old in data and new not in data:
            data[new] = data[old]

    env_aliases = {
        "enable_qwen": ("ARC_V8_ENABLE_QWEN", "ARC_ENABLE_LLM_SEMANTIC_ADVISOR"),
        "qwen_backend": ("ARC_V8_QWEN_BACKEND", "ARC_LLM_ADVISOR_BACKEND"),
        "qwen_model_path": ("ARC_QWEN_MODEL_PATH", "ARC_LLM_MODEL_PATH"),
        "qwen_llama_cli_path": ("ARC_QWEN_LLAMA_CLI_PATH",),
        "qwen_ollama_base_url": ("ARC_QWEN_OLLAMA_BASE_URL",),
        "qwen_ollama_model": ("ARC_QWEN_OLLAMA_MODEL",),
        "qwen_ollama_keep_alive": ("ARC_QWEN_OLLAMA_KEEP_ALIVE",),
        "qwen_vllm_base_url": ("ARC_QWEN_VLLM_BASE_URL", "OPENAI_BASE_URL"),
        "qwen_vllm_api_key": ("ARC_QWEN_VLLM_API_KEY", "OPENAI_API_KEY"),
        "qwen_vllm_model": ("ARC_QWEN_VLLM_MODEL",),
        "qwen_llama_device": ("ARC_QWEN_LLAMA_DEVICE",),
        "qwen_split_mode": ("ARC_QWEN_SPLIT_MODE", "LLAMA_ARG_SPLIT_MODE"),
        "qwen_tensor_split": ("ARC_QWEN_TENSOR_SPLIT", "LLAMA_ARG_TENSOR_SPLIT"),
        "qwen_gpu_layers": ("ARC_QWEN_GPU_LAYERS",),
        "qwen_timeout_seconds": ("ARC_QWEN_TIMEOUT_SECONDS", "LCLD_QWEN_TIMEOUT_SECONDS"),
        "qwen_context_tokens": ("ARC_QWEN_CONTEXT_TOKENS",),
        "qwen_minimum_acceptance_context_tokens": ("ARC_QWEN_MINIMUM_ACCEPTANCE_CONTEXT_TOKENS",),
        "qwen_max_input_tokens": ("ARC_QWEN_MAX_INPUT_TOKENS",),
        "qwen_max_output_tokens": ("ARC_QWEN_MAX_OUTPUT_TOKENS",),
        "qwen_reserved_runtime_margin_tokens": ("ARC_QWEN_RESERVED_RUNTIME_MARGIN_TOKENS", "ARC_QWEN_CONTEXT_RESERVED_MARGIN"),
        "qwen_temperature": ("ARC_QWEN_TEMPERATURE",),
        "qwen_top_k": ("ARC_QWEN_TOP_K",),
        "qwen_top_p": ("ARC_QWEN_TOP_P",),
        "qwen_min_p": ("ARC_QWEN_MIN_P",),
        "qwen_presence_penalty": ("ARC_QWEN_PRESENCE_PENALTY",),
        "qwen_repeat_penalty": ("ARC_QWEN_REPEAT_PENALTY",),
        "qwen_seed": ("ARC_QWEN_SEED",),
        "qwen_enable_thinking": ("ARC_QWEN_ENABLE_THINKING",),
        "qwen_reasoning_mode": ("ARC_QWEN_REASONING_MODE",),
        "qwen_reasoning_budget_tokens": ("ARC_QWEN_REASONING_BUDGET_TOKENS",),
        "qwen_spec_type": ("ARC_QWEN_SPEC_TYPE",),
        "qwen_spec_draft_n_max": ("ARC_QWEN_SPEC_DRAFT_N_MAX",),
        "qwen_trace_dir": ("ARC_QWEN_TRACE_DIR",),
        "qwen_empty_output_retry_enabled": ("ARC_QWEN_EMPTY_OUTPUT_RETRY_ENABLED",),
        "include_component_graph_in_qwen_packet": ("ARC_V8_INCLUDE_COMPONENT_GRAPH",),
        "max_components_in_packet": ("ARC_V8_MAX_COMPONENTS_IN_PACKET",),
        "max_qwen_calls_per_game": ("ARC_V8_MAX_QWEN_CALLS_PER_GAME",),
        "max_primary_qwen_calls_per_level": ("ARC_MAX_QWEN_PRIMARY_CALLS_PER_LEVEL", "ARC_V8_MAX_PRIMARY_QWEN_CALLS_PER_LEVEL"),
        "max_reserve_qwen_calls_per_level": ("ARC_MAX_QWEN_REPLAN_CALLS_PER_LEVEL", "ARC_MAX_QWEN_RESERVE_CALLS_PER_LEVEL", "ARC_V8_MAX_RESERVE_QWEN_CALLS_PER_LEVEL"),
        "max_coordinate_qwen_calls_per_level": ("ARC_MAX_QWEN_COORDINATE_CALLS_PER_LEVEL", "ARC_V8_MAX_COORDINATE_QWEN_CALLS_PER_LEVEL"),
        "max_total_qwen_calls_per_level": ("ARC_MAX_TOTAL_QWEN_CALLS_PER_LEVEL", "ARC_V8_MAX_TOTAL_QWEN_CALLS_PER_LEVEL"),
        "max_actions_per_game": ("LCLD_MAX_ACTIONS_PER_GAME", "ARC_V8_MAX_ACTIONS_PER_GAME"),
        "max_actions_per_level": ("LCLD_MAX_ACTIONS_PER_LEVEL", "ARC_V8_MAX_ACTIONS_PER_LEVEL"),
        "max_level_attempts": ("LCLD_MAX_LEVEL_ATTEMPTS", "ARC_V8_MAX_LEVEL_ATTEMPTS"),
        "game_wall_clock_limit_seconds": ("LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS", "ARC_V8_GAME_WALL_CLOCK_LIMIT_SECONDS"),
        "reset_on_game_over": ("LCLD_RESET_ON_GAME_OVER", "ARC_V8_RESET_ON_GAME_OVER"),
        "qwen_require_runtime": ("ARC_V8_REQUIRE_QWEN_RUNTIME", "LCLD_REQUIRE_QWEN_RUNTIME"),
        "prompt_compaction_strategy": ("ARC_LLM_CONTEXT_STRATEGY",),
    }
    normalized: dict[str, Any] = {}
    for f in fields(V8Config):
        if f.name in data:
            normalized[f.name] = data[f.name]
        for env_name in env_aliases.get(f.name, ()):
            raw = os.environ.get(env_name)
            if raw not in (None, ""):
                normalized[f.name] = raw
                break

    for f in fields(V8Config):
        if f.name not in normalized:
            continue
        current = getattr(cfg, f.name)
        target = type(current) if current is not None else str
        cfg = replace(cfg, **{f.name: _coerce(normalized[f.name], target, current)})

    if cfg.qwen_backend in {"qwen_local", "llama_cli"} and not cfg.qwen_model_path:
        cfg = replace(cfg, qwen_model_path=os.environ.get("ARC_QWEN_MODEL_PATH") or "Code/qwen/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
    if cfg.prompt_compaction_strategy not in {"normal", "aggressive", "minimal"}:
        cfg = replace(cfg, prompt_compaction_strategy="normal")
    cfg = replace(cfg, qwen_enable_thinking=False, qwen_reasoning_mode="off", qwen_reasoning_budget_tokens=0)
    return cfg


def default_config_dict() -> dict[str, Any]:
    return {f.name: getattr(V8Config(), f.name) for f in fields(V8Config)}
