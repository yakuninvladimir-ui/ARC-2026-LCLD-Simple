from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .config import V8Config
from .types import QwenRole


class QwenBackendError(RuntimeError):
    pass


class QwenClient:
    def call(self, role: QwenRole, packet: dict[str, Any], config: V8Config) -> dict[str, Any]:
        if not config.enable_qwen or config.qwen_backend == "disabled":
            return {}
        if config.qwen_backend == "fake":
            return FakeQwenClient().call(role, packet, config)
        if config.qwen_backend in {"qwen_local", "llama_cli"}:
            result = self._call_llama_cli_once(role, packet, config)
            if result or not config.qwen_empty_output_retry_enabled or not _packet_has_semantic_content(packet):
                return result
            retry_packet = compact_packet(packet, "minimal")
            return self._call_llama_cli_once(role, retry_packet, config)
        if config.qwen_require_runtime:
            raise QwenBackendError(f"unsupported qwen backend: {config.qwen_backend}")
        return {}

    def _call_llama_cli_once(self, role: QwenRole, packet: dict[str, Any], config: V8Config) -> dict[str, Any]:
        model_path = Path(str(config.qwen_model_path or os.environ.get("ARC_QWEN_MODEL_PATH") or ""))
        llama_cli = _resolve_llama_cli(config.qwen_llama_cli_path or os.environ.get("ARC_QWEN_LLAMA_CLI_PATH") or "llama-cli")
        if not model_path.exists():
            if config.qwen_require_runtime:
                raise QwenBackendError(f"Qwen model path does not exist: {model_path}")
            return {}
        if llama_cli is None:
            if config.qwen_require_runtime:
                raise QwenBackendError("llama-cli path is not executable or not found")
            return {}
        completion_cli = _resolve_llama_completion(llama_cli)
        generation_cli = completion_cli or llama_cli
        constrained_output = completion_cli is not None
        compacted = _fit_packet_to_budget(role, packet, config)
        prompt = _prompt(role, compacted, config)
        execution_prompt = _qwen_no_thinking_completion_prompt(prompt) if constrained_output else prompt
        output_schema = _allowed_output_schema(role, compacted)
        cmd = [
            str(generation_cli),
            "-m", str(model_path),
            "-n", str(config.qwen_max_output_tokens),
            "-c", str(config.qwen_context_tokens),
            "--temp", str(config.qwen_temperature),
            "--top-k", str(config.qwen_top_k),
            "--top-p", str(config.qwen_top_p),
            "--min-p", str(config.qwen_min_p),
            "--presence-penalty", str(config.qwen_presence_penalty),
            "--repeat-penalty", str(config.qwen_repeat_penalty),
            "--seed", str(config.qwen_seed),
            "--no-display-prompt",
            "--simple-io",
            "--reasoning", "off",
            "--reasoning-budget", "0",
        ]
        if constrained_output:
            # llama-cli injects chat-template control tokens before sampling. Those
            # tokens are outside the JSON grammar, so constrained decoding must use
            # the raw-completion frontend supplied by the same llama.cpp build.
            cmd.append("--no-conversation")
        else:
            cmd.extend([
                "--log-disable",
                "--single-turn",
                "--chat-template-kwargs", json.dumps({"enable_thinking": False}, separators=(",", ":")),
            ])
        if config.qwen_llama_device:
            cmd.extend(["--device", str(config.qwen_llama_device)])
        if config.qwen_split_mode:
            cmd.extend(["--split-mode", str(config.qwen_split_mode)])
        if config.qwen_tensor_split:
            cmd.extend(["--tensor-split", str(config.qwen_tensor_split)])
        if str(getattr(config, "qwen_spec_type", "") or ""):
            cmd.extend(["--spec-type", str(config.qwen_spec_type)])
            if int(getattr(config, "qwen_spec_draft_n_max", 0) or 0) > 0:
                cmd.extend(["--spec-draft-n-max", str(int(config.qwen_spec_draft_n_max))])
        cmd.extend(["-ngl", str(max(0, int(config.qwen_gpu_layers)))])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".txt", delete=False) as handle:
            handle.write(execution_prompt)
            prompt_path = handle.name
        schema_path: str | None = None
        if constrained_output:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".schema.json", delete=False) as handle:
                json.dump(output_schema, handle, ensure_ascii=True, separators=(",", ":"))
                schema_path = handle.name
            cmd.extend(["--json-schema-file", schema_path])
        trace_base = _write_qwen_trace_input(role, compacted, execution_prompt, cmd, config)
        started = time.monotonic()
        try:
            proc = subprocess.run([*cmd, "-f", prompt_path], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=config.qwen_timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            salvaged = _extract_json(_coerce_process_text(exc.stdout), role=role)
            _write_qwen_trace_output(trace_base, _coerce_process_text(exc.stdout), _coerce_process_text(exc.stderr), salvaged, "timeout", elapsed_seconds=elapsed)
            if salvaged:
                return salvaged
            if config.qwen_require_runtime:
                raise QwenBackendError("llama_cli_timeout") from exc
            return {}
        finally:
            try:
                Path(prompt_path).unlink()
            except OSError:
                pass
            if schema_path is not None:
                try:
                    Path(schema_path).unlink()
                except OSError:
                    pass
        if proc.returncode != 0:
            elapsed = time.monotonic() - started
            text = (proc.stderr or proc.stdout)[-1600:]
            _write_qwen_trace_output(trace_base, proc.stdout, proc.stderr, {}, f"returncode_{proc.returncode}", elapsed_seconds=elapsed)
            if config.qwen_require_runtime:
                if "context" in text.lower() and ("exceed" in text.lower() or "overflow" in text.lower()):
                    raise QwenBackendError("prompt_context_overflow_prevented")
                raise QwenBackendError(f"llama_cli_failed rc={proc.returncode}: {text}")
            return {}
        extracted = _extract_json(proc.stdout, role=role)
        elapsed = time.monotonic() - started
        _write_qwen_trace_output(trace_base, proc.stdout, proc.stderr, extracted, "json_schema_valid" if extracted else "parse_or_schema_invalid", elapsed_seconds=elapsed)
        return extracted


def _resolve_llama_cli(value: str | os.PathLike[str] | None) -> Path | None:
    if value:
        candidate = Path(str(value)).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        resolved = shutil.which(str(value))
        if resolved:
            return Path(resolved)
    for name in ("llama-cli", "llama-cli.exe", "llama-completion", "llama-completion.exe"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def _resolve_llama_completion(llama_cli: Path) -> Path | None:
    name = llama_cli.name.lower()
    if name in {"llama-completion", "llama-completion.exe"}:
        return llama_cli
    return None


def _qwen_no_thinking_completion_prompt(prompt: str) -> str:
    # This is the Qwen3.5/3.6 chat-template rendering observed with
    # enable_thinking=false. The empty thinking block is input context, so JSON
    # constrained generation begins directly at the assistant's final answer.
    return (
        "<|im_start|>user\n"
        f"{prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


class FakeQwenClient(QwenClient):
    def call(self, role: QwenRole, packet: dict[str, Any], config: V8Config) -> dict[str, Any]:
        if role is QwenRole.COORDINATE:
            constraints = packet.get("execution_constraints", {})
            actions = constraints.get("allowed_action_ids", []) or []
            targets = packet.get("scene", {}).get("coordinate_candidates", []) or []
            if not actions or not targets:
                raise QwenBackendError("coordinate packet has no executable action or candidate")
            target_ids = [str(t["id"]) for t in targets[: min(3, len(targets))] if isinstance(t, dict) and "id" in t]
            return {
                "schema_version": "v8.4.coordinate_plan",
                "decision": "PLAN",
                "mechanism_hypothesis": "The coordinate action should reveal a state or configuration effect at the selected salient candidate.",
                "coordinate_action_id": str(actions[0]),
                "candidate_sequence": [{
                    "coordinate_candidate_id": cid,
                } for cid in target_ids],
                "completion_criterion": "Verifier observes useful coordinate-target evidence.",
                "confidence": 0.2,
            }
        actions = packet.get("execution_constraints", {}).get("allowed_action_ids", []) or []
        if not actions:
            raise QwenBackendError("semantic packet has no executable action")
        action_id = str(actions[0])
        action_run: dict[str, Any] = {"action_id": action_id, "repeat": 1}
        surface = {
            str(item.get("id")): item
            for item in packet.get("action_surface", {}).get("actions", [])
            if isinstance(item, dict) and item.get("id") is not None
        }
        coordinate_ids = sorted(str(value) for value in packet.get("execution_constraints", {}).get("allowed_coordinate_candidate_ids", []) or [])
        if surface.get(action_id, {}).get("kind") == "coordinate" and coordinate_ids:
            action_run["coordinate_candidate_id"] = coordinate_ids[0]
        return {
            "schema_version": "v8.7.semantic_trajectories",
            "decision": "PROPOSE",
            "hypotheses": [{
                "id": "h1",
                "family": "other",
                "objective": {
                    "kind": "other",
                    "source_objects": [],
                    "reference_objects": [],
                    "description": "Test one grounded observed action.",
                },
                "relations": [],
                "basis": "One allowed action is available for a grounded probe.",
                "actions": [action_id],
                "action_runs": [action_run],
                "status": "complete_candidate",
                "uncertainty": "level goal unknown",
                "confidence": 0.1,
            }],
        }


def _prompt(role: QwenRole, packet: dict[str, Any], config: V8Config | None = None) -> str:
    if role is QwenRole.COORDINATE:
        header = "You select grounded coordinate probes for one interactive ARC-AGI-3 game level."
        task = (
            "TASK:\n"
            "Choose a short ordered sequence of distinct existing coordinate_candidate_id values for one execution of the allowed coordinate action on each candidate before a separate goal-planning call.\n"
            "This is active mechanism research, not level solving: a known win condition or target transformation is not required. Select only candidates whose described object, region, or relation makes one click informative and its result easy to attribute.\n"
            "Treat the coordinate action as a target-dependent point interaction with unknown semantics. Its x,y payload selects a target; it is not a direction or a movement distance. Do not describe it as translation unless prior target-local coordinate evidence in MEMORY explicitly observed translation.\n"
            "Use all observed non-coordinate effects in ACTION_MODEL, including movement, color/state, configuration, and action-surface effects, as evidence for mechanism_hypothesis. Absence of a target-local visible effect after one click is useful evidence: continue only with the next distinct candidate, never retry the same candidate in this attempt."
        )
        precondition = (
            "PROBE RULES:\n"
            "- Use only coordinate candidates and action IDs supplied in OBSERVATION_PACKET_JSON.\n"
            "- execution_constraints.allowed_coordinate_candidate_ids is the authoritative whitelist: every ID present in it is allowed.\n"
            "- candidate_sequence may contain several candidates, but all coordinate_candidate_id values must be distinct. Each item means exactly one click on that candidate.\n"
            "- Several candidate IDs can describe the same physical location through object and relation views. Choose at most one candidate for each location_xy; those IDs are alternative descriptions of one click target, not separate probes.\n"
            "- Never click the same candidate twice in one level attempt. A candidate may be reconsidered only after RESET, in the next attempt with the accumulated evidence.\n"
            "- Avoid a candidate listed in MEMORY as having no target-local effect in the same environmental configuration.\n"
            "- Return PLAN with a grounded mechanism_hypothesis and at least one whitelisted candidate, even when confidence is low."
        )
    else:
        allowed_action_ids = {
            str(action_id)
            for action_id in packet.get("execution_constraints", {}).get("allowed_action_ids", []) or []
        }
        coordinate_action_rules = ""
        if "ACTION6" in allowed_action_ids:
            coordinate_action_rules = (
                "- A coordinate action run is one point interaction: it must include coordinate_candidate_id from execution_constraints.allowed_coordinate_candidate_ids and repeat must equal 1. Never encode click count or spatial distance as repeat. Omit coordinate_candidate_id for non-coordinate runs; use separate runs for different coordinate targets.\n"
                "- A coordinate_candidate_id and its physical location_xy may each appear at most once in the entire trajectory and must not repeat a target already listed in this attempt's coordinate evidence. Different IDs at the same location are one target. Any further click there is allowed only after RESET.\n"
                "- No target-local effect is evidence against that target in the current attempt, not an invitation to retry.\n"
            )
        header = (
            "You propose valid complete full action trajectories for a single interactive game level. "
            "Trajectories are not limited to movement: they may include any action from execution_constraints.allowed_action_ids "
            "that changes the available-action surface, environmental configuration, or an object's position, state, "
            "or color as needed to achieve the level goal."
        )
        task = (
            "TASK:\n"
            "This is the only semantic model call for the current level attempt. Infer the level-transition objective and return 1-3 genuinely distinct grounded hypotheses.\n"
            "If memory.level_attempts.previous_failed_attempts is nonempty, this is a fresh RESET attempt: use its execution and verifier failure reasons, preserve confirmed mechanics, and do not repeat an unchanged failed trajectory.\n"
            "Every proposed hypothesis must be a complete_candidate with the full ordered action sequence from the current state; do not return a prefix or request later re-planning.\n"
            "One hypothesis may cover several source/reference pairs and necessary intermediate configurations. Do not split a multi-object solution into separate model calls.\n"
            "For spatial placement by an observed translation action, compute each required delta from current coordinates, divide it by the observed per-action delta, and include the exact repeat count for every required axis. This arithmetic applies only to actions whose ACTION_MODEL effect explicitly contains translation; never convert a spatial delta into repeated coordinate clicks. State the arithmetic briefly in basis.\n"
            "If a supported control-group switch is required, include all useful actions in the current group, then the switch trigger once, then actions from the resulting group. Do not switch early when progress is still required in the current group.\n"
            "Account for simultaneous effects. Do not use cancelling pairs unless an explicitly stated intermediate interaction requires them.\n"
            "action_runs is the complete ordered trajectory encoded without manual token repetition. Its first action_id must be executable now; each repeat is the TOTAL number of consecutive executions in that run, including its first execution. Never subtract one for a separate first-action slot.\n"
            "Return final conclusions only, keep basis to at most 3 short sentences and 360 characters, and use only supplied IDs and observed effects.\n"
            "Return one hypothesis when only one strong interpretation is grounded. When evidence is uncertain, return the best grounded complete_candidate with lower confidence so verifier results can update memory."
        )
        precondition = (
            "READING AND CONTROL RULES:\n"
            "- Read SCENE.priority_facts_not_goals and control_state_transition_candidates first, then SCENE.component_graph, SCENE.objects/control_groups, ACTION_MODEL/ACTION_SURFACE, and finally CURRENT_FRAME for visual context.\n"
            "- Priority facts are strong observed correspondences, but they are explicitly not preselected goals.\n"
            "- SCENE.component_graph partitions every frame cell into same-color 4-connected components. A component is geometric evidence, not automatically one gameplay object; multicolor objects may span several components. Use only its object_refs in response object fields.\n"
            "- SCENE.object_segmentation defines canonical planning objects. Nested same-color detail suppressed there remains visible in component_graph and CURRENT_FRAME, but is not a separate gameplay object.\n"
            "- In component_graph, shape_hash is exact translation-normalized occupancy; parent is topological enclosure. Detailed tracked-object masks are in SCENE.objects, and CURRENT_FRAME is authoritative for unlinked component detail.\n"
            "- If a correspondence fact provides movable_object_id and reference_object_id, preserve those roles when choosing that fact: the movable object belongs in source_objects and the static object belongs in reference_objects.\n"
            "- source_to_reference_delta_xy is the remaining displacement from the movable object's current centroid to the reference centroid. observed_source_translation_by_action gives one-step deltas in the same coordinates.\n"
            "- Exact geometry means the occupied mask and orientation are identical. Different geometry_group_id values are not identical geometry.\n"
            "- Use only effects explicitly listed in ACTION_MODEL.\n"
            "- ACTION_SURFACE availability is observed legality in a state, not intrinsic action meaning. A direction can be absent because the currently affected geometry is blocked by a boundary, and can appear after movement without a controller-mode change.\n"
            "- SCENE.control_groups identifies objects translated simultaneously in each chronological action context; exact vectors remain in ACTION_MODEL.\n"
            "- Reciprocal isolated visual transitions synchronized with an action-surface change are state-marker evidence. They may support a control-group switch but are not a goal.\n"
            "- A SUPPORTED_CONTROL_GROUP_SWITCH is mechanics, not a goal. Simulate the active group across the whole proposed sequence: before the trigger use current-group vectors; after one trigger use the opposite-group vectors.\n"
            "- If an action lists only translation, do not claim rotation, recoloring, reshaping, selection, or a geometry-group change.\n"
            "- A shape_area_or_visibility_change is not translation even if its centroid changed because one edge grew or shrank.\n"
            "- The first trajectory action must be available_now. ACTION_SURFACE.observed_transitions is chronological and directed from available_before to available_after.\n"
            "- A later action from another observed surface is allowed only when the preceding sequence establishes that surface through a supplied transition.\n"
            f"{coordinate_action_rules}"
            "- Action effects are state-scoped. The executor re-observes and verifies after every action, but the model must still provide the complete expected trajectory in this response.\n"
            "\n"
            "UNCERTAINTY:\n"
            "- The agent has already performed action probing.\n"
            "- The level-transition goal is unknown.\n"
            "- Use uncertainty to lower confidence, but do not replace a complete trajectory with a prefix or an empty response."
        )
    schema = _allowed_output_schema(role, packet)
    parts = [
        header,
        "",
        "DATA AUTHORITY:",
        "- CURRENT_FRAME is the latest exact official observation, encoded as categorical palette rows 0-F.",
        "- SCENE is a deterministic compact parse and may be incomplete; its coordinates and exact-geometry groups are factual.",
        "- SCENE.component_graph is a deterministic all-color topology view. Its background candidates and gameplay roles are explicitly hypotheses, not facts.",
        "- ACTION_MODEL contains verifier-observed action effects, including translations, local state changes, and action-surface transitions.",
        "- MEMORY contains chronological evidence and same-game prior-level summaries; it is cleared between games.",
        "- For object shape, shape_geometry.occupied_mask_rows and SCENE.exact_geometry_groups are authoritative for exact geometry and orientation.",
        "- Same bbox size, area, or fill_ratio does not mean same shape; exact shape correspondence requires the same occupied mask/orientation or an explicit same_shape relation.",
        "- You may propose action IDs and candidate trajectories inside the JSON response.",
        "- You do not execute or authorize actions.",
        "- The agent validates the candidate before execution and re-observes after every executed step.",
        "",
        precondition,
        "",
        task,
        "",
        "OBSERVATION_PACKET_JSON=",
        json.dumps(packet, ensure_ascii=False, sort_keys=False, separators=(",", ":")),
        "",
        "ALLOWED_OUTPUT_JSON_SCHEMA=",
        json.dumps(schema, ensure_ascii=False, sort_keys=False, separators=(",", ":")),
        "",
        "Return exactly one valid JSON object.",
        "Do not return markdown.",
        "Do not return prose outside JSON.",
        "RETURN_JSON_ONLY",
    ]
    return "\n".join(parts)


def _allowed_output_schema(role: QwenRole, packet: dict[str, Any] | None = None) -> dict[str, Any]:
    constraints = (packet or {}).get("execution_constraints", {}) if isinstance(packet, dict) else {}
    action_ids = [str(v) for v in constraints.get("allowed_action_ids", [])]
    surface_actions = (packet or {}).get("action_surface", {}).get("actions", []) if isinstance(packet, dict) else []
    current_action_ids = [
        str(item.get("id"))
        for item in surface_actions
        if isinstance(item, dict) and item.get("id") is not None and item.get("available_now") and item.get("planning_allowed")
    ]
    if not current_action_ids:
        current_action_ids = list(action_ids)
    object_ids = [str(v) for v in constraints.get("allowed_object_ids", [])]
    relation_ids = [str(v) for v in constraints.get("allowed_relation_ids", [])]
    coordinate_ids = [str(v) for v in constraints.get("allowed_coordinate_candidate_ids", [])]
    max_steps = int(constraints.get("max_plan_steps") or 50)
    if role is QwenRole.COORDINATE:
        coordinate_step = {
            "type": "object",
            "additionalProperties": False,
            "required": ["coordinate_candidate_id"],
            "properties": {
                "coordinate_candidate_id": _enum_schema(coordinate_ids),
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "decision", "mechanism_hypothesis", "coordinate_action_id", "candidate_sequence", "completion_criterion", "confidence"],
            "properties": {
                "schema_version": {"const": "v8.4.coordinate_plan"},
                "decision": {"const": "PLAN"},
                "mechanism_hypothesis": {"type": "string", "minLength": 1, "maxLength": 240},
                "coordinate_action_id": _enum_schema(action_ids),
                "candidate_sequence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": min(max_steps, max(1, len(coordinate_ids))),
                    "uniqueItems": True,
                    "items": coordinate_step,
                },
                "completion_criterion": {"type": "string", "maxLength": 240},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
    action_run_properties: dict[str, Any] = {
        "action_id": _enum_schema(action_ids),
        "repeat": {
            "type": "integer",
            "minimum": 1,
            "maximum": max_steps,
            "description": "Total consecutive executions. Must be exactly 1 when action_id is a coordinate action.",
        },
    }
    if coordinate_ids:
        action_run_properties["coordinate_candidate_id"] = {
            **_enum_schema(coordinate_ids),
            "description": "Required only for a coordinate action. Each coordinate run is one click and therefore has repeat=1.",
        }
    objective_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "source_objects", "reference_objects", "description"],
        "properties": {
            "kind": {"enum": ["match_or_overlap", "relative_arrangement", "containment", "connection", "pattern_or_state", "select_or_activate", "surface_change", "other"]},
            "source_objects": {**_array_enum_schema(object_ids), "maxItems": min(6, max(0, len(object_ids)))},
            "reference_objects": {**_array_enum_schema(object_ids), "maxItems": min(6, max(0, len(object_ids)))},
            "description": {"type": "string", "minLength": 1, "maxLength": 240},
        },
    }
    hypothesis_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "family",
            "objective",
            "relations",
            "basis",
            "action_runs",
            "status",
            "uncertainty",
            "confidence",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1, "maxLength": 48},
            "family": {
                "enum": [
                    "object_correspondence",
                    "spatial_configuration",
                    "pattern_transformation",
                    "interaction_sequence",
                    "action_surface_change",
                    "other",
                ]
            },
            "objective": objective_schema,
            "relations": {**_array_enum_schema(relation_ids), "maxItems": min(6, max(0, len(relation_ids)))},
            "basis": {"type": "string", "minLength": 1, "maxLength": 360},
            "action_runs": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_steps,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action_id", "repeat"],
                    "properties": action_run_properties,
                },
            },
            "status": {
                "const": "complete_candidate"
            },
            "uncertainty": {"type": "string", "maxLength": 240},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "decision", "hypotheses"],
        "properties": {
            "schema_version": {"const": "v8.7.semantic_trajectories"},
            "decision": {"const": "PROPOSE"},
            "hypotheses": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": hypothesis_schema,
            },
        },
    }


def _enum_schema(values: list[str]) -> dict[str, Any]:
    if values:
        return {"type": "string", "enum": values}
    return {"type": "string", "not": {}}


def _nullable_enum(values: list[str]) -> dict[str, Any]:
    if values:
        return {"anyOf": [{"type": "string", "enum": values}, {"type": "null"}]}
    return {"type": "null"}


def _array_enum_schema(values: list[str]) -> dict[str, Any]:
    if values:
        return {
            "type": "array",
            "items": {"type": "string", "enum": values},
            "uniqueItems": True,
        }
    return {"type": "array", "maxItems": 0}


def _expected_observation_schema(target_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "target_ids", "description"],
        "properties": {
            "type": {"enum": ["target_change", "object_move", "relation_improvement", "action_surface_change", "score_or_terminal"]},
            "target_ids": _array_enum_schema(target_ids),
            "description": {"type": "string", "maxLength": 240},
        },
    }


def estimate_tokens(text: str) -> int:
    # Conservative model-neutral estimate for compact JSON and ASCII-heavy grids.
    return max(1, (len(text) + 2) // 3)


def _fit_packet_to_budget(role: QwenRole, packet: dict[str, Any], config: V8Config) -> dict[str, Any]:
    strategy = config.prompt_compaction_strategy
    candidate = compact_packet(packet, strategy)
    threshold = int(config.qwen_max_input_tokens * config.prompt_compression_trigger_ratio)
    if estimate_tokens(_prompt(role, candidate, config)) <= threshold:
        return candidate
    candidate = compact_packet(packet, "aggressive")
    if estimate_tokens(_prompt(role, candidate, config)) <= threshold:
        return candidate
    return compact_packet(packet, "minimal")


def compact_packet(packet: dict[str, Any], strategy: str) -> dict[str, Any]:
    out = deepcopy(packet)
    if strategy == "normal":
        return out
    if out.get("schema_version") == "v8.7.semantic_observation":
        return _compact_v87_packet(out, strategy)
    if out.get("schema_version") == "v8.4.observation_packet":
        return _compact_v84_packet(out, strategy)
    return out


def _compact_v87_packet(out: dict[str, Any], strategy: str) -> dict[str, Any]:
    scene = out.setdefault("scene", {})
    object_limit = 18 if strategy == "aggressive" else 12
    relation_limit = 32 if strategy == "aggressive" else 20
    candidate_limit = 48 if strategy == "aggressive" else 32
    scene["objects"] = (scene.get("objects") or [])[:object_limit]
    for obj in scene["objects"]:
        if isinstance(obj, dict):
            obj.pop("local_hex_rows", None)
            geometry = obj.get("shape_geometry")
            if isinstance(geometry, dict) and strategy == "minimal":
                geometry.pop("occupied_mask_rows", None)
    scene["relations"] = (scene.get("relations") or [])[:relation_limit]
    scene["coordinate_candidates"] = (scene.get("coordinate_candidates") or [])[:candidate_limit]
    scene["priority_facts_not_goals"] = (scene.get("priority_facts_not_goals") or [])[:12]
    scene["control_groups"] = (scene.get("control_groups") or [])[:8]
    scene["control_state_transition_candidates"] = (scene.get("control_state_transition_candidates") or [])[:8]
    scene["repeated_object_groups"] = (scene.get("repeated_object_groups") or [])[:12]
    component_graph = scene.get("component_graph")
    if isinstance(component_graph, dict):
        _compact_component_graph(component_graph, 64 if strategy == "aggressive" else 40, strategy)
    memory = out.get("memory")
    if isinstance(memory, dict):
        evidence = memory.get("recent_evidence")
        if isinstance(evidence, dict):
            keep = 8 if strategy == "aggressive" else 4
            evidence["successful_steps"] = (evidence.get("successful_steps") or [])[-keep:]
            evidence["failed_or_irrelevant_steps"] = (evidence.get("failed_or_irrelevant_steps") or [])[-keep:]
            evidence["open_questions"] = (evidence.get("open_questions") or [])[-keep:]
        attempts = memory.get("level_attempts")
        if isinstance(attempts, dict):
            keep = 4 if strategy == "aggressive" else 2
            attempts["previous_failed_attempts"] = (attempts.get("previous_failed_attempts") or [])[-keep:]
    _repair_v87_references(out)
    return out


def _repair_v87_references(packet: dict[str, Any]) -> None:
    scene = packet.setdefault("scene", {})
    object_ids = {item.get("id") for item in scene.get("objects", []) if isinstance(item, dict)}
    component_graph = scene.get("component_graph")
    if isinstance(component_graph, dict):
        for component in component_graph.get("components", []) or []:
            if isinstance(component, dict):
                component["object_refs"] = [
                    object_id for object_id in component.get("object_refs", [])
                    if object_id in object_ids
                ]
    relations = [
        item for item in scene.get("relations", [])
        if isinstance(item, dict) and set(item.get("object_ids") or []).issubset(object_ids)
    ]
    scene["relations"] = relations
    relation_ids = {item.get("id") for item in relations}
    groups = []
    for group in scene.get("exact_geometry_groups", []) or []:
        if not isinstance(group, dict):
            continue
        members = [object_id for object_id in group.get("object_ids", []) if object_id in object_ids]
        if members:
            copy = dict(group)
            copy["object_ids"] = members
            groups.append(copy)
    scene["exact_geometry_groups"] = groups
    scene["priority_facts_not_goals"] = [
        item for item in scene.get("priority_facts_not_goals", [])
        if isinstance(item, dict) and set(item.get("object_ids") or []).issubset(object_ids)
    ]
    control_groups = []
    for item in scene.get("control_groups", []) or []:
        if not isinstance(item, dict):
            continue
        members = [object_id for object_id in item.get("object_ids", []) if object_id in object_ids]
        if len(members) >= 2:
            copy = dict(item)
            copy["object_ids"] = members
            control_groups.append(copy)
    scene["control_groups"] = control_groups
    control_group_ids = {str(item.get("id")) for item in control_groups}
    state_candidates = []
    for item in scene.get("control_state_transition_candidates", []) or []:
        if not isinstance(item, dict):
            continue
        referenced = set()
        for key in (
            "shared_translated_object_ids",
            "before_only_translated_object_ids",
            "after_only_translated_object_ids",
        ):
            referenced.update(str(value) for value in item.get(key) or [])
        for marker in item.get("marker_evidence") or []:
            if isinstance(marker, dict):
                referenced.update(str(value) for value in marker.get("object_ids") or [])
        if not referenced.issubset(object_ids):
            continue
        if str(item.get("before_control_group_id")) not in control_group_ids or str(item.get("after_control_group_id")) not in control_group_ids:
            continue
        state_candidates.append(item)
    scene["control_state_transition_candidates"] = state_candidates
    actions = ((packet.get("action_model") or {}).get("actions") or {})
    if isinstance(actions, dict):
        for action in actions.values():
            if isinstance(action, dict):
                action["observed_control_group_ids"] = [
                    str(value) for value in action.get("observed_control_group_ids") or []
                    if str(value) in control_group_ids
                ]
    scene["repeated_object_groups"] = [
        item for item in scene.get("repeated_object_groups", [])
        if isinstance(item, dict) and set(item.get("object_ids") or []).issubset(object_ids)
    ]
    candidates = [
        item for item in scene.get("coordinate_candidates", [])
        if isinstance(item, dict)
        and (item.get("object_id") is None or item.get("object_id") in object_ids)
        and (item.get("relation_id") is None or item.get("relation_id") in relation_ids)
    ]
    scene["coordinate_candidates"] = candidates
    action_ids = {item.get("id") for item in packet.get("action_surface", {}).get("actions", []) if isinstance(item, dict)}
    constraints = packet.setdefault("execution_constraints", {})
    constraints["allowed_object_ids"] = sorted(str(value) for value in object_ids if value is not None)
    constraints["allowed_relation_ids"] = sorted(str(value) for value in relation_ids if value is not None)
    constraints["allowed_coordinate_candidate_ids"] = sorted(str(item.get("id")) for item in candidates if item.get("id") is not None)
    constraints["allowed_action_ids"] = [action_id for action_id in constraints.get("allowed_action_ids", []) if action_id in action_ids]


def _compact_component_graph(graph: dict[str, Any], limit: int, strategy: str) -> None:
    components = [item for item in graph.get("components", []) if isinstance(item, dict)]
    if len(components) <= limit:
        if strategy == "minimal":
            for component in components:
                if not component.get("object_refs"):
                    component.pop("normalized_shape_runs", None)
                    component.pop("outer_boundary_corners_xy", None)
        return

    ranked = sorted(
        components,
        key=lambda item: (
            -bool(item.get("object_refs")),
            -len(item.get("object_refs") or []),
            -int(item.get("area") or 0),
            str(item.get("id") or ""),
        ),
    )[: max(1, int(limit))]
    retained_ids = {str(item.get("id")) for item in ranked}
    retained = [item for item in components if str(item.get("id")) in retained_ids]
    for component in retained:
        parent = component.get("parent")
        if parent is not None and str(parent) not in retained_ids:
            component["parent"] = None
            component["parent_omitted"] = True
        old_children = list(component.get("children") or [])
        component["children"] = [value for value in old_children if str(value) in retained_ids]
        omitted = len(old_children) - len(component["children"])
        if omitted:
            component["omitted_child_count"] = int(component.get("omitted_child_count") or 0) + omitted
        if strategy == "minimal" and not component.get("object_refs"):
            component.pop("normalized_shape_runs", None)
            component.pop("outer_boundary_corners_xy", None)
    old_edges = [edge for edge in graph.get("adjacency", []) if isinstance(edge, dict)]
    graph["adjacency"] = [
        edge for edge in old_edges
        if str(edge.get("a")) in retained_ids and str(edge.get("b")) in retained_ids
    ]
    graph["same_shape_groups"] = [
        {**group, "component_ids": [value for value in group.get("component_ids", []) if str(value) in retained_ids]}
        for group in graph.get("same_shape_groups", [])
        if isinstance(group, dict)
        and len([value for value in group.get("component_ids", []) if str(value) in retained_ids]) >= 2
    ]
    graph["background_candidates_not_facts"] = [
        item for item in graph.get("background_candidates_not_facts", [])
        if isinstance(item, dict) and str(item.get("component_id")) in retained_ids
    ]
    graph["components"] = retained
    graph["component_count_included"] = len(retained)
    total = int(graph.get("component_count_total") or len(components))
    graph["omitted_component_count"] = max(0, total - len(retained))
    graph["omitted_incident_adjacency_count"] = int(graph.get("omitted_incident_adjacency_count") or 0) + len(old_edges) - len(graph["adjacency"])
    graph["complete"] = len(retained) == total


def _compact_v84_packet(out: dict[str, Any], strategy: str) -> dict[str, Any]:
    if strategy == "aggressive":
        derived = out.setdefault("derived_features", {})
        observed = out.setdefault("observed_facts", {})
        grid = observed.get("grid")
        if isinstance(grid, dict) and isinstance(grid.get("hex_rows"), list):
            grid["hex_rows"] = grid["hex_rows"][:]
        derived["objects"] = (derived.get("objects") or [])[:24]
        for obj in derived.get("objects", []):
            if isinstance(obj, dict):
                obj["local_hex_rows"] = []
        derived.pop("scene_entities", None)
        derived["relations"] = (derived.get("relations") or [])[:48]
        derived["coordinate_candidates"] = (derived.get("coordinate_candidates") or [])[:48]
        evidence = out.get("recent_evidence")
        if isinstance(evidence, dict):
            evidence["successful_steps"] = (evidence.get("successful_steps") or [])[-8:]
            evidence["failed_or_irrelevant_steps"] = (evidence.get("failed_or_irrelevant_steps") or [])[-8:]
            evidence["open_questions"] = (evidence.get("open_questions") or [])[-6:]
        _repair_v84_references(out)
        return out
    minimal = {
        "schema_version": out.get("schema_version"),
        "state": deepcopy(out.get("state")),
        "observed_facts": deepcopy(out.get("observed_facts")),
        "control_model": deepcopy(out.get("control_model")),
        "derived_features": deepcopy(out.get("derived_features")),
        "recent_transition": deepcopy(out.get("recent_transition")),
        "recent_evidence": deepcopy(out.get("recent_evidence")),
        "execution_constraints": deepcopy(out.get("execution_constraints")),
    }
    grid = ((minimal.get("observed_facts") or {}).get("grid") or {})
    if isinstance(grid, dict) and "hex_rows" in grid:
        grid["hex_rows"] = []
    derived = minimal.get("derived_features") or {}
    derived["objects"] = (derived.get("objects") or [])[:12]
    for obj in derived.get("objects", []):
        if isinstance(obj, dict):
            obj["local_hex_rows"] = []
    derived.pop("scene_entities", None)
    derived["relations"] = (derived.get("relations") or [])[:24]
    derived["coordinate_candidates"] = (derived.get("coordinate_candidates") or [])[:32]
    evidence = minimal.get("recent_evidence")
    if isinstance(evidence, dict):
        evidence["successful_steps"] = (evidence.get("successful_steps") or [])[-4:]
        evidence["failed_or_irrelevant_steps"] = (evidence.get("failed_or_irrelevant_steps") or [])[-4:]
        evidence["open_questions"] = (evidence.get("open_questions") or [])[-4:]
    _repair_v84_references(minimal)
    return minimal


def _repair_v84_references(packet: dict[str, Any]) -> None:
    derived = packet.setdefault("derived_features", {})
    object_ids = {item.get("id") for item in derived.get("objects", []) if isinstance(item, dict)}
    relations = []
    for relation in derived.get("relations", []) or []:
        if not isinstance(relation, dict):
            continue
        relation_object_ids = relation.get("object_ids")
        if isinstance(relation_object_ids, list):
            if set(relation_object_ids).issubset(object_ids):
                relations.append(relation)
        elif relation.get("source_object_id") in object_ids and relation.get("target_object_id") in object_ids:
            relations.append(relation)
    derived["relations"] = relations
    relation_ids = {item.get("id") for item in relations}
    for obj in derived.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        geometry = obj.get("shape_geometry")
        if not isinstance(geometry, dict):
            continue
        same_ids = geometry.get("same_exact_geometry_object_ids")
        if isinstance(same_ids, list):
            geometry["same_exact_geometry_object_ids"] = [object_id for object_id in same_ids if object_id in object_ids]
    geometry_groups = []
    for group in derived.get("object_geometry_groups", []) or []:
        if not isinstance(group, dict):
            continue
        group_object_ids = [object_id for object_id in group.get("object_ids", []) if object_id in object_ids]
        if not group_object_ids:
            continue
        copy = dict(group)
        copy["object_ids"] = group_object_ids
        geometry_groups.append(copy)
    derived["object_geometry_groups"] = geometry_groups
    candidates = []
    for candidate in derived.get("coordinate_candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        obj_ok = candidate.get("object_id") is None or candidate.get("object_id") in object_ids
        rel_ok = candidate.get("relation_id") is None or candidate.get("relation_id") in relation_ids
        if obj_ok and rel_ok:
            candidates.append(candidate)
    derived["coordinate_candidates"] = candidates
    action_ids = {item.get("id") for item in (packet.get("observed_facts", {}).get("actions") or []) if isinstance(item, dict)}
    constraints = packet.setdefault("execution_constraints", {})
    constraints["allowed_object_ids"] = sorted(str(v) for v in object_ids if v is not None)
    constraints["allowed_relation_ids"] = sorted(str(v) for v in relation_ids if v is not None)
    constraints["allowed_coordinate_candidate_ids"] = sorted(str(item.get("id")) for item in candidates if item.get("id") is not None)
    constraints["allowed_action_ids"] = [action_id for action_id in constraints.get("allowed_action_ids", []) if action_id in action_ids]


def _extract_json(text: str, *, role: QwenRole | None = None) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}
    return _extract_json_from_region(raw, role=role)


def _extract_json_from_region(raw: str, *, role: QwenRole | None = None) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        normalized = _normalize_qwen_output(value, role=role)
        if normalized:
            return normalized
        return {}
    except json.JSONDecodeError:
        pass
    repaired = _repair_json_trailing_commas(raw)
    if repaired != raw:
        try:
            value = json.loads(repaired)
            normalized = _normalize_qwen_output(value, role=role)
            if normalized:
                return normalized
        except json.JSONDecodeError:
            pass
    parsed = list(_json_object_values(raw))
    for value in reversed(parsed):
        normalized = _normalize_qwen_output(value, role=role)
        if normalized:
            return normalized
    repaired_values = list(_json_object_values_from_repaired_regions(raw))
    for value in reversed(repaired_values):
        normalized = _normalize_qwen_output(value, role=role)
        if normalized:
            return normalized
    salvaged = _salvage_v87_hypothesis_objects(raw, role=role)
    if salvaged:
        return salvaged
    return {}


def _salvage_v87_hypothesis_objects(raw: str, *, role: QwenRole | None) -> dict[str, Any]:
    if role is QwenRole.COORDINATE:
        return {}
    markers = list(re.finditer(r'"schema_version"\s*:\s*"v8\.7\.semantic_trajectories"', raw))
    if not markers:
        return {}
    region = raw[markers[-1].start():]
    hypotheses = []
    seen_ids: set[str] = set()
    for value in _json_object_values(region):
        if not isinstance(value, dict):
            continue
        wrapped = {
            "schema_version": "v8.7.semantic_trajectories",
            "decision": "PROPOSE",
            "hypotheses": [value],
        }
        wrapped = _normalize_split_action_trajectories(wrapped)
        if not _validate_v87_semantic_output(wrapped):
            continue
        normalized_value = wrapped["hypotheses"][0]
        hypothesis_id = str(normalized_value.get("id"))
        if hypothesis_id in seen_ids:
            continue
        seen_ids.add(hypothesis_id)
        hypotheses.append(normalized_value)
        if len(hypotheses) >= 3:
            break
    if not hypotheses:
        return {}
    return {
        "schema_version": "v8.7.semantic_trajectories",
        "decision": "PROPOSE",
        "hypotheses": hypotheses,
    }


def _normalize_qwen_output(value: Any, *, role: QwenRole | None = None) -> dict[str, Any]:
    value = _normalize_split_action_trajectories(value)
    if not isinstance(value, dict) or not _looks_like_qwen_output(value, role=role):
        return {}
    if _valid_qwen_output(value, role=role):
        return value
    if value.get("schema_version") != "v8.7.semantic_trajectories" or str(value.get("decision") or "").upper() != "PROPOSE":
        return {}
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list):
        return {}
    valid_hypotheses = [
        item for item in hypotheses[:3]
        if _validate_v87_semantic_output({
            "schema_version": "v8.7.semantic_trajectories",
            "decision": "PROPOSE",
            "hypotheses": [item],
        })
    ]
    if not valid_hypotheses:
        return {}
    return {
        "schema_version": "v8.7.semantic_trajectories",
        "decision": "PROPOSE",
        "hypotheses": valid_hypotheses,
    }


def _normalize_split_action_trajectories(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("schema_version") != "v8.7.semantic_trajectories":
        return value
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list):
        return value
    changed = False
    normalized_hypotheses = []
    for item in hypotheses:
        if not isinstance(item, dict):
            normalized_hypotheses.append(item)
            continue
        action_runs = item.get("action_runs")
        if isinstance(action_runs, list):
            actions: list[str] = []
            valid_runs = True
            for run in action_runs:
                if (
                    not isinstance(run, dict)
                    or not {"action_id", "repeat"}.issubset(run)
                    or set(run) - {"action_id", "repeat", "coordinate_candidate_id"}
                ):
                    valid_runs = False
                    break
                action_id = run.get("action_id")
                repeat = run.get("repeat")
                coordinate_candidate_id = run.get("coordinate_candidate_id")
                if not isinstance(action_id, str) or not isinstance(repeat, int) or isinstance(repeat, bool) or repeat <= 0:
                    valid_runs = False
                    break
                if coordinate_candidate_id is not None and not isinstance(coordinate_candidate_id, str):
                    valid_runs = False
                    break
                actions.extend([action_id] * repeat)
                if len(actions) > 50:
                    valid_runs = False
                    break
            if valid_runs and actions:
                normalized = dict(item)
                normalized["actions"] = actions
                normalized_hypotheses.append(normalized)
                changed = True
                continue
            normalized_hypotheses.append(item)
            continue
        if "actions" in item:
            normalized_hypotheses.append(item)
            continue
        if "first_action" not in item or "remaining_actions" not in item:
            normalized_hypotheses.append(item)
            continue
        first_action = item.get("first_action")
        remaining_actions = item.get("remaining_actions")
        if not isinstance(first_action, str) or not isinstance(remaining_actions, list):
            normalized_hypotheses.append(item)
            continue
        normalized = dict(item)
        normalized["actions"] = [first_action, *remaining_actions]
        normalized.pop("first_action", None)
        normalized.pop("remaining_actions", None)
        normalized_hypotheses.append(normalized)
        changed = True
    if not changed:
        return value
    normalized_value = dict(value)
    normalized_value["hypotheses"] = normalized_hypotheses
    return normalized_value


def _looks_like_qwen_output(value: dict[str, Any], *, role: QwenRole | None = None) -> bool:
    schema = str(value.get("schema_version") or "")
    semantic = {"v8.2.semantic_output", "v8.3.semantic_output", "v8.4.semantic_plan", "v8.5.semantic_trajectory_hypotheses", "v8.6.semantic_trajectory_hypotheses", "v8.7.semantic_trajectories"}
    coordinate = {"v8.2.coordinate_output", "v8.3.coordinate_output", "v8.4.coordinate_plan"}
    if role is QwenRole.COORDINATE:
        return schema in coordinate
    if role in {QwenRole.PRIMARY, QwenRole.RESERVE}:
        return schema in semantic
    if schema in semantic | coordinate:
        return True
    return False


def _valid_qwen_output(value: dict[str, Any], *, role: QwenRole | None = None) -> bool:
    if not _looks_like_qwen_output(value, role=role):
        return False
    if value.get("schema_version") == "v8.6.semantic_trajectory_hypotheses":
        return _validate_v86_semantic_output(value)
    if value.get("schema_version") == "v8.7.semantic_trajectories":
        return _validate_v87_semantic_output(value)
    return True


def _validate_v86_semantic_output(value: dict[str, Any]) -> bool:
    decision = str(value.get("decision") or "").upper()
    if decision == "ABSTAIN":
        return set(value) <= {"schema_version", "decision", "reason"} and bool(str(value.get("reason") or "")) and len(str(value.get("reason") or "")) <= 360
    if decision != "PROPOSE":
        return False
    if set(value) - {"schema_version", "decision", "hypotheses"}:
        return False
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list) or not (1 <= len(hypotheses) <= 3):
        return False
    allowed_families = {
        "same_shape_alignment",
        "relative_position_alignment",
        "symmetry_or_reflection",
        "containment",
        "connection",
        "pattern_completion",
        "recolor_or_state_match",
        "object_selection",
        "multi_object_configuration",
        "action_surface_change",
        "other",
    }
    required = {"id", "family", "objects", "relations", "basis", "actions", "status", "uncertainty", "confidence"}
    optional = set()
    for item in hypotheses:
        if not isinstance(item, dict) or set(item) - (required | optional) or not required.issubset(item):
            return False
        if not isinstance(item.get("id"), str) or not (1 <= len(item["id"]) <= 64):
            return False
        if item.get("family") not in allowed_families:
            return False
        if not _valid_string_array(item.get("objects"), max_items=6) or not _valid_string_array(item.get("relations"), max_items=6):
            return False
        if not isinstance(item.get("basis"), str) or not (1 <= len(item["basis"]) <= 720):
            return False
        if not _valid_string_array(item.get("actions"), min_items=1, max_items=50, unique=False):
            return False
        if item.get("status") not in {"complete_candidate", "prefix_until_reobservation", "requires_surface_change"}:
            return False
        if not isinstance(item.get("uncertainty"), str) or len(item["uncertainty"]) > 480:
            return False
        try:
            confidence = float(item.get("confidence"))
        except Exception:
            return False
        if not 0.0 <= confidence <= 1.0:
            return False
    return True


def _validate_v87_semantic_output(value: dict[str, Any]) -> bool:
    decision = str(value.get("decision") or "").upper()
    if decision != "PROPOSE" or set(value) - {"schema_version", "decision", "hypotheses"}:
        return False
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list) or not (1 <= len(hypotheses) <= 3):
        return False
    families = {"object_correspondence", "spatial_configuration", "pattern_transformation", "interaction_sequence", "action_surface_change", "other"}
    objective_kinds = {"match_or_overlap", "relative_arrangement", "containment", "connection", "pattern_or_state", "select_or_activate", "surface_change", "other"}
    statuses = {"complete_candidate"}
    required = {"id", "family", "objective", "relations", "basis", "actions", "status", "uncertainty", "confidence"}
    objective_required = {"kind", "source_objects", "reference_objects", "description"}
    for item in hypotheses:
        if not isinstance(item, dict) or frozenset(item) not in {frozenset(required), frozenset(required | {"action_runs"})}:
            return False
        if not isinstance(item.get("id"), str) or not (1 <= len(item["id"]) <= 64):
            return False
        if item.get("family") not in families or item.get("status") not in statuses:
            return False
        objective = item.get("objective")
        if not isinstance(objective, dict) or set(objective) != objective_required:
            return False
        if objective.get("kind") not in objective_kinds:
            return False
        if not _valid_string_array(objective.get("source_objects"), max_items=6):
            return False
        if not _valid_string_array(objective.get("reference_objects"), max_items=6):
            return False
        if not isinstance(objective.get("description"), str) or not (1 <= len(objective["description"]) <= 480):
            return False
        if not _valid_string_array(item.get("relations"), max_items=6):
            return False
        if not isinstance(item.get("basis"), str) or not (1 <= len(item["basis"]) <= 720):
            return False
        if not _valid_string_array(item.get("actions"), min_items=1, max_items=50, unique=False):
            return False
        action_runs = item.get("action_runs")
        if action_runs is not None:
            if not isinstance(action_runs, list) or not action_runs:
                return False
            expanded_actions: list[str] = []
            for run in action_runs:
                if (
                    not isinstance(run, dict)
                    or not {"action_id", "repeat"}.issubset(run)
                    or set(run) - {"action_id", "repeat", "coordinate_candidate_id"}
                ):
                    return False
                action_id = run.get("action_id")
                repeat = run.get("repeat")
                candidate_id = run.get("coordinate_candidate_id")
                if not isinstance(action_id, str) or not action_id:
                    return False
                if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat <= 0:
                    return False
                if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id):
                    return False
                expanded_actions.extend([action_id] * repeat)
                if len(expanded_actions) > 50:
                    return False
            if expanded_actions != item.get("actions"):
                return False
        if not isinstance(item.get("uncertainty"), str) or len(item["uncertainty"]) > 480:
            return False
        try:
            confidence = float(item.get("confidence"))
        except Exception:
            return False
        if not 0.0 <= confidence <= 1.0:
            return False
    return True


def _valid_string_array(value: Any, *, min_items: int = 0, max_items: int = 10**9, unique: bool = True) -> bool:
    if not isinstance(value, list) or not (min_items <= len(value) <= max_items):
        return False
    if unique and len(set(value)) != len(value):
        return False
    return all(isinstance(item, str) and item for item in value)


def _coerce_process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _json_object_values(text: str):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _json_object_values_from_repaired_regions(text: str):
    decoder = json.JSONDecoder()
    for start, end in _candidate_json_regions(text):
        repaired = _repair_json_trailing_commas(text[start:end])
        try:
            value, _end = decoder.raw_decode(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _candidate_json_regions(text: str):
    starts = [idx for idx, char in enumerate(text) if char == "{"]
    ends = [idx + 1 for idx, char in enumerate(text) if char == "}"]
    for start in starts:
        for end in reversed(ends):
            if end <= start:
                break
            yield start, end
            break


def _repair_json_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _packet_has_semantic_content(packet: dict[str, Any]) -> bool:
    if packet.get("schema_version") == "v8.7.semantic_observation":
        scene = packet.get("scene", {})
        return bool(scene.get("objects") or scene.get("relations") or scene.get("coordinate_candidates"))
    if packet.get("schema_version") == "v8.4.observation_packet":
        derived = packet.get("derived_features", {})
        return bool(derived.get("objects") or derived.get("relations") or derived.get("coordinate_candidates"))
    return bool(packet.get("objects") or packet.get("relations") or packet.get("coordinate_action_surface", {}).get("coordinate_targets"))


def _qwen_trace_dir(config: V8Config) -> Path | None:
    raw = config.qwen_trace_dir or os.environ.get("ARC_QWEN_TRACE_DIR")
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def _write_qwen_trace_input(role: QwenRole, packet: dict[str, Any], prompt: str, cmd: list[str], config: V8Config) -> Path | None:
    trace_dir = _qwen_trace_dir(config)
    if trace_dir is None:
        return None
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        state = packet.get("state", {}) if isinstance(packet.get("state"), dict) else {}
        game_id = _safe_trace_part(state.get("game_id", packet.get("game_id", "game")))
        level = _safe_trace_part(state.get("level_index", packet.get("level_index", "level")))
        step = _safe_trace_part(state.get("step_index", packet.get("step_index", "step")))
        stem = f"{time.time_ns()}_{role.value}_g{game_id}_l{level}_s{step}"
        base = trace_dir / stem
        (base.with_suffix(".packet.json")).write_text(json.dumps(packet, ensure_ascii=False, sort_keys=False, indent=2, default=str), encoding="utf-8")
        (base.with_suffix(".prompt.txt")).write_text(prompt, encoding="utf-8", newline="\n")
        meta = {
            "role": role.value,
            "game_id": state.get("game_id", packet.get("game_id")),
            "level_index": state.get("level_index", packet.get("level_index")),
            "step_index": state.get("step_index", packet.get("step_index")),
            "schema_version": packet.get("schema_version"),
            "prompt_token_estimate": estimate_tokens(prompt),
            "context_tokens": config.qwen_context_tokens,
            "max_input_tokens": config.qwen_max_input_tokens,
            "max_output_tokens": config.qwen_max_output_tokens,
            "timeout_seconds": config.qwen_timeout_seconds,
            "command_without_prompt_file": cmd,
        }
        (base.with_suffix(".meta.json")).write_text(json.dumps(meta, ensure_ascii=False, sort_keys=False, indent=2, default=str), encoding="utf-8")
        return base
    except Exception:
        return None


def _write_qwen_trace_output(base: Path | None, stdout: str, stderr: str, extracted: dict[str, Any], status: str, *, elapsed_seconds: float | None = None) -> None:
    if base is None:
        return
    try:
        base.with_suffix(".stdout.txt").write_text(stdout or "", encoding="utf-8")
        base.with_suffix(".stderr.txt").write_text(stderr or "", encoding="utf-8")
        summary = {
            "status": status,
            "elapsed_seconds": round(float(elapsed_seconds), 6) if elapsed_seconds is not None else None,
            "stdout_chars": len(stdout or ""),
            "stderr_chars": len(stderr or ""),
            "extracted_schema_version": extracted.get("schema_version") if isinstance(extracted, dict) else None,
            "hypotheses_count": len(extracted.get("hypotheses") or []) if isinstance(extracted, dict) else 0,
            "coordinate_hypotheses_count": len(extracted.get("coordinate_hypotheses") or []) if isinstance(extracted, dict) else 0,
            "plan_step_count": _semantic_plan_step_count(extracted) if isinstance(extracted, dict) else 0,
            "coordinate_plan_step_count": len(extracted.get("candidate_sequence") or []) if isinstance(extracted, dict) else 0,
            "decision": extracted.get("decision") if isinstance(extracted, dict) else None,
            "extracted": extracted if isinstance(extracted, dict) else {},
        }
        base.with_suffix(".output.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        return


def _semantic_plan_step_count(extracted: dict[str, Any]) -> int:
    direct = extracted.get("steps")
    if isinstance(direct, list):
        return len(direct)
    total = 0
    for hypothesis in extracted.get("hypotheses") or []:
        if isinstance(hypothesis, dict) and isinstance(hypothesis.get("trajectory"), list):
            total += len(hypothesis["trajectory"])
        elif isinstance(hypothesis, dict) and isinstance(hypothesis.get("actions"), list):
            total += len(hypothesis["actions"])
    return total


def _safe_trace_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:48]
