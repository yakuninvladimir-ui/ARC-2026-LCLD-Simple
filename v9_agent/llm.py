from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

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
        if config.qwen_backend == "ollama":
            return self._call_ollama_once(role, packet, config)
        if config.qwen_backend == "vllm":
            return self._call_vllm_once(role, packet, config)
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
            if not _runtime_logs_enabled():
                cmd.append("--log-disable")
            cmd.extend([
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

    def _call_ollama_once(self, role: QwenRole, packet: dict[str, Any], config: V8Config) -> dict[str, Any]:
        images = _packet_image_payloads(packet) if config.qwen_multimodal_enabled else []
        text_packet = _packet_without_embedded_images(packet)
        compacted = _fit_packet_to_budget(role, text_packet, config)
        _set_packet_image_availability(compacted, bool(images))
        output_schema = _allowed_output_schema(role, compacted)
        prompt = _prompt(role, compacted, config)
        enum_hint = _ollama_contract_enum_hint(output_schema)
        if enum_hint:
            prompt = f"{prompt}\n\n{enum_hint}"
        base_url = _normalize_ollama_base_url(config.qwen_ollama_base_url)
        model = str(config.qwen_ollama_model or "qwen_local_3_5")
        endpoint = f"{base_url}/api/chat"
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        thinking_enabled = bool(config.qwen_enable_thinking)
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "think": thinking_enabled,
            "format": output_schema,
            "keep_alive": _ollama_keep_alive_value(config.qwen_ollama_keep_alive),
            "options": {
                "num_ctx": max(1, int(config.qwen_context_tokens)),
                "num_predict": max(1, int(config.qwen_max_output_tokens)),
                "temperature": float(config.qwen_temperature),
                "top_k": max(0, int(config.qwen_top_k)),
                "top_p": float(config.qwen_top_p),
                "min_p": float(config.qwen_min_p),
                "repeat_penalty": float(config.qwen_repeat_penalty),
                "seed": int(config.qwen_seed),
            },
        }
        command_description = [
            "OLLAMA_POST",
            endpoint,
            f"model={model}",
            "stream=false",
            f"think={str(thinking_enabled).lower()}",
            f"num_ctx={int(config.qwen_context_tokens)}",
            f"num_predict={int(config.qwen_max_output_tokens)}",
            f"images={len(images)}",
        ]
        trace_base = _write_qwen_trace_input(role, compacted, prompt, command_description, config)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib_request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib_request.urlopen(request, timeout=max(1, int(config.qwen_timeout_seconds))) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib_error.HTTPError as exc:
            elapsed = time.monotonic() - started
            error_body = exc.read().decode("utf-8", errors="replace")
            status = f"http_{int(exc.code)}"
            _write_qwen_trace_output(
                trace_base,
                "",
                error_body or str(exc),
                {},
                status,
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "ollama", "model": model, "endpoint": endpoint},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError(f"ollama_{status}: {(error_body or str(exc))[-1600:]}") from exc
            return {}
        except (socket.timeout, TimeoutError) as exc:
            elapsed = time.monotonic() - started
            _write_qwen_trace_output(
                trace_base,
                "",
                str(exc),
                {},
                "timeout",
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "ollama", "model": model, "endpoint": endpoint},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError("ollama_timeout") from exc
            return {}
        except urllib_error.URLError as exc:
            elapsed = time.monotonic() - started
            reason = getattr(exc, "reason", exc)
            is_timeout = isinstance(reason, (socket.timeout, TimeoutError))
            status = "timeout" if is_timeout else "connection_error"
            _write_qwen_trace_output(
                trace_base,
                "",
                str(exc),
                {},
                status,
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "ollama", "model": model, "endpoint": endpoint},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError(f"ollama_{status}: {exc}") from exc
            return {}

        elapsed = time.monotonic() - started
        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            _write_qwen_trace_output(
                trace_base,
                raw_response,
                str(exc),
                {},
                "invalid_backend_response",
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "ollama", "model": model, "endpoint": endpoint},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError("ollama_invalid_json_response") from exc
            return {}

        message = response_data.get("message", {}) if isinstance(response_data, dict) else {}
        content = str(message.get("content") or "") if isinstance(message, dict) else ""
        extracted = _extract_json(content, role=role)
        runtime_metadata = _ollama_runtime_metadata(response_data, endpoint=endpoint, model=model)
        status = "json_schema_valid" if extracted else "parse_or_schema_invalid"
        _write_qwen_trace_output(
            trace_base,
            content,
            "",
            extracted,
            status,
            elapsed_seconds=elapsed,
            runtime_metadata=runtime_metadata,
        )
        if not extracted and config.qwen_require_runtime:
            raise QwenBackendError("ollama_parse_or_schema_invalid")
        return extracted

    def _call_vllm_once(self, role: QwenRole, packet: dict[str, Any], config: V8Config) -> dict[str, Any]:
        # Keep the encoded PNG outside the text/compaction packet.  The image is
        # attached once to the HTTP request, while schema and prompt construction
        # operate only on the bounded textual packet.
        images = _packet_image_payloads(packet) if config.qwen_multimodal_enabled else []
        text_packet = _packet_without_embedded_images(packet)
        compacted = _fit_packet_to_budget(role, text_packet, config)
        _set_packet_image_availability(compacted, bool(images))
        prompt = _prompt(role, compacted, config)
        schema_mode = _output_schema_mode()
        output_schema = _vllm_compatible_output_schema(_allowed_output_schema(role, compacted))
        base_url = _normalize_vllm_base_url(config.qwen_vllm_base_url)
        model = str(config.qwen_vllm_model or "vrfai/Qwen3.6-27B-FP8")
        endpoint = f"{base_url}/chat/completions"
        thinking_enabled = bool(config.qwen_enable_thinking)
        content: str | list[dict[str, Any]] = prompt
        if images:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                }
                for image in images
            ]
            content.append({"type": "text", "text": prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": max(1, int(config.qwen_max_output_tokens)),
            "temperature": float(config.qwen_temperature),
            "top_k": max(0, int(config.qwen_top_k)),
            "top_p": float(config.qwen_top_p),
            "min_p": float(config.qwen_min_p),
            "presence_penalty": float(config.qwen_presence_penalty),
            "repetition_penalty": float(config.qwen_repeat_penalty),
            "seed": int(config.qwen_seed),
            "chat_template_kwargs": {"enable_thinking": thinking_enabled},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"lcld_{role.value}_{schema_mode}_response",
                    "schema": output_schema,
                    "strict": True,
                },
            },
        }
        command_description = [
            "VLLM_OPENAI_POST",
            endpoint,
            f"model={model}",
            "stream=false",
            f"enable_thinking={str(thinking_enabled).lower()}",
            f"max_tokens={int(config.qwen_max_output_tokens)}",
            f"schema_mode={schema_mode}",
            f"images={len(images)}",
        ]
        trace_base = _write_qwen_trace_input(role, compacted, prompt, command_description, config)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.qwen_vllm_api_key:
            headers["Authorization"] = f"Bearer {config.qwen_vllm_api_key}"
        request = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib_request.urlopen(request, timeout=max(1, int(config.qwen_timeout_seconds))) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib_error.HTTPError as exc:
            elapsed = time.monotonic() - started
            error_body = exc.read().decode("utf-8", errors="replace")
            status = f"http_{int(exc.code)}"
            _write_qwen_trace_output(
                trace_base,
                "",
                error_body or str(exc),
                {},
                status,
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "vllm", "model": model, "endpoint": endpoint, "images": len(images)},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError(f"vllm_{status}: {(error_body or str(exc))[-1600:]}") from exc
            return {}
        except (socket.timeout, TimeoutError) as exc:
            elapsed = time.monotonic() - started
            _write_qwen_trace_output(
                trace_base,
                "",
                str(exc),
                {},
                "timeout",
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "vllm", "model": model, "endpoint": endpoint, "images": len(images)},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError("vllm_timeout") from exc
            return {}
        except urllib_error.URLError as exc:
            elapsed = time.monotonic() - started
            reason = getattr(exc, "reason", exc)
            is_timeout = isinstance(reason, (socket.timeout, TimeoutError))
            status = "timeout" if is_timeout else "connection_error"
            _write_qwen_trace_output(
                trace_base,
                "",
                str(exc),
                {},
                status,
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "vllm", "model": model, "endpoint": endpoint, "images": len(images)},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError(f"vllm_{status}: {exc}") from exc
            return {}

        elapsed = time.monotonic() - started
        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            _write_qwen_trace_output(
                trace_base,
                raw_response,
                str(exc),
                {},
                "invalid_backend_response",
                elapsed_seconds=elapsed,
                runtime_metadata={"backend": "vllm", "model": model, "endpoint": endpoint, "images": len(images)},
            )
            if config.qwen_require_runtime:
                raise QwenBackendError("vllm_invalid_json_response") from exc
            return {}

        response_object = response_data if isinstance(response_data, dict) else {}
        choices = response_object.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        response_content = str(message.get("content") or "")
        extracted = _extract_json(response_content, role=role)
        usage = response_object.get("usage") if isinstance(response_object.get("usage"), dict) else {}
        runtime_metadata = {
            "backend": "vllm",
            "endpoint": endpoint,
            "model": model,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "image_count": len(images),
            "thinking_enabled": thinking_enabled,
            "output_schema_mode": schema_mode,
            "validated_response_field": "message.content",
            "response_schema_removed_keywords": _count_removed_vllm_schema_keywords(
                _allowed_output_schema(role, compacted),
                output_schema,
            ),
        }
        self.last_call_metrics = dict(runtime_metadata)
        status = "json_schema_valid" if extracted else "parse_or_schema_invalid"
        _write_qwen_trace_output(
            trace_base,
            response_content,
            "",
            extracted,
            status,
            elapsed_seconds=elapsed,
            runtime_metadata=runtime_metadata,
        )
        if not extracted and config.qwen_require_runtime:
            raise QwenBackendError("vllm_parse_or_schema_invalid")
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


def _normalize_ollama_base_url(value: str | None) -> str:
    base_url = str(value or "http://127.0.0.1:11434").strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    return base_url


def _normalize_vllm_base_url(value: str | None) -> str:
    base_url = str(value or "http://127.0.0.1:1234/v1").strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def _vllm_compatible_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove validation-only keywords unsupported by some vLLM decoders."""
    unsupported = {"uniqueItems", "contains", "minContains", "maxContains"}

    def adapt(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: adapt(item)
                for key, item in value.items()
                if key not in unsupported
            }
        if isinstance(value, list):
            return [adapt(item) for item in value]
        return value

    adapted = adapt(schema)
    return adapted if isinstance(adapted, dict) else {}


def _count_removed_vllm_schema_keywords(
    original: dict[str, Any],
    adapted: dict[str, Any],
) -> int:
    del adapted
    unsupported = {"uniqueItems", "contains", "minContains", "maxContains"}

    def count(value: Any) -> int:
        if isinstance(value, dict):
            return sum(1 for key in value if key in unsupported) + sum(count(item) for item in value.values())
        if isinstance(value, list):
            return sum(count(item) for item in value)
        return 0

    return count(original)


def _ollama_keep_alive_value(value: Any) -> int | str:
    text = str(value if value is not None else "-1").strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text or "5m"


def _ollama_contract_enum_hint(schema: dict[str, Any]) -> str:
    """Expose small semantic enums when an Ollama cloud endpoint ignores format."""
    try:
        hypothesis = schema["properties"]["hypotheses"]["items"]
        properties = hypothesis["properties"]
        families = list(properties["family"]["enum"])
        objective_kinds = list(properties["objective"]["properties"]["kind"]["enum"])
        status = str(properties["status"]["const"])
    except (KeyError, TypeError, ValueError):
        return ""
    return (
        "OLLAMA_SCHEMA_ENUM_REQUIREMENTS: Do not invent synonyms. "
        f"family must be exactly one of {json.dumps(families, ensure_ascii=True)}; "
        f"objective.kind must be exactly one of {json.dumps(objective_kinds, ensure_ascii=True)}; "
        f"status must be exactly {json.dumps(status)}."
    )


def _ollama_runtime_metadata(response: Any, *, endpoint: str, model: str) -> dict[str, Any]:
    data = response if isinstance(response, dict) else {}
    message = data.get("message", {}) if isinstance(data.get("message"), dict) else {}
    thinking = str(message.get("thinking") or "")
    content = str(message.get("content") or "")
    return {
        "backend": "ollama",
        "endpoint": endpoint,
        "model": model,
        "done": bool(data.get("done")),
        "done_reason": data.get("done_reason"),
        "total_duration_ns": data.get("total_duration"),
        "load_duration_ns": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
        "eval_count": data.get("eval_count"),
        "eval_duration_ns": data.get("eval_duration"),
        "thinking": thinking,
        "thinking_chars": len(thinking),
        "thinking_token_estimate": estimate_tokens(thinking) if thinking else 0,
        "content_chars": len(content),
        "content_token_estimate": estimate_tokens(content) if content else 0,
    }


def _runtime_logs_enabled() -> bool:
    return os.environ.get("ARC_QWEN_CAPTURE_RUNTIME_LOGS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
            targets = packet.get("action_space", {}).get("coordinate_candidates", []) or packet.get("scene", {}).get("coordinate_candidates", []) or []
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
            for item in (packet.get("action_space", {}).get("actions") or packet.get("action_surface", {}).get("actions", []))
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
            "Choose an ordered sequence of informative coordinate candidates. Each selected candidate is clicked exactly once before the separate primary planning call.\n"
            "Infer no intrinsic meaning from an action ID. Use CURRENT_FRAME_PNG, OBJECT_LAYER, HEX_PATCHES, and factual ACTION_DIFFS.\n"
            "Several candidates may be selected when they are distinct physical locations. Do not retry a location in the same research sequence."
        )
        precondition = (
            "PROBE RULES:\n"
            "- Use only ACTION_SPACE.coordinate_candidates and action IDs supplied in OBSERVATION_PACKET_JSON.\n"
            "- execution_constraints.allowed_coordinate_candidate_ids is the authoritative whitelist: every ID present in it is allowed.\n"
            "- candidate_sequence may contain several candidates, but all coordinate_candidate_id values must be distinct. Each item means exactly one click on that candidate.\n"
            "- Choose at most one candidate for each location_xy. Different IDs at one location are alternative descriptions of one target.\n"
            "- MEMORY.attempts and ACTION_DIFFS identify already tested locations and their factual results.\n"
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
                "- One click per coordinate run does not mean one click per trajectory. When the evidence supports a multi-target configuration, include several ordered coordinate runs with distinct candidate IDs in the same complete trajectory. Do not default to a one-click hypothesis.\n"
                "- A coordinate_candidate_id and its physical location_xy may each appear at most once in the primary trajectory. A prior research click is evidence and does not consume that target for goal execution.\n"
            )
        header = (
            "You propose valid complete full action trajectories for a single interactive game level. "
            "Trajectories are not limited to movement: they may include any action from execution_constraints.allowed_action_ids "
            "that changes the available-action surface, environmental configuration, or an object's position, state, "
            "or color as needed to achieve the level goal."
        )
        task = (
            "TASK:\n"
            "This is the only goal-planning model call for the current level attempt. Infer the level-transition objective and return 1-3 genuinely distinct grounded hypotheses.\n"
            "CURRENT_FRAME_PNG, OBJECT_LAYER, and ACTION_SPACE describe the exact state from which the proposed trajectory will execute. STATE.trajectory_start states whether prior research was restored away; when prior_research_actions_are_history_only is true, probe effects in ACTION_DIFFS are evidence but are NOT already applied to the current frame. Include any observed surface-changing action needed to recreate a required state. ACTION_DIFFS are chronological factual effect groups, not alternate current frames. Each entry's before/after, pixel_diff, object_changes, and synchronous_local_visual_evidence describe the one execution named by observation_summary.sample_step_index; observation_count and step_index_range report how many equivalent executions were grouped.\n"
            "Infer each action's usable effect only from ACTION_DIFFS. Compare before and after object coordinates, bbox, colors, pixel cells, and available_action_ids; do not infer meaning from the action number or from a semantic label.\n"
            "When one transition changes available_action_ids and also changes repeated object-local center cells or local hex patches, treat both as one synchronous fact. Consider a latent active-object/control-state change; do not reduce the action's entire effect to action-surface change alone.\n"
            "For repeated spatial effects, derive the one-step vector from an object's before/after coordinates, start from its coordinates in OBJECT_LAYER, and calculate the complete repeat count needed by the hypothesis. Account for every object changed by the same action.\n"
            "MEMORY.completed_levels contains verified solutions from earlier levels of this game. Use their action ordering and confirmed mechanics as evidence, but historical object IDs are level-local and must not be reused as current IDs.\n"
            "MEMORY.attempts contains completed failed attempts. research_action_runs are evidence collection, not failed goal trajectories and may be reused as a prefix when needed. rejected_proposals gives the exact prior candidate and verifier reason. Never repeat a rejected_proposal or any trajectory marked exact_replay_forbidden unchanged; repair the stated failure. For an availability rejection, insert an observed enabling action before trajectory_step_index_1_based and then continue from the resulting action surface.\n"
            "MEMORY.semantic_feedback reports how prior proposals were bound and evaluated. Preserve corrected source/reference roles, do not exactly replay trajectories marked FORBIDDEN or IRRELEVANT, and use CONFIRMED invariants to refine action counts or ordering.\n"
            "Every proposed hypothesis must be a complete_candidate with the full ordered action sequence from the current state; do not return a prefix or request later re-planning.\n"
            "One hypothesis may cover several source/reference pairs and necessary intermediate configurations. Do not split a multi-object solution into separate model calls.\n"
            "A trajectory may use movement, color/configuration changes, coordinate interactions, or actions that change available_action_ids. Account for simultaneous effects and intermediate configurations.\n"
            "action_runs is the complete ordered trajectory encoded without manual token repetition. Its first action_id must be executable now; each repeat is the TOTAL number of consecutive executions in that run, including its first execution. Never subtract one for a separate first-action slot.\n"
            "Return final conclusions only, keep basis to at most 3 short sentences and 360 characters, and use only supplied IDs and observed effects.\n"
            "Return one hypothesis when only one strong interpretation is grounded. When evidence is uncertain, return the best grounded complete_candidate with lower confidence so verifier results can update memory."
        )
        precondition = (
            "READING RULES:\n"
            "- OBJECT_LAYER.component_graph partitions the current frame into same-color 4-connected components. Components are geometric evidence, not gameplay roles. Use only object IDs in execution_constraints.allowed_object_ids in response fields.\n"
            "- OBJECT_LAYER.objects gives current coordinates, bbox, colors, and geometry. exact_geometry_groups and shape hashes describe geometry only; they are not goals.\n"
            "- HEX_PATCHES are exact current local 0-F crops only for multicolor or segmentation-ambiguous objects. They do not replace the global PNG.\n"
            "- ACTION_DIFFS contains no assigned action semantics. Each flat before/after record is a direct sample execution; observation_summary only groups other executions with the same action, action surface, normalized object deltas, color transitions, and level result. synchronous_local_visual_evidence contains exact bounded 3x3 hex before/after patches from that same transition. In object_changes, shared_attributes applies identically to both before and after; other attributes remain in their respective state. Rare collisions, lifecycle changes, and surface changes remain separate groups. An empty object_changes list does not override a nonempty pixel_diff.\n"
            "- In pixel_diff, sparse_cells_xy lists individual [x,y] cells; row_runs uses y plus an inclusive x range and equal-length before/after palette strings. They are complete exact encodings only when position_data_complete is true. Otherwise changed_cell_count, changed_bbox_xyxy, color_transitions, and the listed evenly sampled runs remain factual but the omitted run positions must not be invented.\n"
            "- available_action_ids records legality in that observed state. A change in that list is a fact, not an explanation of why it changed.\n"
            "- STATE.trajectory_start is authoritative: CURRENT_FRAME_PNG, OBJECT_LAYER, and ACTION_SPACE.current_available_action_ids all refer to the same execution-start state. Research diffs marked history-only must be replayed inside the trajectory when their state is required.\n"
            "- MEMORY.semantic_feedback is reverse-chain annotation, not a new observation source. Deterministic binder records preserve source/reference roles; rebound_after_reset means those roles were deterministically matched to current object IDs. Verifier records distinguish mechanic_result from goal_progress; invariants are usable according to their authority and status. OBSERVED_ONCE and UNRESOLVED are hints, not confirmed rules.\n"
            "- The first trajectory action must occur in ACTION_SPACE.current_available_action_ids. A later action observed on another surface is usable only when the preceding trajectory establishes that surface.\n"
            f"{coordinate_action_rules}"
            "- The executor re-observes and verifies after every action, but the model must provide the complete expected trajectory now.\n"
            "- The goal is unknown. Lower confidence when needed, but never replace a complete trajectory with a prefix or an empty response."
        )
    output_contract = _compact_output_contract(role)
    parts = [
        header,
        "",
        "DATA AUTHORITY:",
        "- CURRENT_FRAME_PNG is the single latest global frame. When available_to_configured_backend is true, its data is attachment current_frame_png; otherwise the text model must rely on OBJECT_LAYER and HEX_PATCHES.",
        "- OBJECT_LAYER is a deterministic current-frame parse. It contains geometry, coordinates, colors, and connectivity, not gameplay-role labels.",
        "- HEX_PATCHES contains exact local categorical cells only where color detail or segmentation is ambiguous.",
        "- ACTION_DIFFS contains direct observed changes in pixels, objects, and available actions. It intentionally contains no action interpretation.",
        "- MEMORY contains verified completed-level solutions, attempt outcomes, confirmed effects, and bounded reverse semantic feedback with explicit authority/status. It contains no historical PNG frames.",
        "- You may propose action IDs and candidate trajectories inside the JSON response.",
        "- You do not execute or authorize actions.",
        "- The agent validates the candidate before execution and re-observes after every executed step.",
        "",
        precondition,
        "",
        task,
        "",
        "OBSERVATION_PACKET_JSON=",
        json.dumps(_packet_for_text_prompt(packet), ensure_ascii=False, sort_keys=False, separators=(",", ":")),
        "",
        "OUTPUT_CONTRACT=",
        output_contract,
        "",
        "Return exactly one valid JSON object.",
        "Do not return markdown.",
        "Do not return prose outside JSON.",
        "RETURN_JSON_ONLY",
    ]
    return "\n".join(parts)


def _packet_without_embedded_images(packet: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a packet without ever copying the potentially large PNG string."""
    projected = dict(packet)
    frame = packet.get("current_frame_png")
    if isinstance(frame, dict):
        projected["current_frame_png"] = {
            key: value
            for key, value in frame.items()
            if key != "data_base64"
        }
    return deepcopy(projected)


def _packet_for_text_prompt(packet: dict[str, Any]) -> dict[str, Any]:
    projected = _packet_without_embedded_images(packet)
    frame = projected.get("current_frame_png")
    if isinstance(frame, dict):
        frame["attachment_status"] = (
            "ATTACHED_IMAGE:current_frame_png"
            if frame.get("available_to_configured_backend")
            else "NOT_ATTACHED:configured_backend_has_no_vision"
        )
    return projected


def _packet_image_payloads(packet: dict[str, Any]) -> list[str]:
    frame = packet.get("current_frame_png")
    if not isinstance(frame, dict):
        return []
    payload = frame.get("data_base64")
    return [str(payload)] if isinstance(payload, str) and payload else []


def _set_packet_image_availability(packet: dict[str, Any], available: bool) -> None:
    frame = packet.get("current_frame_png")
    if isinstance(frame, dict):
        frame["available_to_configured_backend"] = bool(available)


def _compact_output_contract(role: QwenRole) -> str:
    if role is QwenRole.COORDINATE:
        return (
            "Return schema_version=v8.4.coordinate_plan, decision=PLAN, "
            "mechanism_hypothesis, coordinate_action_id, candidate_sequence "
            "of objects containing coordinate_candidate_id, completion_criterion, "
            "and confidence. Use only IDs whitelisted in OBSERVATION_PACKET_JSON."
        )
    return (
        "Return schema_version=v8.7.semantic_trajectories, decision=PROPOSE, and "
        "1-3 hypotheses. Each hypothesis contains id, family, objective with kind/"
        "source_objects/reference_objects/description, relations, basis, action_runs "
        "with action_id/repeat and optional coordinate_candidate_id, "
        "status=complete_candidate, uncertainty, and confidence. Use only IDs "
        "whitelisted in OBSERVATION_PACKET_JSON."
    )

def _output_schema_mode() -> str:
    value = os.environ.get("ARC_QWEN_SCHEMA_MODE", "static").strip().lower()
    if value in {"dynamic", "dynamic_enum", "frame_enum", "strict_enum"}:
        return "dynamic_enum"
    return "static"


def _allowed_output_schema(role: QwenRole, packet: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return either a role-static or frame-specific response schema.

    ``static`` keeps exactly two grammar identities (primary and coordinate).
    Frame-local IDs remain in ``execution_constraints`` and are rejected again
    by HypothesisBank before execution.

    ``dynamic_enum`` preserves the V9 hard constrained-decoding contract: the
    current action/object/relation/coordinate IDs are emitted as JSON Schema
    enums, so hallucinated IDs cannot be decoded. It is retained as an explicit
    A/B mode because each distinct schema can require a separate XGrammar
    compilation and cache entry.
    """
    if _output_schema_mode() == "dynamic_enum":
        return _dynamic_enum_output_schema(role, packet)
    return _static_output_schema(role)


def _static_id_schema(*, max_length: int = 128) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": max_length}


def _static_id_array_schema(*, max_items: int = 6, max_length: int = 128) -> dict[str, Any]:
    return {
        "type": "array",
        "maxItems": max_items,
        "items": _static_id_schema(max_length=max_length),
    }


def _static_output_schema(role: QwenRole) -> dict[str, Any]:
    """Role-static schema used by the controlled competition rerun.

    The schema constrains structure, semantic enums, lengths, numeric ranges,
    and list sizes, but deliberately does not embed frame-local identifiers.
    """
    max_steps = 50
    if role is QwenRole.COORDINATE:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "decision",
                "mechanism_hypothesis",
                "coordinate_action_id",
                "candidate_sequence",
                "completion_criterion",
                "confidence",
            ],
            "properties": {
                "schema_version": {"const": "v8.4.coordinate_plan"},
                "decision": {"const": "PLAN"},
                "mechanism_hypothesis": {"type": "string", "minLength": 1, "maxLength": 240},
                "coordinate_action_id": _static_id_schema(max_length=64),
                "candidate_sequence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_steps,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["coordinate_candidate_id"],
                        "properties": {
                            "coordinate_candidate_id": _static_id_schema(max_length=128),
                        },
                    },
                },
                "completion_criterion": {"type": "string", "maxLength": 240},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }

    action_run_properties: dict[str, Any] = {
        "action_id": _static_id_schema(max_length=64),
        "repeat": {
            "type": "integer",
            "minimum": 1,
            "maximum": max_steps,
            "description": "Total consecutive executions. Must be exactly 1 when action_id is a coordinate action.",
        },
        "coordinate_candidate_id": {
            **_static_id_schema(max_length=128),
            "description": "Optional and valid only for a coordinate action; post-validation enforces the current whitelist.",
        },
    }
    objective_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "source_objects", "reference_objects", "description"],
        "properties": {
            "kind": {
                "enum": [
                    "match_or_overlap",
                    "relative_arrangement",
                    "containment",
                    "connection",
                    "pattern_or_state",
                    "select_or_activate",
                    "surface_change",
                    "other",
                ]
            },
            "source_objects": _static_id_array_schema(max_items=6),
            "reference_objects": _static_id_array_schema(max_items=6),
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
            "relations": _static_id_array_schema(max_items=6),
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
            "status": {"const": "complete_candidate"},
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


def _dynamic_enum_output_schema(role: QwenRole, packet: dict[str, Any] | None = None) -> dict[str, Any]:
    """V9 frame-specific schema that hard-blocks IDs outside execution_constraints."""
    constraints = (packet or {}).get("execution_constraints", {}) if isinstance(packet, dict) else {}
    action_ids = list(dict.fromkeys(str(value) for value in constraints.get("allowed_action_ids", [])))
    object_ids = list(dict.fromkeys(str(value) for value in constraints.get("allowed_object_ids", [])))
    relation_ids = list(dict.fromkeys(str(value) for value in constraints.get("allowed_relation_ids", [])))
    coordinate_ids = list(
        dict.fromkeys(str(value) for value in constraints.get("allowed_coordinate_candidate_ids", []))
    )
    max_steps = max(1, int(constraints.get("max_plan_steps") or 50))
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
            "required": [
                "schema_version",
                "decision",
                "mechanism_hypothesis",
                "coordinate_action_id",
                "candidate_sequence",
                "completion_criterion",
                "confidence",
            ],
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
            "kind": {
                "enum": [
                    "match_or_overlap",
                    "relative_arrangement",
                    "containment",
                    "connection",
                    "pattern_or_state",
                    "select_or_activate",
                    "surface_change",
                    "other",
                ]
            },
            "source_objects": {
                **_array_enum_schema(object_ids),
                "maxItems": min(6, len(object_ids)),
            },
            "reference_objects": {
                **_array_enum_schema(object_ids),
                "maxItems": min(6, len(object_ids)),
            },
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
            "relations": {
                **_array_enum_schema(relation_ids),
                "maxItems": min(6, len(relation_ids)),
            },
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
            "status": {"const": "complete_candidate"},
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
    if out.get("schema_version") == "v8.8.layered_observation":
        return _compact_v88_packet(out, strategy)
    if out.get("schema_version") == "v8.7.semantic_observation":
        return _compact_v87_packet(out, strategy)
    if out.get("schema_version") == "v8.4.observation_packet":
        return _compact_v84_packet(out, strategy)
    return out


def _compact_v88_packet(out: dict[str, Any], strategy: str) -> dict[str, Any]:
    layer = out.setdefault("object_layer", {})
    object_limit = 18 if strategy == "aggressive" else 12
    relation_limit = 28 if strategy == "aggressive" else 18
    layer["objects"] = (layer.get("objects") or [])[:object_limit]
    layer["relations"] = (layer.get("relations") or [])[:relation_limit]
    layer["exact_geometry_groups"] = (layer.get("exact_geometry_groups") or [])[:object_limit]
    graph = layer.get("component_graph")
    if isinstance(graph, dict):
        _compact_component_graph(graph, 64 if strategy == "aggressive" else 40, strategy)
    out["hex_patches"] = (out.get("hex_patches") or [])[: (12 if strategy == "aggressive" else 8)]
    out["action_diffs"] = (out.get("action_diffs") or [])[-(16 if strategy == "aggressive" else 10) :]
    action_space = out.setdefault("action_space", {})
    action_space["coordinate_candidates"] = (action_space.get("coordinate_candidates") or [])[: (48 if strategy == "aggressive" else 32)]
    memory = out.get("memory")
    if isinstance(memory, dict):
        memory["confirmed_effects"] = (memory.get("confirmed_effects") or [])[-(16 if strategy == "aggressive" else 10) :]
        attempts = memory.get("attempts")
        if isinstance(attempts, dict):
            attempts["previous_failed_attempts"] = (attempts.get("previous_failed_attempts") or [])[-(4 if strategy == "aggressive" else 2) :]
            feedback = attempts.get("current_attempt_feedback")
            if isinstance(feedback, dict):
                feedback["hypotheses"] = (feedback.get("hypotheses") or [])[-8:]
                feedback["rejections"] = (feedback.get("rejections") or [])[-8:]
        semantic_feedback = memory.get("semantic_feedback")
        if isinstance(semantic_feedback, dict):
            keep = 6 if strategy == "aggressive" else 4
            semantic_feedback["bindings"] = (semantic_feedback.get("bindings") or [])[:keep]
            semantic_feedback["trajectory_evaluations"] = (semantic_feedback.get("trajectory_evaluations") or [])[:keep]
            semantic_feedback["invariants"] = (semantic_feedback.get("invariants") or [])[: (10 if strategy == "aggressive" else 6)]
    _repair_v88_references(out)
    return out


def _repair_v88_references(packet: dict[str, Any]) -> None:
    layer = packet.setdefault("object_layer", {})
    object_ids = {str(item.get("id")) for item in layer.get("objects") or [] if isinstance(item, dict)}
    relations = [
        item
        for item in layer.get("relations") or []
        if isinstance(item, dict)
        and {str(value) for value in item.get("object_ids") or []}.issubset(object_ids)
    ]
    layer["relations"] = relations
    relation_ids = {str(item.get("id")) for item in relations}
    groups = []
    for group in layer.get("exact_geometry_groups") or []:
        if not isinstance(group, dict):
            continue
        members = [str(value) for value in group.get("object_ids") or [] if str(value) in object_ids]
        if members:
            copy = dict(group)
            copy["object_ids"] = members
            groups.append(copy)
    layer["exact_geometry_groups"] = groups
    graph = layer.get("component_graph")
    if isinstance(graph, dict):
        for component in graph.get("components") or []:
            if isinstance(component, dict):
                component["object_refs"] = [str(value) for value in component.get("object_refs") or [] if str(value) in object_ids]
    packet["hex_patches"] = [
        item for item in packet.get("hex_patches") or []
        if isinstance(item, dict) and str(item.get("object_id")) in object_ids
    ]
    for diff in packet.get("action_diffs") or []:
        if isinstance(diff, dict):
            diff["object_changes"] = [
                item for item in diff.get("object_changes") or []
                if isinstance(item, dict) and str(item.get("object_id")) in object_ids
            ]
    action_space = packet.setdefault("action_space", {})
    candidates = [
        item for item in action_space.get("coordinate_candidates") or []
        if isinstance(item, dict)
        and (item.get("object_id") is None or str(item.get("object_id")) in object_ids)
        and (item.get("relation_id") is None or str(item.get("relation_id")) in relation_ids)
    ]
    action_space["coordinate_candidates"] = candidates
    action_ids = {str(item.get("id")) for item in action_space.get("actions") or [] if isinstance(item, dict)}
    semantic_feedback = (packet.get("memory") or {}).get("semantic_feedback")
    if isinstance(semantic_feedback, dict):
        semantic_feedback["bindings"] = [
            item
            for item in semantic_feedback.get("bindings") or []
            if isinstance(item, dict)
            and set(map(str, item.get("source_object_ids") or [])).issubset(object_ids)
            and set(map(str, item.get("reference_object_ids") or [])).issubset(object_ids)
            and set(map(str, item.get("relation_ids") or [])).issubset(relation_ids)
        ]
        semantic_feedback["invariants"] = [
            item
            for item in semantic_feedback.get("invariants") or []
            if isinstance(item, dict)
            and set(map(str, item.get("subject_object_ids") or [])).issubset(object_ids)
            and (not item.get("action_id") or str(item.get("action_id")) in action_ids)
        ]
        for item in semantic_feedback.get("trajectory_evaluations") or []:
            if not isinstance(item, dict):
                continue
            item["source_object_ids"] = [str(value) for value in item.get("source_object_ids") or [] if str(value) in object_ids]
            item["reference_object_ids"] = [str(value) for value in item.get("reference_object_ids") or [] if str(value) in object_ids]
            item["relation_ids"] = [str(value) for value in item.get("relation_ids") or [] if str(value) in relation_ids]
    constraints = packet.setdefault("execution_constraints", {})
    constraints["allowed_object_ids"] = sorted(object_ids)
    constraints["allowed_relation_ids"] = sorted(relation_ids)
    constraints["allowed_coordinate_candidate_ids"] = sorted(str(item.get("id")) for item in candidates if item.get("id") is not None)
    constraints["allowed_action_ids"] = [str(value) for value in constraints.get("allowed_action_ids") or [] if str(value) in action_ids]


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
            evidence["coordinate_research_history"] = (evidence.get("coordinate_research_history") or [])[-keep:]
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
    if packet.get("schema_version") == "v8.8.layered_observation":
        layer = packet.get("object_layer", {})
        action_space = packet.get("action_space", {})
        return bool(layer.get("objects") or layer.get("relations") or action_space.get("coordinate_candidates"))
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
            "backend": config.qwen_backend,
            "model": (
                config.qwen_ollama_model
                if config.qwen_backend == "ollama"
                else config.qwen_model_path
            ),
            "game_id": state.get("game_id", packet.get("game_id")),
            "level_index": state.get("level_index", packet.get("level_index")),
            "step_index": state.get("step_index", packet.get("step_index")),
            "schema_version": packet.get("schema_version"),
            "prompt_token_estimate": estimate_tokens(prompt),
            "context_tokens": config.qwen_context_tokens,
            "max_input_tokens": config.qwen_max_input_tokens,
            "max_output_tokens": config.qwen_max_output_tokens,
            "timeout_seconds": config.qwen_timeout_seconds,
            "runtime_logs_enabled": (
                _runtime_logs_enabled()
                if config.qwen_backend in {"qwen_local", "llama_cli"}
                else False
            ),
            "command_without_prompt_file": cmd,
        }
        (base.with_suffix(".meta.json")).write_text(json.dumps(meta, ensure_ascii=False, sort_keys=False, indent=2, default=str), encoding="utf-8")
        return base
    except Exception:
        return None


def _write_qwen_trace_output(
    base: Path | None,
    stdout: str,
    stderr: str,
    extracted: dict[str, Any],
    status: str,
    *,
    elapsed_seconds: float | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> None:
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
            "runtime": dict(runtime_metadata or {}),
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
