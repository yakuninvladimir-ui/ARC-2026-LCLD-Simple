import ast
import base64
import importlib.util
import io
import json
from pathlib import Path
import zipfile

from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config, config_from_mapping
from v8_agent.game_adapter import GameAdapter
from v8_agent.hypothesis_bank import HypothesisBank
import v8_agent.llm as llm
from v8_agent.llm import QwenClient
from v8_agent.memory import GameMemory
from v8_agent.qwen_packet import QwenPacketBuilder
from v8_agent.types import ActionEffectRecord, QwenRole


def _semantic_packet():
    memory = GameMemory()
    memory.reset_game("g")
    observation = {
        "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 2]],
        "metadata": {
            "game_id": "g",
            "available_actions": ["ACTION1"],
            "frame_index": 0,
        },
    }
    snapshot = ARGALiteBuilder().build(
        GameAdapter().to_world_state(observation),
        memory,
        V8Config(),
    )
    memory.action_effects[("ACTION1", None)] = ActionEffectRecord(
        "ACTION1", None, "effect", 1, 0.57, snapshot.level_index, snapshot.step_index
    )
    return QwenPacketBuilder().build_semantic_packet(
        snapshot,
        memory,
        HypothesisBank(),
        QwenRole.PRIMARY,
        V8Config(),
    )


def _valid_completion():
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
                "description": "Test the observed action effect.",
            },
            "relations": [],
            "basis": "The action is available and has an observed effect.",
            "action_runs": [{"action_id": "ACTION1", "repeat": 1}],
            "status": "complete_candidate",
            "uncertainty": "The fixture contains limited evidence.",
            "confidence": 0.5,
        }],
    }


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("v83_qwen_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _embedded_payload(unpack_source):
    tree = ast.parse(unpack_source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets):
            return base64.b64decode(ast.literal_eval(node.value))
    raise AssertionError("embedded payload assignment is absent")


def test_qwen_vllm_environment_contract(monkeypatch):
    monkeypatch.setenv("ARC_V8_QWEN_BACKEND", "vllm")
    monkeypatch.setenv("ARC_QWEN_VLLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("ARC_QWEN_VLLM_MODEL", "vrfai/Qwen3.6-27B-FP8")
    monkeypatch.setenv("ARC_QWEN_CONTEXT_TOKENS", "98304")
    monkeypatch.setenv("ARC_QWEN_MAX_INPUT_TOKENS", "65536")
    monkeypatch.setenv("ARC_QWEN_MAX_OUTPUT_TOKENS", "12288")
    monkeypatch.setenv("ARC_QWEN_ENABLE_THINKING", "true")

    config = config_from_mapping()

    assert config.qwen_backend == "vllm"
    assert config.qwen_vllm_model == "vrfai/Qwen3.6-27B-FP8"
    assert config.qwen_context_tokens == 98304
    assert config.qwen_max_input_tokens == 65536
    assert config.qwen_max_output_tokens == 12288
    assert config.qwen_enable_thinking is False


def test_vllm_request_uses_qwen_contract_and_strict_json_schema(monkeypatch, tmp_path):
    captured = {}
    completion = _valid_completion()
    response_payload = {
        "choices": [{
            "message": {"content": json.dumps(completion)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.headers.get("Authorization")
        captured["request"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(llm.urllib_request, "urlopen", fake_urlopen)
    config = V8Config(
        qwen_backend="vllm",
        qwen_require_runtime=True,
        qwen_vllm_base_url="http://127.0.0.1:1234/v1",
        qwen_vllm_api_key="EMPTY",
        qwen_vllm_model="vrfai/Qwen3.6-27B-FP8",
        qwen_context_tokens=98304,
        qwen_max_input_tokens=65536,
        qwen_max_output_tokens=12288,
        qwen_timeout_seconds=500,
        qwen_temperature=0.7,
        qwen_top_k=20,
        qwen_top_p=0.8,
        qwen_presence_penalty=1.5,
        qwen_trace_dir=str(tmp_path),
    )

    client = QwenClient()
    result = client.call(QwenRole.PRIMARY, _semantic_packet(), config)

    request = captured["request"]
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert captured["timeout"] == 500
    assert captured["authorization"] == "Bearer EMPTY"
    assert request["model"] == "vrfai/Qwen3.6-27B-FP8"
    assert request["max_tokens"] == 12288
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "uniqueItems" not in json.dumps(request["response_format"]["json_schema"]["schema"])
    assert result["hypotheses"][0]["actions"] == ["ACTION1"]
    assert client.last_call_metrics["prompt_tokens"] == 1200
    assert client.last_call_metrics["thinking_enabled"] is False
    assert client.last_call_metrics["response_schema_removed_keywords"] > 0


def test_vllm_schema_adapter_removes_validation_only_keywords():
    source = {
        "type": "array",
        "uniqueItems": True,
        "contains": {"type": "string"},
        "minContains": 1,
        "maxContains": 2,
        "items": {"type": "string"},
    }

    adapted = llm._vllm_compatible_output_schema(source)

    assert adapted == {"type": "array", "items": {"type": "string"}}
    assert source["uniqueItems"] is True


def test_generated_qwen_notebook_contract_and_payload():
    builder = _load_builder()
    notebook = builder.build()
    header_source = notebook["cells"][0]["source"]
    unpack_source = notebook["cells"][2]["source"]
    common_source = notebook["cells"][3]["source"]
    phase_b_source = notebook["cells"][4]["source"]
    phase_a_source = notebook["cells"][5]["source"]
    all_source = "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])

    assert notebook["metadata"]["kaggle"]["accelerator"] == "nvidiaRtx6000"
    assert "# ARC Prize 2026 - LCLD Qwen" in header_source
    assert "Creative Commons Attribution 4.0 International License" in header_source
    assert "gpt-oss" not in all_source.lower()
    assert "driessmit1/arc3-vllm-h100-wheelhouse-v3" in common_source
    assert "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot" in common_source
    assert "QWEN_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'" in common_source
    assert "VLLM_MAX_MODEL_LEN = 98304" in common_source
    assert "QWEN_MAX_INPUT_TOKENS = 65536" in common_source
    assert "QWEN_MAX_OUTPUT_TOKENS = 12288" in common_source
    assert "VLLM_MAX_NUM_SEQS = 1" in common_source
    assert "'--tool-call-parser', 'qwen3_coder'" in common_source
    assert "'--reasoning-parser', 'qwen3'" in common_source
    assert "'--enable-prefix-caching'" in common_source
    assert "'chat_template_kwargs': {'enable_thinking': False}" in common_source
    assert "def _assert_expected_cuda_gpu():" in common_source
    assert "def gateway_handshake_or_die():" in common_source
    assert "Kaggle gateway did not become ready within 600s" in common_source
    assert "'curl'" not in common_source
    assert "python-dotenv" not in all_source

    assert "start_model_server=False" in phase_a_source
    assert "start_vllm_server(" not in phase_a_source
    assert "phase_b_model_smoke_or_die()" in phase_b_source
    assert phase_b_source.index("phase_b_model_smoke_or_die()") < phase_b_source.index("arcade.create_scorecard()")
    assert "initial_reset_pending = True" in phase_b_source
    assert "while accepted_actions < max_actions:" in phase_b_source
    assert "LCLD_MAX_ACTIONS_PER_GAME', '200'" in phase_b_source
    assert "LCLD_MAX_ACTIONS_PER_LEVEL', '0'" in phase_b_source
    assert "LCLD_MAX_LEVEL_ATTEMPTS', '4'" in phase_b_source
    assert "LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '6000'" in phase_b_source
    assert ".to_parquet(" not in phase_b_source
    assert "'harness_telemetry': telemetry" in phase_b_source
    assert "if qwen_calls <= 0 and levels_completed <= 0:" in phase_b_source
    assert "json.dumps(payload, indent=2, ensure_ascii=False, default=str)" in phase_b_source
    assert "torch.cuda.empty_cache()" not in phase_b_source

    game_loop_start = phase_b_source.index("[Phase B] starting game")
    game_error_start = phase_b_source.index("            except Exception as exc:", game_loop_start)
    game_error_end = phase_b_source.index("            result =", game_error_start)
    assert "\n                raise\n" in phase_b_source[game_error_start:game_error_end]

    close_call = phase_b_source.index("        _close_shared_scorecard()")
    parquet_validation = phase_b_source.index("        parquet_summary = _validate_phase_b_submission_parquet()")
    assert close_call < parquet_validation
    fatal_start = phase_b_source.index("    except BaseException as exc:")
    fatal_end = phase_b_source.index("else:\n    if RERUN_ENV_TRUE", fatal_start)
    fatal_source = phase_b_source[fatal_start:fatal_end]
    assert "_close_shared_scorecard()" not in fatal_source
    assert "if submission_path.exists():" in fatal_source
    assert "submission_path.unlink()" in fatal_source
    assert "scorecard_abandoned_without_close" in fatal_source
    assert fatal_source.index("submission_path.unlink()") < fatal_source.index("_vllm_log_tail(30000)")

    payload_bytes = _embedded_payload(unpack_source)
    with zipfile.ZipFile(io.BytesIO(payload_bytes)) as archive:
        names = set(archive.namelist())
        backend_source = archive.read("Code/v8_agent/llm.py").decode("utf-8")
        kaggle_agent_source = archive.read("Code/kaggle_agent.py").decode("utf-8")
        memory_source = archive.read("Code/v8_agent/memory.py").decode("utf-8")

    assert "def _call_vllm_once" in backend_source
    assert '"enable_thinking": False' in backend_source
    assert "telemetry = dict(session)" in kaggle_agent_source
    assert "attempt_total_execution" in memory_source
    assert not any(name.endswith("config.old") for name in names)
    assert not any(name.startswith("Code/agent/") for name in names)
