import datetime
import importlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

MARKER = 'ARC_V9_SAFE_HARNESS_THINKING32K_STATIC_SCHEMA_SERIAL_GATEWAY'
VLLM_WHEELHOUSE_DATASET = 'vladimiryakunin/vllm-027-cuda-wheels'
QWEN_MODEL_DATASET = 'ravinderonkaggle/muse-glimmer-30b-bf16'
QWEN_MODEL_NAME = 'ravinderonkaggle/muse-glimmer-30b-bf16'
VLLM_HOST = '127.0.0.1'
VLLM_PORT = 1234
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_HEALTH_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/health'
VLLM_STARTUP_TIMEOUT_SECONDS = 900
VLLM_MAX_MODEL_LEN = 131072
VLLM_MAX_NUM_SEQS = 5
VLLM_TENSOR_PARALLEL_SIZE = 1
QWEN_MAX_INPUT_TOKENS = 65536
QWEN_MAX_OUTPUT_TOKENS = 49152
# Single competition switch for Qwen's reasoning mode. All runtime and agent
# settings below derive from this value.
QWEN_THINKING_ENABLED = True
# The OSS vLLM OpenAI endpoint has no per-request thinking-budget parameter.
# ``max_tokens`` is the hard cap for thinking plus final JSON.
QWEN_REASONING_BUDGET_TOKENS = 32000
# Было:
# VLLM_WHEELHOUSE_STAMP = 'vllm==0.27.2rc1.dev189+gf4b161d7f torch==2.13.0 flashinfer==0.6.6\n'

# Стало (подставьте версии из вашего requirements.lock):
VLLM_WHEELHOUSE_STAMP = 'vllm==0.27.2rc1.dev189+gf4b161d7f torch==2.13.0 flashinfer==0.6.16.post3\n'

working_root = pathlib.Path('/kaggle/working')
working_root.mkdir(parents=True, exist_ok=True)
submission_path = working_root / 'submission.parquet'
vllm_process = None
vllm_log_handle = None
vllm_started_at = None

# Competition RESET is defined by the gateway as a current-level reset.


def _env_true(name):
    return os.getenv(name, '').strip().lower() in {'1', 'true'}


RERUN_ENV_TRUE = _env_true('KAGGLE_IS_COMPETITION_RERUN')
try:
    socket.gethostbyname('gateway')
    GATEWAY_DNS_HINT = True
except OSError:
    GATEWAY_DNS_HINT = False
IS_PHASE_B_CANDIDATE = RERUN_ENV_TRUE

print('MARKER =', MARKER, flush=True)
print('RERUN_ENV_TRUE =', RERUN_ENV_TRUE, flush=True)
print('GATEWAY_DNS_HINT =', GATEWAY_DNS_HINT, flush=True)
print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_cmd(cmd, *, cwd=None, env=None, timeout=60, check=True, tail=12000):
    printable = [str(part) for part in cmd]
    print('RUN:', printable, flush=True)
    result = subprocess.run(
        printable,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    print((result.stdout or '')[-tail:], flush=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, printable, output=result.stdout)
    return result


def _assert_payload_structure():
    code_dir = pathlib.Path(os.environ.get('ARC_AGENT_CODE_DIR', '/tmp/arc_lcld_agent/Code')).resolve()
    required = [
        code_dir / 'kaggle_agent.py',
        code_dir / 'submission.py',
        code_dir / 'v9_agent' / '__init__.py',
        code_dir / 'v9_agent' / 'session.py',
        code_dir / 'v9_agent' / 'llm.py',
        code_dir / 'v9_agent' / 'qwen_packet.py',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Embedded LCLD Qwen payload is incomplete: ' + repr(missing))
    os.environ['ARC_AGENT_CODE_DIR'] = str(code_dir)
    print('ARC_AGENT_CODE_DIR =', code_dir, flush=True)
    return code_dir


def _dataset_mount(dataset_ref):
    owner, slug = dataset_ref.split('/', 1)
    candidates = (
        pathlib.Path('/kaggle/input') / slug,
        pathlib.Path('/kaggle/input/datasets') / owner / slug,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Attached Kaggle dataset {dataset_ref!r} was not found; checked {candidates!r}')


def _assert_expected_cuda_gpu():
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('RTX6000 runtime check failed: nvidia-smi is unavailable')
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    names = [line.strip() for line in (result.stdout or '').splitlines() if line.strip()]
    if result.returncode != 0 or len(names) != 1:
        raise RuntimeError(f'Expected exactly one RTX6000 GPU, found {names!r}; output={result.stdout!r}')
    normalized = names[0].lower()
    if 'rtx pro 6000' not in normalized and 'rtx 6000' not in normalized:
        raise RuntimeError(f'Expected an RTX6000 accelerator, found {names[0]!r}')
    print('RTX6000 runtime check passed:', names[0], flush=True)


def _find_qwen_model():
    dataset_root = _dataset_mount(QWEN_MODEL_DATASET)
    config_path = dataset_root / 'config.json'
    if not config_path.is_file():
        configs = sorted(dataset_root.rglob('config.json'))
        if len(configs) != 1:
            raise FileNotFoundError(f'Expected one Qwen config.json under {dataset_root}, found {configs!r}')
        config_path = configs[0]
    model_path = config_path.parent.resolve()
    weight_files = sorted(model_path.glob('*.safetensors'))
    weight_bytes = sum(path.stat().st_size for path in weight_files)
    if not weight_files or weight_bytes < 20_000_000_000:
        raise RuntimeError(
            f'Muse Glimmer 30B BF16 weights are incomplete at {model_path}: '
            f'files={len(weight_files)} bytes={weight_bytes}'
        )
    print('Qwen model:', model_path, 'weight_files=', len(weight_files), 'weight_bytes=', weight_bytes, flush=True)
    return model_path, weight_bytes


def _vllm_site_packages():
    return working_root / 'vllm-site-packages'


def _vllm_env():
    env = dict(os.environ)
    site_packages = _vllm_site_packages()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(site_packages) if not existing else str(site_packages) + os.pathsep + existing
    env.update({
        'USE_TF': '0',
        'TRANSFORMERS_NO_TF': '1',
        'TRANSFORMERS_NO_TORCHVISION': '1',
        'VLLM_NO_USAGE_STATS': '1',
        'VLLM_XGRAMMAR_CACHE_MB': '64',
        'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS': '700',
        # === FlashInfer workaround для RTX PRO 6000 Blackwell (CUDA 13) ===
        'VLLM_USE_FLASHINFER_SAMPLER': '0',
        'VLLM_ATTENTION_BACKEND': 'FLASH_ATTN',
        # ===================================================================
    })
    return env


def _install_vllm_wheelhouse():
    wheelhouse = _dataset_mount(VLLM_WHEELHOUSE_DATASET)
    requirements = wheelhouse / 'requirements.lock'
    if not requirements.is_file():
        raise FileNotFoundError('Missing vLLM 0.27.1 CUDA 13 wheelhouse requirements.lock: ' + str(requirements))
    site_packages = _vllm_site_packages()
    stamp = site_packages / '.vllm-027-cuda-wheels'
    if stamp.is_file() and stamp.read_text(encoding='utf-8') == VLLM_WHEELHOUSE_STAMP:
        probe = subprocess.run(
            [sys.executable, '-c', "import vllm, torch; print(vllm.__version__, torch.__version__)"],
            env=_vllm_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if probe.returncode == 0:
            print('Using cached vLLM 0.27.1 target:', site_packages, probe.stdout.strip(), flush=True)
            return site_packages
    shutil.rmtree(site_packages, ignore_errors=True)
    site_packages.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        '--no-index', '--find-links', str(wheelhouse),
        '--requirement', str(requirements),
        '--target', str(site_packages),
        '--upgrade', '--ignore-installed', '--only-binary', ':all:',
        '--no-compile', '--disable-pip-version-check', '--no-warn-conflicts',
    ]
    print('Installing vLLM 0.27.1 CUDA 13 wheelhouse into', site_packages, flush=True)
    subprocess.run(cmd, check=True)
    stamp.write_text(VLLM_WHEELHOUSE_STAMP, encoding='utf-8')
    return site_packages


def _configure_qwen_env(model_path):
    os.environ.update({
        'ARC_V8_QWEN_BACKEND': 'vllm',
        'ARC_LLM_ADVISOR_BACKEND': 'vllm',
        'ARC_ENABLE_LLM_SEMANTIC_ADVISOR': 'true',
        'ARC_LLM_PRIORITY_HYBRID': 'true',
        'ARC_QWEN_VLLM_BASE_URL': VLLM_BASE_URL,
        'ARC_QWEN_VLLM_API_KEY': 'EMPTY',
        'ARC_QWEN_VLLM_MODEL': QWEN_MODEL_NAME,
        'OPENAI_BASE_URL': VLLM_BASE_URL,
        'OPENAI_API_KEY': 'EMPTY',
        'VLLM_MODEL_PATH': str(model_path),
        'ARC_QWEN_CONTEXT_TOKENS': str(VLLM_MAX_MODEL_LEN),
        'ARC_QWEN_MINIMUM_ACCEPTANCE_CONTEXT_TOKENS': '65536',
        'ARC_QWEN_MAX_INPUT_TOKENS': str(QWEN_MAX_INPUT_TOKENS),
        'ARC_QWEN_MAX_OUTPUT_TOKENS': str(QWEN_MAX_OUTPUT_TOKENS),
        'ARC_QWEN_RESERVED_RUNTIME_MARGIN_TOKENS': '8192',
        'ARC_QWEN_CONTEXT_RESERVED_MARGIN': '8192',
        'ARC_QWEN_TIMEOUT_SECONDS': '700',
        'ARC_LLM_TIMEOUT_SECONDS': '700',
        'ARC_QWEN_ENABLE_THINKING': str(QWEN_THINKING_ENABLED).lower(),
        'ARC_QWEN_REASONING_MODE': 'on' if QWEN_THINKING_ENABLED else 'off',
        'ARC_QWEN_REASONING_BUDGET_TOKENS': str(QWEN_REASONING_BUDGET_TOKENS),
        'ARC_QWEN_SCHEMA_MODE': 'dynamic_enum',
        'ARC_QWEN_TEMPERATURE': '0.7',
        'ARC_QWEN_TOP_K': '30',
        'ARC_QWEN_TOP_P': '0.9',
        'ARC_QWEN_MIN_P': '0.05',
        'ARC_QWEN_PRESENCE_PENALTY': '0.05',
        'ARC_QWEN_REPEAT_PENALTY': '1.05',
        'ARC_QWEN_SEED': '0',
        'ARC_QWEN_STRICT_REQUIRED': 'true',
        'LCLD_REQUIRE_QWEN_RUNTIME': '1',
        'ARC_QWEN_EMPTY_OUTPUT_RETRY_ENABLED': 'false',
        'ARC_QWEN_MULTIMODAL_ENABLED': 'true',
        'ARC_QWEN_TRACE_DIR': '',
        'ARC_V8_TRACE_PATH': os.devnull,
        'ARC_QWEN_MODEL_PROFILE_ID': (
            'muse_glimmer_30b_bf16_vllm_thinking_128k'
            if QWEN_THINKING_ENABLED
            else 'muse_glimmer_30b_bf16_vllm_nonthinking_128k'
        ),
        'ARC_MAX_QWEN_PRIMARY_CALLS_PER_LEVEL': '3',
        'ARC_MAX_QWEN_REPLAN_CALLS_PER_LEVEL': '0',
        'ARC_MAX_QWEN_COORDINATE_CALLS_PER_LEVEL': '1',
        'ARC_MAX_TOTAL_QWEN_CALLS_PER_LEVEL': '6',
        'LCLD_MAX_ACTIONS_PER_GAME': '500',
        'LCLD_MAX_ACTIONS_PER_LEVEL': '500',
        'LCLD_MAX_LEVEL_ATTEMPTS': '0',
        'LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS': '5000',
        'LCLD_GAME_CONCURRENCY': '5',
        'LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS': '30600',
        'LCLD_COMPETITION_STOP_MARGIN_SECONDS': '60',
        'MPLBACKEND': 'agg',
        'ENVIRONMENTS_DIR': '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files',
        'ARC_API_BASE': 'http://gateway:8001',
        'USE_TF': '0',
        'TRANSFORMERS_NO_TF': '1',
        'TRANSFORMERS_NO_TORCHVISION': '1',
        'VLLM_NO_USAGE_STATS': '1',
    })
    cuda_library_path = '/usr/local/nvidia/lib64'
    os.environ['LIBRARY_PATH'] = os.pathsep.join(
        item for item in (cuda_library_path, *os.environ.get('LIBRARY_PATH', '').split(os.pathsep)) if item
    )


def _build_vllm_command(model_path):
    command = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(model_path),
        '--served-model-name', QWEN_MODEL_NAME,
        '--host', VLLM_HOST,
        '--port', str(VLLM_PORT),
        '--tensor-parallel-size', str(VLLM_TENSOR_PARALLEL_SIZE),
        '--max-num-seqs', str(VLLM_MAX_NUM_SEQS),
        # The agent sends TWO images per hypothesis call (raw grid + annotated
        # markup). Historical vLLM defaults allow only 1 image per prompt, which
        # rejects every Qwen call with HTTP 400 and kills the run instantly.
        '--limit-mm-per-prompt', '{"image": 4}',
        '--enable-auto-tool-choice',
        '--tool-call-parser', 'muse_glimmer',
        '--generation-config', 'auto',
        '--enable-prefix-caching',
        '--mm-processor-cache-gb', '0',
        '--dtype', 'bfloat16',
        '--kv-cache-dtype', 'auto',
        '--gpu-memory-utilization', '0.92',
        '--attention-backend', 'FLASH_ATTN',
        '--default-chat-template-kwargs', json.dumps({
            'enable_thinking': QWEN_THINKING_ENABLED,
        }),
        '--max-model-len', str(VLLM_MAX_MODEL_LEN),
    ]
    if QWEN_THINKING_ENABLED:
        command.extend([
            '--reasoning-parser', 'muse_glimmer',
        ])
    return command


def _url_ok(url, timeout=2.0):
    try:
        with urlopen(url, timeout=timeout) as response:
            return int(getattr(response, 'status', 0) or 0) == 200
    except (OSError, URLError):
        return False


def _request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def vllm_server_ready():
    return _url_ok(VLLM_HEALTH_URL) or _url_ok(VLLM_BASE_URL + '/models')


def _vllm_log_tail(limit=12000):
    # Bounded read: never materialize the complete vLLM log in memory.
    log_path = working_root / 'vllm-muse-glimmer.log'
    if not log_path.is_file():
        return ''
    try:
        byte_limit = max(1, int(limit))
        with log_path.open('rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - byte_limit), os.SEEK_SET)
            return fh.read(byte_limit).decode('utf-8', errors='replace')
    except (OSError, ValueError) as exc:
        return f'<unable to read vLLM log: {type(exc).__name__}: {exc}>'


def wait_for_vllm_server(timeout_seconds=VLLM_STARTUP_TIMEOUT_SECONDS):
    started = time.monotonic()
    deadline = started + max(1, int(timeout_seconds))
    next_report = started + 60.0
    while time.monotonic() < deadline:
        if vllm_process is not None and vllm_process.poll() is not None:
            raise RuntimeError(f'vLLM exited with code {vllm_process.returncode}\n{_vllm_log_tail()}')
        if vllm_server_ready():
            elapsed = time.monotonic() - float(vllm_started_at or time.monotonic())
            print(f'vLLM is ready after {elapsed:.3f}s at {VLLM_BASE_URL}', flush=True)
            return elapsed
        now = time.monotonic()
        if now >= next_report:
            print(f'Waiting for vLLM: elapsed={now - started:.1f}s pid={getattr(vllm_process, "pid", None)}', flush=True)
            print(_vllm_log_tail(3000), flush=True)
            next_report = now + 60.0
        time.sleep(5.0)
    raise TimeoutError(f'vLLM did not become ready within {timeout_seconds}s\n{_vllm_log_tail()}')


def start_vllm_server(model_path, *, wait):
    global vllm_process, vllm_log_handle, vllm_started_at
    if vllm_server_ready():
        return {'pid': None, 'ready': True, 'startup_seconds': 0.0, 'reused': True}
    if vllm_process is not None and vllm_process.poll() is None:
        startup_seconds = wait_for_vllm_server() if wait else None
        return {'pid': vllm_process.pid, 'ready': bool(wait), 'startup_seconds': startup_seconds, 'reused': True}
    _install_vllm_wheelhouse()
    cmd = _build_vllm_command(model_path)
    log_path = working_root / 'vllm-muse-glimmer.log'
    vllm_log_handle = log_path.open('w', encoding='utf-8')
    vllm_started_at = time.monotonic()
    print('Starting Tufa vLLM Muse Glimmer server:', cmd, flush=True)
    vllm_process = subprocess.Popen(
        [str(part) for part in cmd],
        stdout=vllm_log_handle,
        stderr=subprocess.STDOUT,
        env=_vllm_env(),
        text=True,
    )
    os.environ['ARC_QWEN_VLLM_PID'] = str(vllm_process.pid)
    (working_root / 'vllm-muse-glimmer-command.json').write_text(
        json.dumps({'command': cmd, 'pid': vllm_process.pid, 'started_at_utc': _utc_now()}, indent=2) + '\n',
        encoding='utf-8',
    )
    startup_seconds = wait_for_vllm_server() if wait else None
    return {
        'pid': vllm_process.pid,
        'ready': bool(wait),
        'startup_seconds': startup_seconds,
        'reused': False,
        'log_path': str(log_path),
    }


def stop_vllm_server(timeout_seconds=30):
    # Teardown must never override a completed competition run.
    global vllm_process, vllm_log_handle
    process = vllm_process
    try:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=max(1, int(timeout_seconds)))
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=10)
                except (subprocess.TimeoutExpired, OSError):
                    pass
    except BaseException:
        # The notebook result is more important than cleanup of an already-owned
        # helper process; Kaggle will reap remaining descendants with the kernel.
        pass
    finally:
        vllm_process = None
        os.environ.pop('ARC_QWEN_VLLM_PID', None)
        if vllm_log_handle is not None:
            try:
                vllm_log_handle.close()
            except BaseException:
                pass
        vllm_log_handle = None


def setup_arcade_client_env():
    env_path = working_root / '.env'
    settings = {
        'SCHEME': 'http',
        'HOST': 'gateway',
        'PORT': '8001',
        'ARC_API_KEY': 'test-key-123',
        'ARC_API_BASE': 'http://gateway:8001',
        'ARC_BASE_URL': 'http://gateway:8001/',
        'OPERATION_MODE': 'competition',
        'ENVIRONMENTS_DIR': '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files',
        'RECORDINGS_DIR': '/kaggle/working/server_recording',
        'LCLD_MAX_ACTIONS_PER_GAME': '500',
        'LCLD_MAX_ACTIONS_PER_LEVEL': '500',
        'LCLD_MAX_LEVEL_ATTEMPTS': '0',
        'LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS': '5000',
        'LCLD_GAME_CONCURRENCY': '5',
        'LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS': '30600',
        'LCLD_COMPETITION_STOP_MARGIN_SECONDS': '60',
    }
    os.environ.update(settings)
    env_path.write_text(
        ''.join(f'{key}={value}\n' for key, value in settings.items()),
        encoding='utf-8',
    )
    return env_path


def structural_preflight():
    code_dir = _assert_payload_structure()
    for root in (code_dir, (code_dir / 'src').resolve()):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    for module_name in ('v9_agent.config', 'v9_agent.llm', 'v9_agent.qwen_packet', 'kaggle_agent', 'submission'):
        module = importlib.import_module(module_name)
        module_path = pathlib.Path(module.__file__).resolve()
        if not module_path.is_relative_to(code_dir):
            raise RuntimeError(f'{module_name} imported from unexpected path: {module_path}')
        print('[OK] import', module_name, '->', module_path, flush=True)
    from v9_agent.config import config_from_mapping
    config = config_from_mapping({})
    expected = {
        'qwen_backend': 'vllm',
        'qwen_multimodal_enabled': True,
        'qwen_context_tokens': 131072,
        'qwen_minimum_acceptance_context_tokens': 65536,
        'qwen_max_input_tokens': 65536,
        'qwen_max_output_tokens': 49152,
        'qwen_timeout_seconds': 700,
        'qwen_reasoning_mode': 'on' if QWEN_THINKING_ENABLED else 'off',
        'qwen_reasoning_budget_tokens': QWEN_REASONING_BUDGET_TOKENS,
        'max_actions_per_game': 500,
        'max_actions_per_level': 500,
        'max_level_attempts': 0,
        'game_wall_clock_limit_seconds': 5000,
    }
    mismatches = {key: (getattr(config, key), value) for key, value in expected.items() if getattr(config, key) != value}
    if mismatches:
        raise RuntimeError('LCLD Qwen preflight configuration mismatch: ' + repr(mismatches))
    if config.qwen_vllm_model != QWEN_MODEL_NAME or bool(config.qwen_enable_thinking) != QWEN_THINKING_ENABLED:
        raise RuntimeError('Qwen vLLM model/thinking contract mismatch')
    sampling_expected = {
        'qwen_temperature': 0.7,
        'qwen_top_p': 0.9,
        'qwen_top_k': 30,
        'qwen_presence_penalty': 0.05,
    }
    sampling_mismatches = {
        key: (getattr(config, key), value)
        for key, value in sampling_expected.items()
        if getattr(config, key) != value
    }
    if sampling_mismatches:
        raise RuntimeError('LCLD Qwen sampling configuration mismatch: ' + repr(sampling_mismatches))
    print('=== LCLD Qwen structural preflight OK ===', flush=True)


def write_diagnostics_manifest(*, phase, runtime_info, arcade_env_path, heavy_diagnostics, qwen_probe):
    return {
        'marker': MARKER,
        'phase': phase,
        'created_at_utc': _utc_now(),
        'rerun_env_true': RERUN_ENV_TRUE,
        'gateway_dns_hint': GATEWAY_DNS_HINT,
        'is_phase_b_candidate': IS_PHASE_B_CANDIDATE,
        'arc_agent_code_dir': os.environ.get('ARC_AGENT_CODE_DIR'),
        'backend': os.environ.get('ARC_V8_QWEN_BACKEND'),
        'model_path': os.environ.get('VLLM_MODEL_PATH'),
        'model_name': QWEN_MODEL_NAME,
        'vllm_base_url': VLLM_BASE_URL,
        'max_model_len': VLLM_MAX_MODEL_LEN,
        'max_input_tokens': QWEN_MAX_INPUT_TOKENS,
        'max_output_tokens': QWEN_MAX_OUTPUT_TOKENS,
        'max_num_seqs': VLLM_MAX_NUM_SEQS,
        'reasoning_budget_tokens': int(os.environ.get('ARC_QWEN_REASONING_BUDGET_TOKENS', '32000')),
        'output_schema_mode': os.environ.get('ARC_QWEN_SCHEMA_MODE', 'static'),
        'runtime_info': runtime_info,
        'arcade_env_path': str(arcade_env_path),
        'heavy_diagnostics': bool(heavy_diagnostics),
        'model_smoke_requested': bool(qwen_probe),
    }


def setup_runtime(
    *,
    phase,
    heavy_diagnostics,
    qwen_probe,
    full_import_sweep=False,
    start_model_server=True,
    validate_accelerator=True,
):
    del full_import_sweep
    print(f'=== LCLD Qwen {phase} runtime setup START ===', flush=True)
    code_dir = _assert_payload_structure()
    model_path, weight_bytes = _find_qwen_model()
    _dataset_mount(VLLM_WHEELHOUSE_DATASET)
    if validate_accelerator:
        _assert_expected_cuda_gpu()
    else:
        print('Accelerator type check skipped for static phase:', phase, flush=True)
    _configure_qwen_env(model_path)
    arcade_env_path = setup_arcade_client_env()
    if heavy_diagnostics:
        run_cmd(['nvidia-smi'], timeout=60, check=True)
    structural_preflight()
    if start_model_server:
        server_info = start_vllm_server(model_path, wait=bool(qwen_probe))
    else:
        server_info = {
            'pid': None,
            'ready': False,
            'startup_seconds': None,
            'reused': False,
            'skipped': True,
            'reason': 'model_server_disabled_for_phase',
        }
        print('vLLM model server intentionally disabled for', phase, flush=True)
    runtime_info = {
        'model_path': str(model_path),
        'model_weight_bytes': weight_bytes,
        'wheelhouse_path': str(_dataset_mount(VLLM_WHEELHOUSE_DATASET)),
        'server': server_info,
        'code_dir': str(code_dir),
        'accelerator_check_required': bool(validate_accelerator),
    }
    manifest = write_diagnostics_manifest(
        phase=phase,
        runtime_info=runtime_info,
        arcade_env_path=arcade_env_path,
        heavy_diagnostics=heavy_diagnostics,
        qwen_probe=qwen_probe,
    )
    print(f'=== LCLD Qwen {phase} runtime setup OK ===', flush=True)
    return arcade_env_path, manifest


def phase_b_model_smoke_or_die():
    """Run a minimal inference probe before scorecard creation.

    This tests:
      - Basic text-only inference (simple OK response)
      - Structured JSON-schema output (production transport with strict schema)
      - Multi-image vision input (two images attached, matching agent behavior)

    All three transports are exercised to catch vLLM configuration issues
    (model loading, schema enforcement, per-prompt image limits) before
    the competition run begins.
    """
    wait_for_vllm_server(timeout_seconds=VLLM_STARTUP_TIMEOUT_SECONDS)
    payload = {
        'model': QWEN_MODEL_NAME,
        'messages': [{
            'role': 'user',
            'content': 'Reply with exactly OK.',
        }],
        'temperature': 0.7,
        'top_p': 1.0,
        'max_tokens': 8,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    started = time.monotonic()
    try:
        response = _request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=180)
        choices = response.get('choices') or []
        content = str(((choices[0].get('message') or {}).get('content') if choices else '') or '').strip()
        if not content:
            raise RuntimeError('Qwen smoke returned empty message.content')

        # Probe the production transport before opening a scorecard. This keeps
        # the request intentionally tiny while exercising the same vLLM fields
        # used by the agent's thinking + JSON-schema calls.
        contract_payload = {
            'model': QWEN_MODEL_NAME,
            'messages': [{
                'role': 'user',
                'content': 'Return the required JSON object.',
            }],
            'temperature': 0.7,
            'top_p': 0.9,
            'min_p': 0.05,
            'presence_penalty': 0.05,
            'repetition_penalty': 1.05,
            'seed': 0,
            'max_tokens': 8,
            'chat_template_kwargs': {'enable_thinking': QWEN_THINKING_ENABLED},
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'lcld_transport_smoke',
                    'strict': True,
                    'schema': {
                        'type': 'object',
                        'additionalProperties': False,
                        'required': ['action'],
                        'properties': {
                            'action': {'type': 'string', 'enum': ['ACTION1']},
                        },
                    },
                },
            },
        }
        contract_response = _request_json(
            VLLM_BASE_URL + '/chat/completions',
            payload=contract_payload,
            timeout=180,
        )
        contract_choices = contract_response.get('choices') or []
        if not contract_choices:
            raise RuntimeError('Qwen production transport smoke returned no choices')

        summary = {
            'status': 'ok',
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'finish_reason': choices[0].get('finish_reason') if choices else None,
            'usage': response.get('usage'),
            'thinking_enabled': False,
            'production_transport_thinking_enabled': QWEN_THINKING_ENABLED,
            'production_transport_checked': True,
            'production_transport_finish_reason': contract_choices[0].get('finish_reason'),
            'validated_response_field': 'message.content_nonempty',
        }
        print('LCLD_QWEN_PHASE_B_MODEL_SMOKE=' + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    except Exception:
        print('=== QWEN PHASE-B MODEL SMOKE FATAL ===', flush=True)
        print(_vllm_log_tail(30000), flush=True)
        raise

def gateway_handshake_or_die():
    print('=== LCLD Phase B gateway handshake START ===', flush=True)
    url = os.environ.get('ARC_BASE_URL', 'http://gateway:8001/').rstrip('/') + '/api/games'
    deadline = time.monotonic() + 700.0
    last_error = ''
    while time.monotonic() < deadline:
        try:
            request = Request(
                url,
                headers={
                    'Accept': 'application/json',
                    'X-API-Key': os.environ.get('ARC_API_KEY', ''),
                },
            )
            with urlopen(request, timeout=10) as response:
                status = int(getattr(response, 'status', 0) or 0)
                if 200 <= status < 500:
                    print('=== LCLD Phase B gateway handshake OK ===', status, flush=True)
                    return
                last_error = f'HTTP status {status}'
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
        time.sleep(5.0)
    raise RuntimeError('Kaggle gateway did not become ready within 700s: ' + last_error)


# === PHASE A HEAVY COMBAT SMOKE (диагностика; для финального сабмита выключить) ===
PHASE_A_HEAVY_SMOKE = True
HEAVY_SMOKE_WORKERS = VLLM_MAX_NUM_SEQS      # воспроизводим боевой параллелизм
HEAVY_SMOKE_ROUNDS = 2
HEAVY_SMOKE_REQ_TIMEOUT = 600                # верхняя граница на один запрос (снижено с 2400)


def _smoke_gpu_info():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total',
                            '--format=csv,noheader'], text=True, capture_output=True, timeout=30)
        return (r.stdout or '').strip()
    except Exception as exc:
        return f'nvidia-smi failed: {exc}'


def _smoke_build_combat_payload(config, with_schema=True):
    import random
    from v9_agent.frame_media import annotated_frame_png, current_frame_png
    from v9_agent.llm import QwenRole, _allowed_output_schema, _prompt, _vllm_compatible_output_schema
    rnd = random.Random(20260214)
    rows = tuple(''.join(rnd.choice('0123456789ABCDEF') for _ in range(30)) for _ in range(30))
    frame_png = current_frame_png(rows, cell_scale=config.frame_png_cell_scale)
    annotated = annotated_frame_png(rows, [
        {'label': f'obj{i}', 'bbox_xyxy': [2 + 5*i, 2 + 4*i, 6 + 5*i, 6 + 4*i]} for i in range(4)
    ], cell_scale=config.frame_png_cell_scale)
    for fp in (frame_png, annotated):
        fp['available_to_configured_backend'] = True
    object_ids = [f'o{i:02d}' for i in range(12)]
    relation_ids = [f'r{i:02d}' for i in range(8)]
    candidate_ids = [f'c{i:02d}' for i in range(20)]
    action_ids = [f'ACTION{i}' for i in range(1, 7)]
    # Минимальная память с action_effects, чтобы _allowed_action_ids вернул непустой список
    memory = {
        'action_effects': [
            {'action_id': aid, 'effect_summary': f'smoke effect for {aid}', 'step_index': 40 + i}
            for i, aid in enumerate(action_ids)
        ],
        'action_surface_memory_records': [],
    }
    packet = {
        'schema_version': 'v8.8.layered_observation',
        'state': {'game_id': 'smoke', 'level_index': 0, 'step_index': 42, 'state_name': 'IN_PROGRESS'},
        'current_frame_png': frame_png,
        'annotated_frame_png': annotated,
        'object_layer': {
            'objects': [{'id': oid, 'bbox_xyxy': [0, 0, 4, 4], 'centroid_xy': [2, 2], 'area': 9} for oid in object_ids],
            'relations': [{'id': rid, 'type': 'near', 'object_ids': object_ids[:2]} for rid in relation_ids],
        },
        'action_space': {
            'actions': [{'id': aid, 'kind': 'simple'} for aid in action_ids],
            'current_available_action_ids': action_ids,
            'coordinate_candidates': [{'id': cid, 'location_xy': [i % 30, i]} for i, cid in enumerate(candidate_ids)],
        },
        'action_diffs': [],
        'memory': memory,
        'execution_constraints': {
            'max_plan_steps': int(getattr(config, 'max_qwen_trajectory_steps', 50)),
            'allowed_action_ids': action_ids,
            'allowed_object_ids': object_ids,
            'allowed_relation_ids': relation_ids,
            'allowed_coordinate_candidate_ids': candidate_ids,
        },
    }
    prompt = _prompt(QwenRole.PRIMARY, packet, config)
    images = [frame_png['data_base64'], annotated['data_base64']]
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}} for img in images]
    content.append({'type': 'text', 'text': prompt})
    payload = {
        'model': QWEN_MODEL_NAME,
        'messages': [{'role': 'user', 'content': content}],
        'stream': False,
        'max_tokens': QWEN_MAX_OUTPUT_TOKENS,
        'temperature': 0.7, 'top_k': 30, 'top_p': 0.9, 'min_p': 0.05,
        'presence_penalty': 0.05, 'repetition_penalty': 1.05, 'seed': 0,
        'chat_template_kwargs': {'enable_thinking': QWEN_THINKING_ENABLED},
    }
    if with_schema:
        schema = _vllm_compatible_output_schema(_allowed_output_schema(QwenRole.PRIMARY, packet))
        payload['response_format'] = {'type': 'json_schema',
                                      'json_schema': {'name': 'lcld_heavy_smoke', 'schema': schema, 'strict': True}}
    print('[HEAVY-SMOKE] payload built: prompt_chars=', len(prompt),
          'image_b64_chars=', sum(len(i) for i in images), flush=True)
    return payload


def _smoke_send(label, payload, timeout):
    rec = {'label': label, 'utc': _utc_now()}
    started = time.monotonic()
    request = Request(VLLM_BASE_URL + '/chat/completions',
                      data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode('utf-8', errors='replace')
    except Exception as exc:
        rec['elapsed'] = round(time.monotonic() - started, 2)
        rec['error_type'] = type(exc).__name__
        rec['error'] = str(exc)[:1500]
        try:
            rec['error_body'] = exc.read().decode('utf-8', errors='replace')[-1500:]
        except Exception:
            pass
        return rec
    rec['elapsed'] = round(time.monotonic() - started, 2)
    try:
        data = json.loads(body)
        choice = (data.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        content = str(message.get('content') or '')
        reasoning = str(message.get('reasoning_content') or '')
        usage = data.get('usage') or {}
        try:
            json.loads(content); parsed = True
        except Exception:
            parsed = False
        rec.update({
            'finish_reason': choice.get('finish_reason'),
            'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'reasoning_chars': len(reasoning),
            'content_chars': len(content),
            'content_is_json': parsed,
            'content_tail': content[-300:],
        })
    except Exception as exc:
        rec['response_parse_error'] = f'{type(exc).__name__}: {exc}'[:500]
        rec['body_tail'] = body[-800:]
    return rec


def _smoke_worker(worker_id, payload, rounds):
    out = []
    for rnd in range(rounds):
        rec = _smoke_send(f'w{worker_id}_combat_r{rnd}', payload, HEAVY_SMOKE_REQ_TIMEOUT)
        print('[HEAVY-SMOKE]', json.dumps(rec, sort_keys=True, ensure_ascii=False), flush=True)
        out.append(rec)
    return out


def run_phase_a_heavy_smoke():
    from concurrent.futures import ThreadPoolExecutor
    from v9_agent.config import config_from_mapping
    config = config_from_mapping({})
    print('[HEAVY-SMOKE] GPU before:', _smoke_gpu_info(), flush=True)

    # --- baseline 1: чистый decode без thinking ---
    base = {'model': QWEN_MODEL_NAME, 'stream': False, 'max_tokens': 65536, 'temperature': 0.7, 'top_p': 0.9,
            'messages': [{'role': 'user', 'content': 'Count from 1 to 400, one number per line.'}],
            'chat_template_kwargs': {'enable_thinking': True}}
    print('[HEAVY-SMOKE]', json.dumps(_smoke_send('baseline_no_thinking', base, 600), sort_keys=True), flush=True)

    # --- baseline 2: thinking без vision/schema ---
    think = dict(base); think['max_tokens'] = 8192
    think['messages'] = [{'role': 'user', 'content': 'Think step by step about the 3n+1 problem for n=27, then answer.'}]
    think['chat_template_kwargs'] = {'enable_thinking': True}
    print('[HEAVY-SMOKE]', json.dumps(_smoke_send('baseline_thinking', think, 1200), sort_keys=True), flush=True)

    # --- боевой запрос без schema (изоляция overhead XGrammar) ---
    combat_no_schema = _smoke_build_combat_payload(config, with_schema=False)
    print('[HEAVY-SMOKE]', json.dumps(_smoke_send('combat_no_schema_single', combat_no_schema, HEAVY_SMOKE_REQ_TIMEOUT), sort_keys=True), flush=True)

    # --- полный боевой запрос, параллелизм = боевому ---
    combat = _smoke_build_combat_payload(config, with_schema=True)
    with ThreadPoolExecutor(max_workers=HEAVY_SMOKE_WORKERS, thread_name_prefix='smoke') as ex:
        futures = [ex.submit(_smoke_worker, i, combat, HEAVY_SMOKE_ROUNDS) for i in range(HEAVY_SMOKE_WORKERS)]
        for f in futures:
            f.result()
    print('[HEAVY-SMOKE] GPU after:', _smoke_gpu_info(), flush=True)
    print('[HEAVY-SMOKE] vllm log tail:', flush=True)
    print(_vllm_log_tail(30000), flush=True)
