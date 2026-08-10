from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import gc
import json
import os
import pathlib
import sys
import threading
import time


def _env_true(name):
    return os.getenv(name, '').strip().lower() in {'1', 'true'}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


MARKER = os.environ.get(
    'LCLD_BUILD_MARKER',
    'ARC_V8_3_QWEN36_27B_FP8_STABLE_ARCADE_LIFECYCLE',
)
VLLM_MAX_NUM_SEQS = max(1, int(os.environ.get('LCLD_VLLM_MAX_NUM_SEQS', '5')))
working_root = pathlib.Path(os.environ.get('LCLD_WORKING_ROOT', '/kaggle/working')).resolve()
working_root.mkdir(parents=True, exist_ok=True)
submission_path = working_root / 'submission.parquet'
RERUN_ENV_TRUE = _env_true('KAGGLE_IS_COMPETITION_RERUN')
IS_PHASE_B_CANDIDATE = RERUN_ENV_TRUE and _env_true('LCLD_GAMEPLAY_CHILD')


def _vllm_log_tail(limit=12000):
    # Bounded fallback only; gameplay never reads the full server log.
    log_path = working_root / 'vllm-qwen36.log'
    if not log_path.is_file():
        return ''
    try:
        byte_limit = max(1, int(limit))
        with log_path.open('rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - byte_limit), os.SEEK_SET)
            return fh.read(byte_limit).decode('utf-8', errors='replace')
    except (OSError, ValueError):
        return ''


print('=== LCLD Phase B isolated gameplay child ===', flush=True)
print('RERUN_ENV_TRUE =', RERUN_ENV_TRUE, flush=True)
print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)

if IS_PHASE_B_CANDIDATE:
    phase_started = time.monotonic()
    trace_path = working_root / 'lcld_direct_agent_trace.log'
    result_path = working_root / 'lcld_competition_scorecard_results.json'
    failure_path = working_root / 'lcld_phase_b_failure.json'
    arcade = None
    scorecard_id = None
    scorecard_closed = False
    scorecard_close_attempted = False
    scorecard_close_disposition = 'not_attempted'
    scorecard_close_error = ''
    scorecard_close_lock = threading.Lock()
    gateway_make_reset_event = threading.Event()
    gateway_make_reset_lock = threading.Lock()
    gateway_make_reset_count = 0
    accepted_gateway_action_event = threading.Event()
    accepted_gateway_action_lock = threading.Lock()
    accepted_gateway_action_count = 0
    env_infos = []
    results_by_index = {}

    for stale_path in (trace_path, result_path, failure_path):
        if stale_path.exists():
            print('Removing stale Phase-B artifact:', stale_path, stale_path.stat().st_size, 'bytes', flush=True)
            stale_path.unlink()

    def _trace(event, **fields):
        # Contest reruns expose only the final status. Keep tracing disabled so
        # diagnostics cannot consume disk, memory, or alter control flow.
        del event, fields
        return None


    def _record_gateway_action():
        global accepted_gateway_action_count
        with accepted_gateway_action_lock:
            accepted_gateway_action_count += 1
            accepted_gateway_action_event.set()

    def _record_gateway_make_reset():
        global gateway_make_reset_count
        with gateway_make_reset_lock:
            gateway_make_reset_count += 1
            gateway_make_reset_event.set()

    def _gateway_activity_count():
        with gateway_make_reset_lock:
            make_count = int(gateway_make_reset_count)
        with accepted_gateway_action_lock:
            action_count = int(accepted_gateway_action_count)
        return make_count + action_count

    def _compact_harness_telemetry(telemetry):
        source = dict(telemetry or {})
        allowed = (
            'qwen_calls_this_game',
            'qwen_primary_calls_by_level',
            'qwen_coordinate_calls_by_level',
            'qwen_reserve_calls_by_level',
            'qwen_total_calls_by_level',
            'level_attempt_index_by_level',
            'action_count_by_level',
            'max_level_attempts',
            'terminal_level_limit',
            'failed_memory_count',
            'irrelevant_memory_count',
            'object_applicability_memory_count',
            'game_over_reset_count',
            'pending_official_transition',
            'current_attempt_index',
            'attempt_history_count',
        )
        compact = {key: source.get(key) for key in allowed if key in source}
        compact['level_attempt_record_count'] = len(source.get('level_attempt_records') or [])
        return compact

    def _result_log_summary(result):
        telemetry = dict(result.get('telemetry_summary') or {})
        return {
            key: result.get(key)
            for key in (
                'game_id', 'status', 'action_count', 'proposed_action_count',
                'rejected_action_count', 'game_over_reset_count',
                'levels_completed', 'final_state', 'stop_reason',
                'elapsed_seconds', 'error_type', 'error',
            )
        } | {
            'qwen_calls_this_game': int(telemetry.get('qwen_calls_this_game', 0) or 0),
            'level_attempt_record_count': int(telemetry.get('level_attempt_record_count', 0) or 0),
        }

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
            frame = getattr(raw, 'frame', None)
            if frame is not None:
                converted_frame = []
                for row in frame:
                    converted_frame.append(row.tolist() if hasattr(row, 'tolist') else list(row))
                return FrameData(
                    game_id=getattr(raw, 'game_id', ''),
                    frame=converted_frame,
                    state=getattr(raw, 'state', None),
                    levels_completed=getattr(raw, 'levels_completed', 0),
                    win_levels=getattr(raw, 'win_levels', 0),
                    guid=getattr(raw, 'guid', None),
                    full_reset=getattr(raw, 'full_reset', False),
                    available_actions=getattr(raw, 'available_actions', ()),
                )
        except Exception as exc:
            _trace('frame_validation_fallback', exc_type=type(exc).__name__, error=str(exc)[:1000])
        # The direct agent only needs the standard FrameData-like attributes.
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

    def _state(frame):
        return _state_name(getattr(frame, 'state', ''))

    def _terminal_reason(frame):
        state_name = _state(frame)
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

    def _observation(frame, frame_index, game_id):
        world_json = frame_to_world_json(frame)
        state_name = _state(frame)
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
        }
        for key in ('levels_completed', 'win_levels'):
            if hasattr(frame, key):
                metadata[key] = getattr(frame, key)
        return {
            'frame': world_json['grid'],
            'grid': _frame_grid_to_2d(world_json['grid']),
            'metadata': metadata,
        }

    def _direct_config():
        config = default_config()
        config.update({
            'allow_in_memory_env': True,
            'environment_adapter': None,
            'external_action_effect_research': True,
            'action_effect_exploration_before_qwen': True,
            'qwen_context_tokens': int(os.environ.get('ARC_QWEN_CONTEXT_TOKENS', '131072')),
            'qwen_minimum_acceptance_context_tokens': 65536,
            'qwen_max_input_tokens': int(os.environ.get('ARC_QWEN_MAX_INPUT_TOKENS', '65536')),
            'qwen_max_output_tokens': int(os.environ.get('ARC_QWEN_MAX_OUTPUT_TOKENS', '49152')),
            'qwen_enable_thinking': True,
            'qwen_reasoning_mode': 'on',
            'qwen_reasoning_budget_tokens': int(os.environ.get('ARC_QWEN_REASONING_BUDGET_TOKENS', '32000')),
            'qwen_temperature': 0.6,
            'qwen_top_p': 0.95,
            'qwen_top_k': 20,
            'qwen_presence_penalty': 0.0,
            'qwen_strict_required': True,
            'qwen_timeout_seconds': int(os.environ.get('ARC_QWEN_TIMEOUT_SECONDS', '800')),
            'llm_timeout_seconds': int(os.environ.get('ARC_LLM_TIMEOUT_SECONDS', '800')),
            'action_selection_timeout_s': 5000.0,
            'major_cycle_wall_clock_budget_seconds': 5000,
            'total_game_wall_clock_limit_seconds': 5000,
            'max_level_attempts': int(os.environ.get('LCLD_MAX_LEVEL_ATTEMPTS', '4')),
            'max_actions_per_level': int(os.environ.get('LCLD_MAX_ACTIONS_PER_LEVEL', '500')),
        })
        return config

    def _cleanup_delegate(delegate):
        cleanup = getattr(delegate, '_cleanup_old_session', None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception as exc:
                _trace('delegate_cleanup_warning', exc_type=type(exc).__name__, error=str(exc)[:1000])
        for attr in ('last_session', 'last_observation', 'last_pipeline_candidate', 'last_native_action', '_adapter_checkpoint'):
            if hasattr(delegate, attr):
                try:
                    setattr(delegate, attr, None)
                except Exception:
                    pass

    def _close_environment(env):
        if env is None:
            return
        close = getattr(env, 'close', None)
        if callable(close):
            try:
                close()
            except BaseException:
                pass
        session = getattr(env, '_session', None)
        session_close = getattr(session, 'close', None)
        if callable(session_close):
            try:
                session_close()
            except BaseException:
                pass

    class DirectGameFailure(RuntimeError):
        def __init__(self, message, *, metrics):
            super().__init__(message)
            self.metrics = dict(metrics)

    def _run_direct_game(env, game_id, initial_frame, abort_event=None):
        config = _direct_config()
        delegate = ARC_AGI_Agent(config)
        max_actions = max(1, int(os.getenv('LCLD_MAX_ACTIONS_PER_GAME', '500')))
        game_wall_limit = max(0.0, float(os.getenv('LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '5000')))
        started = time.monotonic()
        accepted_actions = 0
        proposed_actions = 0
        rejected_actions = 0
        game_over_resets = 0
        frame_index = 0
        latest = _frame_data(initial_frame)
        stop_reason = ''

        _trace(
            'direct_agent_init',
            game_id=game_id,
            initial_state=_state(latest),
            initial_guid=getattr(latest, 'guid', None),
            max_actions=max_actions,
        )

        try:
            while accepted_actions < max_actions:
                if abort_event is not None and abort_event.is_set():
                    stop_reason = 'parallel_abort'
                    break
                stop_reason = _terminal_reason(latest)
                if stop_reason:
                    break
                if game_wall_limit > 0 and time.monotonic() - started >= game_wall_limit:
                    stop_reason = 'game_wall_clock_limit'
                    break

                state_name = _state(latest)
                observation = _observation(latest, frame_index, game_id)
                try:
                    if state_name == 'GAME_OVER':
                        native_action = delegate.reset_after_game_over(observation, config)
                        game_over_resets += 1
                    else:
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
                        state=_state(latest),
                        accepted_action_count=accepted_actions,
                    )
                    break
                action_id, action_data, reasoning = arcade_step_args(native_action)

                proposed_actions += 1
                _trace(
                    'gateway_step_proposed',
                    game_id=game_id,
                    proposed_action_count=proposed_actions,
                    accepted_action_count=accepted_actions,
                    action=str(getattr(action_id, 'name', action_id)),
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
                        action=str(getattr(action_id, 'name', action_id)),
                        exc_type=type(exc).__name__,
                        error=str(exc)[:2000],
                    )
                    raise

                # Count only after the gateway returned a non-null, usable next frame.
                accepted_actions += 1
                _record_gateway_action()
                frame_index += 1
                latest = next_frame
                _trace(
                    'gateway_step_accepted',
                    game_id=game_id,
                    proposed_action_count=proposed_actions,
                    accepted_action_count=accepted_actions,
                    rejected_action_count=rejected_actions,
                    action=str(getattr(action_id, 'name', action_id)),
                    state_after=_state(latest),
                    guid_after=getattr(latest, 'guid', None),
                    levels_completed=getattr(latest, 'levels_completed', 0),
                )

            if not stop_reason:
                stop_reason = 'max_actions' if accepted_actions >= max_actions else 'loop_exit'
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

    def _write_results(status, results, game_count):
        payload = {
            'marker': MARKER,
            'status': status,
            'created_at_utc': _utc_now(),
            'execution_path': 'isolated_child_parallel_games_direct_ARC_AGI_Agent_act_to_env_step',
            'game_concurrency': min(
                max(1, int(os.environ.get('LCLD_GAME_CONCURRENCY', '5'))),
                max(1, int(game_count)),
            ),
            'vllm_max_num_seqs': int(VLLM_MAX_NUM_SEQS),
            'qwen_timeout_seconds': int(os.environ.get('ARC_QWEN_TIMEOUT_SECONDS', '800')),
            'game_wall_clock_limit_seconds': int(float(
                os.environ.get('LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '5000')
            )),
            'max_level_attempts': int(os.environ.get('LCLD_MAX_LEVEL_ATTEMPTS', '4')),
            'max_actions_per_game': int(os.environ.get('LCLD_MAX_ACTIONS_PER_GAME', '500')),
            'max_actions_per_level': int(os.environ.get('LCLD_MAX_ACTIONS_PER_LEVEL', '500')),
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
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n',
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

    try:
        (working_root / 'arc_phase_marker.txt').write_text(
            f'marker={MARKER}\n'
            'phase=PHASE_B_DIRECT_ARCADE_STARTED\n'
            f'KAGGLE_IS_COMPETITION_RERUN={os.getenv("KAGGLE_IS_COMPETITION_RERUN")!r}\n',
            encoding='utf-8',
        )

        (working_root / 'server_recording').mkdir(parents=True, exist_ok=True)
        for required_env in ('ARC_BASE_URL', 'ARC_API_KEY'):
            if not os.environ.get(required_env):
                raise RuntimeError('competition environment key is missing: ' + required_env)

        code_dir = pathlib.Path(os.environ['ARC_AGENT_CODE_DIR']).resolve()
        # Insert Code first then src so src has final import precedence.
        for import_root in (code_dir, (code_dir / 'src').resolve()):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))

        import arc_agi
        from kaggle_agent import ARC_AGI_Agent, arcade_step_args
        from submission import default_config, frame_to_world_json, _frame_grid_to_2d, _state_name

        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=os.environ['ARC_BASE_URL'],
            environments_dir='',
        )
        env_infos = list(arcade.available_environments)
        if not env_infos:
            raise RuntimeError('Kaggle gateway returned no available environments')
        scorecard_id = arcade.create_scorecard()
        if not scorecard_id:
            raise RuntimeError('competition Arcade returned an empty scorecard id')
        _trace('competition_scorecard_opened', scorecard_id=scorecard_id, game_count=len(env_infos))

        print(
            '[Phase B] direct ARC_AGI_Agent scoring: '
            f'{len(env_infos)} environments; no MyAgent/framework loop; '
            'all games share one explicit competition scorecard',
            flush=True,
        )

        wall_limit = max(0, int(os.getenv('LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS', '30600')))
        stop_margin = max(0, int(os.getenv('LCLD_COMPETITION_STOP_MARGIN_SECONDS', '60')))
        soft_deadline = phase_started + wall_limit - stop_margin if wall_limit > 0 else None
        game_concurrency = min(
            max(1, int(os.getenv('LCLD_GAME_CONCURRENCY', '5'))),
            max(1, int(VLLM_MAX_NUM_SEQS)),
            max(1, len(env_infos)),
        )
        if game_concurrency != min(max(1, int(VLLM_MAX_NUM_SEQS)), max(1, len(env_infos))):
            raise RuntimeError(
                'competition worker count must equal active vLLM sequence capacity: '
                f'workers={game_concurrency} max_num_seqs={VLLM_MAX_NUM_SEQS}'
            )
        abort_event = threading.Event()

        # Restore the fresh-kernel gateway ordering exactly: every Arcade.make()
        # and its implicit RESET completes serially, and the initial frame is
        # captured, before any gameplay worker can issue env.step(ACTION/RESET).
        prepared_games = []
        print(
            f'[Phase B] serially preparing {len(env_infos)} environments before '
            f'launching {game_concurrency} gameplay workers',
            flush=True,
        )
        for index, env_info in enumerate(env_infos):
            game_id = _game_id(env_info)
            env = None
            started = time.monotonic()
            try:
                env = arcade.make(game_id, scorecard_id=scorecard_id)
                if env is None:
                    raise RuntimeError('Arcade.make returned None for ' + game_id)
                _record_gateway_make_reset()
                initial_frame = _current_frame(env)
                prepared_games.append((index, game_id, env, initial_frame))
                print(
                    f'[Phase B] prepared game {index + 1}/{len(env_infos)}: {game_id}',
                    flush=True,
                )
            except Exception as exc:
                _close_environment(env)
                results_by_index[index] = {
                    'game_id': game_id,
                    'status': 'failed',
                    'action_count': 0,
                    'proposed_action_count': 0,
                    'rejected_action_count': 0,
                    'game_over_reset_count': 0,
                    'levels_completed': 0,
                    'final_state': '',
                    'final_guid': '',
                    'stop_reason': 'serial_environment_preparation_failure',
                    'elapsed_seconds': round(time.monotonic() - started, 3),
                    'error_type': type(exc).__name__,
                    'error': str(exc)[:2000],
                }
                print(
                    f'[Phase B] serial preparation failed for {game_id}: '
                    f'{type(exc).__name__}: {exc}',
                    flush=True,
                )

        scheduled_games = list(prepared_games)
        _trace(
            'serial_environments_prepared',
            requested_game_count=len(env_infos),
            prepared_game_count=len(scheduled_games),
            preparation_failure_count=len(results_by_index),
            worker_count=game_concurrency,
            vllm_max_num_seqs=VLLM_MAX_NUM_SEQS,
        )
        if results_by_index:
            _write_results(
                'serial_environment_preparation_complete',
                [results_by_index[key] for key in sorted(results_by_index)],
                len(env_infos),
            )

        def _play_competition_game(index, game_id, env, initial_frame):
            started = time.monotonic()
            status = 'completed'
            error_type = ''
            error_text = ''
            metrics = {
                'action_count': 0,
                'proposed_action_count': 0,
                'rejected_action_count': 0,
                'game_over_reset_count': 0,
                'levels_completed': 0,
                'final_state': '',
                'final_guid': '',
                'stop_reason': '',
            }

            print(
                f'[Phase B] starting prepared game {index + 1}/{len(env_infos)}: {game_id} '
                f'worker={threading.current_thread().name}',
                flush=True,
            )
            try:
                metrics = _run_direct_game(
                    env,
                    game_id,
                    initial_frame,
                    abort_event=abort_event,
                )
                telemetry = metrics.get('telemetry_summary') if isinstance(metrics, dict) else None
                qwen_calls = int((telemetry or {}).get('qwen_calls_this_game', 0) or 0)
                levels_completed = int(metrics.get('levels_completed', 0) or 0)
                if qwen_calls <= 0 and levels_completed <= 0:
                    raise RuntimeError(
                        'game completed without a Qwen call or observed level progress: ' + game_id
                    )
            except Exception as exc:
                status = 'failed'
                error_type = type(exc).__name__
                error_text = str(exc)
                failure_metrics = getattr(exc, 'metrics', None)
                if isinstance(failure_metrics, dict):
                    metrics.update(failure_metrics)
                print(
                    f'[Phase B] game {game_id} failed after {time.monotonic() - started:.1f}s: '
                    f'{error_type}: {error_text}',
                    flush=True,
                )
                _trace(
                    'game_failure_isolated',
                    game_id=game_id,
                    exc_type=error_type,
                    error=error_text[:2000],
                    accepted_action_count=int(metrics.get('action_count', 0) or 0),
                )
            finally:
                _close_environment(env)

            return {
                'game_id': game_id,
                'status': status,
                **metrics,
                'elapsed_seconds': round(time.monotonic() - started, 3),
                'error_type': error_type,
                'error': error_text[:2000],
            }

        print(
            f'[Phase B] launching {len(scheduled_games)} prepared games with '
            f'{game_concurrency} worker threads and vLLM max-num-seqs={VLLM_MAX_NUM_SEQS}',
            flush=True,
        )
        executor = ThreadPoolExecutor(max_workers=game_concurrency, thread_name_prefix='lcld-game')
        pending = {}
        next_prepared = 0
        deadline_reached = False
        try:
            while next_prepared < len(scheduled_games) and len(pending) < game_concurrency:
                index, game_id, env, initial_frame = scheduled_games[next_prepared]
                pending[executor.submit(
                    _play_competition_game,
                    index,
                    game_id,
                    env,
                    initial_frame,
                )] = (index, game_id)
                next_prepared += 1

            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    index, game_id = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(
                            f'[Phase B] isolated worker-boundary failure for {game_id}: '
                            f'{type(exc).__name__}: {exc}',
                            flush=True,
                        )
                        result = {
                            'game_id': game_id,
                            'status': 'failed',
                            'action_count': 0,
                            'proposed_action_count': 0,
                            'rejected_action_count': 0,
                            'game_over_reset_count': 0,
                            'levels_completed': 0,
                            'final_state': '',
                            'final_guid': '',
                            'stop_reason': 'worker_boundary_exception',
                            'elapsed_seconds': 0.0,
                            'error_type': type(exc).__name__,
                            'error': str(exc)[:2000],
                        }
                    results_by_index[index] = result
                    results = [results_by_index[key] for key in sorted(results_by_index)]
                    progress = _write_results('in_progress', results, len(env_infos))
                    print(
                        '[Phase B] game result:', _result_log_summary(result),
                        'aggregate_accepted_actions=', progress['total_actions'],
                        'aggregate_rejected_actions=', progress['total_rejected_actions'],
                        flush=True,
                    )

                while next_prepared < len(scheduled_games) and len(pending) < game_concurrency:
                    if soft_deadline is not None and time.monotonic() >= soft_deadline:
                        deadline_reached = True
                        break
                    index, game_id, env, initial_frame = scheduled_games[next_prepared]
                    pending[executor.submit(
                        _play_competition_game,
                        index,
                        game_id,
                        env,
                        initial_frame,
                    )] = (index, game_id)
                    next_prepared += 1
        except BaseException:
            abort_event.set()
            for future in pending:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            # Close any serially prepared environment that was never submitted.
            for _, _, env, _ in scheduled_games[next_prepared:]:
                _close_environment(env)
        gc.collect()

        if next_prepared < len(scheduled_games):
            deadline_reached = True
            for index, game_id, env, initial_frame in scheduled_games[next_prepared:]:
                results_by_index[index] = {
                    'game_id': game_id,
                    'status': 'skipped_global_deadline',
                    'action_count': 0,
                    'proposed_action_count': 0,
                    'rejected_action_count': 0,
                    'game_over_reset_count': 0,
                    'levels_completed': 0,
                    'final_state': '',
                    'final_guid': '',
                    'stop_reason': 'global_deadline',
                    'elapsed_seconds': 0.0,
                    'error_type': '',
                    'error': '',
                }
        if deadline_reached:
            print('[Phase B] soft wall-clock deadline reached; no new games submitted', flush=True)
        results = [results_by_index[key] for key in sorted(results_by_index)]
        final_payload = _write_results('games_attempted_competition_scorecard_open', results, len(env_infos))
        if final_payload['attempted_games'] <= 0:
            raise RuntimeError('no Kaggle environments were attempted')
        if accepted_gateway_action_count <= 0:
            raise RuntimeError(
                'all games failed before an env.step returned a frame; '
                'refusing to finalize a zero-action scorecard'
            )
        _close_shared_scorecard()
        final_payload = _write_results('competition_scorecard_finalization_attempted', results, len(env_infos))

        (working_root / 'arc_phase_marker.txt').write_text(
            f'marker={MARKER}\n'
            'phase=PHASE_B_DIRECT_GAMES_ATTEMPTED_SCORECARD_FINALIZATION_ATTEMPTED\n'
            f'game_count={final_payload["game_count"]}\n'
            f'attempted_games={final_payload["attempted_games"]}\n'
            f'completed_games={final_payload["completed_games"]}\n'
            f'failed_games={final_payload["failed_games"]}\n'
            f'gateway_make_resets={final_payload["gateway_make_reset_count"]}\n'
            f'accepted_actions={final_payload["total_actions"]}\n'
            f'rejected_actions={final_payload["total_rejected_actions"]}\n'
            f'scorecard_close_disposition={final_payload["scorecard_close_disposition"]}\n',
            encoding='utf-8',
        )

        print('=== LCLD PHASE B DIRECT GAMEPLAY COMPLETE ===', flush=True)
        print(json.dumps({
            key: final_payload[key]
            for key in (
                'game_count', 'attempted_games', 'completed_games', 'failed_games',
                'skipped_games', 'total_actions', 'total_proposed_actions',
                'total_rejected_actions', 'total_game_over_resets', 'levels_completed_observed',
                'gateway_make_reset_count', 'gateway_activity_count',
                'scorecard_close_disposition',
            )
        }, indent=2), flush=True)

    except BaseException as exc:
        print('=== PHASE B ORCHESTRATION FAILURE ===', flush=True)
        accepted_count = int(accepted_gateway_action_count)
        make_reset_count = int(gateway_make_reset_count)
        activity_count = _gateway_activity_count()

        if scorecard_id and accepted_count > 0:
            _close_shared_scorecard()

        if accepted_count > 0 and scorecard_id:
            _trace(
                'partial_scorecard_close_started',
                scorecard_id=scorecard_id,
                gateway_make_reset_count=make_reset_count,
                accepted_action_count=accepted_count,
                original_error_type=type(exc).__name__,
            )
            partial_results = [results_by_index[key] for key in sorted(results_by_index)]
            recovery_status = 'partial_scorecard_preserved_after_orchestration_failure'
            try:
                _write_results(recovery_status, partial_results, len(env_infos))
            except Exception as result_exc:
                print(
                    'Partial scorecard result-manifest warning:',
                    type(result_exc).__name__,
                    result_exc,
                    flush=True,
                )
                recovery_manifest = {
                    'marker': MARKER,
                    'status': recovery_status,
                    'created_at_utc': _utc_now(),
                    'game_count': int(len(env_infos)),
                    'attempted_games': int(len(partial_results)),
                    'completed_games': sum(1 for item in partial_results if item.get('status') == 'completed'),
                    'failed_games': sum(1 for item in partial_results if item.get('status') == 'failed'),
                    'skipped_games': 0,
                    'total_actions': sum(int(item.get('action_count', 0) or 0) for item in partial_results),
                    'gateway_make_reset_count': make_reset_count,
                    'gateway_accepted_action_count': accepted_count,
                    'gateway_activity_count': activity_count,
                    'levels_completed_observed': sum(
                        int(item.get('levels_completed', 0) or 0) for item in partial_results
                    ),
                    'results': partial_results,
                    'explicit_scorecard_opened': True,
                    'explicit_scorecard_closed': bool(scorecard_closed),
                    'scorecard_close_attempted': bool(scorecard_close_attempted),
                    'scorecard_close_disposition': scorecard_close_disposition,
                    'scorecard_close_error': scorecard_close_error,
                    'partial_scorecard_preserved': True,
                    'phase_b_parquet_created_by_notebook': False,
                }
                recovery_tmp = result_path.with_suffix('.json.recovery.tmp')
                recovery_tmp.write_text(
                    json.dumps(recovery_manifest, indent=2, ensure_ascii=False, default=str) + '\n',
                    encoding='utf-8',
                )
                recovery_tmp.replace(result_path)

            failure_payload = {
                'marker': MARKER,
                'phase': 'PHASE_B_PARTIAL_SCORECARD_PRESERVED',
                'error_type': type(exc).__name__,
                'error': str(exc),
                'gateway_make_reset_count': make_reset_count,
                'gateway_accepted_action_count': accepted_count,
                'gateway_activity_count': activity_count,
                'scorecard_id_present': True,
                'scorecard_closed': bool(scorecard_closed),
                'scorecard_close_attempted': bool(scorecard_close_attempted),
                'scorecard_close_disposition': scorecard_close_disposition,
                'scorecard_close_error': scorecard_close_error,
                'partial_scorecard_preserved': True,
            }
            try:
                failure_path.write_text(
                    json.dumps(failure_payload, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8',
                )
            except Exception:
                pass
            print(
                'Recovered Phase-B orchestration failure after real gateway activity:',
                json.dumps(failure_payload, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        else:
            try:
                failure_path.write_text(
                    json.dumps({
                        'marker': MARKER,
                        'phase': 'PHASE_B_FATAL_ZERO_ACCEPTED_ACTIONS',
                        'error_type': type(exc).__name__,
                        'error': str(exc),
                        'gateway_make_reset_count': make_reset_count,
                        'gateway_accepted_action_count': accepted_count,
                        'gateway_activity_count': activity_count,
                        'scorecard_id_present': bool(scorecard_id),
                        'scorecard_closed': bool(scorecard_closed),
                        'scorecard_close_attempted': bool(scorecard_close_attempted),
                        'scorecard_close_disposition': scorecard_close_disposition,
                        'scorecard_close_error': scorecard_close_error,
                    }, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8',
                )
            except Exception:
                pass
            print(
                'Fatal Phase-B failure before any env.step returned a frame; '
                'the zero-action scorecard was not finalized.',
                flush=True,
            )
            raise
else:
    if RERUN_ENV_TRUE:
        raise RuntimeError('KAGGLE_IS_COMPETITION_RERUN is present but the Phase-B gate did not run')
    print('Phase B skipped: KAGGLE_IS_COMPETITION_RERUN is absent.', flush=True)
