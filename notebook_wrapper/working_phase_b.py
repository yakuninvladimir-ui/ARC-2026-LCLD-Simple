print('=== LCLD Phase B Tufa-derived competition lifecycle gate ===', flush=True)
print('RERUN_ENV_TRUE =', RERUN_ENV_TRUE, flush=True)
print('GATEWAY_DNS_HINT =', GATEWAY_DNS_HINT, flush=True)
print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)

if IS_PHASE_B_CANDIDATE:
    import gc
    import traceback

    phase_started = time.monotonic()
    trace_path = working_root / 'lcld_direct_agent_trace.log'
    result_path = working_root / 'lcld_competition_scorecard_results.json'
    failure_path = working_root / 'lcld_phase_b_failure.json'
    arcade = None
    scorecard_id = None
    scorecard_closed = False

    for stale_path in (submission_path, trace_path, result_path, failure_path):
        if stale_path.exists():
            print('Removing stale Phase-B artifact:', stale_path, stale_path.stat().st_size, 'bytes', flush=True)
            stale_path.unlink()

    def _trace(event, **fields):
        payload = {
            'time_utc': _utc_now(),
            'event': event,
            **fields,
        }
        with trace_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + '\n')

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
            'qwen_context_tokens': int(os.environ.get('ARC_QWEN_CONTEXT_TOKENS', '98304')),
            'qwen_minimum_acceptance_context_tokens': 65536,
            'qwen_max_input_tokens': int(os.environ.get('ARC_QWEN_MAX_INPUT_TOKENS', '65536')),
            'qwen_max_output_tokens': int(os.environ.get('ARC_QWEN_MAX_OUTPUT_TOKENS', '12288')),
            'qwen_reasoning_mode': 'off',
            'qwen_reasoning_budget_tokens': 0,
            'qwen_strict_required': True,
            'qwen_timeout_seconds': int(os.environ.get('ARC_QWEN_TIMEOUT_SECONDS', '500')),
            'llm_timeout_seconds': int(os.environ.get('ARC_LLM_TIMEOUT_SECONDS', '500')),
            'action_selection_timeout_s': 6000.0,
            'major_cycle_wall_clock_budget_seconds': 6000,
            'total_game_wall_clock_limit_seconds': 6000,
            'max_level_attempts': int(os.environ.get('LCLD_MAX_LEVEL_ATTEMPTS', '4')),
            'max_actions_per_level': int(os.environ.get('LCLD_MAX_ACTIONS_PER_LEVEL', '0')),
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
        gc.collect()

    class DirectGameFailure(RuntimeError):
        def __init__(self, message, *, metrics):
            super().__init__(message)
            self.metrics = dict(metrics)

    def _run_direct_game(env, game_id):
        from arcengine import GameAction

        config = _direct_config()
        delegate = ARC_AGI_Agent(config)
        max_actions = max(1, int(os.getenv('LCLD_MAX_ACTIONS_PER_GAME', '200')))
        game_wall_limit = max(0.0, float(os.getenv('LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS', '6000')))
        started = time.monotonic()
        accepted_actions = 0
        proposed_actions = 0
        rejected_actions = 0
        frame_index = 0
        latest = _current_frame(env)
        stop_reason = ''
        initial_reset_pending = True

        _trace(
            'direct_agent_init',
            game_id=game_id,
            initial_state=_state(latest),
            initial_guid=getattr(latest, 'guid', None),
            max_actions=max_actions,
        )

        try:
            while accepted_actions < max_actions:
                stop_reason = _terminal_reason(latest)
                if stop_reason:
                    break
                if game_wall_limit > 0 and time.monotonic() - started >= game_wall_limit:
                    stop_reason = 'game_wall_clock_limit'
                    break

                state_name = _state(latest)
                if initial_reset_pending:
                    action_id = GameAction.RESET
                    action_data = {}
                    reasoning = {
                        'agent': 'arc_lcld',
                        'source': 'unconditional_initial_reset',
                        'generated_tokens': 0,
                    }
                else:
                    observation = _observation(latest, frame_index, game_id)
                    try:
                        if state_name == 'GAME_OVER':
                            native_action = delegate.reset_after_game_over(observation, config)
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
                frame_index += 1
                latest = next_frame
                initial_reset_pending = False
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
            return {
                'action_count': int(accepted_actions),
                'proposed_action_count': int(proposed_actions),
                'rejected_action_count': int(rejected_actions),
                'levels_completed': int(getattr(latest, 'levels_completed', 0) or 0),
                'final_state': _state(latest),
                'final_guid': str(getattr(latest, 'guid', '') or ''),
                'stop_reason': stop_reason,
            }
        except Exception as exc:
            failure_metrics = {
                'action_count': int(accepted_actions),
                'proposed_action_count': int(proposed_actions),
                'rejected_action_count': int(rejected_actions),
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
            'execution_path': 'direct_ARC_AGI_Agent_act_to_env_step',
            'max_level_attempts': int(os.environ.get('LCLD_MAX_LEVEL_ATTEMPTS', '4')),
            'max_actions_per_level': int(os.environ.get('LCLD_MAX_ACTIONS_PER_LEVEL', '0')),
            'game_count': int(game_count),
            'attempted_games': sum(1 for item in results if item.get('status') != 'skipped_global_deadline'),
            'completed_games': sum(1 for item in results if item.get('status') == 'completed'),
            'failed_games': sum(1 for item in results if item.get('status') == 'failed'),
            'skipped_games': sum(1 for item in results if item.get('status') == 'skipped_global_deadline'),
            'total_actions': sum(int(item.get('action_count', 0) or 0) for item in results),
            'total_proposed_actions': sum(int(item.get('proposed_action_count', 0) or 0) for item in results),
            'total_rejected_actions': sum(int(item.get('rejected_action_count', 0) or 0) for item in results),
            'levels_completed_observed': sum(int(item.get('levels_completed', 0) or 0) for item in results),
            'results': results,
            'scorecard_owner': 'notebook_shared_competition_scorecard',
            'explicit_scorecard_opened': bool(scorecard_id),
            'explicit_scorecard_closed': bool(scorecard_closed),
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
        global scorecard_closed
        if arcade is None or not scorecard_id or scorecard_closed:
            return None
        closed = arcade.close_scorecard(scorecard_id)
        if closed is None:
            raise RuntimeError('competition scorecard close returned None')
        scorecard_closed = True
        _trace('competition_scorecard_closed', scorecard_id=scorecard_id)
        return closed

    def _validate_phase_b_submission_parquet():
        import pandas as pd

        deadline = time.monotonic() + 30.0
        last_error = ''
        while time.monotonic() < deadline:
            if submission_path.is_file():
                try:
                    frame = pd.read_parquet(submission_path)
                    required = {'row_id', 'game_id', 'end_of_game', 'score'}
                    missing = sorted(required - set(frame.columns))
                    if missing:
                        raise RuntimeError('competition parquet missing columns: ' + repr(missing))
                    if frame.empty:
                        raise RuntimeError('competition parquet contains zero rows')
                    if frame['game_id'].isna().all() or frame['score'].isna().all():
                        raise RuntimeError('competition parquet has no usable game_id/score rows')
                    summary = {
                        'rows': int(len(frame)),
                        'bytes': int(submission_path.stat().st_size),
                        'columns': list(frame.columns),
                    }
                    print('LCLD_PHASE_B_PARQUET_VALID=' + json.dumps(summary, sort_keys=True), flush=True)
                    return summary
                except Exception as exc:
                    last_error = f'{type(exc).__name__}: {exc}'
            time.sleep(1.0)
        raise RuntimeError(
            'scorecard closed without a nonempty valid submission.parquet; last_error=' + last_error
        )

    try:
        (working_root / 'arc_phase_marker.txt').write_text(
            f'marker={MARKER}\n'
            'phase=PHASE_B_DIRECT_ARCADE_STARTED\n'
            f'KAGGLE_IS_COMPETITION_RERUN={os.getenv("KAGGLE_IS_COMPETITION_RERUN")!r}\n',
            encoding='utf-8',
        )

        arcade_env_path, manifest = setup_runtime(
            phase='phase_b_qwen_direct_arcade',
            heavy_diagnostics=False,
            qwen_probe=False,
            full_import_sweep=False,
            validate_accelerator=True,
        )
        # vLLM loads asynchronously while gateway setup and imports run. A short
        # model gate runs before scorecard creation; every game then starts with
        # an unconditional RESET before the delegate can request a full inference.
        gateway_handshake_or_die()
        (working_root / 'server_recording').mkdir(parents=True, exist_ok=True)
        for required_env in ('ARC_BASE_URL', 'ARC_API_KEY', 'ONLY_RESET_LEVELS'):
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

        model_smoke = phase_b_model_smoke_or_die()
        if scorecard_id is not None:
            raise RuntimeError('scorecard was opened before the Phase-B model smoke completed')

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

        wall_limit = max(0, int(os.getenv('LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS', '0')))
        stop_margin = max(0, int(os.getenv('LCLD_COMPETITION_STOP_MARGIN_SECONDS', '0')))
        soft_deadline = phase_started + wall_limit - stop_margin if wall_limit > 0 else None
        results = []

        for index, env_info in enumerate(env_infos):
            game_id = _game_id(env_info)
            if soft_deadline is not None and time.monotonic() >= soft_deadline:
                for remaining in env_infos[index:]:
                    results.append({
                        'game_id': _game_id(remaining),
                        'status': 'skipped_global_deadline',
                        'action_count': 0,
                        'proposed_action_count': 0,
                        'rejected_action_count': 0,
                        'levels_completed': 0,
                        'final_state': '',
                        'final_guid': '',
                        'stop_reason': 'global_deadline',
                        'elapsed_seconds': 0.0,
                        'error_type': '',
                        'error': '',
                    })
                print('[Phase B] soft wall-clock deadline reached; stopping before next game', flush=True)
                break

            started = time.monotonic()
            status = 'completed'
            error_type = ''
            error_text = ''
            metrics = {
                'action_count': 0,
                'proposed_action_count': 0,
                'rejected_action_count': 0,
                'levels_completed': 0,
                'final_state': '',
                'final_guid': '',
                'stop_reason': '',
            }

            print(f'[Phase B] starting game {index + 1}/{len(env_infos)}: {game_id}', flush=True)
            try:
                env = arcade.make(game_id, scorecard_id=scorecard_id)
                if env is None:
                    raise RuntimeError('Arcade.make returned None for ' + game_id)
                metrics = _run_direct_game(env, game_id)
                telemetry = metrics.get('harness_telemetry') if isinstance(metrics, dict) else None
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
                print('=== vLLM LOG TAIL AFTER GAME ERROR ===', flush=True)
                print(_vllm_log_tail(30000), flush=True)
                failure_metrics = getattr(exc, 'metrics', None)
                if isinstance(failure_metrics, dict):
                    metrics.update(failure_metrics)
                print(
                    f'[Phase B] game {game_id} failed after {time.monotonic() - started:.1f}s: '
                    f'{error_type}: {error_text}',
                    flush=True,
                )
                traceback.print_exc()
                raise

            result = {
                'game_id': game_id,
                'status': status,
                **metrics,
                'elapsed_seconds': round(time.monotonic() - started, 3),
                'error_type': error_type,
                'error': error_text[:2000],
            }
            results.append(result)
            progress = _write_results('in_progress', results, len(env_infos))
            print(
                '[Phase B] game result:', result,
                'aggregate_accepted_actions=', progress['total_actions'],
                'aggregate_rejected_actions=', progress['total_rejected_actions'],
                flush=True,
            )

        final_payload = _write_results('games_attempted_competition_scorecard_open', results, len(env_infos))
        if final_payload['attempted_games'] <= 0:
            raise RuntimeError('no Kaggle environments were attempted')
        if final_payload['total_actions'] <= 0:
            raise RuntimeError(
                'all games failed before any gateway-accepted action; '
                'action_count is incremented only after env.step returns a frame'
            )
        if not trace_path.exists():
            raise FileNotFoundError('direct-agent trace was not created')
        trace_text = trace_path.read_text(encoding='utf-8', errors='replace')
        for required_marker in ('direct_agent_init', 'gateway_step_proposed', 'gateway_step_accepted'):
            if required_marker not in trace_text:
                raise RuntimeError('required direct-agent trace marker is absent: ' + required_marker)

        _close_shared_scorecard()
        parquet_summary = _validate_phase_b_submission_parquet()
        final_payload = _write_results('competition_scorecard_closed', results, len(env_infos))

        (working_root / 'arc_phase_marker.txt').write_text(
            f'marker={MARKER}\n'
            'phase=PHASE_B_DIRECT_GAMES_ATTEMPTED_SCORECARD_CLOSED\n'
            f'game_count={final_payload["game_count"]}\n'
            f'attempted_games={final_payload["attempted_games"]}\n'
            f'completed_games={final_payload["completed_games"]}\n'
            f'failed_games={final_payload["failed_games"]}\n'
            f'accepted_actions={final_payload["total_actions"]}\n'
            f'rejected_actions={final_payload["total_rejected_actions"]}\n',
            encoding='utf-8',
        )

        print('=== LCLD PHASE B DIRECT GAMEPLAY COMPLETE; COMPETITION SCORECARD CLOSED ===', flush=True)
        print(json.dumps({
            key: final_payload[key]
            for key in (
                'game_count', 'attempted_games', 'completed_games', 'failed_games',
                'skipped_games', 'total_actions', 'total_proposed_actions',
                'total_rejected_actions', 'levels_completed_observed',
            )
        }, indent=2), flush=True)

    except BaseException as exc:
        parquet_deleted = False
        if submission_path.exists():
            try:
                submission_path.unlink()
                parquet_deleted = True
                print('Deleted Phase-B parquet on fatal path:', submission_path, flush=True)
            except OSError as unlink_exc:
                print('Failed to delete Phase-B parquet:', type(unlink_exc).__name__, unlink_exc, flush=True)
        print('=== PHASE B FATAL: vLLM LOG TAIL ===', flush=True)
        print(_vllm_log_tail(30000), flush=True)
        print(
            'Fatal Phase-B failure: abandoning the open scorecard without close_scorecard '
            'so Kaggle cannot finalize a zero-result parquet.',
            flush=True,
        )
        try:
            failure_path.write_text(
                json.dumps({
                    'marker': MARKER,
                    'phase': 'PHASE_B_FATAL_FAILURE',
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                    'scorecard_id_present': bool(scorecard_id),
                    'scorecard_closed': bool(scorecard_closed),
                    'scorecard_abandoned_without_close': bool(scorecard_id and not scorecard_closed),
                    'phase_b_parquet_deleted': parquet_deleted,
                }, indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
        except Exception:
            pass
        raise
else:
    if RERUN_ENV_TRUE:
        raise RuntimeError('KAGGLE_IS_COMPETITION_RERUN is present but the Phase-B gate did not run')
    print('Phase B skipped: KAGGLE_IS_COMPETITION_RERUN is absent.', flush=True)
