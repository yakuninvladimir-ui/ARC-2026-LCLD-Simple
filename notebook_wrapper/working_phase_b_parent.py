print('=== LCLD Phase B isolated gameplay supervisor ===', flush=True)
print('RERUN_ENV_TRUE =', RERUN_ENV_TRUE, flush=True)
print('GATEWAY_DNS_HINT =', GATEWAY_DNS_HINT, flush=True)
print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)

if IS_PHASE_B_CANDIDATE:
    child_result_path = working_root / 'lcld_competition_scorecard_results.json'
    child_failure_path = working_root / 'lcld_phase_b_failure.json'
    child_script_path = pathlib.Path(os.environ['ARC_AGENT_CODE_DIR']).resolve() / 'lcld_competition_child.py'

    try:
        (working_root / 'arc_phase_marker.txt').write_text(
            f'marker={MARKER}\n'
            'phase=PHASE_B_ISOLATED_SUPERVISOR_STARTED\n'
            f'KAGGLE_IS_COMPETITION_RERUN={os.getenv("KAGGLE_IS_COMPETITION_RERUN")!r}\n',
            encoding='utf-8',
        )

        setup_runtime(
            phase='phase_b_qwen_isolated_supervisor',
            heavy_diagnostics=False,
            qwen_probe=False,
            full_import_sweep=False,
            validate_accelerator=True,
        )
        gateway_handshake_or_die()
        for required_env in ('ARC_BASE_URL', 'ARC_API_KEY'):
            if not os.environ.get(required_env):
                raise RuntimeError('competition environment key is missing: ' + required_env)
        if not child_script_path.is_file():
            raise FileNotFoundError('embedded competition child is missing: ' + str(child_script_path))

        # Validate the model before opening the scorecard.
        phase_b_model_smoke_or_die()

        child_env = dict(os.environ)
        child_env.update({
            'LCLD_GAMEPLAY_CHILD': '1',
            'LCLD_BUILD_MARKER': MARKER,
            'LCLD_VLLM_MAX_NUM_SEQS': str(VLLM_MAX_NUM_SEQS),
            'LCLD_WORKING_ROOT': str(working_root),
            # The contest does not expose these files. Avoid their I/O and their
            # failure-amplification paths entirely.
            'ARC_QWEN_TRACE_DIR': '',
            'ARC_V8_TRACE_PATH': os.devnull,
        })
        print(
            '[Phase B] launching isolated gameplay process:',
            sys.executable,
            child_script_path,
            flush=True,
        )
        child = subprocess.run(
            [sys.executable, '-u', str(child_script_path)],
            cwd=working_root,
            env=child_env,
            check=False,
        )

        # The result manifest is the primary child/parent commit record. A late
        # nonzero exit must not invalidate a scorecard that was already closed and
        # committed by the child.
        child_result = None
        if child_result_path.is_file():
            try:
                loaded = json.loads(child_result_path.read_text(encoding='utf-8'))
                if isinstance(loaded, dict):
                    child_result = loaded
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                child_result = None

        if child_result is None:
            failure_summary = ''
            if child_failure_path.is_file():
                try:
                    failure_summary = child_failure_path.read_text(
                        encoding='utf-8', errors='replace'
                    )[-8000:]
                except OSError:
                    failure_summary = ''
            raise RuntimeError(
                f'isolated gameplay process exited with code {child.returncode} '
                f'without a valid result manifest: {failure_summary}'
            )

        accepted_actions = int(child_result.get('gateway_accepted_action_count', 0) or 0)
        make_resets = int(child_result.get('gateway_make_reset_count', 0) or 0)
        gateway_activity = int(
            child_result.get('gateway_activity_count', make_resets + accepted_actions) or 0
        )
        if gateway_activity <= 0:
            raise RuntimeError(
                f'isolated gameplay process exited with code {child.returncode} '
                'without gateway activity'
            )
        if accepted_actions <= 0:
            raise RuntimeError(
                f'isolated gameplay process exited with code {child.returncode} '
                'without an accepted gameplay action; refusing to finalize a zero-action scorecard'
            )

        close_disposition = str(child_result.get('scorecard_close_disposition') or '')
        if not bool(child_result.get('scorecard_close_attempted')):
            raise RuntimeError(
                f'isolated gameplay process exited with code {child.returncode} '
                'without attempting scorecard finalization'
            )
        # The gateway can autonomously finalize an already-active scorecard.
        # At this point gateway_activity has already been verified above, so a
        # 404/409/410 from the explicit close is evidence of that terminal
        # state, not an empty or uncommitted run.
        valid_close_dispositions = {
            'closed',
            'closed_no_payload',
            'missing_or_auto_closed',
        }
        if close_disposition not in valid_close_dispositions:
            raise RuntimeError(
                f'isolated gameplay process exited with code {child.returncode} '
                f'without confirmed scorecard finalization: {close_disposition!r}'
            )

        print('LCLD_ISOLATED_GAMEPLAY_RESULT=' + json.dumps({
            key: child_result.get(key)
            for key in (
                'status', 'game_count', 'attempted_games', 'completed_games',
                'failed_games', 'gateway_make_reset_count', 'gateway_accepted_action_count',
                'gateway_activity_count', 'levels_completed_observed',
                'scorecard_close_attempted', 'scorecard_close_disposition',
            )
        } | {'child_returncode': int(child.returncode)}, ensure_ascii=False, sort_keys=True), flush=True)
        (working_root / 'arc_phase_marker.txt').write_text(
            f'marker={MARKER}\n'
            'phase=PHASE_B_ISOLATED_GAMEPLAY_COMPLETE\n'
            f'child_returncode={child.returncode}\n'
            f'status={child_result.get("status")}\n'
            f'gateway_make_resets={make_resets}\n'
            f'gateway_accepted_actions={accepted_actions}\n'
            f'gateway_activity_count={gateway_activity}\n'
            f'scorecard_close_disposition={close_disposition}\n',
            encoding='utf-8',
        )
    finally:
        # This function is deliberately noexcept: teardown must not override a
        # completed scorecard or a valid child result.
        stop_vllm_server()
else:
    if RERUN_ENV_TRUE:
        raise RuntimeError('KAGGLE_IS_COMPETITION_RERUN is present but the Phase-B gate did not run')
    print('Phase B skipped: KAGGLE_IS_COMPETITION_RERUN is absent.', flush=True)
