print('=== LCLD Phase A Qwen static preflight; model server disabled ===', flush=True)
print('RERUN_ENV_TRUE =', RERUN_ENV_TRUE, flush=True)
print('IS_PHASE_B_CANDIDATE =', IS_PHASE_B_CANDIDATE, flush=True)

if not IS_PHASE_B_CANDIDATE:
    import pandas as pd

    if submission_path.exists():
        raise RuntimeError('NON-RERUN CONTAMINATION: submission.parquet exists before Qwen Phase-A preflight')

    phase_marker_path = working_root / 'arc_phase_marker.txt'
    phase_marker_path.write_text(
        f'marker={MARKER}\n'
        'phase=PHASE_A_QWEN_STATIC_PREFLIGHT_STARTED_NO_MODEL_SERVER\n'
        f'KAGGLE_IS_COMPETITION_RERUN={os.getenv("KAGGLE_IS_COMPETITION_RERUN")!r}\n',
        encoding='utf-8',
    )

    try:
        arcade_env_path, manifest = setup_runtime(
            phase='phase_a_qwen_static_preflight',
            heavy_diagnostics=True,
            qwen_probe=False,
            full_import_sweep=False,
            start_model_server=False,
        )
        server_info = manifest['runtime_info']['server']
        if not server_info.get('skipped') or server_info.get('pid') is not None:
            raise RuntimeError('Phase A unexpectedly started the vLLM model server')

        phase_marker_path.write_text(
            f'marker={MARKER}\n'
            'phase=PHASE_A_QWEN_STATIC_PREFLIGHT_COMPLETE_NO_MODEL_SERVER\n'
            'model_server_started=false\n'
            'model_completion_requested=false\n',
            encoding='utf-8',
        )

        submission = pd.DataFrame(
            data=[['1_0', '1', True, 1]],
            columns=['row_id', 'game_id', 'end_of_game', 'score'],
        )
        submission.to_parquet(submission_path, index=False)
        read_back = pd.read_parquet(submission_path)
        if read_back.empty or int(read_back.iloc[0]['score']) != 1:
            raise RuntimeError('Phase-A dummy submission write/read validation failed')
        print('LCLD_QWEN_PHASE_A_STATIC_SUMMARY=' + json.dumps({
            'status': 'static_preflight_complete',
            'model_server_started': False,
            'model_completion_requested': False,
            'model_name_for_phase_b': QWEN_MODEL_NAME,
            'context_tokens_for_phase_b': VLLM_MAX_MODEL_LEN,
            'max_input_tokens_for_phase_b': QWEN_MAX_INPUT_TOKENS,
            'max_output_tokens_for_phase_b': QWEN_MAX_OUTPUT_TOKENS,
        }, ensure_ascii=False, sort_keys=True), flush=True)
        print('=== LCLD PHASE A STATIC PREFLIGHT OK; MODEL SERVER NOT STARTED ===', flush=True)
        print(read_back.head().to_string(index=False), flush=True)
    except BaseException as exc:
        print('LCLD_QWEN_PHASE_A_STATIC_FATAL=' + json.dumps({
            'error_type': type(exc).__name__,
            'error': str(exc)[:4000],
        }, ensure_ascii=False, sort_keys=True), flush=True)
        raise
else:
    print('Phase-A Qwen static preflight skipped: this is a Phase-B competition rerun.', flush=True)
