# Competition Wrapper Audit - 2026-07-15

## Scope

Audited:

- `D:/download/tufa-labs-duck-harness.ipynb` and the matching Tufa launcher sources under `duck-harness-main`;
- the standard `ARC-AGI-3-Agents` Arcade, action, reset, and scorecard lifecycle;
- `build_notebook.py`, `notebook_wrapper/`, the embedded agent payload, and the generated production notebook.

## Result

The generated notebook follows the required competition lifecycle and the requested Tufa model runtime:

- `KAGGLE_IS_COMPETITION_RERUN` accepts only `1` or `true`, matching Tufa;
- `ONLY_RESET_LEVELS=true` is set before the competition client is imported;
- `arc-agi` is installed only from the official offline competition wheelhouse;
- model and wheelhouse datasets are the Tufa sources;
- vLLM is installed into an isolated target and launched as one persistent OpenAI-compatible server;
- Qwen uses the Tufa server flags, non-thinking request control, and one sequence;
- one explicit scorecard is shared by every game;
- every game is created with that scorecard and starts with one unconditional `RESET`;
- `GAME_OVER` receives one level `RESET` before another agent/model cycle;
- every accepted non-initial action is committed through `observe_action_result()`;
- limits are 500 seconds per Qwen call, 6000 seconds per game, 200 accepted actions per game, and four attempts per level;
- no per-level action limit and no global competition deadline are active.

Intentional differences from Tufa are the requested `98304` model context, `65536` input cap, `12288` output cap, and `--max-num-seqs 1`. Tufa's published default context is `65536`.

## Parquet Failure Policy

Phase A writes the required nonempty one-row dummy parquet only for a non-rerun notebook validation. It does not start vLLM.
Its static preflight does not invoke `nvidia-smi` or enforce the Phase-B RTX6000 type because Kaggle may run Phase A without that accelerator. Phase B performs the strict one-RTX6000 check before model startup.

Phase B has no `to_parquet()` path. Its order is:

1. Remove stale parquet.
2. Start vLLM and pass a real structured-output smoke test.
3. Connect to the gateway and create one scorecard.
4. Play all games.
5. Close the scorecard once.
6. Read back and validate a nonempty standard parquet.

Any Phase-B exception is re-raised. The fatal handler does not call `close_scorecard`; it first removes any parquet artifact and then prints diagnostics. This closes the prior deterministic failure mode in which one accepted initial reset was enough to close a zero-result scorecard after a Qwen timeout.

## Verification

- `python -m pytest tests -q`: 106 passed.
- Production notebook rebuilt successfully: 211529 bytes, 25 embedded Python files.
- All generated code cells and all embedded Python files compile.
- Static order check passes: model smoke -> scorecard create -> game make/step -> normal close -> parquet validation.

## Residual Risks

- The real RTX6000, attached datasets, gateway, and Qwen FP8 model cannot be executed locally. The Phase-B smoke is therefore the first hardware validation.
- A hard `SIGKILL`, host OOM, or kernel loss cannot run Python cleanup. No notebook code can guarantee cleanup in that case. The notebook minimizes this risk by not creating a scorecard until model readiness succeeds and by never writing a Phase-B parquet itself.
- There is intentionally no global soft deadline. If cumulative game time reaches Kaggle's external notebook deadline before normal scorecard close, the run will fail instead of publishing a partial parquet.
