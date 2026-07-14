# ARC-2026 LCLD Simple

Current source snapshot of the LCLD Qwen agent for the ARC Prize 2026 / ARC-AGI-3 interactive competition.

The repository contains the agent core, reproducible Kaggle notebook builder, competition wrapper sources, specifications, tests, and validation notes. It intentionally excludes model weights, ARC environment files, vLLM wheels, `llama.cpp`, generated notebooks, Kaggle datasets, and local traces.

## Architecture

```text
official frame and available actions
  -> grid/object parsing and component graph
  -> action-effect and action-surface memory
  -> bounded Qwen hypothesis and complete trajectory proposal
  -> trajectory normalization and lightweight verification
  -> deterministic execution
  -> official transition ingestion and retry memory
```

Qwen generates hypotheses and complete action trajectories. The agent owns action research, validates current action availability and coordinate targets, executes accepted trajectories, observes every official transition, and retains failure evidence across level resets. There is no general-purpose DSL or autonomous symbolic planner.

## Competition Profile

- accelerator: one NVIDIA RTX6000;
- model: `vrfai/Qwen3.6-27B-FP8`;
- model dataset: `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`;
- vLLM dataset: `driessmit1/arc3-vllm-h100-wheelhouse-v3`;
- one persistent vLLM server, tensor parallel size 1, maximum one sequence;
- model context / maximum input / maximum output: `98304 / 65536 / 12288`;
- non-thinking requests with structured JSON output;
- Qwen timeout: 500 seconds;
- game wall-clock limit: 6000 seconds;
- maximum 200 accepted actions per game;
- maximum four attempts per level, with no per-level action limit;
- calls per level: primary 1, coordinate 1, reserve 0, total 2;
- no notebook-wide competition deadline.

The model and wheelhouse datasets are referenced by the generated Kaggle metadata but are not redistributed here.

## Competition Lifecycle

Phase A performs static validation without loading the model or invoking GPU/accelerator validation, so it can run on Kaggle's CPU or P100 allocation. It writes the required nonempty validation parquet only outside a competition rerun. Phase B still requires exactly one RTX6000 before starting vLLM.

Phase B follows this order:

```text
remove stale parquet
  -> start vLLM
  -> real structured-output model smoke
  -> create one shared scorecard
  -> make game
  -> unconditional initial RESET
  -> act -> env.step -> observe_action_result
  -> one RESET after GAME_OVER when another attempt is allowed
  -> close scorecard after all games
  -> validate the gateway-generated nonempty parquet
```

Phase B never writes a fallback `submission.parquet`. Any model, gateway, game, or agent exception abandons the scorecard without calling `close_scorecard`, removes any parquet artifact, prints the vLLM log tail, and re-raises. This intentionally turns infrastructure failures into failed notebook runs instead of zero-result submissions.

The active competition path uses generated `kaggle_agent.py` and `submission.py` compatibility modules. It does not use `agent/my_agent.py`.

## Repository Layout

```text
v8_agent/                            active agent runtime
notebook_wrapper/                    Phase A/B and vLLM wrapper sources
build_notebook.py                    production notebook builder
tests/                               structural and behavioral regression suite
ARCHITECTURAL_SPECIFICATION_V6_2.md  baseline architecture
ENGINEERING_SPECIFICATION_V6_2.md    baseline engineering contract
ARCHITECTURAL_SPECIFICATION_V8_3.md  current architecture
ENGINEERING_SPECIFICATION_V8_3.md    current engineering contract
CHANGELOG_V8_3.md                    implementation history
VALIDATION_V8_3.txt                  frozen competition contract
COMPETITION_WRAPPER_AUDIT_2026-07-15.md
```

## Build

Python 3.12 or newer is recommended.

```bash
python build_notebook.py
```

Output:

```text
notebooks/arc-prize-2026-lcld-qwen.ipynb
notebooks/kernel-metadata.json
```

The generated metadata attaches the ARC Prize competition, the Tufa vLLM wheelhouse, and the Qwen FP8 snapshot. Generated artifacts remain ignored by Git.

## Test

```bash
python -m pytest tests -q
```

The current snapshot passes 106 tests and compiles all generated notebook cells and embedded Python payload files. This does not replace an actual Kaggle RTX6000/gateway run.

## License

Repository source and documentation are available under the [MIT License](LICENSE). The generated notebook header also carries the author's CC-BY-4.0 attribution notice. External models, ARC environments, Kaggle assets, and the Tufa vLLM wheelhouse are not included and remain under their respective licenses.
