# ARC-2026 LCLD Simple

Compact object-centric agent for the ARC Prize 2026 / ARC-AGI-3 interactive
competition. This repository contains the V8.3 competition source snapshot,
its Kaggle wrapper and notebook builder, and the architectural and engineering
specifications used to develop it.

The repository intentionally does **not** include a Qwen model, `llama.cpp`
binaries, ARC environment files, Kaggle datasets, or generated notebooks.

## Architecture

The runtime keeps the semantic layer deliberately small:

```text
official frame and available actions
  -> grid/object parsing and component graph
  -> action-effect and action-surface memory
  -> one bounded Qwen proposal
  -> trajectory normalization and verifier checks
  -> deterministic execution
  -> official transition ingestion and memory update
```

Qwen proposes complete action trajectories. The agent retains control of the
official action loop, validates action availability and trajectory grounding,
tracks effects against the current frame, and records failures for a later
level attempt. There is no general-purpose DSL, binder, or autonomous symbolic
planner in this version.

## Repository Layout

```text
agent/my_agent.py                    competition-compatible adapter
v8_agent/                            active V8.3 runtime
build_notebook.py                    offline Kaggle notebook builder
ARCHITECTURAL_SPECIFICATION_V6_2.md  baseline architecture
ENGINEERING_SPECIFICATION_V6_2.md    baseline engineering contract
ARCHITECTURAL_SPECIFICATION_V8_3.md  current architecture
ENGINEERING_SPECIFICATION_V8_3.md    current engineering contract
CHANGELOG_V8_3.md                    V8.3 change summary
```

## Competition Profile

The checked-in notebook builder is configured for:

- Kaggle accelerator: two NVIDIA T4 GPUs;
- Qwen execution: `CUDA0,CUDA1`, layer split `1,1`;
- context/input/output limits: `98304 / 65536 / 4096` tokens;
- Qwen timeout: 500 seconds per call;
- game timeout: 5000 seconds;
- no per-level timeout;
- call budgets supplied by the competition wrapper: primary 1, coordinate 1,
  reserve 1, total 3 per level.

The external runtime dataset must provide a compatible Qwen GGUF model and a
Linux `llama-cli` build with its adjacent CUDA libraries. Those assets have
their own licenses and are not covered or redistributed by this repository.

## Build

Python 3.12 or newer is recommended. Building the notebook itself uses only the
Python standard library:

```bash
python build_notebook.py
```

The command creates:

```text
notebooks/arc-prize-2026-lcld-qwen.ipynb
notebooks/kernel-metadata.json
```

Before publishing the notebook on Kaggle, configure the runtime dataset named
in `build_notebook.py` or change it to your own dataset containing the model
and `llama-cli`. The ARC-AGI-3 competition environment is supplied by Kaggle at
runtime and is not part of this source tree.

## Runtime Contract

The direct competition loop uses `ARC_AGI_Agent` from `agent/my_agent.py`.
Every accepted action must be followed by ingestion of the official result:

```text
act(before) -> environment.step(action) -> observe_action_result(after)
```

Available actions are read from the latest official frame. Coordinate action
rules are included in the Qwen packet only when `ACTION6` is available.
`ACTION7` is treated as semantic undo and is not used for routine probing.

## Status

This is an experimental competition agent. Structural tests and local harness
runs do not establish an official score or compatibility with every hidden
game. No model weights or benchmark results are bundled with this snapshot.

## License

The source code and documentation in this repository are released under the
[MIT License](LICENSE). External models, `llama.cpp`, ARC-AGI-3 environments,
and Kaggle competition assets are not included and remain subject to their own
terms.

