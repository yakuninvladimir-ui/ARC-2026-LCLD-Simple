# V9 Reverse-Semantic Agent

`v9_agent` is an isolated derivative of the current V8 agent. It keeps the V8
observation packet and execution policy, while adding a bounded semantic
backreaction path for later Qwen calls.

## Dual-view (patch A)

Qwen and the offline trajectory verifier share the same planning-object set:

| Consumer | Media |
|----------|-------|
| Qwen visual | `annotated_frame_png` (grid + planning-object bboxes/labels). Raw `current_frame_png` remains in the packet metadata; multimodal backends attach **one** image (annotated preferred) to avoid double vision tokens. |
| Offline verifier | `verifier_packet` with `full_grid_hex_rows`, `planning_objects`, `action_probe_summaries`, and execution constraints. **Not serialized into the model prompt.** |

### ID policy

| Layer | Role | Legal in trajectories? |
|-------|------|------------------------|
| `object_layer.objects` / `verifier_packet.planning_objects` | Tracked connected regions | **Yes** (planning IDs) |
| `component_graph` | Same-color 4-connected geometry evidence | **No** (observation only) |

## Trajectory contour (patch B)

1. **Goal hash** — resolve `expected_final_state_hash` from model goal_spec, prior POSITIVE evaluations, simulation, or soft start-state anchor.
2. **Rebind** — after ACCEPT / CORRECTED / PARTIAL, every step is rebound via `VerificationBinder` on the current snapshot.
3. **PARTIAL** — memory-supported prefix executes; remainder is left for reobserve.
4. **Hybrid accept** — unknown memory transition alone does not reject a grounded objective with a legal first action; empiric `TransitionJudge` decides.

Hard REJECT remains for illegal actions, invalid coordinate candidates, empty plans, and ungrounded objectives with no path.

## Runtime flow

1. Parse frame → annotated PNG + object layer + component graph + action diffs + offline `verifier_packet`.
2. Qwen proposes 1–3 trajectories using whitelisted IDs only.
3. Offline `TrajectoryVerifier` scores ACCEPT / CORRECTED / PARTIAL / hybrid / REJECT.
4. Empiric step + trajectory judges update `MEMORY.semantic_feedback`.
5. A trajectory that finishes with POSITIVE progress or a confirmed visible-effect mechanic is chained into `confirmed_continuation` (top-priority queue) and exploited until it stalls — no extra Qwen call. Alternative-hypothesis RESET fires only for lineages that produced no continuation.

## Synthetic integration

`test_integration_cycle.py` drives the full loop (probe-first research → fake Qwen → offline contour ACCEPT → verified execution → confirmed continuations → WIN → terminal guard) against a scripted deterministic environment; `test_empiric_loop.py` locks the binder target guards, preflight repeat rule, and continuation enqueue conditions.

## Evidence authority

- `OFFICIAL_OBSERVATION` — direct transitions and action invariants.
- `DETERMINISTIC_BINDER` — ID/role binding and supported goal metrics.
- `DETERMINISTIC_VERIFIER` — offline contour + empiric mechanic/trajectory judgments.
- Qwen output is a proposal until environment evidence supports it.

## Competition harness notes (wrapper / builder)

Canonical runtime settings live in `working_common_v9.py` (builder embeds this, not `working_common.py`):

| Knob | V9 value | Role |
|------|----------|------|
| `LCLD_MAX_ACTIONS_PER_GAME` | 200 | Soft game action budget |
| `LCLD_MAX_ACTIONS_PER_LEVEL` | 200 | Per-level action budget |
| `LCLD_MAX_LEVEL_ATTEMPTS` | 0 | Disabled (wall-clock is terminal) |
| `LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS` | 6000 | Primary per-game stop |
| `ARC_QWEN_TIMEOUT_SECONDS` | 700 | Single model call |
| `ARC_QWEN_SCHEMA_MODE` | `dynamic_enum` | Constrained decoding over live IDs |
| `VLLM_MAX_NUM_SEQS` / concurrency | 4 | Must match worker count |
| Qwen calls / level | primary 1 + coordinate 1 | Total 2 |

`build_notebook_v9.py` adapts `working_phase_b.py` so the competition child:
- does **not** call implicit `env.reset()` after `Arcade.make`
- calls `observe_action_result` after every accepted gateway step
- treats GAME_OVER as one RESET, then continues (Tufa loop)

Capped `verified_semantic` queue size is intentionally **not** used: game action and wall-clock limits already bound loopbacks.

## Deliberate limits

- No restored forward DSL compiler / ARGA search.
- Reverse vocabulary is descriptive and bounded.
- Goal metrics stay small (centroid, bbox gap, containment, palette/shape).
