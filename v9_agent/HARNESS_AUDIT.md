# Wrapper / builder technical audit

## Verdict

Harness is broadly coherent for competition Phase B. Skip C (verified_semantic cap).
Main residual risks are config drift between files and prompt token waste (fixed for verifier_packet).

## Limits matrix (canonical = `working_common_v9.py` + builder adaptation)

| Source | actions/game | actions/level | level attempts | game wall | qwen timeout | schema | sampling |
|--------|-------------:|--------------:|---------------:|----------:|-------------:|--------|----------|
| `working_common_v9.py` env | 150 | 150 | 0 | 6000 | 600 | dynamic_enum | T=0.4 k=30 pp=0.05 |
| Builder adapted `_direct_config` | 150 | 150 | 0 | 6000 | 600 | (from env) | **T=0.2 k=20 pp=0.0** (hardcoded) |
| Raw `working_phase_b.py` fallbacks | 200 | 200 | 4 | 9000 | 800 | — | T=0.6 |
| Legacy `working_common.py` | 200 | 200 | 4 | 8000 | 800 | static | T=0.6 |

**Issue:** builder `_direct_config` overrides sampling away from `working_common_v9`. Prefer reading env vars so parent and child share one sampling contract.

**Non-issue:** raw `working_phase_b.py` fallbacks differ — parent always configures env before launch; builder rewrites the child. Only dangerous if the raw file is executed alone.

## Structural correctness

Good:
- Serial `Arcade.make` before workers (gateway RESET ordering).
- Worker count forced equal to `VLLM_MAX_NUM_SEQS`.
- Scorecard close tolerates 404/409/410 auto-close.
- Zero accepted actions refuses finalization.
- GAME_OVER → single RESET in adapted child (Tufa invariant).
- `observe_action_result` after accepted step in adapted child.
- No implicit `env.reset()` after make in adapted child.
- Payload size safety cap 985KB notebook.

Watch:
- Required payload list omits `trajectory.py` / `verifier_packet.py` / `frame_media.py`. Missing files fail only at first Qwen call, not in structural preflight. Recommend adding them to `_assert_payload_structure`.
- Dual PNG was about to send two vision attachments; policy now prefers **annotated only**.
- `verifier_packet` was embedded in the model JSON; stripped from text path (session still sees the full packet object).

## Why C is unnecessary

`max_actions_per_game` / wall-clock already bound any verified_semantic loopback. A queue cap would discard offline-corrected plans that still deserve one empiric try after PARTIAL/hybrid.

## Recommended follow-ups (optional, small)

1. Builder `_direct_config`: read temperature/top_k/presence from env (match v9).
2. Payload required list: include `v9_agent/trajectory.py`, `verifier_packet.py`, `frame_media.py`.
3. Align raw `working_phase_b.py` fallbacks with v9 for offline readability (not required for notebook build path).

## Status update (2026-08-07)

- Follow-up 1: **done** — child `_direct_config` reads sampling/limits from env (200/200/700/T=0.4/k=30/pp=0.05); no residual 0.2/150 hardcodes in the built notebook.
- Follow-up 2: **done** — `trajectory.py`, `verifier_packet.py`, `frame_media.py` and `planning_set.py` added to all three payload required lists (source, unpack cell, Phase A structural preflight).
- Dual PNG: **resolved** — `_packet_image_payloads` attaches exactly one image (annotated preferred, raw fallback); text prompt marks the raw frame `NOT_ATTACHED:annotated_frame_preferred_single_image`.
- Contour B: prefix salvage now precedes hard REJECT — a memory-supported, surface-legal prefix executes as `PARTIAL` even when a later step is surface-illegal (`prefix_simulated_clipped_before_illegal_step`); planner alternatives that fail surface validation fall through to prefix salvage instead of rejecting outright.
- Canonical env (working_common_v9): actions/game 200, actions/level 200, attempts 0, wall 6000, qwen timeout 700, dynamic_enum, T=0.4/k=30/pp=0.05. The limits matrix above reflects the pre-fix audit state.
