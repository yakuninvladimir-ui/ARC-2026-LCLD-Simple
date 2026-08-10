# Trajectory Contour B patch

Enhances offline trajectory verification before empiric execution.

## Changes

### 4. Goal hash resolution (`resolve_goal_hash`)
Priority:
1. `goal_spec.expected_final_state_hash` / `goal_signature` / `final_grid_hash`
2. `HypothesisItem.expected_final_state_hash` from model output
3. end-state of a prior **POSITIVE** trajectory evaluation for the same hypothesis
4. simulated final state (when full memory simulation succeeds)
5. soft anchor = current `grid_hash` (metric targets cannot invent a concrete grid)

ACCEPT / CORRECTED / PARTIAL / hybrid all stamp `expected_final_state_hash` when possible.

### 5. Rebind after CORRECTED / ACCEPT / PARTIAL
`session._call_qwen_role` now calls `rebind_plan_to_test_steps(...)`:
- rebuilds `TestStep`s from the offline plan
- preserves original target_object / relation / contract metadata when actions align
- attaches `semantic_binding`
- runs `VerificationBinder.bind` so before-metrics and contracts match the current snapshot

### 6. PARTIAL acceptance
If steps `0 .. k-1` simulate cleanly and step `k` has no memory transition:
- status = `PARTIAL`
- plan = prefix only
- hypothesis moves to `verified_semantic` and executes the prefix
- environment reobservation continues the normal loop

### 7. Hybrid empiric accept
If the offline contour cannot simulate or correct, **do not reject** when:
- first action is legal on the current surface / verifier_packet constraints
- objective is grounded (`SemanticBindingStatus` GROUNDED or PARTIAL, or legacy non-empty plan)

Status stays `ACCEPT` with `reason=hybrid_empiric_execution` and lower confidence.
The empiric `TransitionJudge` decides after the real environment transition.

Hard REJECT remains for:
- unknown / unavailable action IDs
- invalid coordinate candidate IDs
- empty trajectories
- ungrounded objective with no simulatable path

## Files

- `trajectory.py` — goal hash, PARTIAL, hybrid, rebind helpers
- `session.py` — applies PARTIAL + rebind on accept/correct/partial
- `test_contour_b.py` — unit smoke tests (stdlib-only where possible)

## Install

Copy into `v9_agent/` over the dual-view (A) baseline.
