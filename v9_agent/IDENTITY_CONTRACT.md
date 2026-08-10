# Dual-view identity + Contour B contract (patch 2026-08-05)

## Roles

| Component | Role |
|-----------|------|
| `GameSession` | Sole mutable owner of snapshot, memory, bank, `_planning_set` |
| `planning_set.PlanningSet` | Canonical id vocabulary for one snapshot cycle |
| `TrajectoryVerifier` | Pure repair advisor (`ACCEPT`/`CORRECTED`/`PARTIAL`/`PASSTHROUGH`/`REJECT`) |
| `TransitionJudge` | Empiric post-step authority |
| Qwen | Hypothesis proposer only |

## Contour B statuses

| Status | Meaning | Queue effect |
|--------|---------|--------------|
| `ACCEPT` / `CORRECTED` / `PARTIAL` | Repair applied (non-empty steps) | boost → `verified_semantic` |
| `PASSTHROUGH` | Syntax ok, no offline sim | **stay** on `semantic_test_queue` |
| `REJECT` | Illegal action / bad candidate / empty plan | `rejected` only |

Hard reject for `no_simulation_and_no_alternative` is **removed**.

## Goal hash

Never invent goal from current `grid_hash` (no `start_state_anchor`).

## Queue priority

```
confirmed_rules
→ verified_semantic   (boost within semantic tier)
→ semantic_test_queue
→ coordinate_test_queue
→ fallback_exploration_queue
```

## Confirmed continuation (patch 2026-08-07)

- A trajectory that finishes with POSITIVE goal progress, or with a MATCH
  mechanic plus at least one visibly effective step, is chained into a
  `confirmed_continuation` item in `confirmed_rules` — directed exploitation
  of the confirmed effect without another Qwen call.
- Continuations carry `proposal_batch_id=""`, so they never trigger the
  alternative-entry RESET; confirmed progress is not destroyed to retry
  untried siblings. The sibling RESET fires only when no continuation was
  enqueued (stalled or exhausted lineages).
- Self-limiting: first non-confirming empiric judgment ends the lineage;
  steps that left the action surface trim the plan; per-level cap
  (`_MAX_PROGRESS_CONTINUATIONS_PER_LEVEL = 48`) bounds exploitation.
- Contract-kind guards: explicit/expected-type OBJECT_DISPLACEMENT,
  RELATION_ERROR_DECREASE and LOCAL_TARGET_CHANGE are honored only when their
  measurable targets exist; targetless kinds degrade to
  ACTION_EFFECT_DISCOVERY (no guaranteed-MISMATCH poison contracts).
- Preflight repeat suppression fires only when the last identical action had
  no visible effect (or worse); repeating a visibly effective action is
  exploitation, not epistemic waste.
- Trajectory evaluations are attributed to the executing item (continuation
  id), not to the original proposer id stored on the semantic binding.
- Probe-diff target inference: when a v87 hypothesis names no source objects
  but the probe diffs show which planning objects changed under the action,
  the step binds those objects (model alias → internal tracked id) so the
  empiric judge measures a named displacement instead of a generic effect.
- Dead-attempt reset: when Qwen was spent on the attempt, only
  fallback/liveness actions remain, no recent step showed positive progress,
  and the flail streak reaches 8, the session emits `failed_attempt_reset`
  (`no_executable_hypothesis_fallback_exhausted`) instead of flailing until
  the wall clock; the failure record feeds the next Qwen packet.

## Files in this patch

- `planning_set.py` (new)
- `trajectory.py`
- `session.py`
- `hypothesis_bank.py`
- `IDENTITY_CONTRACT.md` (this file)
