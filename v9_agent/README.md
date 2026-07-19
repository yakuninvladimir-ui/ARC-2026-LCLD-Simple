# V9 Reverse-Semantic Agent

`v9_agent` is an isolated derivative of the current V8 agent. It keeps the V8
observation packet and execution policy, while adding a bounded semantic
backreaction path for later Qwen calls.

## Runtime flow

1. The environment is parsed into the existing current-frame PNG, object layer,
   component graph, selective hex patches, and factual action diffs.
2. Qwen proposes one to three complete action trajectories using whitelisted IDs.
3. `SemanticBindingResult` preserves the proposed source, reference, and relation
   roles and binds the objective to one cheap deterministic metric when possible.
4. The step verifier checks whether the expected action mechanic occurred.
5. The trajectory verifier separately compares the bound goal at trajectory start
   and end. A movement is not goal progress merely because an object moved.
6. Binder results, trajectory evaluations, and observed invariants are returned in
   `MEMORY.semantic_feedback` on later model calls.

Attempt-local object IDs are rebound after a level reset using the original
binding-time geometry, palette, mask, and position fingerprints. Rebound records
always expose current whitelisted IDs and carry `rebound_after_reset=true`.

Repeated factual action diffs are grouped by normalized effect. One real
before/after sample is retained with an observation count and step range;
collisions, lifecycle changes, coordinate targets, action-surface changes, and
level results remain separate groups. Large pixel diffs keep bounded row-run
samples and explicitly mark incomplete positional coverage.

## Evidence authority

- `OFFICIAL_OBSERVATION`: direct transition evidence and derived action invariants.
- `DETERMINISTIC_BINDER`: ID/role binding and a supported goal metric.
- `DETERMINISTIC_VERIFIER`: mechanic and trajectory-level judgments.
- Qwen output remains a proposal until environment evidence supports it.

An invariant is `OBSERVED_ONCE` after one transition and becomes `CONFIRMED` only
after repeated identical evidence. Conflicting parameterizations are marked
`CONTRADICTED`.

## Deliberate limits

- No forward DSL compiler, planner, binder-driven trajectory generation, or ARGA
  search was restored.
- The reverse vocabulary is descriptive and bounded: translation, shape
  preservation, color change, co-motion, opposite-axis motion, action-surface
  change, and no-visible-effect observations.
- Goal metrics are intentionally small: centroid distance, bbox gap, containment
  error, and palette/shape mismatch.
- Mirror, rotation, and repeated-pattern relations are candidates from current
  geometry, never asserted game goals.

`V9Config` aliases `V8Config` so existing local integration can switch package
imports without changing configuration keys. The competition package is not
modified by this directory.
