# ARC-AGI-3 Compact Verified Hypothesis Agent
# Architectural Specification
# Version 8.3

## 0. Purpose

Version 8.3 is the production-oriented continuation of the deliberately compact V8.2 architecture. It restores the verifier, memory, identity, and competition-lifecycle invariants lost during the simplification from V7.7, without restoring the V7.7 general DSL/compiler/router stack.

The architecture targets ARC-AGI-3 environments with:

- hidden game-local rules;
- dynamic action surfaces;
- simple actions with locally inferred semantics;
- coordinate actions using `x = column`, `y = row`;
- sparse score or terminal feedback;
- passive or mixed visual transitions;
- repeated level attempts after `GAME_OVER`;
- strict offline competition execution;
- a local Qwen backend used only as a semantic proposer.

V8.3 is not:

- a raw-pixel policy;
- an unrestricted LLM controller;
- an LLM-authorized action executor;
- a general-purpose symbolic DSL;
- a rollback search engine;
- a restoration of the V7.7 PlanGraph/compiler hierarchy.

V8.3 is:

- a compact object-centric proposer/judge agent;
- a fixed-contract verification architecture;
- a stateful but replayable hypothesis loop;
- an explicit official-transition ingestion pipeline;
- a competition-safe agent in which `GAME_OVER` requests `RESET` and timeout belongs to the outer harness.

## 1. Authority hierarchy

The normative authority order is:

1. competition lifecycle and official action boundary;
2. action availability and coordinate-payload validation;
3. explicit official-transition commit;
4. fixed `VerificationContract` evaluation;
5. failed and irrelevant attempt memory;
6. persistent object/relation identity;
7. grounded hypothesis bank;
8. deterministic action exploration;
9. Qwen semantic proposals;
10. safe fallback.

The Qwen proposer may rank hypotheses and bounded tests. It may not emit raw coordinates, invent IDs, revise action availability, authorize execution, or convert visible grid change into semantic truth.

## 2. Core loop

```text
Official stable observation
-> GameAdapter
-> ARGALite perception
-> persistent object-track assignment
-> stable relation graph and coordinate candidates
-> process previous pending official transition
-> update failed / irrelevant / action-effect memory
-> choose Qwen role if budget and state require it
-> validate Qwen JSON and existing IDs
-> bind next test step to one fixed VerificationContract
-> preflight legality and repeat suppression
-> emit exactly one action and create pending token
-> environment executes action
-> observe_action_result(after_observation)
-> commit that exact official transition
-> verifier evaluates only the selected contract
-> next iteration
```

The active recursive state is external to Qwen. The local model is stateless across calls.

## 3. Competition lifecycle contract

### 3.1 Official states

V8.3 distinguishes:

- `NOT_STARTED` / `NOT_PLAYED`: emit `RESET` to start;
- normal playing state: select a legal available action;
- successful terminal state: outer harness stops;
- `GAME_OVER`: emit `RESET` and replay the current level;
- wall-clock or action budget exhaustion: outer harness stops without another reset.

### 3.2 GAME_OVER semantics

`GAME_OVER` is not a semantic terminal for the agent. In competition mode, a reset after `GAME_OVER` is treated as a level reset. Therefore:

```text
GAME_OVER and remaining outer budget
-> RESET
-> commit reset transition
-> continue same game/level attempt
```

There is no reset-count terminal condition. Reset counters are telemetry only. The outer harness may terminate because the game wall-clock budget or configured action budget is exhausted. There is no per-level wall-clock timeout.

### 3.3 Official-transition invariant

Every agent-emitted action has exactly one pending transition token:

```text
emit action A from state S
-> pending(token, S, A)
-> receive official state S'
-> observe_action_result(S')
-> commit token exactly once
```

The agent must not emit another action while an uncommitted transition from an identical official state remains pending. Duplicate `observe_action_result` calls are ignored and counted. Telemetry must expose the true pending state.

## 4. State model

### 4.1 Canonical grid

- rectangular grid;
- dimensions from 1 to 64 in each axis;
- palette IDs `0..15`;
- canonical hexadecimal row encoding;
- internal geometry in row/column order;
- action payload coordinates in `x = column`, `y = row` order.

### 4.2 ARGALite snapshot

An `ARGALiteSnapshot` contains:

- game, level, and step identity;
- state name, score, progress, and `GAME_OVER` flags;
- complete canonical grid representation;
- object records;
- relation records;
- coordinate target candidates;
- available and coordinate action IDs;
- deterministic grid, object, relation, snapshot, and semantic-state hashes.

The snapshot is the only state supplied to policy, binder, verifier, and Qwen packet construction.

## 5. Object perception and persistent identity

### 5.1 Extraction

The baseline extractor supports:

- border-dominant or modal background estimation;
- 4-connected foreground components;
- optional multicolor component merging;
- local palette masks;
- shape signatures;
- color histograms;
- holes and topology signatures;
- symmetry hints;
- border contact;
- line, frame, sparse, singleton, hollow, and multicolor tags.

### 5.2 Track identity

Frame-local bounding boxes cannot define semantic identity because motion changes the bounding box. V8.3 therefore separates:

- `frame_object_id`: identity in one parsed frame;
- `track_id` / canonical `object_id`: persistent level-local identity.

Tracks are assigned deterministically using compatible shape, color, area, topology, and minimum centroid displacement. A moved object should keep the same canonical ID. Track memory is reset on a true level boundary, not on every frame.

### 5.3 Stable relation identity

A relation ID is derived from:

- relation type;
- sorted persistent endpoint IDs;
- metric name where required.

The current metric value is not part of relation identity. Consequently, a relation can preserve its ID while its error changes.

## 6. Relation graph

The compact relation vocabulary includes:

- `same_color`;
- `same_shape`;
- `translated_shape`;
- `aligned_row`;
- `aligned_col`;
- `near`;
- `separated_by_gap`;
- `inside` / `frame_contains`;
- `line_continuation`;
- `button_like_structure`;
- `unique_symbol_pair`.

Relations that guide goal verification expose a measurable metric such as centroid distance, row/column offset, gap distance, line endpoint distance, or containment outside distance.

The graph is a deterministic observation abstraction. It is not semantic truth.

## 7. Fixed verification contracts

V8.3 replaces the general V7.7 DSL with seven fixed contract kinds:

1. `NO_OP_TEST`
2. `LOCAL_TARGET_CHANGE`
3. `OBJECT_DISPLACEMENT`
4. `RELATION_ERROR_DECREASE`
5. `ACTION_SURFACE_CHANGE`
6. `SCORE_OR_TERMINAL`
7. `ACTION_EFFECT_DISCOVERY`

A contract records its target object/relation IDs, candidate target ID, target region, metric, before-value, expected effect, semantic question, and target signature.

### 7.1 Binding

`VerificationBinder` performs only:

- existing-ID validation;
- relation endpoint recovery;
- contract selection;
- current metric capture;
- target-region capture;
- semantic-question registration.

It does not search programs or compile arbitrary operators.

### 7.2 Contract judgment

#### Relation error

For `RELATION_ERROR_DECREASE`:

```text
error_delta = error_before - error_after
```

- positive above epsilon: `TRUE`, relevant, positive progress;
- negative below minus epsilon: `FALSE`, relevant, negative progress;
- unchanged: unknown and irrelevant;
- unavailable: unknown and undecided.

#### Object displacement

A persistent target track must move beyond epsilon. Visual activity elsewhere does not satisfy the contract.

#### Local target change

Changed cells must overlap the target object's before/after region by at least the configured fraction. An unrelated change is undecided, not confirmation.

#### Action surface

The set of available actions must change.

#### Score or terminal

A positive score delta, successful terminal transition, level index increase, or levels-completed increase provides positive evidence.

#### Action-effect discovery and no-op

A typed action-linked effect or no-effect may resolve a semantic question, but it does not prove a goal hypothesis. Repeating an already resolved no-effect from an equivalent state becomes irrelevant.

### 7.3 Prohibited inference

The following implication is forbidden:

```text
action-linked visible grid change -> hypothesis TRUE
```

Grid change is evidence only after linkage to the active contract.

## 8. Information gain

V8.3 uses a small verifier-side information-gain model rather than raw grid-difference magnitude.

A `SemanticQuestion` has:

- a finite domain;
- a question type;
- a target signature;
- observed categorical counts;
- an optional resolved outcome.

The first stable typed observation can reduce normalized entropy. Repeated identical evidence after resolution produces zero or near-zero gain. Unrelated visual change cannot update a question.

Information gain is used for epistemic relevance only. It cannot override legality, negative progress, failed memory, or a contract mismatch.

## 9. Memory

`GameMemory` keeps:

- all transition judgments;
- failed events;
- irrelevant events;
- coordinate effect records;
- action-effect records;
- semantic questions;
- state-scoped candidate attempt counts;
- level-local simple-action discovery counts;
- coordinate probe budget counts;
- persistent object tracks.

Failed and irrelevant outcomes are separate:

- failed: invalid, contradicted, negative, or impossible;
- irrelevant: legal but no progress and no remaining information value.

Repeat suppression is scoped by candidate signature plus semantic state signature. The same action can be reconsidered after the state, target, relation error, or action surface changes.

## 10. Hypothesis bank

The bank stores short bounded hypotheses and test plans. It accepts V8.3 JSON and retains limited V8.2 schema compatibility for migration.

Before insertion it rejects:

- invented object IDs;
- invented relation IDs;
- invented coordinate candidate IDs;
- unavailable actions;
- raw coordinates when disabled;
- malformed schemas.

Before execution it rebinds the next step against the current snapshot. A nonempty queue is not sufficient: `has_executable_candidate(current_snapshot)` is the routing criterion. Stale targets are invalidated instead of blocking replanning.

Positive concrete tests are not promoted into an unrestricted reusable program. Remaining bounded steps may continue; game-local action effects live in memory.

## 11. Deterministic exploration and fallback

When no grounded Qwen candidate is executable:

1. probe an untried simple action with `ACTION_EFFECT_DISCOVERY`;
2. probe a salient coordinate candidate with a target-specific contract;
3. choose the least-tried legal state-scoped candidate;
4. use a least-used `exhaustion_revisit` only after every informative candidate is exhausted.

All candidate sources share the same preflight, attempt accounting, pending-token, and transition-verification paths. The internal `exhaustion_revisit` flag may relax repeat and coordinate-probe suppression only after normal candidate exhaustion; it never changes semantic judgment and is counted in telemetry. This preserves liveness until the outer timeout/action budget instead of crashing in an all-no-effect state.

## 12. Qwen proposer

### 12.1 Roles

- primary semantic proposal at a new level;
- coordinate affordance research when coordinate semantics remain unresolved;
- reserve replanning after stalls, contradictions, or exhaustion.

### 12.2 Output restrictions

Qwen returns strict JSON with existing IDs and short test plans. It may not emit final actions or raw coordinates. Coordinate proposals select existing candidate IDs.

### 12.3 Prompt-tail priority

The final prompt sections are emitted in this literal order:

1. `previous_attempts_feedback`;
2. `semantic_candidate_menu`;
3. `current_question_set`;
4. `allowed_output_schema`;
5. strict instruction and `RETURN_JSON_ONLY`.

The renderer does not alphabetically sort packet keys. Under context pressure, normal, aggressive, and minimal compaction preserve action surface, valid IDs, current feedback, contract menu, questions, schema, and final instruction.

### 12.4 Backend

The reference competition backend is `llama-cli` with an offline GGUF model. The fake backend is used by tests. A failed optional Qwen call falls back to deterministic exploration unless runtime configuration marks Qwen as required.

## 13. Preflight boundary

Before emission, `PreflightJudge` verifies:

- action is available, except state-required `RESET`;
- coordinate payload is complete and within both grid and `0..63` bounds;
- coordinate candidate ID exists when required;
- coordinate probe budget remains;
- an exact known no-effect coordinate repeat is suppressed;
- a failed or irrelevant equivalent candidate is not repeated from the same semantic state.

Only a preflight-valid candidate reaches the official environment.

## 14. Competition wrapper

The generated direct-Arcade shim must:

- keep one `GameSession` per game;
- propagate dynamic Qwen timeout values into the active session;
- call `act()` once per proposed agent action;
- call `observe_action_result()` immediately after every gateway-accepted agent action;
- expose true pending-transition telemetry;
- reject loop exit with an unconsumed pending transition;
- treat `GAME_OVER` as a replay request;
- stop on success or outer budget exhaustion;
- preserve the proven direct `env.step()` orchestration pattern from the V7.7 competition package.

Phase A remains a structural/runtime setup path and does not run the heavy Qwen context smoke probe.

## 15. Explicitly excluded complexity

V8.3 does not contain:

- a general graph DSL;
- general operator compilation;
- a separate hypothesis router hierarchy;
- speculative rollback search;
- multi-step plan graphs with cursor stacks;
- unrestricted whole-grid coordinate enumeration;
- LLM chain-of-thought storage;
- cross-game semantic memory.

New complexity should be added only when a failing benchmark demonstrates that one of these exclusions is the limiting factor.

## 16. Acceptance criteria

V8.3 is architecturally conformant when:

- every official action has at most one pending transition;
- accepted transitions are committed exactly once;
- `GAME_OVER` produces `RESET` while outer budget remains;
- generic visible change never confirms an unbound hypothesis;
- relation progress requires measured error reduction;
- object IDs survive ordinary motion;
- stale candidates do not block replanning;
- coordinate budgets are source-independent;
- failed and irrelevant memories are distinct;
- prompt-tail priority is literal;
- all tests run without a real Qwen model;
- generated competition shim compiles and contains the real commit path.
