# ARC-AGI-3 LCLD Reverse-Semantic Trajectory Agent
# Architectural Specification
# Version 9.0

Status: implementation-aligned specification of the supplied competition notebook

Date: 2026-07-17

Companion document: `ENGINEERING_SPECIFICATION_V9.md`

## 0. Purpose

This document defines the complete architecture of the Version 9.0 ARC-AGI-3
LCLD agent contained in the supplied competition notebook. It is a standalone
normative description of the agent's world model, action-research protocol,
multimodal Qwen interface, reverse-semantic binding, trajectory verification,
memory, reset recursion, and direct competition orchestration.

The architecture targets interactive ARC environments with:

- hidden game-local mechanics;
- dynamic and state-dependent action surfaces;
- discrete and coordinate-bearing actions;
- sparse level-completion and terminal feedback;
- object identity that may change at the frame level after motion or reset;
- actions whose immediate visual effect is not equivalent to goal progress;
- multiple plausible semantic explanations of the same scene;
- strict offline competition execution;
- bounded model calls, accepted actions, attempts, and wall-clock time.

The central design is a **reverse-semantic trajectory architecture**:

1. official transitions establish what actions actually do;
2. deterministic perception exposes objects, relations, components, and exact
   geometry without assigning a goal;
3. Qwen proposes one to three complete semantic trajectories from one restored
   entry state;
4. a deterministic binder maps each proposed objective to current object and
   relation identities and, where possible, to a cheap measurable goal metric;
5. every primitive action is checked before execution and judged after the
   official next frame arrives;
6. the trajectory-level verifier separates mechanic conformity from semantic
   goal progress;
7. unsuccessful alternatives are compared from the same entry state through a
   level `RESET`, while observed mechanics and verifier evidence remain in
   memory.

The architecture is not:

- an unrestricted language-model controller;
- a raw-pixel end-to-end policy;
- a model-authorized action executor;
- a forward simulator or rollback search engine;
- a general symbolic DSL and compiler stack;
- an RL training system;
- a system that treats any visible movement as success;
- a system that treats model confidence as evidence;
- a system that assumes fixed meanings for `ACTION1` through `ACTION7`.

The architecture is:

- official-observation centered;
- deterministic at every action boundary outside model inference;
- object-centric and component-aware;
- multimodal but ID-grounded;
- trajectory-first rather than single-step semantic planning;
- reverse-semantic, because semantic invariants are induced from observed
  effects and fed back into later proposals;
- reset-comparative, because sibling hypotheses are evaluated from a restored
  common state;
- replayable at the level of packet construction, action choice, transition
  ingestion, judgment, and memory update.

### 0.1 Normative artifact snapshot

The supplied artifact inspected for this specification has the following
identity:

```text
notebook: arc-prize-2026-lcld-qwen-v9(1).ipynb
notebook SHA-256: 12917cc19473f47ccfcf9d551d7296ba331a0168deb57e1a2f3584e999b1f023
embedded payload ZIP SHA-256: c128de29006dd2b25e85ab577c385bee2fe99a48c9249659ac57812b2fef67cf
embedded payload bytes: 145125
embedded active source files: 29
embedded Python package version: 9.0.0
```

The active payload compiles successfully under the available Python runtime.
A local fake-backend smoke test successfully created a session, emitted one
research action, ingested the corresponding official transition exactly once,
and cleared the pending transition token. No real ARC gateway run and no real
Qwen inference run were executed as part of this documentation audit.

### 0.2 Status vocabulary

The following terms are normative:

- **official**: obtained from the environment frame or from a transition formed
  by an accepted environment action;
- **deterministic**: reproducible from explicit inputs without a model call;
- **proposed**: emitted by Qwen and not yet accepted as fact;
- **grounded**: all required IDs exist in the current whitelisted state and the
  binder has assigned an executable interpretation;
- **partial**: the proposal is syntactically valid but lacks enough grounded
  entities or a supported metric for full semantic verification;
- **rejected**: invalid, hallucinated, unreachable, stale, duplicate, or
  contradicted before or during execution;
- **observed once**: supported by one official transition;
- **confirmed**: supported by repeated compatible official evidence;
- **contradicted**: incompatible observed variants exist for the same scoped
  invariant;
- **irrelevant**: valid evidence that did not affect the bound semantic goal;
- **unresolved**: insufficient evidence for a required/forbidden/irrelevant
  classification.

## 1. Normative authority hierarchy

The architecture has the following strict authority order:

```text
1. Official ARC frame, accepted action result, and current action surface
2. Competition lifecycle, payload, coordinate, and reset legality
3. Pending-transition and exactly-once ingestion contract
4. Deterministic preflight validation
5. Deterministic post-action transition judgment
6. Trajectory-level semantic evaluation
7. Officially derived action effects and reverse-semantic invariants
8. Persistent game-scoped memory and attempt records
9. Deterministic perception, tracking, relations, and component graph
10. Deterministic semantic binding and metric baseline
11. Qwen trajectory or coordinate proposal
12. Heuristic exploration order
```

No lower layer may overwrite a higher layer.

Hard prohibitions:

1. Qwen cannot emit an authorized primitive action directly.
2. Qwen cannot invent action, object, relation, geometry-group, or coordinate
   candidate IDs.
3. Qwen cannot emit raw coordinates when raw coordinates are disabled.
4. Qwen cannot revise official action availability.
5. Qwen cannot assign final truth, relevance, validity, or progress.
6. A pending action cannot be followed by another action before its official
   result is ingested.
7. A predicted or prompt-internal state cannot be committed as an observed
   transition.
8. Pixel change alone cannot establish semantic progress.
9. Immediate mechanic match alone cannot establish trajectory success.
10. A level or terminal frame cannot be used to infer ordinary same-level
    object mechanics for the final action.
11. A reset may restore an entry state, but it may not erase established
    game-scoped action evidence.
12. An orchestration timeout may not be converted into an environment reset.

## 2. Active layer model

The runtime is organized into eight layers and four cross-cutting systems.

```text
Layer 1 — Official Runtime Boundary
  frame, metadata, current available actions, accepted next frame, RESET

Layer 2 — Canonical Observation Adapter
  normalized grid, state, level, score, terminal flags, deterministic hash

Layer 3 — Deterministic Visual Semantics
  ARGALite objects, persistent tracks, relations, component graph, PNG, hex

Layer 4 — Causal Action Research
  typed simple-action probes, coordinate research, official action diffs

Layer 5 — Multimodal Qwen Proposal
  one coordinate plan or one batch of 1-3 complete semantic trajectories

Layer 6 — Reverse-Semantic Grounding
  role binding, supported goal operator, cheap metric, reset-safe rebinding

Layer 7 — Primitive and Trajectory Verification
  preflight, official step judgment, mechanic result, terminal goal evaluation

Layer 8 — Attempt and Competition Orchestration
  research restoration, sibling alternatives, failed-attempt RESET, limits

Cross-cutting — game-scoped evidence and attempt memory
Cross-cutting — prompt compaction and strict packet-specific JSON schema
Cross-cutting — telemetry and exactly-once transition accounting
Cross-cutting — isolated competition workers and shared scorecard lifecycle
```

Compact ownership contract:

```text
Official runtime = what happened and what is currently legal
Perception = what is visibly present and structurally related
Action research = what actions have officially been observed to do
Qwen = which complete semantic explanations are worth executing
Binder = which current entities and metric make an explanation testable
Preflight = whether a primitive action is legal and non-duplicative now
Step judge = whether the expected local mechanic occurred
Trajectory evaluator = whether the bound semantic objective improved
Memory = what remains true, failed, irrelevant, or unresolved
Orchestrator = when to reset, retry, stop, or continue another game
```

## 3. Official competition lifecycle

### 3.1 Official state classes

The architecture distinguishes:

- `NOT_STARTED` / `NOT_PLAYED`: an initial `RESET` is emitted to start;
- normal playing state: the session may research, propose, or execute;
- successful terminal state: the outer harness stops the game;
- `GAME_OVER`: the current level attempt may be recovered by one legal reset;
- action, attempt, game-clock, or orchestration exhaustion: the outer harness
  stops without converting exhaustion into an environment event.

Success terminal states recognized by the direct harness include explicit
states such as `WIN`, `WON`, `DONE`, `TERMINAL`, and `VICTORY`, and the metadata
condition `levels_completed >= win_levels` when both values exist.

`GAME_OVER` is not itself a successful terminal state.

### 3.2 Reset classes

Reset actions have distinct semantics:

1. **initial reset** — starts a not-yet-started game;
2. **research-entry reset** — removes exploratory state changes before the
   primary semantic trajectory call;
3. **alternative reset** — restores the common proposal entry state before a
   sibling trajectory is tried;
4. **failed-attempt reset** — begins another attempt after `GAME_OVER` or a
   bounded semantic failure;
5. **competition recovery reset** — the direct harness performs exactly one
   recovery reset after an official `GAME_OVER` before any additional analyzer
   or model call.

A reset transition is committed as lifecycle evidence but is not judged as an
ordinary semantic action. Execution-suppression state may be cleared; observed
mechanics, action effects, semantic bindings, failed trajectories, and
invariants remain available unless a true game boundary occurs.

### 3.3 Attempt and level boundaries

A level boundary is recognized through official evidence, primarily:

- level index change;
- increase in `levels_completed`;
- session memory marking a newly observed level.

At a true level boundary, the session resets level-local proposal queues,
active trajectory state, attempt index, and per-level Qwen budget. Game-scoped
action mechanics and completed-level evidence persist.

At an attempt boundary within the same level:

- the attempt index increments;
- attempt-scoped exact-repeat suppression is cleared;
- coordinate candidate click suppression is cleared;
- Qwen per-attempt scheduling state is reset;
- action-effect and invariant evidence remains;
- the failed attempt is stored with the complete ordered action trajectory.

### 3.4 Outer limits

The competition contract bounds:

- accepted actions per game;
- attempts per level;
- Qwen calls per game and level;
- coordinate probes;
- game wall-clock time;
- optional global submission deadline.

The outer harness, not the semantic verifier, owns these termination decisions.

## 4. Official-transition invariant

Every session-emitted non-lifecycle action creates exactly one pending token:

```text
o_t + candidate a_t
-> preflight approval
-> emit a_t and register pending(token, o_t, a_t)
-> environment executes accepted action
-> receive official o_t+1
-> observe_action_result(o_t+1)
-> commit token exactly once
-> judge transition and update memory
```

The session must not emit another action while an identical official state has
an uncommitted pending transition. Duplicate ingestion calls are ignored and
counted. The action token binds:

- the before snapshot;
- the exact candidate action and coordinate payload;
- the hypothesis and trajectory step;
- the fixed verification contract;
- the semantic binding identifier, where present.

The official transition pair is the sole factual commit boundary. A model
response, deterministic expectation, or in-memory construction cannot replace
`(o_t, a_t, o_t+1)`.

## 5. Canonical state model

### 5.1 Grid and coordinate contract

The logical grid contract is:

- rectangular;
- height and width in `1..64`;
- palette IDs in `0..15`;
- internal geometry in `(row, column)` order;
- action payload coordinates in `x = column`, `y = row` order;
- top-left origin;
- maximum payload coordinate `63` on either axis;
- canonical hexadecimal row encoding for exact serialization and hashing.

Rendered RGB is presentation-only. The logical palette index is authoritative.

### 5.2 World state

A canonical `WorldState` contains:

- `game_id`, `level_index`, and monotonic `step_index`;
- normalized grid;
- current `available_actions`;
- planning action IDs and undo action IDs;
- possible action universe supplied by the adapter;
- score;
- normalized state name;
- success terminal and `game_over` flags;
- `levels_completed`, `win_levels`, and `full_reset` metadata;
- raw normalized metadata;
- deterministic state hash.

The current official action surface is authoritative for immediate execution.
The possible-action universe is contextual evidence, not permission to execute
an unavailable action.

### 5.3 ARGALite snapshot

An `ARGALiteSnapshot` is the only semantic state supplied to policy, binder,
verifier, memory summarization, and Qwen packet construction. It contains:

- the complete canonical grid and hexadecimal rows;
- tracked objects;
- stable relations;
- coordinate target candidates;
- available, coordinate, undo, planning, and possible action IDs;
- grid, object, relation, component, snapshot, and semantic-state hashes;
- official lifecycle metadata.

## 6. Deterministic visual semantics

### 6.1 Background and object extraction

The object extractor:

- estimates background by border dominance when the dominant border color has
  sufficient support;
- otherwise uses the global modal palette value;
- extracts 4-connected non-background components;
- optionally merges adjacent multicolor foreground cells into one component;
- may recover sufficiently large same-color objects inside a merged component;
- computes exact local masks and canonical hashes.

Object features include:

- frame-local and persistent identity;
- bounding box and centroid;
- area;
- palette set and histogram;
- local hexadecimal mask rows;
- shape and topology signatures;
- hole count;
- border contact;
- horizontal, vertical, and rotational symmetry hints;
- tags including singleton, sparse, line, frame, hollow, and multicolor;
- deterministic salience.

Perception does not assign game roles such as player, key, goal, button, or
obstacle as facts.

### 6.2 Persistent object identity

Frame-local connected components are not semantic identity. Motion changes a
bounding box, and reset reconstruction can change frame-local ordering.

The memory layer therefore assigns persistent level-local track IDs by
minimizing a deterministic compatibility score over:

- shape signature;
- palette overlap and exact palette match;
- relative area change;
- topology signature;
- centroid displacement.

A new track is created only when no compatible track is available. Track IDs
are deterministic hashes scoped by game, level, shape, color, and sequence.

Track identity is a heuristic continuity mechanism. It is not permitted to
silently override contradictory current-frame geometry.

### 6.3 Reset-safe semantic rebinding

A semantic binding stores object fingerprints containing current geometry,
palette, mask, topology, and position information. After a reset, the binder
maps old attempt-local IDs onto the current snapshot by deterministic
fingerprint scoring. A match must exceed a fixed threshold. Each current object
can satisfy at most one old object reference.

Relation references are rebound only after their endpoint objects are rebound,
and only when relation type, endpoint set, and metric name agree.

Rebinding reports:

- old-to-current object map;
- old-to-current relation map;
- unresolved counts;
- whether all required references remain current;
- `rebound_after_reset` when any identity changed.

A sibling trajectory is executable only if its required binding can be
reconstructed on the restored state.

### 6.4 Relation graph

The deterministic relation vocabulary includes structural and directional
relations such as:

- `same_color`;
- `same_shape`;
- `translated_shape`;
- `mirror_candidate`;
- `rotation_candidate`;
- `aligned_row` and `aligned_col`;
- `left_of`, `right_of`, `above`, and `below`;
- `near` and `separated_by_gap`;
- `contains` and `frame_contains`;
- `line_continuation`;
- `button_like_structure`;
- `unique_symbol_pair`;
- `repeated_pattern`.

A relation ID is based on relation type, persistent endpoints, and metric name.
Its current metric value is not part of relation identity, so the same relation
may persist while its error changes.

Candidate relations such as mirror, rotation, or repeated pattern are visible
structural evidence only. They are not automatically supported goal operators.

### 6.5 Component graph

The component graph covers every same-color 4-connected region of the current
frame, including background and nested regions. It records:

- component color, area, bounding box, and centroid;
- topology and nesting;
- adjacency and shared-edge counts;
- shape and color hashes;
- compact shape runs and boundary corners;
- references to tracked objects when a component overlaps them.

Component IDs are observation-local and may not be used directly as action or
semantic target IDs. The component graph provides exact local structural
context; the object layer remains the targetable identity layer.

### 6.6 Frame media and exact geometry

The current logical grid is rendered to a deterministic PNG using the local ARC
palette and a fixed integer cell scale. The PNG is transported as an
out-of-band multimodal attachment when the backend supports vision.

The text packet carries:

- PNG dimensions and hash;
- exact full-grid hexadecimal representation when enabled;
- selected object-local masks;
- bounded rectangular hexadecimal patches;
- exact geometry groups for objects with matching masks or structural
  signatures.

The image improves visual grouping but does not replace exact symbolic data or
ID grounding.

## 7. Action surface and causal research

### 7.1 Action ontology

Action IDs are opaque game-local labels. No global directional or interaction
meaning is assigned to `ACTION1` through `ACTION7`.

The current action surface is divided into:

- currently available actions;
- coordinate-bearing actions;
- undo actions known from the environment contract;
- possible actions that may appear on another surface.

An action may only be emitted when it is currently available, except `RESET`
under the explicit lifecycle rules.

### 7.2 Mandatory action research

Before the primary semantic trajectory call, every currently exposed gameplay
action must have typed evidence or be intrinsically known as undo. Missing
simple actions are probed deterministically.

The action-research protocol supports:

- one-shot probes for co-visible unknown actions;
- a control/new-action/control sequence when a new action appears on a later
  surface and a known control action can bracket it;
- coordinate-specific research after a coordinate action is identified;
- exact before/after surface and object evidence;
- a research-entry reset before semantic execution when research changed the
  state.

Research actions collect causal evidence. They are not treated as a failed goal
trajectory merely because they did not complete the level.

### 7.3 Coordinate research

Coordinate research is a separate bounded protocol. Candidate targets are
generated deterministically from:

- tracked object centers and salient cells;
- relation-based locations;
- occupied cells;
- centers of empty regions;
- other bounded structural slots.

Qwen may return only an ordered sequence of candidate IDs. Raw `x,y` output is
rejected when disabled. Each coordinate candidate is clicked at most once per
coordinate plan and each physical location is unique within the plan.

Official coordinate evidence records:

- candidate ID and exact `x,y` payload;
- clicked cell before and after;
- target-cell change;
- local and global visual change;
- object and relation effects;
- action-surface change;
- terminal or level progress;
- repeat-suppression signature.

### 7.4 Action-diff memory

Official action diffs are grouped by normalized effect. A group keeps one real
sample plus:

- observation count;
- step-index range;
- bounded changed cells or row runs;
- color-transition counts;
- object motion and lifecycle changes;
- action-surface additions and removals;
- synchronous local visual evidence;
- coordinate target evidence where applicable.

Collisions, lifecycle transitions, coordinate effects, action-surface changes,
level transitions, and terminal results are represented separately when their
semantics differ. A truncated diff is marked as incomplete rather than being
presented as exhaustive.

## 8. Multimodal Qwen information contract

### 8.1 Canonical packet

The Qwen packet is a single layered observation with these top-level sections:

```text
state
current_frame_png
object_layer
hex_patches
action_space
action_diffs
memory
execution_constraints
```

The active wire identifier is `v8.8.layered_observation`. It is a frozen
interface identifier used by the V9 implementation; the identifier does not
change the authority or semantics defined by this document.

The packet enforces one consistent coordinate contract and one set of
frame-local aliases. The model sees compact aliases; its output is translated
back to internal IDs before validation.

### 8.2 Packet truth classes

The packet separates four evidence authorities:

1. `OFFICIAL_OBSERVATION` — frames and accepted transitions;
2. `DETERMINISTIC_BINDER` — role and metric grounding;
3. `DETERMINISTIC_VERIFIER` — step and trajectory judgments;
4. `QWEN_PROPOSAL` — unverified semantic suggestions.

The model is explicitly told that:

- current frame, object layer, and current action surface describe one exact
  execution-start state;
- research diffs may be history-only after state restoration;
- action diffs are chronological factual effect groups, not alternate current
  worlds;
- completed attempts and verifier feedback are factual memory;
- a model explanation is not evidence merely because it is coherent.

### 8.3 Reference whitelist

Each packet carries exact allowed sets for:

- action IDs;
- object IDs;
- relation IDs;
- coordinate candidate IDs;
- maximum plan steps;
- raw-coordinate permission.

Packet construction validates all internal references. Constrained decoding
uses a packet-specific JSON schema enumerating the same IDs. The hypothesis bank
performs the same validation again before execution.

### 8.4 Context discipline

The prompt builder applies bounded compaction when estimated input size exceeds
the configured ratio. It may reduce:

- low-salience objects and relations;
- old attempt detail;
- redundant action-diff samples;
- component shape detail;
- older semantic feedback.

It must preserve:

- current state and action surface;
- current whitelists;
- trajectory-start semantics;
- failed exact trajectories and reason codes;
- current semantic bindings and reset rebinding status;
- the output contract and strict JSON instruction at the prompt tail.

The multimodal image is never duplicated inside the textual packet.

## 9. Qwen roles and scheduling

### 9.1 Coordinate role

The coordinate role is used when a coordinate action remains unresearched. It
returns one bounded ordered sequence of distinct coordinate candidate IDs and a
mechanism hypothesis. The current output wire identifier is
`v8.4.coordinate_plan`.

The coordinate role is exploratory. It does not define the final semantic goal.

### 9.2 Primary semantic role

The primary role is called only after mandatory action research is complete and
the execution entry state has been restored when required. It returns one to
three complete candidate trajectories. The current output wire identifier is
`v8.7.semantic_trajectories`.

Each hypothesis contains:

- a unique proposal-local ID;
- a semantic family;
- a grounded objective description;
- source and reference object IDs;
- supporting relation IDs;
- factual basis;
- a complete sequence of action runs;
- uncertainty and confidence;
- `status = complete_candidate`.

### 9.3 Reserve role

The type system supports a reserve role, but the supplied active session does
not schedule reserve calls under the production configuration. Semantic
recovery is performed through sibling alternatives and bounded attempt reset,
not through an automatically selected second proposal role.

### 9.4 Call budget

The production schedule permits, by default:

- one coordinate call per level when required;
- one primary semantic call per level attempt;
- zero reserve calls;
- two total Qwen calls per level;
- twenty calls per game.

A model invocation counts against budget even when it times out or produces
malformed output. When strict runtime is enabled, a backend failure is fatal to
the affected game worker rather than silently converted into an ungrounded
policy.

## 10. Complete trajectory hypothesis contract

### 10.1 Trajectory completeness

A semantic proposal must describe the full expected route from the current
restored state. It may not return only a promising first action.

`action_runs` encodes consecutive repetitions without token-by-token manual
expansion:

```text
action_id
repeat = total number of consecutive executions
coordinate_candidate_id = required for coordinate actions
```

For coordinate actions, `repeat` must equal one. A coordinate candidate and its
physical location may each occur at most once in a trajectory.

The first action must be currently available. A later action that exists on a
different surface is valid only when an earlier run has an observed basis for
creating that surface.

### 10.2 Proposal batch

The one-to-three hypotheses returned by a primary call form one proposal batch.
They are alternatives over one common entry state. The bank preserves their
batch identity and executes them sequentially.

The proposal validator rejects a candidate when it:

- uses an unknown ID;
- starts with an unavailable action;
- uses an unreachable later action surface;
- violates coordinate uniqueness or repeat rules;
- exceeds the trajectory-step limit;
- contains a vacuous or ungroundable objective;
- contradicts known action effects;
- repeats a failed exact trajectory unchanged;
- omits an observed enabling action needed for a later surface;
- fails exact correspondence or control-context checks.

### 10.3 Active trajectory execution

The chosen hypothesis is expanded into primitive `TestStep` objects. Each step
carries a fixed verification contract and the shared semantic binding. The
session re-observes the world and judges every accepted action before emitting
the next step.

A mechanic mismatch terminates the active trajectory. A completed sequence is
then evaluated against its bound semantic objective and official level state.

## 11. Reverse-semantic binding

### 11.1 Objective vocabulary

The model-level objective kinds map to bounded goal operators:

- match or overlap;
- relative arrangement;
- containment;
- connection;
- pattern or state completion;
- select or activate;
- action-surface change;
- affordance or relation probe;
- other unresolved intent.

The deterministic operator vocabulary includes:

- `MOVE_TOWARD`;
- `ALIGN`;
- `MATCH_GEOMETRY`;
- `OVERLAP`;
- `CONNECT`;
- `BRIDGE_GAP`;
- `EXTEND_LINE`;
- `CONTAIN`;
- `COMPLETE_PATTERN`;
- `MATCH_STATE`;
- `ACTIVATE`;
- `CHANGE_ACTION_SURFACE`;
- `PROBE_AFFORDANCE`;
- `PROBE_RELATION`;
- `OTHER`.

The binder may infer missing source or reference objects only from explicitly
cited relation endpoints. It may not invent an object.

### 11.2 Binding status

A binding is:

- `GROUNDED` when all required entities exist and the objective has a supported
  deterministic interpretation;
- `PARTIAL` when IDs are valid but a reference or metric is unavailable;
- `REJECTED` when IDs are unknown or required source objects are absent.

The binding stores:

- source, reference, and relation roles;
- inferred roles;
- goal operator;
- metric specification and baseline;
- evidence references;
- semantic-state and action-surface signatures;
- game and level scope.

### 11.3 Supported goal metrics

The architecture uses only cheap deterministic metrics:

1. **Centroid distance** for movement, alignment, and overlap:

\[
E_{\mathrm{centroid}}
= \frac{1}{|S|}\sum_{s\in S}\min_{r\in R}
\sqrt{(y_s-y_r)^2+(x_s-x_r)^2}.
\]

2. **Bounding-box gap** for connection, bridging, and line extension:

\[
d_r=\max(0, r^B_0-r^A_1-1, r^A_0-r^B_1-1),
\]

\[
d_c=\max(0, c^B_0-c^A_1-1, c^A_0-c^B_1-1),
\]

\[
E_{\mathrm{gap}}=\frac{1}{|S|}\sum_{s\in S}\min_{r\in R}
\sqrt{d_r^2+d_c^2}.
\]

3. **Containment outside distance** for containment:

\[
E_{\mathrm{contain}}=
\max(0,r^C_0-r^I_0)+
\max(0,c^C_0-c^I_0)+
\max(0,r^I_1-r^C_1)+
\max(0,c^I_1-c^C_1).
\]

4. **Palette-shape mismatch** for pattern and state matching:

\[
E_{\mathrm{match}} =
\mathbf{1}[\sigma_s\ne\sigma_r]
+ \frac{1}{2}\sum_{k}
\left|\frac{h_s(k)}{\sum_j h_s(j)}-
      \frac{h_r(k)}{\sum_j h_r(j)}\right|.
\]

For multiple references, each source uses its minimum pairwise error and the
binding averages over sources.

These metrics are deliberately narrow. Unsupported semantic ideas remain
partial or unresolved rather than being forced into a misleading proxy.

## 12. Fixed verification contracts

### 12.1 Contract kinds

Every primitive step is bound to one fixed contract kind:

- `NO_OP_TEST`;
- `LOCAL_TARGET_CHANGE`;
- `OBJECT_DISPLACEMENT`;
- `RELATION_ERROR_DECREASE`;
- `ACTION_SURFACE_CHANGE`;
- `SCORE_OR_TERMINAL`;
- `ACTION_EFFECT_DISCOVERY`.

A contract may carry:

- target object and relation IDs;
- target coordinate candidate and region;
- before metric;
- expected effect;
- registered question;
- target signature;
- semantic binding ID;
- source and reference roles;
- semantic metric name, direction, target, and epsilon.

The contract is fixed before the action is emitted. The verifier may not choose
a favorable interpretation after observing the result.

### 12.2 Preflight

Preflight rejects an action when:

- the action is not currently available;
- `RESET` is not legal in the current lifecycle state;
- coordinate order or bounds are invalid;
- a coordinate candidate ID is required but absent;
- the candidate is not current or does not match the payload;
- the same coordinate was already tested without effect under the same scoped
  state;
- candidate or action repeat limits are exceeded;
- the exact same state-action-contract signature is suppressed;
- the proposal references stale entities;
- the trajectory step is inconsistent with the current action surface.

### 12.3 Step judgment

The transition judge evaluates official before and after snapshots. Its global
precedence is:

1. `GAME_OVER` without level progress — negative and forbidden;
2. level completion, success terminal, or positive official score evidence —
   positive and required;
3. selected fixed contract;
4. semantic binding metric, where defined;
5. unresolved or irrelevant when evidence does not support stronger judgment.

A `Judgment` separates:

- truth: `TRUE`, `FALSE`, `UNKNOWN`;
- relevance: `RELEVANT`, `IRRELEVANT`, `UNDECIDED`;
- validity: `VALID`, `INVALID`, `UNCHECKED`;
- progress: `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `UNKNOWN`;
- mechanic result: `MATCH`, `MISMATCH`, `UNKNOWN`;
- semantic judgment: `REQUIRED`, `FORBIDDEN`, `IRRELEVANT`, `UNRESOLVED`;
- attribution: action-linked, passive-possible, mixed/uncertain, or no visible
  change.

### 12.4 Mechanic and goal separation

A primitive action can match the expected mechanism without improving the
semantic objective. Conversely, a route may produce semantic progress even
when one local contract was weak, provided official terminal evidence or the
bound end metric establishes progress.

Therefore:

```text
step mechanic result != trajectory semantic result
visible change != goal progress
movement != success
```

## 13. Trajectory-level evaluation

A trajectory is evaluated over:

- one start snapshot;
- one end snapshot;
- the complete ordered set of official step judgments;
- the exact executed action IDs;
- one semantic binding.

Precedence:

1. official level progress or successful terminal -> `REQUIRED`, positive;
2. `GAME_OVER` -> `FORBIDDEN`, negative;
3. supported bound metric improves -> `REQUIRED`, positive;
4. supported bound metric worsens -> `FORBIDDEN`, negative;
5. supported bound metric is unchanged -> `IRRELEVANT`, neutral;
6. no metric and any mechanic mismatch -> `FORBIDDEN`, negative;
7. otherwise -> `UNRESOLVED`.

For a minimization metric, improvement is:

\[
\Delta E = E_{\mathrm{before}} - E_{\mathrm{after}}.
\]

For maximization:

\[
\Delta E = E_{\mathrm{after}} - E_{\mathrm{before}}.
\]

For a target value \(T\):

\[
\Delta E = |E_{\mathrm{before}}-T|-|E_{\mathrm{after}}-T|.
\]

With epsilon \(\varepsilon\):

```text
Delta E > epsilon   -> REQUIRED / POSITIVE
Delta E < -epsilon  -> FORBIDDEN / NEGATIVE
otherwise           -> IRRELEVANT / NEUTRAL
```

The trajectory record includes the first mechanic-divergence step, start/end
signatures, error values, action sequence, target roles, evidence contract IDs,
and final reason code.

## 14. Sibling-alternative reset recursion

### 14.1 Common-entry comparison

All hypotheses in one proposal batch are intended to start from the same
restored state. If an alternative completes without level success and sibling
candidates remain, the hypothesis bank creates an alternative-reset request.

The session then:

1. emits a legal `RESET`;
2. waits for the official restored frame;
3. acknowledges the reset transition;
4. clears only attempt-local execution suppression;
5. rebinds remaining hypotheses to the restored objects and relations;
6. drops alternatives whose bindings cannot be recovered;
7. executes the next valid sibling.

This is a controlled comparative experiment, not rollback simulation. Each
alternative is executed in the real environment and receives real official
frames.

### 14.2 Preserved evidence

Across an alternative reset, the architecture preserves:

- action research;
- official action diffs;
- object and relation fingerprints;
- semantic binding records;
- previous trajectory evaluations;
- failed exact action sequences;
- reverse-semantic invariants;
- completed-level memory;
- model and verifier telemetry.

It clears:

- active pending action;
- current trajectory cursor;
- same-state action repeat suppression;
- transient queue state that refers to the old frame IDs.

### 14.3 Exact replay prohibition

A complete failed trajectory from an attempt entry is marked
`exact_replay_forbidden` unless it produced official level progress. A later
proposal must alter the route or repair the stated failure. Research prefixes
are not automatically forbidden because they were evidence collection rather
than goal execution.

## 15. Reverse-semantic invariants

From non-terminal, same-level official transitions, the deterministic layer may
derive bounded invariants:

- `TRANSLATES_BY`;
- `PRESERVES_SHAPE`;
- `CHANGES_COLOR`;
- `CO_MOVES_WITH`;
- `MOVES_OPPOSITE_ON_AXIS`;
- `CHANGES_ACTION_SURFACE`;
- `NO_VISIBLE_EFFECT_IN_STATE`.

Each invariant is scoped by:

- action ID;
- subject object IDs;
- current action-surface/control-context signature;
- parameter variant;
- official evidence references.

One observation yields `OBSERVED_ONCE`. Repeated compatible evidence reaches
`CONFIRMED` at the configured count. Competing variants for the same scoped
base invariant yield `CONTRADICTED`; the system must not average contradictory
mechanics into a false rule.

No ordinary object or surface invariant is derived when the transition crosses
a level boundary, reaches a terminal state, or enters `GAME_OVER`, because the
after frame may belong to a different semantic scene.

## 16. Memory architecture

### 16.1 Memory scopes

The memory system separates:

- game-scoped official action effects;
- level-scoped object tracks and semantic bindings;
- attempt-scoped execution suppression;
- coordinate-effect history;
- action-surface history;
- failed and irrelevant events;
- completed-level trajectories;
- reverse-semantic invariants;
- semantic questions and posterior states;
- model proposal and trajectory evaluation history.

### 16.2 Attempt record

A failed attempt record contains:

- attempt identity and reset trigger;
- Qwen-call count;
- complete ordered environment trajectory from entry to reset;
- research actions separated as provenance but not as a fictitious state
  boundary;
- goal execution sequence;
- clicked-cell transitions;
- aggregate color transitions;
- visible-effect and target-cell statistics;
- level-progress and `GAME_OVER` outcome;
- executed hypotheses and verifier reasons;
- exact-replay prohibition.

The authoritative attempt trajectory includes all accepted environment actions
in order. Research and goal labels do not imply that the environment state was
reset between phases unless an official reset actually occurred.

### 16.3 Semantic feedback

The Qwen memory section exposes bounded deterministic feedback:

- recent semantic bindings with current rebinding status;
- trajectory evaluations;
- confirmed, observed, or contradicted invariants;
- failed exact proposals and reason codes;
- completed-level successful action runs;
- current unresolved semantic questions.

A Qwen explanation is stored only as proposal provenance. Confirmation status
comes from deterministic official evidence.

## 17. Semantic questions and information gain

The agent registers typed questions over finite domains, including:

- action effect;
- affordance;
- controllability;
- relation relevance;
- action-surface change;
- terminal progress.

Observed information gain is posterior uncertainty reduction, not the size of
the grid diff.

For a normalized probability vector \(p\) over \(n>1\) outcomes:

\[
H(p)=-\frac{\sum_{i=1}^{n}p_i\ln p_i}{\ln n}.
\]

The observed information gain is:

\[
IG_{\mathrm{obs}}=\max(0,H(p_{\mathrm{before}})-H(p_{\mathrm{after}})).
\]

An unresolved question starts uniform. A resolved outcome receives posterior
mass `0.92`, with total tail mass `0.08` distributed over other outcomes.
Contradictory typed evidence reopens the question instead of forcing false
certainty. A raw visual change that does not update a registered semantic
question has zero semantic information gain.

## 18. Failure and irrelevance semantics

The architecture does not collapse all non-success into one category.

- **Forbidden**: the route worsened the bound goal, caused game over, violated a
  required mechanic, or contradicted official constraints.
- **Irrelevant**: the route was legal and measurable but left the bound goal
  unchanged.
- **Unresolved**: the route lacked a supported metric and produced no decisive
  official outcome.
- **Invalid**: the proposal or primitive action violated schema, ID, surface,
  coordinate, repeat, or lifecycle contracts.

These distinctions drive different memory behavior:

- forbidden exact routes are suppressed;
- irrelevant routes are remembered as non-progress for the particular binding;
- unresolved evidence may remain useful for mechanism discovery;
- invalid proposals are not executed and are reported with precise reason
  codes.

## 19. Competition orchestration

### 19.1 Direct child process

The competition path uses a dedicated child process rather than framework-level
agent orchestration. The child:

- creates game environments lazily;
- runs multiple game workers concurrently;
- shares one explicit scorecard;
- bounds accepted actions and game time;
- commits every accepted transition immediately;
- isolates one game's failure from other games;
- writes an atomic result manifest.

The default child concurrency is greater than model sequence concurrency. Model
requests are therefore naturally queued by the shared vLLM server.

### 19.2 GAME_OVER recovery

After an official `GAME_OVER`, the child performs exactly one `RESET` before any
new model or analyzer call. If the reset frame remains `GAME_OVER`, the game
stops with a specific persistence reason. A rejected gateway step is a game
failure and is not treated as an observed transition.

### 19.3 Scorecard lifecycle

One scorecard is shared across workers. Close outcomes distinguish:

- successful close;
- close without a payload;
- already-closed responses;
- nonfatal close error recorded in the result manifest.

If orchestration fails after gateway activity, partial results are preserved and
the scorecard is closed when possible. A fatal failure before any gateway
activity is re-raised.

### 19.4 Phase separation

The notebook has two execution modes:

- **static preflight mode**: validates imports and output shape without starting
  the model server;
- **competition mode**: installs the isolated runtime, starts vLLM, performs a
  strict multimodal JSON smoke request, then launches the child supervisor.

The model server is stopped in a finalization block.

## 20. Production model profile

The supplied competition notebook is configured for:

```text
model: vrfai/Qwen3.6-27B-FP8
backend: vLLM OpenAI-compatible server
context length: 131072 tokens
maximum packet input: 65536 tokens
maximum generated output: 49152 tokens
reserved runtime margin: 8192 tokens
thinking: enabled
multimodal current-frame PNG: enabled
sampling temperature: 0.6
sampling top_k: 20
top_p: 0.95
presence penalty: 0.0
repeat penalty: 1.0
model-server max concurrent sequences: 4
```

The agent package also implements fake, Ollama, local/`llama-cli`, and generic
vLLM backends. Tests and deterministic validation must not require the real
model.

## 21. Telemetry contract

Required telemetry includes:

- game, level, attempt, and step identity;
- action and reset counts;
- alternative-reset count;
- pending transition token state;
- successful, duplicate, and missing transition ingestions;
- Qwen calls by role and scope;
- model errors, malformed outputs, and strict-runtime failures;
- research completion and missing action IDs;
- active hypothesis and proposal batch;
- preflight rejections by reason;
- step truth/relevance/validity/progress;
- mechanic and semantic judgments;
- trajectory evaluations;
- failed, irrelevant, coordinate, and invariant memory counts;
- rebound object and relation counts;
- attempt and exact-replay suppression counts;
- game termination reason;
- scorecard close disposition;
- result-manifest path and atomic-write status.

Telemetry must report what actually happened. A degraded or failed model call
must not be labeled as successful semantic reasoning.

## 22. Risk model

### 22.1 Semantic proxy risk

The supported metrics are intentionally cheap. Centroid distance, bounding-box
gap, containment error, and palette-shape mismatch can fail to represent the
true game objective. The verifier must return unresolved or irrelevant rather
than overclaim semantic truth.

### 22.2 Object identity risk

Track and reset-rebinding heuristics can confuse repeated or visually identical
objects. Ambiguous bindings must be reported as unresolved; they may not be
silently forced.

### 22.3 Alternative-reset risk

The environment may not restore an identical entry state, or a reset may alter
hidden state. Rebinding and state signatures are therefore mandatory before a
sibling trajectory is executed.

### 22.4 Action-surface risk

A model may propose an action observed on another surface without reproducing
the enabling transition. Surface reachability validation and per-step
re-observation are mandatory.

### 22.5 Prompt compression risk

Aggressive compaction can remove the evidence that differentiates two
hypotheses. Whitelists, failed-route reasons, current state, bindings, and the
output schema have preservation priority.

### 22.6 Multimodal interpretation risk

The PNG may encourage semantic overinterpretation. Exact IDs, masks, relations,
hex patches, and action diffs remain the grounding authority.

### 22.7 Concurrency risk

Many game workers share a smaller number of model sequences and one scorecard.
Game isolation, bounded requests, atomic manifests, and final scorecard handling
are required.

## 23. Explicit non-claims and known limitations

The supplied implementation does not claim:

- a complete ARC semantic ontology;
- exact physical simulation;
- general rollback or search over hypothetical states;
- a general-purpose DSL compiler;
- a proof that reset restores hidden environment state;
- robust identity resolution for arbitrarily many identical objects;
- direct metric support for mirror, rotation, or arbitrary repeated-pattern
  completion;
- automatic scheduling of a reserve semantic model role;
- guaranteed recovery when all three model trajectories are invalid;
- guaranteed semantic correctness of Qwen's natural-language basis;
- successful execution on the live competition gateway from the local audit.

The active deterministic fallback is intentionally narrow. When no legal
research action, confirmed step, coordinate step, or validated semantic
trajectory exists, the session prefers bounded reset or explicit failure over
unverifiable blind play.

## 24. Acceptance criteria

The architecture is conformant only when all of the following hold:

1. The current official action surface governs every primitive action.
2. Every non-lifecycle emitted action has one pending token and one official
   commit.
3. Duplicate result ingestion cannot double-count evidence.
4. Grid and coordinate contracts are validated before use.
5. Qwen receives deterministic IDs and cannot authorize execution.
6. Raw coordinates are rejected unless explicitly enabled.
7. Mandatory exposed-action research precedes the primary semantic call.
8. Research state is restored before trajectory comparison when required.
9. The primary response contains one to three complete trajectories.
10. All model IDs are packet-whitelisted and validated twice.
11. Each trajectory has one deterministic semantic binding.
12. Each primitive step has one fixed verification contract.
13. Official step judgments separate mechanic result from semantic progress.
14. Trajectory evaluation uses terminal evidence or one predeclared metric.
15. Unchanged metric is classified as irrelevant, not forbidden.
16. Failed exact trajectories are not replayed unchanged.
17. Sibling alternatives are reset to and rebound against a common entry state.
18. Official mechanics and invariants persist across alternative resets.
19. Contradictory invariant evidence does not become a confirmed rule.
20. Information gain is entropy reduction over a typed semantic question.
21. `GAME_OVER` and orchestration timeout remain distinct termination classes.
22. The direct harness performs exactly one recovery reset before further
    reasoning after `GAME_OVER`.
23. Accepted gateway transitions are ingested immediately.
24. The real model is not required for deterministic tests.
25. The embedded source compiles and imports without network access.

## 25. Change discipline

Any change to the following requires synchronized updates to code, packet
schemas, tests, and both V9 specifications:

- authority hierarchy;
- pending-transition lifecycle;
- grid or coordinate contract;
- action research gating;
- packet whitelists or alias translation;
- Qwen output schemas;
- semantic objective or metric vocabulary;
- binding and reset-rebinding rules;
- preflight or transition-judge precedence;
- alternative-reset state preservation;
- attempt and game termination rules;
- competition model profile;
- notebook installation or scorecard lifecycle.

Wire identifiers and public compatibility names may remain frozen even when the
implementation version changes. Their literal names must never be used to infer
a different authority model than the one defined here.
