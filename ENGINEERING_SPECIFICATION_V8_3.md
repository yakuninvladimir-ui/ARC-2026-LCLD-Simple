# ARC-AGI-3 Compact Verified Hypothesis Agent
# Engineering Specification
# Version 8.3

## 0. Engineering objective

Implement the V8.3 architecture as a small active Python package that preserves the working V7.7 direct competition boundary while replacing V8.2's overpermissive transition judgment with fixed, grounded verification contracts.

Target properties:

- Python 3.12+ compatible;
- offline competition execution;
- deterministic parsing and hashing;
- optional local Qwen through `llama-cli`;
- no real model required by tests;
- one active path, with compatibility shims kept minimal;
- generated notebook below the competition size limit;
- no general DSL/compiler/router stack.

## 1. Repository structure

```text
agent/
  my_agent.py

v8_agent/
  __init__.py
  action_adapter.py
  action_effects.py
  action_explorer.py
  arga_lite.py
  config.py
  coordinate_research.py
  deliberation.py
  game_adapter.py
  hypothesis_bank.py
  judge.py
  llm.py
  logging.py
  memory.py
  observe.py
  policy.py
  qwen_packet.py
  qwen_roles.py
  relations.py
  session.py
  types.py
  verification.py

tests/
  test_v8_*.py
  test_v83_*.py

build_notebook.py
build_notebook_smoke.py
notebooks/
ARCHITECTURAL_SPECIFICATION_V8_3.md
ENGINEERING_SPECIFICATION_V8_3.md
README_V8_3.md
CHANGELOG_V8_3.md
```

`coordinate_research.py` and `action_adapter.py` are compatibility facades only. Active behavior resides in the modules listed above.

## 2. Configuration contract

The public configuration dataclass remains named `V8Config` for package compatibility.

### 2.1 Qwen

```text
enable_qwen
qwen_backend: disabled | fake | qwen_local | llama_cli
qwen_model_path
qwen_llama_cli_path
qwen_temperature
qwen_seed
qwen_timeout_seconds
qwen_context_tokens
qwen_max_input_tokens
qwen_max_output_tokens
qwen_require_runtime
qwen_empty_output_retry_enabled
prompt_compaction_strategy: normal | aggressive | minimal
prompt_compression_trigger_ratio
prompt_tail_priority_enabled
```

### 2.2 Qwen budgets

```text
max_qwen_calls_per_game
max_primary_qwen_calls_per_level
max_reserve_qwen_calls_per_level
max_coordinate_qwen_calls_per_level
max_total_qwen_calls_per_level
min_steps_between_qwen_calls
qwen_stall_threshold
```

### 2.3 Verification and exploration

```text
max_test_plan_length_default
max_simple_action_probes_per_level
max_coordinate_candidates_in_packet
max_coordinate_probes_per_level
max_coordinate_probe_repeats_per_signature
max_same_state_action_repeats
allow_qwen_raw_coordinates = false
require_coordinate_candidate_id = true
reject_hallucinated_ids = true
relation_error_epsilon
local_target_overlap_min_fraction
information_gain_min_threshold
```

### 2.4 Competition

```text
max_actions_per_game
game_wall_clock_limit_seconds
reset_on_game_over = true
max_game_over_resets_per_game = 0
max_game_over_resets_per_level = 0
```

The reset-limit fields are compatibility telemetry and must not terminate gameplay. Zero means unlimited.

Frozen competition defaults:

```text
qwen_timeout_seconds = 350
game_wall_clock_limit_seconds = 5000
per-level wall-clock deadline = disabled
competition/global wall-clock deadline = disabled
```

The game clock persists across levels and GAME_OVER resets. No level clock is created or restarted.

### 2.5 Compatibility aliases

The loader accepts V7.7/V8.2 names such as:

- `llm_advisor_backend`;
- `llm_timeout_seconds`;
- `max_qwen_calls_per_level`;
- `max_qwen_primary_calls_per_level`;
- `max_qwen_replan_calls_per_level`;
- `max_qwen_coordinate_calls_per_level`.

Dynamic step configuration must update the active `GameSession`, not merely a wrapper dictionary.

## 3. Core data types

### 3.1 CandidateAction

Required fields:

```text
action_id
x | None
y | None
coordinate_candidate_id | None
hypothesis_id | None
reason
source
verification_contract | None
```

`to_arc_action()` emits both `action_id` and `id`, optional coordinate data, and bounded reasoning metadata. `suppression_signature` includes action, coordinate candidate or coordinates, and contract kind.

### 3.2 VerificationContract

```text
contract_id
kind
target_object_ids
target_relation_ids
target_coordinate_candidate_id
target_region_rc
metric_name
before_metric
expected_effect
question_id
question_type
target_signature
```

### 3.3 PendingAction

```text
before_snapshot
action
hypothesis_id
reason
token_id
```

A session has zero or one pending action.

### 3.4 Judgment

```text
truth: TRUE | FALSE | UNKNOWN
relevance: RELEVANT | IRRELEVANT | UNDECIDED
validity: VALID | INVALID | UNCHECKED
progress: POSITIVE | NEGATIVE | NEUTRAL | UNKNOWN
attribution
action
before_hash
after_hash
contract_kind
error_before
error_after
error_delta
observed_information_gain
question_id
reason_code
observed_delta
```

## 4. Observation adapter

`GameAdapter.to_world_state()` must:

- accept `grid` or frame-like input;
- canonicalize integer palette cells;
- read available actions from metadata;
- normalize state names;
- detect `GAME_OVER` independently of success terminal;
- detect success terminal from explicit successful states or level completion;
- retain score, levels completed, win levels, reset indicators, and raw metadata;
- build deterministic state hash.

`GAME_OVER` must set `game_over=True` and must not set the agent-loop success terminal flag solely because of its state name.

## 5. ARGALite implementation

### 5.1 Background

Use border dominance when sufficiently strong; otherwise use global mode. Background estimation must be deterministic.

### 5.2 Components

Use 4-connectivity. When `merge_multicolor_components=True`, adjacent non-background colors form one component. Otherwise, components are color-specific.

### 5.3 Object features

Compute:

- bbox and centroid;
- area and color histogram;
- local color mask and binary shape mask;
- shape signature independent of position;
- hole count;
- topology signature;
- horizontal/vertical symmetry;
- border contacts;
- semantic tags;
- salience score.

### 5.4 Persistent matching

`GameMemory.assign_object_tracks()` performs deterministic greedy matching over compatible prior tracks. A candidate pair is admissible only when shape/topology/color/area compatibility is sufficient. Sort pair scores and allocate each frame object and prior track at most once.

New track IDs are deterministic hashes of game, level, intrinsic object features, and a level-local sequence number.

### 5.5 Coordinate targets

Generate bounded targets from:

- centroid;
- bbox center;
- corners and edge midpoints;
- selected relation hotspots;
- last changed region;
- largest empty-region center;
- grid center.

A candidate ID is derived from stable target semantics, not current coordinates. Coordinates remain current payload data.

## 6. Relation model

`relations.build_relations()` must produce deterministic, globally capped relation records. Relation identity must exclude the mutable metric value.

Required helpers:

- lookup relation by stable ID;
- recompute metric after transition using persistent endpoints;
- recover endpoints when relation is absent but objects survive.

At minimum, centroid distance, row/column deltas, bbox gap, line endpoint gap, and containment outside distance must be supported where applicable.

## 7. Verification binder

`VerificationBinder.bind(step, snapshot, hypothesis_id)` returns a contract or `None`.

Binding algorithm:

1. validate target IDs against current snapshot;
2. when relation targeted, recover endpoint object IDs and metric;
3. when object targeted, capture bbox and intrinsic target signature;
4. when coordinate candidate targeted, validate candidate and recover its object/relation linkage;
5. choose explicit contract kind if valid;
6. otherwise infer from target and expected-observation text;
7. derive semantic question type and deterministic question ID;
8. return fixed contract.

No action is emitted by the binder.

## 8. Transition judge

### 8.1 Global evidence

Compute:

- changed cells and bbox;
- color-transition histogram;
- score delta;
- level and levels-completed delta;
- terminal and game-over delta;
- coordinate target proximity;
- broad attribution category.

### 8.2 Precedence

Evaluation precedence:

1. reset transition;
2. newly observed `GAME_OVER` after non-reset action -> negative;
3. successful terminal, level progress, or positive score -> positive;
4. unbound candidate -> never semantic confirmation;
5. selected fixed contract.

### 8.3 Contract rules

Implement the exact contract behavior described in the architectural specification. In particular, the default/unbound path for any visible change returns:

```text
truth = UNKNOWN
relevance = UNDECIDED
progress = UNKNOWN
reason = unbound_visible_change_not_semantic_proof
```

### 8.4 Information gain

`action_effects.py` maintains categorical question domains and normalized entropy reduction. Posterior updates are permitted only from typed contract outcomes. Repeating the same resolved outcome yields zero gain.

## 9. Memory engineering

### 9.1 Attempt accounting

Centralize accounting in `GameSession._emit()` for all sources. Store attempt count under:

```text
hash(candidate.suppression_signature, snapshot.semantic_state_signature)
```

This prevents exact same-state loops while permitting reconsideration after new evidence.

Simple-action discovery counts may remain level-global to ensure initial coverage. Coordinate probe totals are level-local and source-independent.

### 9.2 Judgment storage

`add_judgment()` must:

- append a general event;
- append to failed or irrelevant memory according to semantics;
- update last changed-region center;
- update action-effect record;
- record coordinate effect and exact no-effect suppression signature;
- preserve verifier contract and information-gain metadata.

### 9.3 Qwen summaries

Expose compact:

- recent transitions;
- confirmed, rejected, unknown, and irrelevant hypotheses;
- do-not-repeat hints;
- known action effects;
- unresolved semantic questions;
- prior coordinate probes;
- `previous_attempts_feedback`.

## 10. Hypothesis-bank engineering

### 10.1 Parsing

Accept `v8.3.semantic_output` and `v8.3.coordinate_output`. V8.2 schema names may be accepted during migration, but all generated packets and fake outputs use V8.3.

Reject raw coordinates when disabled and every invented ID.

### 10.2 Executability

`has_executable_candidate(snapshot)` must dry-run the same binder path used by actual candidate selection. It must not return true solely because a queue contains an item.

### 10.3 Update

- positive verified progress increases priority/confidence;
- false or invalid marks the item rejected;
- irrelevant decreases priority and consumes the tested step;
- unknown decreases priority slightly;
- consumed, invalid, and expired items are purged.

## 11. Qwen packet and backend

### 11.1 Packet schemas

Semantic packet:

```text
v8.3.semantic_level_packet
```

Coordinate packet:

```text
v8.3.coordinate_research_packet
```

Outputs:

```text
v8.3.semantic_output
v8.3.coordinate_output
```

### 11.2 Tail renderer

`llm._prompt()` must manually separate broad context and tail sections. It must use `sort_keys=False`. The final line is a strict JSON-only instruction.

### 11.3 Compaction

- normal: preserve packet;
- aggressive: cap objects, relations, transitions, and memory lists; remove local masks;
- minimal: preserve lifecycle, action surface, valid IDs, tail sections, compact grid header, top objects, and top relations.

### 11.4 Invocation

`llama-cli` command includes:

```text
--model
--temp
--seed
--n-predict
--ctx-size
--no-display-prompt
--simple-io
--prompt
```

Timeout uses the current session configuration. Parse either the entire response as JSON or the first balanced valid JSON object. If a nonempty semantic packet yields empty output and retry is enabled, retry once with minimal compaction.

## 12. Session engineering

### 12.1 act()

1. adapt observation and build snapshot;
2. if pending exists, compatibility-commit only when observation has advanced;
3. apply level boundary after commit;
4. if `GAME_OVER`, emit reset;
5. if success terminal, return guard reset only for compatibility; outer harness should already stop;
6. if not started, emit reset;
7. run role selection and optional Qwen call;
8. choose candidate;
9. preflight;
10. fallback if required;
11. `_emit()` and return official action dictionary.

### 12.2 observe_action_result()

- return false and increment duplicate count if no pending action exists;
- require after-observation when pending exists;
- build after snapshot;
- commit pending transition exactly once;
- apply level boundary;
- update latest snapshot;
- return true.

### 12.3 reset transitions

A reset transition is ingested and clears pending state but is not passed through semantic action verification. The prior failing action has already been committed on the frame that reported `GAME_OVER`.

### 12.4 GAME_OVER

`_handle_game_over()`:

- requires reset support configuration;
- increments telemetry counters;
- clears level-local concrete hypotheses;
- emits `RESET` with a pending token;
- never terminates because of reset count.

## 13. Competition notebook engineering

### 13.1 Generated shim

The generated `ARC_AGI_Agent` delegates to one `GameSession`. Its `act()` must call `update_runtime_config()` before `session.act()`. Its `observe_action_result()` must call the real session method. Telemetry must delegate to the session.

### 13.2 Direct loop

For every accepted agent action:

```text
native_action = delegate.act(before_observation, dynamic_config)
env.step(action)
delegate.observe_action_result(after_observation)
```

Initial start reset may be emitted directly by the loop because no agent pending token exists yet.

On `GAME_OVER`, the loop does not break. It calls the delegate, which emits reset. The loop stops on environment success, wall-clock timeout, configured action cap, or fatal error.

Before returning, `pending_official_transition` must be false. When all ordinary candidates are exhausted, the session uses a least-used internal liveness revisit rather than raising; this path remains verifier-observed and is exposed in telemetry.

### 13.3 Phase A

Phase A performs package extraction, structural preflight, optional light runtime diagnostics, and a dummy submission file. It does not execute the previous heavy/max-context Qwen smoke call.

## 14. Test requirements

The test suite must cover:

- 1x1 and 64x64 grids;
- palette validation;
- bounded coordinate targets;
- object track stability after movement;
- stable relation ID with changing metric;
- invented-ID rejection;
- stale candidate executability;
- no-op unknown/irrelevant behavior;
- generic visual change not confirming a hypothesis;
- relation error decrease positive judgment;
- target-external change not confirming local affordance;
- first typed no-effect information gain and repeated irrelevance;
- state-scoped repeat accounting;
- explicit transition commit and duplicate guard;
- dynamic timeout propagation;
- replayable `GAME_OVER` reset;
- prompt-tail literal order;
- minimal compaction preservation;
- generated competition shim compilation and commit-path presence;
- no-crash least-used liveness revisits after epistemic exhaustion;
- fake Qwen role budgets.

Tests must pass without `arcengine`, the competition gateway, or a real model.

## 15. Build and validation commands

```bash
python -m compileall -q agent v8_agent build_notebook.py build_notebook_smoke.py
pytest -q
python build_notebook_smoke.py
python build_notebook.py
```

Run the main builder last so the final notebook is the production V8.3 notebook.

## 16. Definition of done

The release is complete when:

- all tests pass;
- all Python files compile;
- generated notebook builds below `MAX_NOTEBOOK_BYTES`;
- generated shim compiles;
- notebook payload includes all active V8.3 modules;
- no V8.2 user-facing markers remain;
- archive contains V8.3 specs, README, changelog, source, tests, and notebook;
- archive excludes caches and generated bytecode.
