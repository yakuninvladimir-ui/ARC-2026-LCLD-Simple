# ARC-AGI-3 LCLD Reverse-Semantic Trajectory Agent
# Engineering Specification
# Version 9.0

Status: implementation-aligned engineering contract of the supplied competition notebook

Date: 2026-07-17

Companion document: `ARCHITECTURAL_SPECIFICATION_V9.md`

## 0. Engineering objective

Implement and maintain the supplied V9 ARC-AGI-3 agent as one small active
Python package with:

- deterministic observation normalization and hashing;
- object-centric ARGALite parsing;
- persistent level-local object tracks;
- stable relation identities;
- an exact same-color component graph;
- deterministic current-frame PNG and hexadecimal geometry views;
- mandatory typed action research over the official action surface;
- a strict multimodal Qwen packet;
- packet-specific constrained JSON decoding;
- one to three complete semantic trajectories per primary call;
- deterministic reverse-semantic binding to current IDs;
- reset-safe rebinding of sibling alternatives;
- fixed primitive verification contracts;
- exactly-once ingestion of accepted official transitions;
- trajectory-level semantic evaluation;
- game-scoped action, invariant, attempt, and failure memory;
- isolated direct competition orchestration;
- no network dependency for tests.

The active implementation must remain Python 3.12-compatible, offline during
competition inference, deterministic outside model generation, and auditable at
every primitive action boundary.

### 0.1 Artifact and validation snapshot

```text
notebook: arc-prize-2026-lcld-qwen-v9(1).ipynb
notebook SHA-256: 12917cc19473f47ccfcf9d551d7296ba331a0168deb57e1a2f3584e999b1f023
embedded payload ZIP SHA-256: c128de29006dd2b25e85ab577c385bee2fe99a48c9249659ac57812b2fef67cf
embedded payload bytes: 145125
active embedded source files: 29
package version: 9.0.0
source compileall: PASS
fake-backend session creation: PASS
one-action pending-token commit smoke: PASS
real Qwen inference: NOT EXECUTED IN THIS AUDIT
real ARC gateway/scorecard run: NOT EXECUTED IN THIS AUDIT
```

The specification must not claim live competition success based only on static
compilation or fake-backend smoke tests.

### 0.2 Active-path rule

Every active module must be:

- imported by the competition path or by an active package facade;
- covered by deterministic unit/integration tests; or
- explicitly documented as a compatibility facade.

The runtime must not contain a parallel inactive policy, verifier, memory, or
packet builder whose behavior can diverge silently from the competition path.

## 1. Active source layout

The embedded source tree is:

```text
Code/
  submission.py
  kaggle_agent.py
  lcld_competition_child.py
  v9_agent/
    __init__.py
    action_adapter.py
    action_effects.py
    action_explorer.py
    arga_lite.py
    component_graph.py
    config.py
    coordinate_research.py
    deliberation.py
    frame_media.py
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
    reverse_semantics.py
    session.py
    types.py
    verification.py
    README.md
```

### 1.1 Module responsibilities

| Module | Active responsibility |
|---|---|
| `submission.py` | Competition default configuration and ARC frame normalization helpers. |
| `kaggle_agent.py` | Framework-compatible agent wrapper, explicit transition ingestion, reset guard, telemetry, cleanup. |
| `lcld_competition_child.py` | Isolated direct game loop, worker concurrency, scorecard lifecycle, result manifest. |
| `v9_agent/__init__.py` | Public API, package version, `V9Config` alias. |
| `config.py` | Frozen configuration dataclass, environment and mapping overrides, clamping. |
| `types.py` | All enums and immutable transport dataclasses. |
| `game_adapter.py` | Raw observation to canonical `WorldState`. |
| `observe.py` | Stable hashing and low-level observation utilities. |
| `arga_lite.py` | Background estimation, objects, persistent-track integration, coordinate candidates, snapshots. |
| `relations.py` | Deterministic relation generation and relation hashing. |
| `component_graph.py` | Full-frame same-color component topology and compact geometry. |
| `frame_media.py` | Deterministic PNG encoding and palette rendering. |
| `action_explorer.py` | Typed simple-action and coordinate exploration order. |
| `coordinate_research.py` | Compatibility facade for coordinate research interfaces. |
| `action_effects.py` | Typed semantic questions, posterior entropy, action-effect merge. |
| `qwen_packet.py` | Layered packet, aliases, action diffs, memory views, compaction inputs. |
| `llm.py` | Prompt rendering, strict schemas, backend invocation, image transport, extraction, traces. |
| `qwen_roles.py` | Per-game/per-level model budget state. |
| `hypothesis_bank.py` | Output parsing, validation, binding, route expansion, active trajectory, sibling reset. |
| `reverse_semantics.py` | Objective binding, metric computation, trajectory evaluation, invariant derivation. |
| `verification.py` | Contract binder and preflight legality/repeat suppression. |
| `judge.py` | Official transition comparison and fixed-contract judgment. |
| `memory.py` | Game memory, tracks, attempts, action diffs, semantic feedback, rebinding. |
| `policy.py` | Priority selection among confirmed, coordinate, semantic, and research queues. |
| `deliberation.py` | Bounded deterministic deliberation helpers. |
| `action_adapter.py` | Candidate-to-ARC action and payload compatibility facade. |
| `logging.py` | Structured trace and telemetry helpers. |
| `session.py` | Main state machine and exactly-once official transition commit. |

### 1.2 Public package API

`v9_agent.__init__` must export:

```python
V8Config
V9Config
config_from_mapping
default_config_dict
GameSession
LevelRunLimitReached
__version__
```

`V9Config` is the public V9 alias of the implementation class named `V8Config`.
`__version__` is `9.0.0`. The class name and active `v8.*` wire identifiers are
frozen compatibility identifiers. Code must not infer architecture version or
behavior from those literal names.

## 2. Configuration contract

### 2.1 Configuration object

The configuration is an immutable slotted dataclass. `config_from_mapping()`
constructs a new normalized instance from:

1. dataclass defaults;
2. mapping values and accepted aliases;
3. environment-variable overrides;
4. backend-specific clamping.

A live `GameSession.update_runtime_config()` must replace the active session
configuration, recreate or update dependent Qwen components, and preserve
already committed game evidence unless the caller explicitly starts a new game.

### 2.2 Qwen backend fields

```text
enable_qwen: bool = true
qwen_backend: disabled | fake | ollama | vllm | qwen_local | llama_cli
qwen_model_path: str | null
qwen_llama_cli_path: str | null
qwen_ollama_base_url: http://127.0.0.1:11434
qwen_ollama_model: qwen_local_3_5
qwen_ollama_keep_alive: -1
qwen_vllm_base_url: http://127.0.0.1:1234/v1
qwen_vllm_api_key: EMPTY
qwen_vllm_model: vrfai/Qwen3.6-27B-FP8
qwen_llama_device: str | null
qwen_split_mode: str
qwen_tensor_split: str
qwen_gpu_layers: 999
qwen_temperature: 0.0 dataclass / 0.6 competition override
qwen_top_k: 20
qwen_top_p: 0.95
qwen_min_p: 0.0
qwen_presence_penalty: 1.5 dataclass / 0.0 competition override
qwen_repeat_penalty: 1.0
qwen_seed: 0
qwen_timeout_seconds: 1000
qwen_context_tokens: 131072
qwen_minimum_acceptance_context_tokens: 65536
qwen_max_input_tokens: 65536
qwen_max_output_tokens: 49152
qwen_reserved_runtime_margin_tokens: 8192
qwen_enable_thinking: false dataclass / true competition override
qwen_reasoning_mode: off dataclass / on competition override
qwen_reasoning_budget_tokens: 0
qwen_spec_type: str
qwen_spec_draft_n_max: 0
qwen_require_runtime: false dataclass / true competition override
qwen_trace_dir: str | null
qwen_empty_output_retry_enabled: false
qwen_multimodal_enabled: false dataclass / true competition override
```

`config_from_mapping()` must disable thinking unless the selected backend is
`vllm` and `qwen_enable_thinking` is true. When thinking is disabled it must set:

```text
qwen_enable_thinking = false
qwen_reasoning_mode = off
qwen_reasoning_budget_tokens = 0
```

### 2.3 Prompt and packet fields

```text
prompt_compaction_strategy: normal | aggressive | minimal = normal
prompt_compression_trigger_ratio: 0.82
prompt_tail_priority_enabled: true
include_full_grid_in_qwen_packet: true
include_object_local_masks: true
include_component_graph_in_qwen_packet: true
frame_png_cell_scale: 8
max_hex_patches_in_packet: 16
max_hex_patch_side: 24
max_action_diffs_in_packet: 20
max_changed_cells_per_action_diff: 256
max_changed_row_runs_per_action_diff: 6
max_components_in_packet: 96
max_component_shape_runs: 32
max_component_boundary_corners: 32
max_objects_in_packet: 64
max_object_masks_in_packet: 32
max_relations_in_packet: 192
max_semantic_objects_in_packet: 24
max_coordinate_objects_in_packet: 48
max_semantic_relations_in_packet: 24
max_semantic_groups_in_packet: 24
max_recent_transitions_in_packet: 12
max_memory_notes_in_packet: 20
max_active_hypotheses_in_packet: 30
max_semantic_feedback_bindings: 8
max_semantic_feedback_trajectories: 8
max_semantic_feedback_invariants: 12
semantic_invariant_confirmation_count: 2
```

### 2.4 Qwen call budgets

```text
max_qwen_calls_per_game: 20
max_primary_qwen_calls_per_level: 1
max_reserve_qwen_calls_per_level: 0
max_coordinate_qwen_calls_per_level: 1
max_total_qwen_calls_per_level: 2
min_steps_between_qwen_calls: 3
qwen_stall_threshold: 3
```

Every invocation attempt consumes its role and scope budget, including timeout,
empty, malformed, or schema-invalid responses.

### 2.5 Planning and exploration limits

```text
max_test_plan_length_default: 3
max_test_plan_length_confirmed_rule: 5
max_qwen_trajectory_steps: 50
max_simple_action_probes_per_level: 10
max_action_memory_records_in_packet: 40
max_visual_change_groups_in_memory: 12
max_visual_change_locations_per_group: 16
max_clue_patterns_in_packet: 64
max_coordinate_candidates_in_packet: 96
max_coordinate_probes_per_level: 24
max_coordinate_probe_repeats_per_signature: 1
max_same_state_action_repeats: 1
reject_unchanged_failed_trajectories: true
```

### 2.6 Safety and verification thresholds

```text
allow_qwen_raw_coordinates: false
require_coordinate_candidate_id: true
require_json_only: true
reject_hallucinated_ids: true
passive_attribution_enabled: true
passive_change_threshold_cells: 3
mixed_change_ratio_threshold: 0.50
relation_error_epsilon: 1e-6
local_target_overlap_min_fraction: 0.10
information_gain_min_threshold: 0.03
merge_multicolor_components: true
preserve_object_tracks: true
default_coordinate_action_id: ACTION6
```

### 2.7 Competition limits

```text
max_actions_per_game: 200
max_actions_per_level: 0              # disabled
max_level_attempts: 4
game_wall_clock_limit_seconds: 8000
reset_on_game_over: true
max_game_over_resets_per_game: 0      # unlimited telemetry counter
max_game_over_resets_per_level: 0     # unlimited telemetry counter
```

The reset-count fields do not terminate play. Action and wall-clock limits are
owned by the outer harness.

### 2.8 Required environment variables

The loader must support at least these active environment names:

```text
ARC_ENABLE_LLM_SEMANTIC_ADVISOR
ARC_LLM_ADVISOR_BACKEND
ARC_QWEN_MODEL_PATH
ARC_LLM_MODEL_PATH
ARC_QWEN_LLAMA_CLI_PATH
ARC_QWEN_LLAMA_DEVICE
ARC_QWEN_SPLIT_MODE
ARC_QWEN_TENSOR_SPLIT
ARC_QWEN_GPU_LAYERS
ARC_QWEN_TIMEOUT_SECONDS
ARC_QWEN_CONTEXT_TOKENS
ARC_QWEN_MINIMUM_ACCEPTANCE_CONTEXT_TOKENS
ARC_QWEN_MAX_INPUT_TOKENS
ARC_QWEN_MAX_OUTPUT_TOKENS
ARC_QWEN_RESERVED_RUNTIME_MARGIN_TOKENS
ARC_QWEN_TEMPERATURE
ARC_QWEN_TOP_K
ARC_QWEN_TOP_P
ARC_QWEN_MIN_P
ARC_QWEN_PRESENCE_PENALTY
ARC_QWEN_REPEAT_PENALTY
ARC_QWEN_SEED
ARC_QWEN_ENABLE_THINKING
ARC_QWEN_REASONING_MODE
ARC_QWEN_REASONING_BUDGET_TOKENS
ARC_QWEN_TRACE_DIR
ARC_QWEN_MULTIMODAL_ENABLED
ARC_QWEN_VLLM_BASE_URL
ARC_QWEN_VLLM_API_KEY
ARC_QWEN_VLLM_MODEL
ARC_MAX_QWEN_PRIMARY_CALLS_PER_LEVEL
ARC_MAX_QWEN_REPLAN_CALLS_PER_LEVEL
ARC_MAX_QWEN_COORDINATE_CALLS_PER_LEVEL
ARC_MAX_TOTAL_QWEN_CALLS_PER_LEVEL
LCLD_REQUIRE_QWEN_RUNTIME
LCLD_MAX_ACTIONS_PER_GAME
LCLD_MAX_LEVEL_ATTEMPTS
LCLD_MAX_ACTIONS_PER_LEVEL
LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS
LCLD_RESET_ON_GAME_OVER
```

Names beginning with `ARC_V8_` remain accepted by the code as public
compatibility keys. They must map to the V9 fields above and must not create a
second configuration path.

## 3. Core enums and data types

All policy, verifier, memory, and transport state must use the dataclasses and
enums in `types.py` rather than ad hoc dictionaries, except at explicit JSON or
ARC API boundaries.

### 3.1 Evidence and judgment enums

```python
TriTruth = TRUE | FALSE | UNKNOWN
Relevance = RELEVANT | IRRELEVANT | UNDECIDED
Validity = VALID | INVALID | UNCHECKED
Progress = POSITIVE | NEGATIVE | NEUTRAL | UNKNOWN
Attribution = ACTION_LINKED | PASSIVE_POSSIBLE | MIXED_OR_UNCERTAIN | NO_VISIBLE_CHANGE
MechanicResult = MATCH | MISMATCH | UNKNOWN
SemanticJudgment = REQUIRED | FORBIDDEN | IRRELEVANT | UNRESOLVED
EvidenceAuthority = OFFICIAL_OBSERVATION | DETERMINISTIC_BINDER | DETERMINISTIC_VERIFIER | QWEN_PROPOSAL
EvidenceStatus = OBSERVED_ONCE | CONFIRMED | CONTRADICTED | IRRELEVANT | UNRESOLVED
```

### 3.2 Qwen and binding enums

```python
QwenRole = PRIMARY | RESERVE | COORDINATE
BindingStatus = GROUNDED | PARTIAL | REJECTED
MetricDirection = MINIMIZE | MAXIMIZE | TARGET | BOOLEAN
```

### 3.3 Verification contract kinds

```python
VerificationContractKind =
  NO_OP_TEST |
  LOCAL_TARGET_CHANGE |
  OBJECT_DISPLACEMENT |
  RELATION_ERROR_DECREASE |
  ACTION_SURFACE_CHANGE |
  SCORE_OR_TERMINAL |
  ACTION_EFFECT_DISCOVERY
```

### 3.4 Goal operators

```python
GoalOperator =
  MOVE_TOWARD | ALIGN | MATCH_GEOMETRY | OVERLAP |
  CONNECT | BRIDGE_GAP | EXTEND_LINE | CONTAIN |
  COMPLETE_PATTERN | MATCH_STATE | ACTIVATE |
  CHANGE_ACTION_SURFACE | PROBE_AFFORDANCE |
  PROBE_RELATION | OTHER
```

### 3.5 Required dataclasses

The following dataclasses are normative:

- `WorldState`;
- `ObjectRecord`;
- `RelationRecord`;
- `CoordinateTargetCandidate`;
- `ARGALiteSnapshot`;
- `SemanticObjective`;
- `GoalMetricSpec`;
- `SemanticBindingResult`;
- `VerificationContract`;
- `CandidateAction`;
- `PendingAction`;
- `PreflightResult`;
- `Judgment`;
- `MemoryEvent`;
- `CoordinateEffectRecord`;
- `ActionEffectRecord`;
- `SemanticQuestion`;
- `TestStep`;
- `HypothesisItem`;
- `TrajectoryEvaluation`;
- `QwenBudgetState`.

### 3.6 CandidateAction contract

`CandidateAction` contains:

```text
action_id
x | null
y | null
coordinate_candidate_id | null
hypothesis_id | null
reason
source
verification_contract | null
allow_exhaustion_revisit
```

`to_arc_action()` emits both `action_id` and `id`, optional coordinate payload,
and bounded reasoning metadata. Its suppression signature must include action,
coordinate candidate or exact coordinate, hypothesis/contract scope, and any
field required to distinguish legitimate alternative execution.

### 3.7 PendingAction contract

A session has zero or one `PendingAction`:

```text
before_snapshot
action
hypothesis_id
reason
token_id
```

The token ID must be deterministic from the before state, action, session
sequence, and relevant contract identity. It must never be exposed as model
input authority.

### 3.8 Judgment contract

`Judgment` must carry at least:

```text
truth
relevance
validity
progress
attribution
mechanic_result
semantic_judgment
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
affected_objects
affected_relations
terminal_delta
```

`observed_delta` is a bounded factual transition record, not an explanation.

## 4. Observation adapter

`GameAdapter.to_world_state()` must:

1. accept a normalized 2D grid or a temporal frame structure;
2. collapse temporal axes to the last visible 2D frame;
3. convert all cells to integers;
4. reject nonrectangular grids;
5. reject grid dimensions outside `1..64`;
6. reject palette values outside `0..15`;
7. normalize action IDs and state names;
8. read current available actions from official metadata;
9. distinguish `GAME_OVER` from success terminal;
10. retain score, `levels_completed`, `win_levels`, `full_reset`, and raw
    metadata;
11. assign a monotonic session step index;
12. build a deterministic state hash.

`submission.frame_to_world_json()` must accept ARC `FrameData.frame` represented
as a temporal list of 2D grids or as an already normalized 2D grid. It must not
interpret the temporal axis as RGB channels or a 3D world.

### 4.1 Action normalization

Action normalization must map:

```text
0 -> RESET
1..7 -> ACTION1..ACTION7
Enum.name/value -> uppercase terminal token
numeric string -> same numeric mapping
```

Unknown action strings may pass through only as normalized uppercase opaque IDs.
No semantic name is attached.

## 5. ARGALite implementation

### 5.1 Background estimation

The parser computes border counts. If one border color occupies at least `0.45`
of the border cells, it becomes the background. Otherwise the global modal color is used. Ties must resolve
deterministically by palette ID.

### 5.2 Connected components

Use 4-connectivity. With `merge_multicolor_components=true`, adjacent
non-background cells may form one multicolor object. The parser may additionally
recover same-color subobjects with area at least the implementation threshold.
With merging disabled, connected components are color-specific.

### 5.3 Object features

For each object compute:

```text
frame_object_id
bbox_rc = (r0, c0, r1, c1)
centroid_rc
area
colors
color_histogram
shape_signature
local_mask_hex_rows
holes
symmetry
border_contact
tags
topology_signature
stable_hash
salience
```

The shape signature must be translation-invariant. The frame object ID may
include observation-local ordering; the persistent object ID may not.

### 5.4 Persistent track assignment

`GameMemory.assign_object_tracks()` performs deterministic bipartite greedy
matching over all compatible object-track pairs. Pair order is sorted by:

```text
(match_score, track_id, frame_object_index)
```

The score is:

\[
D + 8R_A + P_C + P_S + P_T,
\]

where:

- \(D\) is centroid displacement;
- \(R_A=|A_{new}-A_{old}|/\max(1,A_{old})\);
- \(P_C=0\) for exact colors, `2` for color overlap, otherwise `3` when shape
  compatibility still permits matching;
- \(P_S=0\) for exact shape, otherwise `4`;
- \(P_T=0\) for exact topology, otherwise `1`.

A pair is inadmissible if neither color overlap nor exact shape exists. A
non-exact-shape pair is inadmissible when relative area difference exceeds
`0.55`.

Unmatched objects receive a new track hash scoped by game, level, shape,
palette, and deterministic sequence.

### 5.5 Snapshot hashes

The snapshot builder must compute separate deterministic hashes for:

- grid;
- object set;
- relation set;
- component graph where enabled;
- complete snapshot;
- semantic state.

A relation's current numeric value must not be included in its identity hash,
but may be included in the current relation-state hash.

## 6. Relation model

The relation builder must be deterministic and bounded. Required relation types
include:

```text
same_color
same_shape
translated_shape
mirror_candidate
rotation_candidate
aligned_row
aligned_col
left_of
right_of
above
below
near
separated_by_gap
contains
frame_contains
line_continuation
button_like_structure
unique_symbol_pair
repeated_pattern
```

Each `RelationRecord` contains:

```text
relation_id
relation_type
a
b
metric_name | null
value | null
confidence
salience
signature
```

`relation_id` is derived from relation type, a canonical endpoint order, and
metric name. The metric value is state, not identity.

Relations used in Qwen output must refer only to tracked object IDs in the
current packet whitelist.

## 7. Component graph and frame media

### 7.1 Component graph

`component_graph.py` must build an observation-local same-color 4-connected
component graph over the full grid. Each component may include:

- palette value;
- area;
- bounding box and centroid;
- compact row/shape runs;
- topology and nesting depth;
- parent/child containment;
- adjacent component IDs and shared-edge counts;
- boundary corners;
- shape/color hashes;
- overlapping tracked object references.

Packet caps:

```text
max components: 96
max shape runs per component: 32
max boundary corners per component: 32
```

Component IDs are never legal Qwen action or objective IDs. Only object
references exposed by the component may be used for grounding.

### 7.2 PNG media

`current_frame_png()` must:

- render the exact palette grid with no interpolation;
- use integer cell scale, default `8`;
- use a deterministic local ARC palette;
- return width, height, MIME type, hash, and base64 payload;
- mark transport as an out-of-band multimodal attachment;
- state whether the configured backend actually receives the image.

The textual prompt must omit `data_base64`; the backend request attaches the
same payload as an image data URL when vision is enabled.

### 7.3 Hex patches

The packet builder may include at most 16 exact patches, each with side at most
24 cells. Patches must identify their source object/group and coordinate bounds.
When local mask or patch detail is omitted due to budget, the omission must be
explicit.

## 8. Action research engineering

### 8.1 Research completeness

`GameMemory.action_research_status(snapshot)` returns:

```text
required_action_ids
researched_action_ids
missing_action_ids
missing_simple_action_ids
missing_coordinate_action_ids
intrinsically_known_undo_action_ids
```

Required actions are all currently available non-reset gameplay actions. An
action counts as researched when an `ActionEffectRecord` has confidence at least
`0.45`, or when the environment contract identifies it as undo.

The primary semantic model call is forbidden while `missing_action_ids` is
nonempty.

### 8.2 Simple action probes

`action_explorer.py` must emit typed `ACTION_EFFECT_DISCOVERY` candidates. The
research order is deterministic.

Supported probe modes:

- `initial_action_probe`;
- `action_probe_control_before`;
- `action_probe_new_action`;
- `action_probe_control_after`.

A control/new/control sequence is permitted when a new action appears and a
known action can test whether control semantics changed. Co-visible unknown
actions on the first surface are ordinary one-shot probes.

The per-level simple-probe cap is 10 by default.

### 8.3 Research-entry reset

After mandatory action research, if accepted research actions changed the
current state or the active control cycle requires restoration, the session
must emit `research_entry_reset` before calling the primary semantic role.

The reset result must be ingested before packet construction. The packet must
mark prior research actions as history-only when their effects are not applied
to the restored frame.

### 8.4 Coordinate candidates

Coordinate candidates are deterministic records:

```text
candidate_id
x
y
source
object_id | null
relation_id | null
region_signature
target_signature
reason
salience
```

Candidate generation may use object centroids, relation anchors, occupied cells,
empty-region centers, and bounded structural slots. Candidate IDs must be stable
for the current semantic state.

### 8.5 Coordinate plan validation

The coordinate Qwen response must satisfy:

```json
{
  "schema_version": "v8.4.coordinate_plan",
  "decision": "PLAN",
  "mechanism_hypothesis": "...",
  "coordinate_action_id": "<whitelisted action>",
  "candidate_sequence": [
    {"coordinate_candidate_id": "<whitelisted candidate>"}
  ],
  "completion_criterion": "...",
  "confidence": 0.0
}
```

Requirements:

- at least one candidate;
- no duplicate candidate IDs;
- no duplicate physical locations;
- current coordinate action ID only;
- no raw coordinates;
- at most configured trajectory steps;
- one execution per candidate.

## 9. Qwen layered packet engineering

### 9.1 Packet schema

`LayeredQwenPacketBuilder` must produce:

```json
{
  "schema_version": "v8.8.layered_observation",
  "state": {},
  "current_frame_png": {},
  "object_layer": {
    "coordinate_system": "x=column,y=row; origin=top_left",
    "segmentation_contract": "...",
    "objects": [],
    "exact_geometry_groups": [],
    "relations": [],
    "component_graph": {}
  },
  "hex_patches": [],
  "action_space": {
    "actions": [],
    "current_available_action_ids": [],
    "undo_action_ids": [],
    "possible_action_ids": [],
    "coordinate_candidates": []
  },
  "action_diffs": [],
  "memory": {
    "confirmed_effects": [],
    "completed_levels": [],
    "attempts": [],
    "semantic_feedback": {}
  },
  "execution_constraints": {
    "max_plan_steps": 50,
    "allowed_action_ids": [],
    "allowed_object_ids": [],
    "allowed_relation_ids": [],
    "allowed_coordinate_candidate_ids": [],
    "raw_coordinates_allowed": false
  }
}
```

The active schema identifier is literal and must not be renamed without a
migration layer and packet replay tests.

### 9.2 Focus selection

The builder selects a bounded focus subset based on:

- object salience;
- coordinate role versus semantic role;
- relation support;
- memory references;
- failed bindings and active alternatives;
- exact geometry groups;
- coordinate candidate coverage.

Limits for the semantic role are 24 objects and 24 relations by default. The
coordinate role may include up to 48 objects. Total parsed objects may be up to
64 and total relations up to 192 before focus projection.

### 9.3 Alias translation

Internal IDs may be long hashes. The packet builder assigns frame-local aliases
to objects, relations, candidates, and geometry groups. It must:

1. build a one-to-one alias map;
2. rewrite every packet reference;
3. validate no dangling aliases remain;
4. translate model output back to internal IDs;
5. reject an alias that was not in the packet;
6. preserve failed-attempt and semantic-feedback references when the referenced
   entity remains in focus;
7. remove or explicitly mark stale references when compaction excludes them.

### 9.4 Action diffs

An action-diff sample must retain:

- action and target identity;
- before and after state/surface metadata;
- sample step index;
- grouped observation count and step range;
- bounded pixel changes or row runs;
- color-transition histogram;
- object lifecycle and motion deltas;
- synchronous local visual evidence;
- coordinate-cell before/after where applicable;
- level, terminal, and game-over deltas;
- explicit incomplete-coverage flag when truncated.

Grouping may collapse repeated normalized effects, but it may not fabricate an
average frame or combine semantically incompatible outcomes.

### 9.5 Memory packet

The memory view must keep these categories separate:

- confirmed effects;
- completed-level evidence;
- failed attempts;
- rejected proposals;
- research runs;
- complete attempt execution;
- semantic bindings;
- trajectory evaluations;
- invariant variants and status.

The packet must state that research provenance does not create an implicit
state reset.

## 10. Prompt renderer and model backend

### 10.1 Prompt tail

The final prompt tail must contain, in order of high preservation priority:

1. current execution constraints and whitelists;
2. exact role-specific output contract;
3. packet-specific allowed JSON schema summary;
4. instruction to return JSON only.

Under compaction, the renderer must preserve the tail and current state before
old low-salience geometry.

### 10.2 Semantic role instruction

The primary prompt must state:

- the current frame and action surface are the trajectory entry state;
- history-only research effects are not currently applied;
- the goal is unknown;
- the model must return one to three distinct complete trajectories;
- each action run's `repeat` is the total consecutive count;
- coordinate runs have `repeat=1` and unique targets;
- the first action is currently executable;
- later-surface actions require an earlier observed enabling action;
- rejected exact trajectories may not be repeated unchanged;
- movement or visible change alone is not goal completion;
- the executor will re-observe after every action, but the route must be complete
  at proposal time.

### 10.3 Token estimation and compaction

The implementation estimates prompt tokens approximately as:

```text
estimated_tokens = (character_count + 2) // 3
```

Compaction triggers at:

\[
\text{estimated input} >
\text{compression ratio}\times\text{maximum input tokens}.
\]

Strategies:

- `normal`: retain balanced object, relation, memory, and diff detail;
- `aggressive`: reduce historical and low-salience structural detail earlier;
- `minimal`: preserve only the current executable state, critical evidence,
  whitelists, and output contract.

After every compaction pass, reference validation must run again.

### 10.4 Backends

Required backends:

- `fake` — deterministic tests;
- `ollama` — local HTTP generation;
- `vllm` — OpenAI-compatible chat completions with strict response schema;
- `qwen_local` / `llama_cli` — local command execution;
- `disabled` — no model.

The vLLM request must support:

- system and user messages;
- attached current-frame PNG as a data URL when enabled;
- strict JSON schema response format;
- temperature, top-k, top-p, min-p, presence and repeat penalties;
- thinking controls;
- timeout;
- optional bounded trace output.

### 10.5 Strict-runtime behavior

When `qwen_require_runtime=true`:

- connection failure is fatal to the affected game;
- timeout is fatal to the affected game;
- malformed or schema-invalid response is not converted into an ungrounded
  deterministic action;
- the failure is surfaced in telemetry and the child result manifest.

When strict runtime is false, the session may continue only through already
validated deterministic research or confirmed queues. It must not claim a
successful semantic model call.

## 11. Semantic output schema

### 11.1 Packet-specific schema

The primary response must be:

```json
{
  "schema_version": "v8.7.semantic_trajectories",
  "decision": "PROPOSE",
  "hypotheses": [
    {
      "id": "...",
      "family": "object_correspondence | spatial_configuration | pattern_transformation | interaction_sequence | action_surface_change | other",
      "objective": {
        "kind": "match_or_overlap | relative_arrangement | containment | connection | pattern_or_state | select_or_activate | surface_change | other",
        "source_objects": ["<allowed object>"],
        "reference_objects": ["<allowed object>"],
        "description": "..."
      },
      "relations": ["<allowed relation>"],
      "basis": "...",
      "action_runs": [
        {
          "action_id": "<allowed action>",
          "repeat": 1,
          "coordinate_candidate_id": "<allowed coordinate candidate when needed>"
        }
      ],
      "status": "complete_candidate",
      "uncertainty": "...",
      "confidence": 0.0
    }
  ]
}
```

The array length is `1..3`. Every enum and ID list is specialized to the current
packet. No additional properties are permitted.

### 11.2 vLLM schema normalization

When the serving backend does not support selected JSON Schema keywords, the
request adapter may remove unsupported declarative constraints such as
`uniqueItems` or `contains`. This does not weaken runtime validation:
`HypothesisBank` must recheck uniqueness, counts, IDs, and semantic constraints
in Python before any candidate is enqueued.

### 11.3 Output extraction

The extractor must:

- prefer the assistant's structured content field;
- tolerate fenced JSON only when it can extract one unambiguous object;
- locate the active schema marker when extra thinking text exists;
- reject empty or incomplete objects;
- preserve raw response metadata for trace when tracing is enabled;
- never execute a partially parsed action list.

## 12. Hypothesis-bank engineering

### 12.1 Parse and validate

For each semantic hypothesis, the bank must validate:

- decision and active schema;
- `complete_candidate` status;
- unique proposal-local ID;
- allowed family and objective kind;
- all source/reference/relation IDs;
- nonempty factual basis;
- action-run count and expanded step count;
- current first action;
- coordinate candidate requirements;
- coordinate candidate and physical-location uniqueness;
- coordinate repeat exactly one;
- later action-surface reachability;
- observed action-effect consistency;
- control-context consistency;
- inverse-pair and correspondence completeness rules;
- failed exact-trajectory suppression;
- non-vacuous semantic objective.

The entire response need not be rejected because one sibling is invalid. Valid
siblings may be retained, but all rejection reasons must be recorded.

### 12.2 Proposal batch

All accepted siblings receive one deterministic `proposal_batch_id`. The bank
queues them in deterministic rank order. Rank may use confidence and validation
quality, but may not override legality or binder rejection.

### 12.3 Expansion

Each `action_run` is expanded into primitive `TestStep` objects up to
`max_qwen_trajectory_steps`. Every step receives:

- action ID and optional candidate;
- 1-based trajectory index;
- hypothesis and proposal batch IDs;
- expected local effect where available;
- one fixed `VerificationContract`;
- shared semantic binding ID.

### 12.4 Active trajectory update

After each official judgment:

- `MISMATCH` stops and rejects the trajectory;
- an unavailable next action invalidates the remaining route;
- a legal mechanic match advances the cursor;
- level success completes the trajectory immediately;
- `GAME_OVER` terminates it as forbidden;
- end-of-route triggers `evaluate_trajectory()`.

### 12.5 Sibling reset

If an unsuccessful completed hypothesis has remaining siblings in the same
proposal batch, the bank creates:

```text
reason = sibling_alternative_requires_entry_reset
proposal_batch_id
completed_hypothesis_id
remaining_hypothesis_ids
```

After the official reset, `rebind_pending_alternatives()` must:

- resolve old binding IDs against current objects and relations;
- rewrite source, reference, inferred, and relation IDs;
- recompute current metric baseline;
- set `after_alternative_reset=true`;
- reject a sibling when required references cannot be resolved.

## 13. Reverse-semantic binding engineering

### 13.1 `bind_semantic_objective()`

Input:

```text
SemanticObjective
current ARGALiteSnapshot
hypothesis_id
```

Algorithm:

1. deduplicate source, reference, and relation IDs;
2. reject unknown object or relation IDs;
3. collect endpoints of cited relations;
4. infer a missing source or reference role only from those endpoints;
5. map objective kind and relation evidence to `GoalOperator`;
6. choose a supported metric when source and reference roles exist;
7. classify `GROUNDED`, `PARTIAL`, or `REJECTED`;
8. compute a binding ID from hypothesis, roles, operator, and semantic-state
   signature;
9. compute the initial metric baseline;
10. store game, level, state, surface, and evidence scope.

### 13.2 Metric definitions

For source set \(S\) and reference set \(R\), every supported pairwise metric is
reduced as:

\[
E(S,R)=\frac{1}{|S|}\sum_{s\in S}\min_{r\in R} e(s,r).
\]

#### Centroid distance

\[
e_{c}(s,r)=\sqrt{(y_s-y_r)^2+(x_s-x_r)^2}.
\]

#### Bounding-box gap

For boxes \(A=(r^A_0,c^A_0,r^A_1,c^A_1)\) and
\(B=(r^B_0,c^B_0,r^B_1,c^B_1)\):

\[
d_r=\max(0,r^B_0-r^A_1-1,r^A_0-r^B_1-1),
\]

\[
d_c=\max(0,c^B_0-c^A_1-1,c^A_0-c^B_1-1),
\]

\[
e_g(A,B)=\sqrt{d_r^2+d_c^2}.
\]

#### Containment outside distance

For container \(C\) and inner box \(I\):

\[
e_{in}(C,I)=
\max(0,r^C_0-r^I_0)+
\max(0,c^C_0-c^I_0)+
\max(0,r^I_1-r^C_1)+
\max(0,c^I_1-c^C_1).
\]

#### Palette-shape mismatch

\[
e_m(s,r)=\mathbf 1[\sigma_s\ne\sigma_r]
+\frac12\sum_k|p_s(k)-p_r(k)|.
\]

`GoalMetricSpec.computation_version` is `v9.0`.

### 13.3 Improvement function

```python
MINIMIZE: before - after
MAXIMIZE: after - before
TARGET: abs(before - target) - abs(after - target)
BOOLEAN: after - before
```

The default metric epsilon is `1e-6`.

### 13.4 Transition semantic result

For a contract with a metric:

```text
improvement > epsilon  -> Progress.POSITIVE, SemanticJudgment.REQUIRED
improvement < -epsilon -> Progress.NEGATIVE, SemanticJudgment.FORBIDDEN
otherwise              -> Progress.NEUTRAL, SemanticJudgment.IRRELEVANT
```

When either metric value is unavailable, return `UNKNOWN/UNRESOLVED` rather than
substituting another metric after observation.

## 14. Verification binder and preflight

### 14.1 Contract binder

The binder creates fixed contracts from:

- research intent;
- coordinate target;
- action effect evidence;
- relation target;
- semantic binding;
- terminal/score intent.

It may resolve relation endpoints and current target regions. It is not a route
planner or general program compiler.

### 14.2 Stable questions

A semantic question ID must be a stable hash of:

```text
question_type
action_id
target_signature
contract scope
```

Default domains:

```text
action effect / affordance: effect, no_effect, negative_effect
controllability: moved, not_moved, ambiguous
relation relevance: error_decreased, unchanged, error_increased
action surface: changed, unchanged
terminal progress: progress, no_progress, negative
```

### 14.3 Preflight rules

`PreflightValidator` must check:

1. lifecycle legality for reset;
2. current action availability;
3. coordinate presence symmetry: both `x` and `y` or neither;
4. `0 <= x,y <= 63` and within current grid;
5. `x=column`, `y=row` conversion;
6. current candidate ID existence;
7. candidate-to-coordinate equality;
8. per-level coordinate probe cap;
9. per-signature coordinate repeat cap;
10. per-attempt candidate-click uniqueness;
11. same-state action repeat cap;
12. exact no-effect suppression;
13. stale semantic binding or relation target;
14. active trajectory action-surface consistency.

The result must be a `PreflightResult` with a machine-readable reason code.

## 15. Transition judge

### 15.1 Input

`TransitionJudge` receives:

```text
before ARGALiteSnapshot
action CandidateAction
after ARGALiteSnapshot
selected VerificationContract
current GameMemory/config context
```

It must rebuild deterministic after-state semantics before evaluation.

### 15.2 Observed delta

The judge computes and bounds:

- changed-cell count and locations;
- changed bounding box and center;
- row runs when cell lists are truncated;
- palette transitions;
- persisted, appeared, disappeared, moved, recolored, and reshaped objects;
- relation value changes;
- target overlap/local change;
- available/planning/undo action surfaces before and after;
- score, level, terminal, game-over, and state-name deltas;
- clicked-cell before/after;
- target and affected IDs;
- passive/mixed attribution evidence.

### 15.3 Precedence

Judgment precedence is:

1. reset -> unknown/neutral lifecycle result;
2. `GAME_OVER` -> false, relevant, negative, mechanic mismatch, forbidden;
3. success terminal, level progress, or positive score evidence -> true,
   relevant, positive, required;
4. selected contract kind;
5. semantic binding metric overlay;
6. unresolved fallback.

### 15.4 Contract rules

- `NO_OP_TEST`: no visible effect is a mechanic match; visible effect is a
  mismatch unless expected otherwise.
- `LOCAL_TARGET_CHANGE`: target-local overlap must meet the configured minimum
  fraction.
- `OBJECT_DISPLACEMENT`: selected object centroid or tracked geometry must move.
- `RELATION_ERROR_DECREASE`: the preselected relation metric must decrease by
  more than epsilon.
- `ACTION_SURFACE_CHANGE`: available/planning surface must change.
- `SCORE_OR_TERMINAL`: official score, level, or terminal evidence is required.
- `ACTION_EFFECT_DISCOVERY`: classify typed effect/no-effect/negative-effect;
  this does not by itself establish the hidden semantic goal.

### 15.5 Attribution

When passive attribution is enabled, the judge classifies:

- `ACTION_LINKED` when changes overlap the target or match expected local
  mechanics;
- `PASSIVE_POSSIBLE` when a small distant change can occur independently;
- `MIXED_OR_UNCERTAIN` when target and far changes coexist beyond ratio
  thresholds;
- `NO_VISIBLE_CHANGE` when no logical cell changes.

Attribution is evidence quality, not semantic progress.

## 16. Information gain engineering

### 16.1 Normalized entropy

For probabilities \(p_i\):

\[
H(p)=-\frac{\sum_i p_i\ln p_i}{\ln n}, \quad n>1.
\]

If the domain is empty, singleton, or has nonpositive total mass, entropy is
zero.

### 16.2 Posterior model

- unresolved question: uniform distribution;
- resolved question: `0.92` on the resolved outcome;
- remaining `0.08` uniformly distributed over other outcomes;
- contradictory outcome: reopen by setting `resolved_outcome=None`.

### 16.3 Information gain

```python
observed_information_gain = max(0.0, H_before - H_after)
```

An outcome outside the registered domain produces zero information gain and no
question update. Grid-diff magnitude is not used as a substitute.

### 16.4 Action-effect confidence

`merge_effect_record()` increments evidence count and computes:

\[
confidence = \min(0.98, 0.45 + 0.12\times evidence\_count).
\]

The record is scoped by action and optional target signature and stores level
and last step.

## 17. Memory engineering

### 17.1 `GameMemory` collections

Required stores:

```text
events
failed_events
irrelevant_events
coordinate_effects
action_effects
action_memory_records
action_surface_memory_records
level_attempt_records
object_applicability_memory
semantic_bindings
semantic_binding_contexts
trajectory_evaluations
semantic_invariants
semantic_questions
simple_action_attempts_by_level
action_attempts_by_signature
coordinate_probe_counts_by_level
coordinate_probe_signature_counts
coordinate_candidates_clicked_this_attempt
tracks_by_level
current_attempt_by_level
attempt_action_offsets_by_level
attempt_entry_source_by_level
```

### 17.2 Scope reset rules

`reset_game(game_id)` clears all stores and establishes a new game identity.

`mark_observed_level()` creates level-scoped track and research buckets.

`begin_level_attempt(retry=false)` establishes the action-memory offset.

`begin_level_attempt(retry=true)` additionally clears:

- same-state action attempt signatures;
- per-attempt coordinate clicks;
- per-level coordinate probe counts for the new attempt;
- coordinate probe signature counts.

It must retain action effects, semantic bindings, trajectory evaluations, and
invariants.

`begin_hypothesis_alternative()` clears only execution suppression needed to
allow a sibling route with an overlapping prefix.

### 17.3 Failed attempt record

`record_level_attempt_failure()` must compute:

- complete action list from attempt entry;
- research and goal provenance;
- ordered step evidence;
- coordinate target-cell transitions;
- aggregate color transitions;
- visible effect counts;
- level-progress and game-over outcome;
- executed hypothesis records;
- failure subset;
- verifier feedback;
- `exact_replay_forbidden`.

The complete attempt scope string must make clear that there was no implicit
reset between ordered steps.

### 17.4 Semantic binding context

`record_semantic_binding(binding, snapshot)` stores:

- binding itself;
- object fingerprints for source, reference, and inferred IDs;
- relation fingerprints for cited and inferred relations;
- state signature.

Fingerprints must be deterministic and sufficient for reset rebinding.

### 17.5 Reset rebinding

`resolve_semantic_binding_references()` must:

1. accept exact current IDs first;
2. score missing old IDs against unused current objects;
3. require score at least `70.0`;
4. enforce one-to-one current object use;
5. rebind relations only after endpoints are mapped;
6. require relation type and metric name equality;
7. return mapped roles and completeness flags.

A partial rebind may be retained as evidence, but an active sibling requiring
unresolved source/reference/relation IDs must not execute.

### 17.6 Invariant consolidation

Invariant records are keyed by base predicate/action/subjects/control surface.
Each parameter variant stores count and bounded evidence references.

Status projection:

```text
one compatible observation -> OBSERVED_ONCE
same variant count >= semantic_invariant_confirmation_count -> CONFIRMED
multiple incompatible variants for same base -> CONTRADICTED
```

The Qwen feedback view must expose the status and authority, not only the most
recent parameter value.

### 17.7 Semantic feedback schema

The memory feedback object uses schema label `v9.reverse_semantic_feedback` and
contains bounded:

- bindings;
- trajectory evaluations;
- invariants;
- current-reference and rebound flags;
- evidence authorities;
- exact failure reasons.

Qwen proposals may cite these records, but cannot alter them.

## 18. Session state machine

### 18.1 Construction

`GameSession.__init__()` creates:

- normalized config;
- game adapter and ARGALite parser;
- memory;
- action explorer;
- hypothesis bank;
- verification binder and preflight;
- transition judge;
- Qwen backend/client and budget state;
- pending-action state;
- level, attempt, action, reset, and telemetry counters.

### 18.2 `act()` order

The normative `act(raw_observation)` sequence is:

1. normalize observation and build current snapshot;
2. detect a new game and reset game-scoped session state;
3. if a pending action exists and the observation is a later official state,
   commit it before further selection;
4. reject action emission from an unchanged state while pending remains;
5. apply true level boundary after commit;
6. if called on a success-terminal frame, return the `terminal_guard` reset-shaped sentinel; the outer direct harness must stop before submitting it;
7. emit initial reset for `NOT_STARTED` / `NOT_PLAYED`;
8. service pending sibling-alternative reset before ordinary policy;
9. service official `GAME_OVER` failed-attempt reset;
10. enforce per-level action limits;
11. continue or start mandatory simple-action research;
12. continue coordinate research when required;
13. restore entry state after research when required;
14. call the permitted Qwen role when budget and readiness allow;
15. select policy candidate by queue priority;
16. run preflight; scan bounded alternatives if rejected;
17. if no candidate remains, either reset a bounded failed attempt or raise an
    explicit no-fallback error;
18. emit one action and register a pending token.

### 18.3 Policy priority

The policy queue order is:

```text
1. confirmed/continuing trajectory step
2. coordinate research step
3. validated semantic trajectory step
4. deterministic simple-action research probe
```

The session does not use a blind random or raw-action fallback. A separate
fallback queue may exist in the bank for compatibility, but the active session
must not populate it with unverified semantic play.

### 18.4 `observe_action_result()`

`observe_action_result(after_observation)` must:

- return false when no pending action exists;
- normalize and parse the official after observation;
- ignore an exact duplicate commit token;
- call the single internal commit path;
- clear pending state exactly once;
- update telemetry;
- return true only when a new official transition was committed.

### 18.5 Official commit

For a non-reset action, `_commit_pending()` must:

1. judge the official transition;
2. record memory event and action diff;
3. update typed semantic question posterior;
4. merge action effect evidence;
5. derive same-level invariants;
6. update active hypothesis mechanic status;
7. append the judgment to trajectory evidence;
8. evaluate a completed route;
9. create sibling-reset request when applicable;
10. update counters and last official snapshot.

For a reset action:

- no ordinary semantic judgment is emitted;
- retry reset begins the next attempt;
- research reset marks the entry state restored;
- alternative reset rebinds remaining siblings and clears only alternative
  execution suppression.

### 18.6 Level boundary

A level boundary resets:

- bank queues and active route;
- attempt index and level action count;
- per-level Qwen budget;
- attempt-scoped suppression;
- level-local research status as required by the new action surface.

It preserves:

- game identity;
- official action-effect evidence;
- completed-level memory;
- game-scoped model call count;
- appropriate cross-level action surface evidence.

### 18.7 Failed-attempt reset

A failed attempt can reset only when:

- reset is configured and legal;
- attempt count is below `max_level_attempts`;
- at least one Qwen call has occurred for the level when semantic failure is the
  trigger;
- outer action and wall-clock limits permit another attempt.

Before emitting reset, the session records the attempt with its exact complete
execution and verifier feedback.

## 19. Framework-compatible agent wrapper

`kaggle_agent.ARC_AGI_Agent` must:

- construct one `GameSession`;
- accept dynamic config updates through the active session;
- normalize framework observations;
- call `session.act()`;
- convert returned action dictionaries to the expected ARC action type;
- expose explicit `observe_action_result()`;
- perform a guarded fallback ingestion in `act()` only when the framework did
  not explicitly ingest the accepted result;
- expose `reset_after_game_over()` that validates current official state and
  returns only a reset action;
- expose bounded telemetry;
- clean up backend resources.

The fallback ingestion path must be idempotent and must not double-commit a
transition already ingested explicitly.

## 20. Direct competition child

### 20.1 Process isolation

`lcld_competition_child.py` runs as a separate process launched by the notebook
supervisor. It owns the live ARC SDK import and direct gateway loop. Model server
failure, game worker failure, and scorecard finalization are isolated from the
notebook setup process.

### 20.2 Concurrency

The child uses a thread pool with default game concurrency `16`. The vLLM server
uses `max_num_seqs=4`. The implementation must tolerate queued model calls and
must enforce per-game wall-clock limits independently.

### 20.3 Environment creation

Each game environment is created lazily and at most once per worker game. The
implicit gateway reset associated with `make()` is counted in orchestration
telemetry. The direct loop must not perform an unconditional second initial
reset.

### 20.4 Accepted action loop

For each game:

1. read the current official frame;
2. stop on success terminal;
3. on `GAME_OVER`, perform exactly one legal reset before any analyzer/model
   call;
4. request one action from the delegate session;
5. enforce action and wall-clock limits;
6. submit to gateway;
7. treat rejection as a game failure;
8. immediately call `observe_action_result()` on the accepted next frame;
9. continue until success or explicit termination.

A game with Qwen enabled that records no Qwen calls and no completed levels is
reported as a failure, preventing a silent empty-agent success path.

### 20.5 Result manifest

The child atomically writes:

```text
/kaggle/working/lcld_competition_scorecard_results.json
```

The manifest must include:

- per-game status and reason;
- actions, resets, attempts, levels, and time;
- Qwen calls and telemetry;
- gateway activity count;
- scorecard close disposition;
- worker exceptions;
- overall success/failure summary.

A temporary file plus atomic rename is required.

### 20.6 Scorecard close

Valid close dispositions:

```text
closed
closed_no_payload
already_closed
```

HTTP-style close outcomes equivalent to already closed may be accepted for
specific conflict/gone statuses. Other close failures are recorded but need not
destroy valid game results.

If a fatal orchestration error occurs after gateway activity, the child must
attempt to close/preserve the partial scorecard and write failure manifests. If
no gateway activity occurred, the fatal error may be re-raised.

## 21. Competition notebook runtime

### 21.1 Notebook phases

The notebook contains six cells and separates:

- dependency and environment setup;
- embedded payload materialization;
- static preflight;
- vLLM runtime setup;
- strict multimodal model smoke;
- isolated competition child supervision.

The phase is selected by the competition rerun environment gate.

### 21.2 Isolated installation

The notebook uses an offline wheelhouse and installs the vLLM stack into an
isolated target directory. The ARC SDK is installed from the competition
wheelhouse. The host environment must not be mutated more than required for the
child path.

The supplied runtime identifies:

```text
vLLM: 0.19.0
PyTorch: 2.10.0
FlashInfer: 0.6.6
```

The notebook checks for the expected NVIDIA accelerator before starting Phase B.

### 21.3 Model server command

The model server is configured with:

```text
model: vrfai/Qwen3.6-27B-FP8
served model name: same
host: 127.0.0.1
port: 1234
tensor parallel size: 1
max model length: 131072
max concurrent sequences: 4
prefix caching: enabled
reasoning parser: qwen3
auto tool choice: enabled
tool-call parser: qwen3_coder
preserve_thinking: true
enable_thinking: true
```

The exact command must be logged in bounded form without exposing secrets.

### 21.4 Model smoke

Before the scorecard run, Phase B sends a small deterministic PNG and a strict
JSON request through the same vLLM endpoint. The smoke must verify:

- server readiness;
- image attachment acceptance;
- assistant content extraction;
- strict response schema;
- thinking configuration compatibility.

A failed smoke must prevent the competition child from starting.

### 21.5 Supervisor result policy

The supervisor treats the child result manifest as primary. A nonzero child exit
may still be accepted only when all of the following are true:

- a valid result manifest exists;
- gateway activity occurred;
- scorecard finalization has an accepted disposition;
- the manifest explicitly records the nonzero cause.

The vLLM process must be stopped in `finally` regardless of child outcome.

## 22. Logging and telemetry

### 22.1 Session telemetry

`GameSession.harness_telemetry()` must expose bounded fields including:

```text
package/session version
game_id
level_index
attempt_index
level_action_count
game_action_count
reset counts by source
alternative reset count
pending action/token flag
observed transition ingestions
duplicate ingestion count
Qwen calls by role, level, game
research status
active hypothesis and batch
memory event counts
failed and irrelevant counts
coordinate effect counts
semantic binding/evaluation/invariant counts
preflight rejection counts
last judgment summary
termination/limit reason
```

### 22.2 Qwen traces

When `qwen_trace_dir` is nonempty, trace records may contain:

- role;
- packet hashes and token estimate;
- prompt after image removal;
- output schema hash;
- backend request metadata;
- raw response in bounded form;
- extracted JSON;
- validation errors;
- timing.

Competition defaults disable persistent prompt/response traces. The supervisor
may preserve only a bounded tail of vLLM logs.

### 22.3 Determinism

All deterministic hashes must use canonical serialization with sorted keys and
stable ordering. Telemetry must distinguish model nondeterminism from action
surface changes, frame changes, or unordered collection serialization.

## 23. Test requirements

### 23.1 Configuration tests

Required tests:

- `V9Config is V8Config` alias behavior;
- environment aliases map to active fields;
- vLLM thinking enabled only under valid conditions;
- non-vLLM thinking is clamped off;
- competition overrides produce 131072/65536/49152 token envelope;
- strict runtime default in submission config;
- no raw coordinates by default;
- reset counters do not create termination limits.

### 23.2 Observation and parsing tests

- temporal frame collapses to final 2D grid;
- normalized 2D grid remains unchanged;
- nonrectangular and out-of-domain grids reject;
- x/y coordinate order is preserved;
- background choice deterministic;
- multicolor merge deterministic;
- object masks, holes, topology, and hashes stable;
- repeated frame parse yields identical snapshot hashes.

### 23.3 Tracking and rebinding tests

- translated object keeps persistent track ID;
- incompatible object receives a new track;
- deterministic pair ordering resolves ties;
- reset with changed frame IDs rebinds source/reference roles;
- relation rebind requires mapped endpoints and exact relation type;
- ambiguous/low-score object remains unresolved;
- one current object cannot satisfy two old IDs.

### 23.4 Component graph tests

- all cells belong to exactly one same-color component;
- nesting and adjacency deterministic;
- shape run and corner caps respected;
- component IDs never enter allowed object IDs;
- object references remain valid after packet aliasing.

### 23.5 Action research tests

- all current gameplay actions become required;
- undo is intrinsically researched;
- unknown simple actions block primary Qwen;
- one-shot and control/new/control ordering deterministic;
- research-entry reset occurs after state-changing probes;
- research records are not labeled as failed goal execution;
- new action on a later surface is researched.

### 23.6 Coordinate tests

- candidates are within grid and 0..63;
- candidate ID maps exactly to x/y;
- raw coordinate model output rejects;
- duplicate candidate and duplicate physical location reject;
- coordinate run repeat greater than one rejects;
- no-effect repeat suppression is attempt-scoped;
- alternative reset permits legitimate sibling coordinate use under restored
  scope.

### 23.7 Packet tests

- packet schema label and sections exact;
- full grid and PNG hashes correspond;
- aliases are one-to-one;
- all references validate before invocation;
- compaction preserves current whitelists and output tail;
- omitted detail is explicit;
- grouped action diff retains one real sample and occurrence metadata;
- history-only research is marked correctly after reset;
- textual prompt excludes base64 image.

### 23.8 Model schema tests

- coordinate response schema enumerates current IDs;
- semantic response allows one to three hypotheses only;
- no additional properties;
- first action availability validation;
- later surface reachability validation;
- expanded action count <= 50;
- hallucinated IDs reject;
- failed exact trajectory rejects unchanged;
- unsupported backend-schema keywords may be stripped without weakening Python
  validation;
- fake backend requires no model.

### 23.9 Reverse-semantic tests

- relation endpoints infer missing roles only when cited;
- unknown IDs reject binding;
- no source rejects object-required objective;
- centroid, gap, containment, and palette-shape metrics match formulas;
- metric baseline recomputed after reset rebind;
- improvement sign correct for all directions;
- unchanged metric returns irrelevant;
- unsupported metric returns unresolved;
- terminal progress overrides metric;
- game over overrides mechanic match.

### 23.10 Invariant tests

- translation, shape preservation, recolor, co-motion, opposite-axis motion,
  surface change, and no-effect observations derive from official transitions;
- no invariant derives across level/terminal/game-over transition;
- repeated same variant confirms;
- competing variants contradict;
- control-surface scope separates otherwise identical action IDs.

### 23.11 Session and ingestion tests

- one emitted action creates one pending token;
- unchanged observation while pending cannot emit again;
- explicit result ingestion commits once;
- duplicate ingestion ignored;
- fallback wrapper ingestion remains idempotent;
- accepted action immediately updates memory;
- reset transition is not judged as ordinary semantic action;
- level boundary resets level-local state only;
- strict Qwen failure surfaces as game error;
- no unverified fallback action is emitted.

### 23.12 Alternative trajectory tests

- primary response queues up to three siblings under one batch;
- first failed sibling requests one reset;
- reset result is ingested before next sibling;
- remaining binding IDs are rebound;
- failed exact route persists in memory;
- action-effect and invariant memory persists;
- execution suppression clears;
- unresolvable sibling is rejected;
- successful level completion prevents sibling execution.

### 23.13 Competition child tests

- one environment creation per game;
- no unconditional initial reset after `make()`;
- exactly one reset after game over before reasoning;
- persisted game over stops with specific reason;
- accepted transition immediately ingested;
- rejected gateway step fails game;
- action and game-clock limits stop without reset;
- game failures isolated;
- result manifest atomic;
- scorecard close disposition validated;
- partial scorecard preserved after post-activity orchestration failure.

### 23.14 Notebook tests

- payload extraction yields exactly the expected active files;
- compileall passes for embedded payload;
- Phase A does not start vLLM;
- Phase A writes valid dummy submission artifact;
- Phase B checks accelerator;
- offline wheelhouse paths exist;
- model server readiness and strict multimodal smoke gate child launch;
- child manifest is primary supervisor result;
- vLLM stops in `finally`.

## 24. Validation commands

### 24.1 Notebook integrity

```bash
sha256sum arc-prize-2026-lcld-qwen-v9.ipynb
```

Expected notebook hash for the supplied artifact:

```text
12917cc19473f47ccfcf9d551d7296ba331a0168deb57e1a2f3584e999b1f023
```

### 24.2 Embedded source extraction

Use a deterministic extraction utility that:

1. loads notebook JSON;
2. locates the embedded base64 ZIP constant;
3. decodes once;
4. validates ZIP integrity;
5. rejects path traversal;
6. extracts into a clean directory;
7. counts non-cache files.

Expected active file count: `29`.

### 24.3 Compilation

```bash
python -m compileall -q extracted/Code
```

### 24.4 Import smoke

```bash
PYTHONPATH=extracted/Code python - <<'PY'
from v9_agent import V9Config, GameSession, __version__
assert __version__ == "9.0.0"
assert V9Config().allow_qwen_raw_coordinates is False
session = GameSession({"qwen_backend": "fake", "qwen_require_runtime": False})
print(session.harness_telemetry())
PY
```

### 24.5 Exactly-once transition smoke

Construct a small legal grid with one available action, call `act()`, supply a
changed official observation to `observe_action_result()`, and assert:

```text
first ingestion = true
second identical ingestion = false
pending_official_transition = false
observed_transition_ingestions = 1
```

### 24.6 Full tests

A production repository should provide:

```bash
pytest -q
```

with no network and no real model requirement. Competition-only tests may use a
fake gateway and fake scorecard but must exercise the same child state machine.

## 25. Known engineering limitations

1. The notebook contains the payload directly; there is no supplied standalone
   repository test suite in the artifact.
2. The public configuration implementation class is named `V8Config`, and
   packet/output wire schemas use `v8.*` identifiers. These are technically
   harmless but require disciplined documentation.
3. Object tracking uses deterministic greedy matching, not global optimal
   assignment or probabilistic identity.
4. Reset rebinding uses a fixed fingerprint threshold and can remain ambiguous
   for identical repeated objects.
5. Only four cheap goal metrics are implemented.
6. Mirror, rotation, and repeated-pattern relations can inform proposals but do
   not receive dedicated exact goal metrics.
7. The reserve Qwen role is represented but not selected by the active session.
8. There is no forward world simulator or rollback evaluator.
9. The direct child uses more game threads than model sequences; queueing can
   consume game wall-clock budget.
10. Competition traces are disabled by default, limiting postmortem prompt
    inspection.
11. The local audit did not execute vLLM, the ARC gateway, or shared scorecard.
12. Exact hidden-state restoration after `RESET` cannot be proven from visible
    frames alone.

## 26. Definition of done

A V9 implementation is complete only when:

- all 29 active payload files compile;
- public imports and alias behavior are stable;
- the grid, palette, and coordinate contracts are enforced;
- action research gates primary semantic planning;
- the current state is restored after research when required;
- the packet is reference-valid and multimodal when configured;
- strict schemas enumerate only current IDs;
- one to three complete trajectories are parsed and validated;
- semantic objectives bind to current entities or reject explicitly;
- supported metrics are computed exactly as specified;
- every primitive action has one fixed contract;
- accepted transitions commit exactly once;
- local mechanic and trajectory goal judgments remain separate;
- failed exact routes are suppressed;
- sibling alternatives reset and rebind to a common entry state;
- official mechanics and invariants survive alternative resets;
- contradictory invariant evidence remains contradicted;
- `GAME_OVER` recovery and timeout termination remain distinct;
- direct competition workers isolate failures and preserve scorecard results;
- deterministic tests pass without a real model;
- live model/gateway validation, when performed, is reported separately from
  static and fake-backend validation.

## 27. Update requirements

Any code change affecting one of these areas requires synchronized updates to
both specifications and tests:

```text
config fields or env aliases
public package exports
wire schema identifiers
packet sections or alias rules
object tracking or reset rebinding
relation or component graph vocabulary
action research gating
coordinate candidate rules
Qwen call schedule
semantic output schema
objective/operator mapping
goal metrics
preflight legality
transition judgment precedence
information-gain posterior
memory scope and attempt records
sibling-reset preservation
pending transition API
competition child lifecycle
notebook model runtime
scorecard result manifest
```

A notebook rebuild is required after any embedded source or competition runtime
change. Post-build validation must inspect and test the embedded payload; a
source-only pass is insufficient.
