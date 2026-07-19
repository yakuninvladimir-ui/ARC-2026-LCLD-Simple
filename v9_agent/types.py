from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TriTruth(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class Relevance(str, Enum):
    RELEVANT = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"
    UNDECIDED = "UNDECIDED"


class Validity(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNCHECKED = "UNCHECKED"


class Progress(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class Attribution(str, Enum):
    ACTION_LINKED = "ACTION_LINKED"
    PASSIVE_POSSIBLE = "PASSIVE_POSSIBLE"
    MIXED_OR_UNCERTAIN = "MIXED_OR_UNCERTAIN"
    NO_VISIBLE_CHANGE = "NO_VISIBLE_CHANGE"


class QwenRole(str, Enum):
    PRIMARY = "primary"
    RESERVE = "reserve"
    COORDINATE = "coordinate"


class VerificationContractKind(str, Enum):
    NO_OP_TEST = "NO_OP_TEST"
    LOCAL_TARGET_CHANGE = "LOCAL_TARGET_CHANGE"
    OBJECT_DISPLACEMENT = "OBJECT_DISPLACEMENT"
    RELATION_ERROR_DECREASE = "RELATION_ERROR_DECREASE"
    ACTION_SURFACE_CHANGE = "ACTION_SURFACE_CHANGE"
    SCORE_OR_TERMINAL = "SCORE_OR_TERMINAL"
    ACTION_EFFECT_DISCOVERY = "ACTION_EFFECT_DISCOVERY"


class BindingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"


class MetricDirection(str, Enum):
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"
    TARGET = "TARGET"
    BOOLEAN = "BOOLEAN"


class MechanicResult(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


class SemanticJudgment(str, Enum):
    REQUIRED = "REQUIRED"
    FORBIDDEN = "FORBIDDEN"
    IRRELEVANT = "IRRELEVANT"
    UNRESOLVED = "UNRESOLVED"


class EvidenceAuthority(str, Enum):
    OFFICIAL_OBSERVATION = "OFFICIAL_OBSERVATION"
    DETERMINISTIC_BINDER = "DETERMINISTIC_BINDER"
    DETERMINISTIC_VERIFIER = "DETERMINISTIC_VERIFIER"
    QWEN_PROPOSAL = "QWEN_PROPOSAL"


class EvidenceStatus(str, Enum):
    OBSERVED_ONCE = "OBSERVED_ONCE"
    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    IRRELEVANT = "IRRELEVANT"
    UNRESOLVED = "UNRESOLVED"


class GoalOperator(str, Enum):
    MOVE_TOWARD = "MOVE_TOWARD"
    ALIGN = "ALIGN"
    MATCH_GEOMETRY = "MATCH_GEOMETRY"
    OVERLAP = "OVERLAP"
    CONNECT = "CONNECT"
    BRIDGE_GAP = "BRIDGE_GAP"
    EXTEND_LINE = "EXTEND_LINE"
    CONTAIN = "CONTAIN"
    COMPLETE_PATTERN = "COMPLETE_PATTERN"
    MATCH_STATE = "MATCH_STATE"
    ACTIVATE = "ACTIVATE"
    CHANGE_ACTION_SURFACE = "CHANGE_ACTION_SURFACE"
    PROBE_AFFORDANCE = "PROBE_AFFORDANCE"
    PROBE_RELATION = "PROBE_RELATION"
    OTHER = "OTHER"


class SemanticQuestionType(str, Enum):
    ACTION_EFFECT = "action_effect"
    AFFORDANCE = "affordance"
    CONTROLLABILITY = "controllability"
    RELATION_RELEVANCE = "relation_relevance"
    ACTION_SURFACE_CHANGE = "action_surface_change"
    TERMINAL_PROGRESS = "terminal_progress"


@dataclass(frozen=True, slots=True)
class WorldState:
    game_id: str
    level_index: int
    step_index: int
    grid: tuple[tuple[int, ...], ...]
    available_actions: tuple[str, ...]
    score: float | None
    terminal: bool
    raw: dict[str, Any]
    state_hash: str
    state_name: str = ""
    levels_completed: int = 0
    win_levels: int | None = None
    game_over: bool = False
    full_reset: bool = False
    planning_action_ids: tuple[str, ...] = ()
    undo_action_ids: tuple[str, ...] = ()
    possible_actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    object_id: str
    bbox_rc: tuple[int, int, int, int]
    centroid_rc: tuple[float, float]
    area: int
    colors: tuple[int, ...]
    color_histogram: dict[int, int]
    shape_signature: str
    local_mask_hex_rows: tuple[str, ...]
    holes: int
    symmetry_hints: tuple[str, ...]
    border_touching: tuple[str, ...]
    tags: tuple[str, ...]
    stable_hash: str
    salience_score: float
    track_id: str = ""
    frame_object_id: str = ""
    topology_signature: str = ""


@dataclass(frozen=True, slots=True)
class RelationRecord:
    relation_id: str
    relation_type: str
    a: str
    b: str
    metric_name: str | None
    metric_value: float | None
    confidence: float
    salience_score: float
    relation_signature: str = ""


@dataclass(frozen=True, slots=True)
class CoordinateTargetCandidate:
    candidate_id: str
    x: int
    y: int
    source: str
    object_id: str | None
    relation_id: str | None
    region_signature: str
    reason: str
    salience_score: float
    target_signature: str = ""


@dataclass(frozen=True, slots=True)
class ARGALiteSnapshot:
    snapshot_id: str
    game_id: str
    level_index: int
    step_index: int
    height: int
    width: int
    palette_ids_seen: tuple[int, ...]
    palette_histogram: dict[int, int]
    coordinate_order: str
    full_grid_hex_rows: tuple[str, ...]
    objects: tuple[ObjectRecord, ...]
    relations: tuple[RelationRecord, ...]
    coordinate_targets: tuple[CoordinateTargetCandidate, ...]
    available_actions: tuple[str, ...]
    coordinate_action_ids: tuple[str, ...]
    grid_hash: str
    object_hash: str
    relation_hash: str
    semantic_state_signature: str
    score: float | None = None
    terminal: bool = False
    state_name: str = ""
    levels_completed: int = 0
    win_levels: int | None = None
    game_over: bool = False
    full_reset: bool = False
    planning_action_ids: tuple[str, ...] = ()
    undo_action_ids: tuple[str, ...] = ()
    possible_actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticObjective:
    kind: str
    source_object_ids: tuple[str, ...] = ()
    reference_object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    description: str = ""
    family: str = "other"
    basis: str = ""


@dataclass(frozen=True, slots=True)
class GoalMetricSpec:
    name: str
    direction: MetricDirection
    target_value: float | None = None
    epsilon: float = 1e-6
    subject_roles: tuple[str, ...] = ("source", "reference")
    computation_version: str = "v9.0"


@dataclass(frozen=True, slots=True)
class SemanticBindingResult:
    binding_id: str
    hypothesis_id: str
    status: BindingStatus
    objective: SemanticObjective
    goal_operator: GoalOperator
    source_object_ids: tuple[str, ...] = ()
    reference_object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    inferred_object_ids: tuple[str, ...] = ()
    inferred_relation_ids: tuple[str, ...] = ()
    metric_spec: GoalMetricSpec | None = None
    baseline_value: float | None = None
    reason_code: str = ""
    evidence_refs: tuple[str, ...] = ()
    state_signature: str = ""
    action_surface_signature: str = ""
    game_id: str = ""
    level_index: int = 0


@dataclass(frozen=True, slots=True)
class VerificationContract:
    contract_id: str
    kind: VerificationContractKind
    target_object_ids: tuple[str, ...] = ()
    target_relation_ids: tuple[str, ...] = ()
    target_coordinate_candidate_id: str | None = None
    target_region_rc: tuple[int, int, int, int] | None = None
    metric_name: str | None = None
    before_metric: float | None = None
    expected_effect: str | None = None
    question_id: str | None = None
    question_type: SemanticQuestionType | None = None
    target_signature: str | None = None
    semantic_binding_id: str | None = None
    goal_operator: GoalOperator | None = None
    source_object_ids: tuple[str, ...] = ()
    reference_object_ids: tuple[str, ...] = ()
    semantic_metric_name: str | None = None
    metric_direction: MetricDirection | None = None
    metric_target_value: float | None = None
    metric_epsilon: float = 1e-6


@dataclass(frozen=True, slots=True)
class CandidateAction:
    action_id: str
    x: int | None = None
    y: int | None = None
    coordinate_candidate_id: str | None = None
    hypothesis_id: str | None = None
    reason: str = ""
    source: str = "policy"
    verification_contract: VerificationContract | None = None
    allow_exhaustion_revisit: bool = False

    def to_arc_action(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.x is not None and self.y is not None:
            data = {"x": int(self.x), "y": int(self.y)}
        contract = self.verification_contract
        return {
            "action_id": self.action_id,
            "id": self.action_id,
            "data": data,
            "reasoning": {
                "agent": "v9_reverse_semantic_hypothesis_agent",
                "source": self.source,
                "reason": self.reason,
                "hypothesis_id": self.hypothesis_id,
                "coordinate_candidate_id": self.coordinate_candidate_id,
                "verification_contract": contract.kind.value if contract else None,
                "verification_contract_id": contract.contract_id if contract else None,
                "exhaustion_revisit": bool(self.allow_exhaustion_revisit),
            },
        }

    @property
    def suppression_signature(self) -> str:
        contract_kind = self.verification_contract.kind.value if self.verification_contract else "unbound"
        if self.coordinate_candidate_id:
            return f"{self.action_id}:candidate:{self.coordinate_candidate_id}:{contract_kind}"
        if self.x is not None and self.y is not None:
            return f"{self.action_id}:xy:{self.x},{self.y}:{contract_kind}"
        return f"{self.action_id}:simple:{contract_kind}"


@dataclass(frozen=True, slots=True)
class PendingAction:
    before_snapshot: ARGALiteSnapshot
    action: CandidateAction
    hypothesis_id: str | None
    reason: str
    token_id: str = ""


@dataclass(frozen=True, slots=True)
class PreflightResult:
    valid: bool
    validity: Validity
    reason_code: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Judgment:
    truth: TriTruth
    relevance: Relevance
    validity: Validity
    progress: Progress
    attribution: Attribution
    reason_code: str
    observed_delta: dict[str, Any]
    affected_objects: tuple[str, ...]
    affected_relations: tuple[str, ...]
    score_delta: float | None
    terminal_delta: bool
    action: CandidateAction
    hypothesis_id: str | None
    before_hash: str
    after_hash: str
    contract_kind: VerificationContractKind | None = None
    error_before: float | None = None
    error_after: float | None = None
    error_delta: float | None = None
    observed_information_gain: float = 0.0
    question_id: str | None = None
    mechanic_result: MechanicResult = MechanicResult.UNKNOWN
    semantic_judgment: SemanticJudgment = SemanticJudgment.UNRESOLVED
    semantic_binding_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    event_id: str
    level_index: int
    step_index: int
    event_type: str
    before_hash: str
    action: dict[str, Any] | None
    after_hash: str | None
    hypothesis_id: str | None
    truth: TriTruth
    relevance: Relevance
    validity: Validity
    progress: Progress
    attribution: Attribution
    reason_code: str
    summary: str
    contract_kind: str | None = None
    information_gain: float = 0.0


@dataclass(frozen=True, slots=True)
class CoordinateEffectRecord:
    coordinate_action_id: str
    candidate_target_id: str | None
    x: int
    y: int
    level_index: int
    step_index: int
    object_id: str | None
    region_signature: str
    observed_effect: str
    truth: TriTruth
    relevance: Relevance
    progress: Progress
    attribution: Attribution
    state_signature: str
    repeat_suppression_signature: str


@dataclass(frozen=True, slots=True)
class ActionEffectRecord:
    action_id: str
    target_signature: str | None
    outcome: str
    evidence_count: int
    confidence: float
    level_index: int
    last_step: int


@dataclass(slots=True)
class SemanticQuestion:
    question_id: str
    question_type: SemanticQuestionType
    domain: tuple[str, ...]
    target_signature: str
    resolved_outcome: str | None = None
    observations: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TestStep:
    kind: str
    action_id: str | None
    target_object_id: str | None = None
    target_relation_id: str | None = None
    target_object_ids: tuple[str, ...] = ()
    target_relation_ids: tuple[str, ...] = ()
    coordinate_candidate_id: str | None = None
    expected_observation: str | None = None
    contract_kind: str | None = None
    question_type: str | None = None
    semantic_binding: SemanticBindingResult | None = None


@dataclass(slots=True)
class HypothesisItem:
    hypothesis_id: str
    source: str
    claim: str
    truth: TriTruth
    relevance: Relevance
    validity: Validity
    progress: Progress
    test_plan: tuple[TestStep, ...]
    cursor: int
    priority: float
    confidence: float
    expiry_step: int | None
    evidence_refs: tuple[str, ...]
    suppression_signature: str
    created_state_signature: str = ""
    proposal_batch_id: str = ""
    semantic_objective: SemanticObjective | None = None
    semantic_binding: SemanticBindingResult | None = None
    trajectory_start_snapshot: ARGALiteSnapshot | None = None
    trajectory_judgments: list[Judgment] = field(default_factory=list)
    executed_action_ids: list[str] = field(default_factory=list)

    def has_next_step(self) -> bool:
        return self.cursor < len(self.test_plan)

    def next_step(self) -> TestStep | None:
        if not self.has_next_step():
            return None
        return self.test_plan[self.cursor]


@dataclass(frozen=True, slots=True)
class TrajectoryEvaluation:
    evaluation_id: str
    hypothesis_id: str
    binding_id: str | None
    level_index: int
    executed_action_ids: tuple[str, ...]
    mechanic_result: MechanicResult
    goal_progress: Progress
    semantic_judgment: SemanticJudgment
    reason_code: str
    error_before: float | None = None
    error_after: float | None = None
    error_delta: float | None = None
    first_divergence_step: int | None = None
    source_object_ids: tuple[str, ...] = ()
    reference_object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    start_state_signature: str = ""
    end_state_signature: str = ""


@dataclass(slots=True)
class QwenBudgetState:
    calls_this_game: int = 0
    primary_calls_by_level: dict[int, int] = field(default_factory=dict)
    reserve_calls_by_level: dict[int, int] = field(default_factory=dict)
    coordinate_calls_by_level: dict[int, int] = field(default_factory=dict)
    total_calls_by_level: dict[int, int] = field(default_factory=dict)
    last_qwen_step: int = -10**9
