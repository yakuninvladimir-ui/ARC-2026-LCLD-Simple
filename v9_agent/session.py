from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from typing import Any, Mapping

from .action_explorer import ActionExplorer
from .arga_lite import ARGALiteBuilder
from .config import V8Config, config_from_mapping
from .deliberation import choose_qwen_role
from .game_adapter import GameAdapter
from .hypothesis_bank import HypothesisBank
from .judge import PreflightJudge, TransitionJudge
from .llm import QwenBackendError, QwenClient
from .logging import runtime_log
from .memory import GameMemory, is_coordinate_research_source
from .observe import stable_hash
from .policy import Policy
from .qwen_packet import QwenPacketBuilder, QwenPacketNotReady
from .qwen_roles import record_qwen_call
from .types import CandidateAction, PendingAction, QwenBudgetState, QwenRole, Validity
from .trajectory import TrajectoryVerifier, rebind_plan_to_test_steps
from .planning_set import (
    PlanningSet,
    assert_dual_view_identity,
    build_planning_set_from_packet,
    build_planning_set_from_snapshot,
)


_ATTEMPT_RETRY_RESET_SOURCES = {"game_over_level_reset", "failed_attempt_reset"}
_ALTERNATIVE_HYPOTHESIS_RESET_SOURCE = "alternative_hypothesis_reset"
_RESEARCH_ENTRY_RESET_SOURCE = "research_entry_reset"


class LevelRunLimitReached(RuntimeError):
    """Normal orchestration terminal for an exhausted level budget."""

    def __init__(self, reason_code: str, *, level_index: int, attempt_index: int, action_count: int, limit: int) -> None:
        self.reason_code = str(reason_code)
        self.level_index = int(level_index)
        self.attempt_index = int(attempt_index)
        self.action_count = int(action_count)
        self.limit = int(limit)
        super().__init__(
            f"{self.reason_code}: level={self.level_index} attempt={self.attempt_index} "
            f"actions={self.action_count} limit={self.limit}"
        )


class GameSession:
    def __init__(self, config: V8Config | Mapping[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, V8Config) else config_from_mapping(dict(config or {}))
        self.adapter = GameAdapter()
        self.arga_lite = ARGALiteBuilder()
        self.packet_builder = QwenPacketBuilder()
        self.qwen = QwenClient()
        self.bank = HypothesisBank()
        self.memory = GameMemory()
        self.explorer = ActionExplorer()
        self.preflight_judge = PreflightJudge()
        self.transition_judge = TransitionJudge()
        self.policy = Policy()
        self.budget = QwenBudgetState()
        self.pending_action: PendingAction | None = None
        self._last_game_id: str | None = None
        self._last_level_index: int | None = None
        self._last_levels_completed: int = 0
        self._action_count_this_game = 0
        self._qwen_successful_calls_this_game = 0
        self._qwen_empty_primary_streak = 0
        self._game_over_resets_this_game = 0
        self._game_over_resets_by_level: dict[int, int] = defaultdict(int)
        self._attempt_index_by_level: dict[int, int] = defaultdict(int)
        self._action_count_by_level: dict[int, int] = defaultdict(int)
        self._terminal_level_limit: dict[str, Any] = {}
        self._observed_transition_ingestions = 0
        self._observed_transition_duplicate_skips = 0
        self._last_committed_token: str | None = None
        self._latest_snapshot = None
        self._exhaustion_revisit_count = 0
        self._last_observation_signature: str | None = None
        self._synthetic_step_index = -1
        self._last_action_selection: dict[str, Any] = {}
        self._alternative_hypothesis_resets_this_game = 0
        # Consecutive fallback/liveness emits per level; a long streak with Qwen
        # already spent and no recent positive progress means the attempt is
        # dead — reset with retained feedback instead of flailing (spec §10-11).
        self._fallback_streak_by_level: dict[int, int] = defaultdict(int)
        # Single owner of dual-view identity for the current snapshot cycle.
        self._planning_set: PlanningSet | None = None
        self._planning_set_snapshot_id: str | None = None

    def update_runtime_config(self, updates: Mapping[str, Any] | None) -> None:
        if not updates:
            return
        merged = asdict(self.config)
        merged.update(dict(updates))
        self.config = config_from_mapping(merged)

    def act(self, raw_observation: Mapping[str, Any]) -> dict[str, Any]:
        state, snapshot = self._prepare_snapshot(raw_observation)

        if self.pending_action is not None:
            # Compatibility fallback for starter loops that do not call observe_action_result.
            if snapshot.snapshot_id == self.pending_action.before_snapshot.snapshot_id or (
                snapshot.grid_hash == self.pending_action.before_snapshot.grid_hash
                and snapshot.step_index == self.pending_action.before_snapshot.step_index
            ):
                raise RuntimeError("cannot emit a new action while the previous official transition is uncommitted")
            self._commit_pending(snapshot)

        is_new_level = self._apply_level_boundary_after_commit(state, snapshot)
        self._latest_snapshot = snapshot

        if state.terminal:
            return CandidateAction("RESET", reason="terminal success observed; outer loop should stop", source="terminal_guard").to_arc_action()
        if state.state_name in {"NOT_STARTED", "NOT_PLAYED"}:
            return self._emit(snapshot, CandidateAction("RESET", reason="initial environment start", source="initial_reset"))
        alternative_reset = self.bank.pending_alternative_reset()
        if alternative_reset is not None:
            runtime_log(
                "alternative_hypothesis_reset_emit",
                level=state.level_index,
                step=state.step_index,
                proposal_batch_id=alternative_reset.get("proposal_batch_id"),
                completed_hypothesis_id=alternative_reset.get("completed_hypothesis_id"),
                remaining_hypothesis_ids=alternative_reset.get("remaining_hypothesis_ids"),
            )
            return self._emit(
                snapshot,
                CandidateAction(
                    "RESET",
                    reason="restore the proposal entry state before executing the next alternative hypothesis",
                    source=_ALTERNATIVE_HYPOTHESIS_RESET_SOURCE,
                ),
            )

        if state.game_over:
            return self._handle_game_over(snapshot)

        control_cycle_pending = self.explorer.has_pending_control_cycle(snapshot)
        if self._should_restore_entry_after_research(snapshot, control_cycle_pending):
            runtime_log(
                "research_entry_reset_emit",
                level=state.level_index,
                attempt_index=self._attempt_index_by_level.get(state.level_index, 0),
                step=state.step_index,
                retained_action_memory=len(self.memory.action_memory_records),
            )
            return self._emit(
                snapshot,
                CandidateAction(
                    "RESET",
                    reason="restore the level entry state after action research before primary planning",
                    source=_RESEARCH_ENTRY_RESET_SOURCE,
                ),
            )
        role = (
            None
            if control_cycle_pending
            else choose_qwen_role(
                state,
                snapshot,
                self.memory,
                self.bank,
                self.budget,
                self.config,
                is_new_level=is_new_level,
            )
        )
        if role is None and self.config.enable_qwen and self.config.qwen_backend != "disabled":
            research = self.memory.action_research_status(snapshot)
            runtime_log(
                "qwen_no_call",
                level=state.level_index,
                step=state.step_index,
                is_new_level=is_new_level,
                coordinate_action_count=len(snapshot.coordinate_action_ids),
                coordinate_research_needed=bool(research["missing_coordinate_action_ids"]),
                required_action_ids=research["required_action_ids"],
                researched_action_ids=research["researched_action_ids"],
                missing_action_ids=research["missing_action_ids"],
                has_executable_candidate=self.bank.has_executable_candidate(snapshot),
                recent_progress_positive=self.memory.recent_progress_positive(),
                recent_unknown_or_irrelevant_count=self.memory.recent_unknown_or_irrelevant_count(),
                qwen_calls_this_game=self.budget.calls_this_game,
                total_calls_this_level=self.budget.total_calls_by_level.get(state.level_index, 0),
                control_probe_cycle_pending=control_cycle_pending,
            )
        if role is not None:
            self._call_qwen_role(role, snapshot, state)

        research = self.memory.action_research_status(snapshot)
        allow_semantic = not bool(research.get("missing_simple_action_ids"))
        candidate = None
        preflight = None
        seen_rejections: set[str] = set()
        for _ in range(max(8, self.config.max_active_hypotheses_in_packet + 8)):
            candidate = self.policy.choose_action(snapshot, self.memory, self.bank, self.explorer, self.config, allow_semantic=allow_semantic)
            if candidate is None:
                # Last-resort memory-aware fallback so empty queues do not hard-crash
                # the competition child before any accepted gateway action.
                fallback = self.policy.safe_fallback(snapshot, self.memory, self.config)
                self._last_action_selection = {
                    "verifier_exhausted": True,
                    "fallback_enabled": True,
                    "fallback_action": fallback.to_arc_action() if fallback is not None else None,
                    "level_index": state.level_index,
                    "step_index": state.step_index,
                    "semantic_queue_count": len(self.bank.semantic_test_queue),
                    "verified_semantic_count": len(self.bank.verified_semantic),
                    "coordinate_queue_count": len(self.bank.coordinate_test_queue),
                    "confirmed_rule_count": len(self.bank.confirmed_rules),
                    "unprobed_actions": self.memory.unprobed_action_effect_ids(snapshot, self.config),
                    "missing_action_ids": research["missing_action_ids"],
                    "qwen_calls_this_game": self.budget.calls_this_game,
                    "qwen_total_calls_this_level": self.budget.total_calls_by_level.get(state.level_index, 0),
                }
                runtime_log(
                    "no_executable_candidate_using_safe_fallback",
                    level=state.level_index,
                    step=state.step_index,
                    fallback_enabled=True,
                    semantic_queue_count=len(self.bank.semantic_test_queue),
                    verified_semantic_count=len(self.bank.verified_semantic),
                    coordinate_queue_count=len(self.bank.coordinate_test_queue),
                    confirmed_rule_count=len(self.bank.confirmed_rules),
                    unprobed_actions=self.memory.unprobed_action_effect_ids(snapshot, self.config),
                    missing_action_ids=research["missing_action_ids"],
                )
                if fallback is not None:
                    if self._should_reset_instead_of_flail(state.level_index, fallback):
                        self._fallback_streak_by_level[state.level_index] = 0
                        return self._emit_attempt_reset(snapshot, "no_executable_hypothesis_fallback_exhausted")
                    self._fallback_streak_by_level[state.level_index] += 1
                    return self._emit(snapshot, fallback)
                if self._can_reset_failed_attempt(state.level_index):
                    return self._emit_attempt_reset(snapshot, "no_executable_verified_hypothesis")
                raise RuntimeError("no verifier-authorized action: no executable hypothesis remains and safe fallback produced nothing")
            preflight = self.preflight_judge.validate(candidate, snapshot, self.memory, self.config)
            if preflight.valid:
                break
            sig = candidate.suppression_signature
            runtime_log("preflight_reject", action=candidate.to_arc_action(), reason=preflight.reason_code, fallback_enabled=False)
            self.bank.reject_candidate(candidate.hypothesis_id, f"preflight_{preflight.reason_code}", snapshot)
            if sig in seen_rejections:
                if self._can_reset_failed_attempt(state.level_index):
                    return self._emit_attempt_reset(snapshot, f"repeated_preflight_rejection:{preflight.reason_code}")
                raise RuntimeError(f"preflight rejected the same candidate twice with fallback disabled: {preflight.reason_code}")
            seen_rejections.add(sig)
        if candidate is None or preflight is None or not preflight.valid:
            reason = preflight.reason_code if preflight is not None else "none"
            if self._can_reset_failed_attempt(state.level_index):
                return self._emit_attempt_reset(snapshot, f"no_legal_verified_candidate:{reason}")
            raise RuntimeError(f"no legal candidate after verified queue scan with fallback disabled: {reason}")
        return self._emit(snapshot, candidate)

    def observe_action_result(self, after_observation: Mapping[str, Any] | None = None) -> bool:
        if self.pending_action is None:
            self._observed_transition_duplicate_skips += 1
            return False
        if after_observation is None:
            raise ValueError("after_observation is required while an official transition is pending")
        state, snapshot = self._prepare_snapshot(after_observation)
        self._commit_pending(snapshot)
        self._apply_level_boundary_after_commit(state, snapshot)
        self._latest_snapshot = snapshot
        return True

    def harness_telemetry(self) -> dict[str, Any]:
        return {
            "observed_transition_ingestions": self._observed_transition_ingestions,
            "observed_transition_duplicate_skips": self._observed_transition_duplicate_skips,
            "pending_official_transition": self.pending_action is not None,
            "pending_transition_token": self.pending_action.token_id if self.pending_action else None,
            "last_committed_transition_token": self._last_committed_token,
            "action_count_this_game": self._action_count_this_game,
            "game_over_reset_count": self._game_over_resets_this_game,
            "alternative_hypothesis_reset_count": self._alternative_hypothesis_resets_this_game,
            "pending_alternative_hypothesis_reset": self.bank.pending_alternative_reset(),
            "failed_memory_count": len(self.memory.failed_events),
            "irrelevant_memory_count": len(self.memory.irrelevant_events),
            "object_applicability_memory_count": len(self.memory.object_applicability_memory),
            "exhaustion_revisit_count": self._exhaustion_revisit_count,
            "last_level_index": self._last_level_index,
            "last_levels_completed": self._last_levels_completed,
            "qwen_calls_this_game": self.budget.calls_this_game,
            "qwen_primary_calls_by_level": dict(self.budget.primary_calls_by_level),
            "qwen_coordinate_calls_by_level": dict(self.budget.coordinate_calls_by_level),
            "qwen_reserve_calls_by_level": dict(self.budget.reserve_calls_by_level),
            "qwen_total_calls_by_level": dict(self.budget.total_calls_by_level),
            "level_attempt_index_by_level": dict(self._attempt_index_by_level),
            "action_count_by_level": dict(self._action_count_by_level),
            "max_level_attempts": self.config.max_level_attempts,
            "max_actions_per_level": self.config.max_actions_per_level,
            "terminal_level_limit": dict(self._terminal_level_limit),
            "level_attempt_records": list(self.memory.level_attempt_records),
            "confirmed_rule_count": len(self.bank.confirmed_rules),
            "verified_semantic_count": len(self.bank.verified_semantic),
            "semantic_queue_count": len(self.bank.semantic_test_queue),
            "coordinate_queue_count": len(self.bank.coordinate_test_queue),
            "fallback_queue_count": len(self.bank.fallback_exploration_queue),
            "action_selection": dict(self._last_action_selection),
            "latest_action_research": self.memory.action_research_status(self._latest_snapshot) if self._latest_snapshot is not None else {},
        }

    def _prepare_snapshot(self, raw_observation: Mapping[str, Any]):
        state = self.adapter.to_world_state(raw_observation)
        state = self._with_monotonic_step_index(state)
        if self._last_game_id is not None and state.game_id != self._last_game_id:
            if self.pending_action is not None:
                raise RuntimeError("game changed with an uncommitted official transition")
            self._reset_for_new_game(state.game_id)
        elif self._last_game_id is None:
            self.memory.reset_game(state.game_id)
            self._last_game_id = state.game_id
        snapshot = self.arga_lite.build(state, self.memory, self.config)
        self._planning_set = build_planning_set_from_snapshot(snapshot)
        self._planning_set_snapshot_id = snapshot.snapshot_id
        return state, snapshot

    def _with_monotonic_step_index(self, state):
        signature = stable_hash((
            state.game_id,
            state.level_index,
            state.levels_completed,
            state.state_name,
            state.grid,
            state.raw.get("metadata", {}) if isinstance(state.raw, dict) else {},
        ), "obs_")
        if signature != self._last_observation_signature:
            self._synthetic_step_index += 1
            self._last_observation_signature = signature
        effective_step = max(int(state.step_index), self._synthetic_step_index)
        if effective_step == state.step_index:
            return state
        raw = dict(state.raw)
        metadata = dict(raw.get("metadata", {}) or {})
        metadata["synthetic_step_index"] = effective_step
        raw["metadata"] = metadata
        return replace(state, step_index=effective_step, raw=raw)

    def _commit_pending(self, after_snapshot) -> None:
        pending = self.pending_action
        if pending is None:
            self._observed_transition_duplicate_skips += 1
            return
        if pending.token_id and pending.token_id == self._last_committed_token:
            self.pending_action = None
            self._observed_transition_duplicate_skips += 1
            return
        retry_reset = pending.action.action_id == "RESET" and pending.action.source in _ATTEMPT_RETRY_RESET_SOURCES
        alternative_reset = (
            pending.action.action_id == "RESET"
            and pending.action.source == _ALTERNATIVE_HYPOTHESIS_RESET_SOURCE
        )
        research_entry_reset = (
            pending.action.action_id == "RESET"
            and pending.action.source == _RESEARCH_ENTRY_RESET_SOURCE
        )
        if pending.action.action_id == "RESET":
            runtime_log(
                "reset_transition_observed",
                level=after_snapshot.level_index,
                step=after_snapshot.step_index,
                source=pending.action.source,
            )
        else:
            judgment = self.transition_judge.evaluate(
                pending.before_snapshot,
                pending.action,
                after_snapshot,
                pending,
                self.memory,
                self.config,
            )
            self.memory.add_judgment(judgment)
            trajectory_evaluation = self.bank.update(judgment, after_snapshot)
            self.memory.add_trajectory_evaluation(trajectory_evaluation)
            if trajectory_evaluation is not None:
                runtime_log(
                    "trajectory_evaluation",
                    hypothesis_id=trajectory_evaluation.hypothesis_id,
                    binding_id=trajectory_evaluation.binding_id,
                    mechanic_result=trajectory_evaluation.mechanic_result.value,
                    goal_progress=trajectory_evaluation.goal_progress.value,
                    semantic_judgment=trajectory_evaluation.semantic_judgment.value,
                    reason=trajectory_evaluation.reason_code,
                    executed_actions=list(trajectory_evaluation.executed_action_ids),
                    error_before=trajectory_evaluation.error_before,
                    error_after=trajectory_evaluation.error_after,
                    error_delta=trajectory_evaluation.error_delta,
                )
        self._last_committed_token = pending.token_id
        self.pending_action = None
        self._observed_transition_ingestions += 1
        if retry_reset:
            self._begin_retry_attempt(after_snapshot.level_index, after_snapshot.step_index, pending.action.source)
        elif alternative_reset:
            rebound_bindings = self.bank.rebind_pending_alternatives(after_snapshot, self.memory)
            for binding in rebound_bindings:
                self.memory.record_semantic_binding(binding, after_snapshot)
            request = self.bank.acknowledge_alternative_reset()
            self.memory.begin_hypothesis_alternative()
            self._alternative_hypothesis_resets_this_game += 1
            runtime_log(
                "alternative_hypothesis_reset_observed",
                level=after_snapshot.level_index,
                step=after_snapshot.step_index,
                proposal_batch_id=(request or {}).get("proposal_batch_id"),
                completed_hypothesis_id=(request or {}).get("completed_hypothesis_id"),
                remaining_hypothesis_ids=(request or {}).get("remaining_hypothesis_ids"),
                rebound_binding_count=len(rebound_bindings),
            )
        elif research_entry_reset:
            self.memory.mark_attempt_entry_restored(after_snapshot.level_index, pending.action.source)
            runtime_log(
                "research_entry_reset_observed",
                level=after_snapshot.level_index,
                attempt_index=self._attempt_index_by_level.get(after_snapshot.level_index, 0),
                step=after_snapshot.step_index,
                retained_action_memory=len(self.memory.action_memory_records),
            )

    def _apply_level_boundary_after_commit(self, state, snapshot) -> bool:
        is_new_level = (
            self.memory.mark_observed_level(state.game_id, state.level_index)
            or self._last_level_index != state.level_index
            or state.levels_completed > self._last_levels_completed
        )
        if is_new_level:
            runtime_log(
                "level_boundary_observed",
                game_id=state.game_id,
                level_index=state.level_index,
                previous_level_index=self._last_level_index,
                levels_completed=state.levels_completed,
                previous_levels_completed=self._last_levels_completed,
                step_index=state.step_index,
                bank_reset=True,
                memory_events=len(self.memory.events),
                object_applicability_memory_count=len(self.memory.object_applicability_memory),
            )
            self.bank.reset_level(state.level_index)
            self._attempt_index_by_level[state.level_index] = 0
            self._action_count_by_level[state.level_index] = 0
            self._terminal_level_limit = {}
            self._reset_qwen_attempt_budget(state.level_index)
            self._qwen_empty_primary_streak = 0
            self.memory.begin_level_attempt(state.level_index, 0, retry=False, entry_source="level_entry")
            self._last_level_index = state.level_index
            self._last_levels_completed = state.levels_completed
        return is_new_level

    def _reset_qwen_attempt_budget(self, level_index: int) -> None:
        self.budget.primary_calls_by_level.pop(level_index, None)
        self.budget.reserve_calls_by_level.pop(level_index, None)
        self.budget.coordinate_calls_by_level.pop(level_index, None)
        self.budget.total_calls_by_level.pop(level_index, None)
        self.budget.last_qwen_step = -10**9

    def _begin_retry_attempt(self, level_index: int, step_index: int, reset_source: str) -> None:
        attempt_index = self._attempt_index_by_level.get(level_index, 0) + 1
        self._attempt_index_by_level[level_index] = attempt_index
        self._fallback_streak_by_level[level_index] = 0
        self._reset_qwen_attempt_budget(level_index)
        self.bank.reset_level(level_index)
        self.memory.begin_level_attempt(level_index, attempt_index, retry=True, entry_source=reset_source)
        runtime_log(
            "level_attempt_started",
            level=level_index,
            attempt_index=attempt_index,
            step=step_index,
            reset_source=reset_source,
            retained_action_memory=len(self.memory.action_memory_records),
            retained_attempt_failures=len(self.memory.level_attempt_records),
        )

    _MAX_FALLBACK_STREAK_BEFORE_ATTEMPT_RESET = 8
    _FALLBACK_SOURCES = {"memory_aware_fallback", "exhaustion_revisit"}

    def _should_reset_instead_of_flail(self, level_index: int, fallback: CandidateAction) -> bool:
        """Dead-attempt detection for the RESET-with-explanation cycle.

        Fires only when the policy has nothing but fallback/liveness actions,
        Qwen was already spent on this attempt, no recent step produced positive
        progress (never reset a line that is still advancing), and the flail
        streak reached the bound. The reset retains verifier/judge feedback in
        memory, so the next Qwen packet explains why the attempt failed.
        """
        if fallback.source not in self._FALLBACK_SOURCES:
            return False
        if not self._can_reset_failed_attempt(level_index):
            return False
        if self.memory.recent_progress_positive():
            return False
        return self._fallback_streak_by_level.get(level_index, 0) + 1 >= self._MAX_FALLBACK_STREAK_BEFORE_ATTEMPT_RESET

    def _can_reset_failed_attempt(self, level_index: int) -> bool:
        # Semantics of max_level_attempts (LCLD_MAX_LEVEL_ATTEMPTS):
        #   0 = unlimited attempts (legacy behavior, no cap enforced here)
        #  >0 = maximum number of attempts allowed per level
        if self.config.max_level_attempts > 0:
            current_attempt = self._attempt_index_by_level.get(level_index, 0)
            if current_attempt >= self.config.max_level_attempts:
                return False
        return (
            self.config.reset_on_game_over
            and self.config.enable_qwen
            and self.config.qwen_backend != "disabled"
            and self.budget.total_calls_by_level.get(level_index, 0) > 0
        )

    def _should_restore_entry_after_research(self, snapshot, control_cycle_pending: bool) -> bool:
        level_index = int(snapshot.level_index)
        if (
            control_cycle_pending
            or not self.config.enable_qwen
            or self.config.qwen_backend == "disabled"
            or self.budget.primary_calls_by_level.get(level_index, 0) > 0
        ):
            return False
        if self.memory.current_attempt_entry_source(level_index) == _RESEARCH_ENTRY_RESET_SOURCE:
            return False
        research = self.memory.action_research_status(snapshot)
        return (
            not research["missing_action_ids"]
            and self.memory.current_attempt_has_research_actions(level_index)
        )

    def _emit_attempt_reset(self, snapshot, reset_trigger: str, *, source: str = "failed_attempt_reset") -> dict[str, Any]:
        level_index = snapshot.level_index
        attempt_index = self._attempt_index_by_level.get(level_index, 0)
        feedback = self.bank.attempt_feedback(self.config.max_memory_notes_in_packet)
        record = self.memory.record_level_attempt_failure(
            level_index,
            attempt_index,
            step_index=snapshot.step_index,
            reset_trigger=reset_trigger,
            qwen_calls=self.budget.total_calls_by_level.get(level_index, 0),
            verifier_feedback=feedback,
        )
        if source == "game_over_level_reset":
            self._game_over_resets_this_game += 1
            self._game_over_resets_by_level[level_index] += 1
        runtime_log(
            "level_attempt_reset_emit",
            level=level_index,
            attempt_index=attempt_index,
            step=snapshot.step_index,
            reset_trigger=reset_trigger,
            qwen_calls=record["qwen_calls"],
            rejection_count=len(feedback.get("rejections") or []),
            retained_action_runs=len(record.get("action_runs") or []),
        )
        candidate = CandidateAction(
            "RESET",
            reason=f"failed level attempt; restart with retained evidence: {reset_trigger}",
            source=source,
        )
        return self._emit(snapshot, candidate)

    def _emit(self, snapshot, candidate: CandidateAction) -> dict[str, Any]:
        if self.pending_action is not None:
            raise RuntimeError("attempted to emit while an official transition is pending")
        if candidate.source not in self._FALLBACK_SOURCES:
            self._fallback_streak_by_level[snapshot.level_index] = 0
        is_coordinate = candidate.action_id in snapshot.coordinate_action_ids or candidate.x is not None or candidate.y is not None
        is_coordinate_research = is_coordinate and is_coordinate_research_source(candidate.source)
        self.memory.mark_emitted_action(
            snapshot.level_index,
            candidate.action_id,
            candidate.suppression_signature,
            state_signature=snapshot.semantic_state_signature,
            is_coordinate=is_coordinate,
            is_coordinate_research=is_coordinate_research,
            coordinate_candidate_id=candidate.coordinate_candidate_id,
            coordinate_x=candidate.x,
            coordinate_y=candidate.y,
        )
        if candidate.allow_exhaustion_revisit:
            self._exhaustion_revisit_count += 1
        token = stable_hash((snapshot.snapshot_id, candidate.suppression_signature, self._action_count_this_game), "pending_")
        self.pending_action = PendingAction(snapshot, candidate, candidate.hypothesis_id, candidate.reason, token)
        self._action_count_this_game += 1
        self._action_count_by_level[snapshot.level_index] += 1
        self._last_action_selection = {
            "verifier_exhausted": False,
            "selected_action_id": candidate.action_id,
            "selected_action_source": candidate.source,
            "hypothesis_id": candidate.hypothesis_id,
            "fallback_enabled": False,
        }
        return candidate.to_arc_action()

    def _call_qwen_role(self, role: QwenRole, snapshot, state) -> None:
        try:
            packet = (
                self.packet_builder.build_coordinate_packet(snapshot, self.memory, self.bank, self.config)
                if role is QwenRole.COORDINATE
                else self.packet_builder.build_semantic_packet(snapshot, self.memory, self.bank, role, self.config)
            )
            # Count the real invocation even when the backend times out or returns malformed output.
            record_qwen_call(role, state.level_index, state.step_index, self.budget)
            # Anti-fixation: when consecutive PRIMARY calls produce only rejected
            # hypotheses (dominant failure mode observed live: the model rephrases
            # the same failed trajectory), escalate sampling temperature to break
            # the attractor. Streak resets on any productive call or level change.
            call_config = self.config
            if role is QwenRole.PRIMARY and self._qwen_empty_primary_streak > 0:
                escalated = min(0.9, float(self.config.qwen_temperature) + 0.2 * self._qwen_empty_primary_streak)
                call_config = replace(self.config, qwen_temperature=escalated)
            proposals = self.qwen.call(role, packet, call_config)
            self._qwen_successful_calls_this_game += 1
            invalid_before = len(self.bank.invalid_rejections)
            semantic_before = len(self.bank.semantic_test_queue)
            coordinate_before = len(self.bank.coordinate_test_queue)
            self.bank.add_qwen_output(role, proposals, snapshot, self.config, packet=packet)
            for item in (
                self.bank.semantic_test_queue
                + self.bank.verified_semantic
                + self.bank.coordinate_test_queue
                + self.bank.confirmed_rules
            ):
                self.memory.record_semantic_binding(item.semantic_binding, snapshot)
            new_rejections = self.bank.invalid_rejections[invalid_before:]
            semantic_added = max(0, len(self.bank.semantic_test_queue) - semantic_before)
            coordinate_added = max(0, len(self.bank.coordinate_test_queue) - coordinate_before)
            if role is QwenRole.PRIMARY:
                if semantic_added + coordinate_added > 0:
                    self._qwen_empty_primary_streak = 0
                else:
                    self._qwen_empty_primary_streak += 1
            runtime_log(
                "qwen_call",
                role=role.value,
                level=state.level_index,
                attempt_index=self._attempt_index_by_level.get(state.level_index, 0),
                step=state.step_index,
                accepted=bool(proposals),
                decision=(proposals or {}).get("decision") if isinstance(proposals, dict) else None,
                schema_version=(proposals or {}).get("schema_version") if isinstance(proposals, dict) else None,
                hypothesis_count=len((proposals or {}).get("hypotheses") or []) if isinstance(proposals, dict) else 0,
                coordinate_hypothesis_count=len((proposals or {}).get("candidate_sequence") or []) if isinstance(proposals, dict) else 0,
                semantic_queue_added=semantic_added,
                coordinate_queue_added=coordinate_added,
                rejection_count=len(new_rejections),
                rejection_reasons=[str(item.get("reason")) for item in new_rejections[-8:] if isinstance(item, dict)],
                empty_primary_streak=self._qwen_empty_primary_streak,
                sampling_temperature=float(call_config.qwen_temperature),
            )

            # Offline Contour B: repair trajectories only. Session owns state;
            # verifier is a pure advisory function. Never promote into confirmed_rules.
            try:
                if isinstance(packet, dict):
                    self._planning_set = build_planning_set_from_packet(packet, snapshot)
                    self._planning_set_snapshot_id = snapshot.snapshot_id
                    identity_violations = assert_dual_view_identity(packet, self._planning_set)
                    if identity_violations:
                        runtime_log(
                            "dual_view_identity_violation",
                            snapshot_id=snapshot.snapshot_id,
                            violations=identity_violations[:5],
                        )
                verifier = TrajectoryVerifier(self.config)
                new_semantic = list(self.bank.semantic_test_queue[semantic_before:])
                for hyp in new_semantic:
                    try:
                        result = verifier.verify_hypothesis(
                            hyp,
                            snapshot,
                            self.memory,
                            self.config,
                            verifier_packet=packet.get("verifier_packet") if isinstance(packet, dict) else None,
                        )
                        runtime_log(
                            "trajectory_verifier_result",
                            hypothesis_id=getattr(hyp, "hypothesis_id", None),
                            status=result.status,
                            reason=result.reason,
                            confidence=result.confidence,
                        )
                        if result.status in ("ACCEPT", "CORRECTED", "PARTIAL") and result.plan is not None:
                            hyp.trajectory_start_snapshot = snapshot
                            original_length = len(hyp.test_plan)
                            original_plan = tuple(hyp.test_plan)
                            if result.plan.steps:
                                hyp.test_plan = rebind_plan_to_test_steps(
                                    list(result.plan.steps),
                                    snapshot,
                                    hyp.hypothesis_id,
                                    semantic_binding=getattr(hyp, "semantic_binding", None),
                                    original_plan=original_plan,
                                )
                                if result.plan.expected_final_state_hash:
                                    hyp.expected_final_state_hash = result.plan.expected_final_state_hash
                                elif not hyp.expected_final_state_hash and isinstance(result.plan.goal, dict):
                                    goal_hash = result.plan.goal.get("expected_final_state_hash")
                                    if isinstance(goal_hash, str) and goal_hash:
                                        hyp.expected_final_state_hash = goal_hash
                                hyp.trajectory_verified = True
                                try:
                                    self.bank.semantic_test_queue.remove(hyp)
                                except ValueError:
                                    pass
                                if hyp not in self.bank.verified_semantic:
                                    self.bank.verified_semantic.append(hyp)
                                runtime_log(
                                    "trajectory_plan_applied",
                                    hypothesis_id=hyp.hypothesis_id,
                                    original_plan_length=original_length,
                                    corrected_plan_length=len(hyp.test_plan),
                                    expected_final_state_hash=hyp.expected_final_state_hash,
                                    status=result.status,
                                    reason=result.reason,
                                    goal_hash_source=(result.details or {}).get("goal_hash_source") if isinstance(result.details, dict) else None,
                                    prefix_steps=(result.details or {}).get("prefix_steps") if isinstance(result.details, dict) else None,
                                )
                            else:
                                # Empty repaired plan: keep original model steps executable.
                                runtime_log(
                                    "trajectory_plan_empty_kept_model",
                                    hypothesis_id=hyp.hypothesis_id,
                                    status=result.status,
                                    reason=result.reason,
                                )
                        elif result.status == "PASSTHROUGH":
                            # Syntax-ok model plan stays on semantic_test_queue for empiric execution.
                            runtime_log(
                                "trajectory_passthrough",
                                hypothesis_id=getattr(hyp, "hypothesis_id", None),
                                reason=result.reason,
                            )
                        elif result.status == "REJECT":
                            # Soft reject: mark invalid so policy skips it, but keep in rejected for feedback.
                            hyp.validity = Validity.INVALID
                            if hyp not in self.bank.rejected:
                                self.bank.rejected.append(hyp)
                            self.bank.invalid_rejections.append({
                                "reason": "trajectory_verifier_reject",
                                "hypothesis_id": hyp.hypothesis_id,
                                "verifier_reason": result.reason,
                                "details": getattr(result, "details", None),
                            })
                    except Exception as exc:
                        runtime_log("trajectory_verifier_error", hypothesis_id=getattr(hyp, "hypothesis_id", None), exc=str(exc)[:400])
            except Exception as exc:
                runtime_log("trajectory_verifier_init_failed", exc_type=type(exc).__name__, exc=str(exc)[:400])

        except QwenBackendError as exc:
            # Fail fast only while the backend has never produced a successful
            # call this game (misconfiguration must stay loud). Once the
            # backend is proven, a single timeout/parse failure degrades to a
            # skipped call instead of killing the whole game worker.
            if self.config.qwen_require_runtime and self._qwen_successful_calls_this_game <= 0:
                raise
            runtime_log(
                "qwen_backend_soft_failure",
                role=role.value,
                level=state.level_index,
                attempt_index=self._attempt_index_by_level.get(state.level_index, 0),
                exc_type=type(exc).__name__,
                exc=str(exc)[:300],
                qwen_successful_calls_this_game=self._qwen_successful_calls_this_game,
            )
        except QwenPacketNotReady as exc:
            runtime_log("qwen_packet_not_ready", role=role.value, reason=str(exc)[:500])
        except Exception as exc:
            if self.config.qwen_require_runtime and self._qwen_successful_calls_this_game <= 0:
                raise
            runtime_log("qwen_soft_failure", role=role.value, level=state.level_index, attempt_index=self._attempt_index_by_level.get(state.level_index, 0), exc_type=type(exc).__name__, exc=str(exc)[:500], qwen_successful_calls_this_game=self._qwen_successful_calls_this_game)

    def _handle_game_over(self, snapshot) -> dict[str, Any]:
        if not self.config.reset_on_game_over:
            raise RuntimeError("GAME_OVER requires RESET under ARC-AGI-3 rules, but reset_on_game_over is disabled")
        runtime_log(
            "game_over_observed",
            level=snapshot.level_index,
            step=snapshot.step_index,
            attempt_index=self._attempt_index_by_level.get(snapshot.level_index, 0),
        )
        return self._emit_attempt_reset(snapshot, "game_over_observed", source="game_over_level_reset")

    def _reset_for_new_game(self, game_id: str) -> None:
        self.memory.reset_game(game_id)
        self.bank = HypothesisBank()
        self.budget = QwenBudgetState()
        self.pending_action = None
        self._last_game_id = game_id
        self._last_level_index = None
        self._last_levels_completed = 0
        self._action_count_this_game = 0
        self._qwen_successful_calls_this_game = 0
        self._qwen_empty_primary_streak = 0
        self._game_over_resets_this_game = 0
        self._game_over_resets_by_level = defaultdict(int)
        self._attempt_index_by_level = defaultdict(int)
        self._action_count_by_level = defaultdict(int)
        self._terminal_level_limit = {}
        self._last_committed_token = None
        self._latest_snapshot = None
        self._exhaustion_revisit_count = 0
        self._last_observation_signature = None
        self._synthetic_step_index = -1
        self._last_action_selection = {}
        self._alternative_hypothesis_resets_this_game = 0
        self._fallback_streak_by_level = defaultdict(int)
