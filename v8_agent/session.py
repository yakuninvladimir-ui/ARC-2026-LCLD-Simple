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
from .memory import GameMemory
from .observe import stable_hash
from .policy import Policy
from .qwen_packet import QwenPacketBuilder, QwenPacketNotReady
from .qwen_roles import can_call_qwen_role, record_qwen_call
from .types import CandidateAction, PendingAction, QwenBudgetState, QwenRole


_ATTEMPT_RETRY_RESET_SOURCES = {"game_over_level_reset", "failed_attempt_reset"}


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
        self._game_over_resets_this_game = 0
        self._game_over_resets_by_level: dict[int, int] = defaultdict(int)
        self._attempt_index_by_level: dict[int, int] = defaultdict(int)
        self._observed_transition_ingestions = 0
        self._observed_transition_duplicate_skips = 0
        self._last_committed_token: str | None = None
        self._latest_snapshot = None
        self._exhaustion_revisit_count = 0
        self._last_observation_signature: str | None = None
        self._synthetic_step_index = -1
        self._last_action_selection: dict[str, Any] = {}

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

        if state.game_over:
            return self._handle_game_over(snapshot)
        if state.terminal:
            return CandidateAction("RESET", reason="terminal success observed; outer loop should stop", source="terminal_guard").to_arc_action()
        if state.state_name in {"NOT_STARTED", "NOT_PLAYED"}:
            return self._emit(snapshot, CandidateAction("RESET", reason="initial environment start", source="initial_reset"))

        role = choose_qwen_role(state, snapshot, self.memory, self.bank, self.budget, self.config, is_new_level=is_new_level)
        if role is None and self.config.enable_qwen and self.config.qwen_backend != "disabled":
            runtime_log(
                "qwen_no_call",
                level=state.level_index,
                step=state.step_index,
                is_new_level=is_new_level,
                coordinate_action_count=len(snapshot.coordinate_action_ids),
                coordinate_research_needed=bool(snapshot.coordinate_action_ids and self.memory.coordinate_research_needed(state.level_index)),
                has_executable_candidate=self.bank.has_executable_candidate(snapshot),
                recent_progress_positive=self.memory.recent_progress_positive(),
                recent_unknown_or_irrelevant_count=self.memory.recent_unknown_or_irrelevant_count(),
                qwen_calls_this_game=self.budget.calls_this_game,
                total_calls_this_level=self.budget.total_calls_by_level.get(state.level_index, 0),
            )
        if role is not None:
            self._call_qwen_role(role, snapshot, state)
            if (
                role is QwenRole.COORDINATE
                and not self.bank.has_executable_candidate(snapshot)
                and can_call_qwen_role(
                    QwenRole.PRIMARY,
                    state.level_index,
                    state.step_index,
                    self.budget,
                    self.config,
                    ignore_spacing=True,
                )
            ):
                runtime_log(
                    "coordinate_primary_handoff",
                    level=state.level_index,
                    step=state.step_index,
                    attempt_index=self._attempt_index_by_level.get(state.level_index, 0),
                    reason="coordinate_call_produced_no_executable_probe",
                )
                self._call_qwen_role(QwenRole.PRIMARY, snapshot, state)

        candidate = None
        preflight = None
        seen_rejections: set[str] = set()
        for _ in range(max(8, self.config.max_active_hypotheses_in_packet + 8)):
            candidate = self.policy.choose_action(snapshot, self.memory, self.bank, self.explorer, self.config)
            if candidate is None:
                self._last_action_selection = {
                    "verifier_exhausted": True,
                    "fallback_enabled": False,
                    "level_index": state.level_index,
                    "step_index": state.step_index,
                    "semantic_queue_count": len(self.bank.semantic_test_queue),
                    "coordinate_queue_count": len(self.bank.coordinate_test_queue),
                    "confirmed_rule_count": len(self.bank.confirmed_rules),
                    "unprobed_actions": self.memory.unprobed_action_effect_ids(snapshot, self.config),
                    "qwen_calls_this_game": self.budget.calls_this_game,
                    "qwen_total_calls_this_level": self.budget.total_calls_by_level.get(state.level_index, 0),
                }
                runtime_log(
                    "no_executable_candidate",
                    level=state.level_index,
                    step=state.step_index,
                    fallback_enabled=False,
                    semantic_queue_count=len(self.bank.semantic_test_queue),
                    coordinate_queue_count=len(self.bank.coordinate_test_queue),
                    confirmed_rule_count=len(self.bank.confirmed_rules),
                    unprobed_actions=self.memory.unprobed_action_effect_ids(snapshot, self.config),
                )
                if self._can_reset_failed_attempt(state.level_index):
                    return self._emit_attempt_reset(snapshot, "no_executable_verified_hypothesis")
                raise RuntimeError("no verifier-authorized action: no executable verified hypothesis remains and deterministic fallback is disabled")
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
            "level_attempt_records": list(self.memory.level_attempt_records),
            "confirmed_rule_count": len(self.bank.confirmed_rules),
            "semantic_queue_count": len(self.bank.semantic_test_queue),
            "coordinate_queue_count": len(self.bank.coordinate_test_queue),
            "fallback_queue_count": len(self.bank.fallback_exploration_queue),
            "action_selection": dict(self._last_action_selection),
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
        if pending.action.action_id == "RESET":
            runtime_log("reset_transition_observed", level=after_snapshot.level_index, step=after_snapshot.step_index)
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
            self.bank.update(judgment)
        self._last_committed_token = pending.token_id
        self.pending_action = None
        self._observed_transition_ingestions += 1
        if retry_reset:
            self._begin_retry_attempt(after_snapshot.level_index, after_snapshot.step_index, pending.action.source)

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
            self._reset_qwen_attempt_budget(state.level_index)
            self.memory.begin_level_attempt(state.level_index, 0, retry=False)
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
        self._reset_qwen_attempt_budget(level_index)
        self.bank.reset_level(level_index)
        self.memory.begin_level_attempt(level_index, attempt_index, retry=True)
        runtime_log(
            "level_attempt_started",
            level=level_index,
            attempt_index=attempt_index,
            step=step_index,
            reset_source=reset_source,
            retained_action_memory=len(self.memory.action_memory_records),
            retained_attempt_failures=len(self.memory.level_attempt_records),
        )

    def _can_reset_failed_attempt(self, level_index: int) -> bool:
        return (
            self.config.reset_on_game_over
            and self.config.enable_qwen
            and self.config.qwen_backend != "disabled"
            and self.budget.total_calls_by_level.get(level_index, 0) > 0
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
        is_coordinate = candidate.action_id in snapshot.coordinate_action_ids or candidate.x is not None or candidate.y is not None
        self.memory.mark_emitted_action(
            snapshot.level_index,
            candidate.action_id,
            candidate.suppression_signature,
            state_signature=snapshot.semantic_state_signature,
            is_coordinate=is_coordinate,
            coordinate_candidate_id=candidate.coordinate_candidate_id,
            coordinate_x=candidate.x,
            coordinate_y=candidate.y,
        )
        if candidate.allow_exhaustion_revisit:
            self._exhaustion_revisit_count += 1
        token = stable_hash((snapshot.snapshot_id, candidate.suppression_signature, self._action_count_this_game), "pending_")
        self.pending_action = PendingAction(snapshot, candidate, candidate.hypothesis_id, candidate.reason, token)
        self._action_count_this_game += 1
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
            proposals = self.qwen.call(role, packet, self.config)
            invalid_before = len(self.bank.invalid_rejections)
            semantic_before = len(self.bank.semantic_test_queue)
            coordinate_before = len(self.bank.coordinate_test_queue)
            self.bank.add_qwen_output(role, proposals, snapshot, self.config, packet=packet)
            new_rejections = self.bank.invalid_rejections[invalid_before:]
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
                semantic_queue_added=max(0, len(self.bank.semantic_test_queue) - semantic_before),
                coordinate_queue_added=max(0, len(self.bank.coordinate_test_queue) - coordinate_before),
                rejection_count=len(new_rejections),
                rejection_reasons=[str(item.get("reason")) for item in new_rejections[-8:] if isinstance(item, dict)],
            )
        except QwenBackendError:
            if self.config.qwen_require_runtime:
                raise
            runtime_log("qwen_backend_soft_failure", role=role.value, level=state.level_index, attempt_index=self._attempt_index_by_level.get(state.level_index, 0))
        except QwenPacketNotReady as exc:
            runtime_log("qwen_packet_not_ready", role=role.value, reason=str(exc)[:500])
        except Exception as exc:
            if self.config.qwen_require_runtime:
                raise
            runtime_log("qwen_soft_failure", role=role.value, level=state.level_index, attempt_index=self._attempt_index_by_level.get(state.level_index, 0), exc_type=type(exc).__name__, exc=str(exc)[:500])

    def _handle_game_over(self, snapshot) -> dict[str, Any]:
        if not self.config.reset_on_game_over:
            raise RuntimeError("GAME_OVER requires RESET under ARC-AGI-3 rules, but reset_on_game_over is disabled")
        self._game_over_resets_this_game += 1
        self._game_over_resets_by_level[snapshot.level_index] += 1
        # No reset limit is terminal. Competition timeout is the only orchestration terminal.
        runtime_log("game_over_reset_emit", level=snapshot.level_index, step=snapshot.step_index, resets_game=self._game_over_resets_this_game, resets_level=self._game_over_resets_by_level[snapshot.level_index])
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
        self._game_over_resets_this_game = 0
        self._game_over_resets_by_level = defaultdict(int)
        self._attempt_index_by_level = defaultdict(int)
        self._last_committed_token = None
        self._latest_snapshot = None
        self._exhaustion_revisit_count = 0
        self._last_observation_signature = None
        self._synthetic_step_index = -1
        self._last_action_selection = {}
