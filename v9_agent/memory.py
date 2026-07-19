from __future__ import annotations

from dataclasses import asdict, replace
from math import hypot
from typing import Any

from .action_effects import default_question_domain, merge_effect_record, resolve_question
from .config import V8Config
from .observe import stable_hash
from .reverse_semantics import derive_invariant_observations
from .types import (
    ActionEffectRecord,
    Attribution,
    CoordinateEffectRecord,
    EvidenceAuthority,
    EvidenceStatus,
    Judgment,
    MemoryEvent,
    ObjectRecord,
    Progress,
    QwenRole,
    Relevance,
    SemanticQuestion,
    SemanticQuestionType,
    SemanticBindingResult,
    TrajectoryEvaluation,
    TriTruth,
    Validity,
)


COORDINATE_RESEARCH_SOURCES = frozenset({
    "coordinate_qwen",
    "deterministic_coordinate_explorer",
    "memory_aware_fallback",
})
ACTION_RESEARCH_SOURCES = frozenset({
    *COORDINATE_RESEARCH_SOURCES,
    "initial_action_probe",
    "action_probe_control_before",
    "action_probe_new_action",
    "action_probe_control_after",
})


def is_coordinate_research_source(source: str | None) -> bool:
    return str(source or "") in COORDINATE_RESEARCH_SOURCES


def is_action_research_source(source: str | None) -> bool:
    return str(source or "") in ACTION_RESEARCH_SOURCES


class GameMemory:
    # Raw transition history is useful for the current retry, but it must not
    # grow with every RESET during a long competition game. Durable summaries
    # (action_effects, invariants, bindings) remain intact.
    _MAX_RAW_ACTION_RECORDS = 512
    _MAX_RAW_EVENTS = 512
    _MAX_ACTION_SURFACE_RECORDS = 192
    _MAX_OBJECT_APPLICABILITY_RECORDS = 256
    _MAX_TRAJECTORY_EVALUATIONS = 96
    _MAX_FAILED_ATTEMPTS_ACTIVE_LEVEL = 24
    _MAX_FAILED_ATTEMPTS_OTHER_LEVEL = 8

    def __init__(self) -> None:
        self.events: list[MemoryEvent] = []
        self.failed_events: list[MemoryEvent] = []
        self.irrelevant_events: list[MemoryEvent] = []
        self.coordinate_effects: list[CoordinateEffectRecord] = []
        self.action_effects: dict[tuple[str, str | None], ActionEffectRecord] = {}
        self.action_memory_records: list[dict[str, Any]] = []
        self.action_surface_memory_records: list[dict[str, Any]] = []
        self.level_attempt_records: list[dict[str, Any]] = []
        self.object_applicability_memory: list[dict[str, Any]] = []
        self.semantic_bindings: dict[str, SemanticBindingResult] = {}
        self.semantic_binding_contexts: dict[str, dict[str, Any]] = {}
        self.trajectory_evaluations: list[TrajectoryEvaluation] = []
        self.semantic_invariants: dict[str, dict[str, Any]] = {}
        self.semantic_questions: dict[str, SemanticQuestion] = {}
        self.simple_action_attempts_by_level: dict[int, dict[str, int]] = {}
        self.action_attempts_by_signature: dict[str, int] = {}
        self.coordinate_probe_counts_by_level: dict[int, int] = {}
        self.coordinate_probe_signature_counts: dict[str, int] = {}
        self.coordinate_candidates_clicked_this_attempt: set[str] = set()
        self.confirmed_rule_contradicted: bool = False
        self._current_game_id: str | None = None
        self._last_level_index: int | None = None
        self._last_changed_center: tuple[int, int] | None = None
        self._tracks_by_level: dict[int, dict[str, dict[str, Any]]] = {}
        self._next_track_index_by_level: dict[int, int] = {}
        self._current_attempt_by_level: dict[int, int] = {}
        self._attempt_action_offsets_by_level: dict[int, int] = {}
        self._attempt_entry_source_by_level: dict[int, str] = {}
        self._next_event_sequence = 1

    def reset_game(self, game_id: str) -> None:
        self.__init__()
        self._current_game_id = game_id

    def is_new_game(self, game_id: str) -> bool:
        return self._current_game_id is not None and self._current_game_id != game_id

    def mark_observed_level(self, game_id: str, level_index: int) -> bool:
        if self._current_game_id is None:
            self._current_game_id = game_id
        is_new = self._last_level_index != level_index
        self._last_level_index = level_index
        if is_new:
            self.simple_action_attempts_by_level.setdefault(level_index, {})
            self.coordinate_probe_counts_by_level.setdefault(level_index, 0)
            self._tracks_by_level.setdefault(level_index, {})
            self._next_track_index_by_level.setdefault(level_index, 0)
        return is_new

    def begin_level_attempt(
        self,
        level_index: int,
        attempt_index: int,
        *,
        retry: bool,
        entry_source: str | None = None,
    ) -> None:
        self._current_attempt_by_level[level_index] = attempt_index
        self._attempt_action_offsets_by_level[level_index] = len(self.action_memory_records)
        self._attempt_entry_source_by_level[level_index] = str(
            entry_source or ("attempt_retry_reset" if retry else "level_entry")
        )
        self.coordinate_candidates_clicked_this_attempt.clear()
        if retry:
            # Mechanics remain game-scoped. Execution suppression is attempt-scoped.
            self.action_attempts_by_signature.clear()
            self.coordinate_probe_counts_by_level[level_index] = 0
            self.coordinate_probe_signature_counts.clear()

    def begin_hypothesis_alternative(self) -> None:
        # A level RESET restores the proposal entry state. Execution suppression
        # must not make the next hypothesis fail merely because its prefix
        # overlaps the previous alternative; all observed evidence is retained.
        self.action_attempts_by_signature.clear()

    def current_attempt_index(self, level_index: int) -> int:
        return self._current_attempt_by_level.get(level_index, 0)

    def mark_attempt_entry_restored(self, level_index: int, source: str) -> None:
        self._attempt_entry_source_by_level[int(level_index)] = str(source)

    def current_attempt_entry_source(self, level_index: int) -> str:
        return self._attempt_entry_source_by_level.get(int(level_index), "level_entry")

    def current_attempt_has_research_actions(self, level_index: int) -> bool:
        start = self._attempt_action_offsets_by_level.get(int(level_index), 0)
        for record in self.action_memory_records[start:]:
            record_level = int(
                record.get("level_index_before")
                if record.get("level_index_before") is not None
                else record.get("level_index", -1)
            )
            if record_level == int(level_index) and is_action_research_source(record.get("source")):
                return True
        return False

    def record_level_attempt_failure(
        self,
        level_index: int,
        attempt_index: int,
        *,
        step_index: int,
        reset_trigger: str,
        qwen_calls: int,
        verifier_feedback: dict[str, Any],
    ) -> dict[str, Any]:
        start = self._attempt_action_offsets_by_level.get(level_index, 0)
        records = [
            record for record in self.action_memory_records[start:]
            if int(
                record.get("level_index_before")
                if record.get("level_index_before") is not None
                else record.get("level_index", -1)
            ) == level_index
        ]
        research_records = [
            record for record in records
            if is_action_research_source(record.get("source"))
        ]
        goal_records = [
            record for record in records
            if not is_action_research_source(record.get("source"))
        ]
        failures = []
        for record in goal_records:
            game_over = bool(record.get("game_over_delta"))
            progress = str(record.get("progress") or "")
            if not game_over and progress == "POSITIVE":
                continue
            failures.append({
                "action_id": record.get("action_id"),
                "source": record.get("source"),
                "hypothesis_id": record.get("hypothesis_id"),
                "semantic_hypothesis": record.get("hypothesis_claim"),
                "coordinate_candidate_id": record.get("coordinate_candidate_id"),
                "coordinate_xy": record.get("coordinate_xy"),
                "clicked_cell_before": record.get("coordinate_cell_before"),
                "clicked_cell_after": record.get("coordinate_cell_after"),
                "changed_cell_count": record.get("changed_cell_count"),
                "changed_color_transitions": (
                    (record.get("changed_color_delta") or {}).get("transitions")
                    if isinstance(record.get("changed_color_delta"), dict)
                    else None
                ),
                "visible_effect_observed": bool(int(record.get("changed_cell_count") or 0) > 0),
                "target_cell_changed": _target_cell_changed(record),
                "observed_motion_vectors": record.get("observed_motion_vectors") or [],
                "action_surface_added": record.get("planning_action_surface_added") or record.get("action_surface_added") or [],
                "action_surface_removed": record.get("planning_action_surface_removed") or record.get("action_surface_removed") or [],
                "reason_code": record.get("reason_code"),
                "verifier_effect_outcome": record.get("effect_outcome"),
                "progress": record.get("progress"),
                "game_over": game_over,
            })
        executed_hypotheses = []
        hypothesis_ids = list(dict.fromkeys(
            str(record.get("hypothesis_id"))
            for record in goal_records
            if record.get("hypothesis_id")
        ))
        for hypothesis_id in hypothesis_ids:
            hypothesis_records = [record for record in goal_records if str(record.get("hypothesis_id") or "") == hypothesis_id]
            claim = next((str(record.get("hypothesis_claim")) for record in hypothesis_records if record.get("hypothesis_claim")), None)
            game_over = any(bool(record.get("game_over_delta")) for record in hypothesis_records)
            level_progress = any(
                bool(record.get("terminal_delta") or record.get("levels_completed_delta") or record.get("level_index_delta"))
                for record in hypothesis_records
            )
            if game_over:
                outcome = "GAME_OVER_DURING_EXECUTION"
            elif level_progress:
                outcome = "LEVEL_PROGRESS_OBSERVED"
            else:
                outcome = "EXHAUSTED_WITHOUT_LEVEL_PROGRESS"
            executed_hypotheses.append({
                "hypothesis_id": hypothesis_id,
                "source": hypothesis_records[0].get("source") if hypothesis_records else None,
                "semantic_hypothesis": claim,
                "action_runs": _compact_attempt_action_runs(hypothesis_records),
                "observed_steps": [_attempt_observed_step(record) for record in hypothesis_records],
                "executed_step_count": len(hypothesis_records),
                "outcome": outcome,
                "last_transition_reason": hypothesis_records[-1].get("reason_code") if hypothesis_records else None,
                "retry_directive": "DO_NOT_REPEAT_THIS_COMPLETE_TRAJECTORY_UNCHANGED" if not level_progress else None,
            })
        item = {
            "level_index": level_index,
            "attempt_index": attempt_index,
            "outcome": "FAILED_RESET_REQUESTED",
            "reset_trigger": reset_trigger,
            "step_index": step_index,
            "qwen_calls": qwen_calls,
            "attempt_start_state_signature": records[0].get("grid_hash_before") if records else None,
            "goal_entry_state_signature": goal_records[0].get("grid_hash_before") if goal_records else None,
            "action_runs": _compact_attempt_action_runs(goal_records),
            "attempt_total_execution": _attempt_total_execution(records),
            "research_phase": {
                "status": "EVIDENCE_COLLECTION_NOT_GOAL_EXECUTION",
                "action_runs": _compact_attempt_action_runs(research_records),
                "observed_steps": [_attempt_observed_step(record) for record in research_records],
            },
            "goal_execution": _attempt_total_execution(goal_records),
            "executed_hypotheses": executed_hypotheses,
            "execution_failures": failures[-12:],
            "verifier_feedback": verifier_feedback,
        }
        self.level_attempt_records.append(item)
        self._compact_retry_history(level_index)
        return item

    def _compact_retry_history(self, active_level_index: int) -> None:
        """Bound retry-only evidence without imposing a retry limit.

        The agent may reset a level until the game wall-clock deadline. Keep
        enough raw evidence for the packet and verifier, while retaining the
        merged mechanics/invariants that summarize older transitions.
        """
        original_action_count = len(self.action_memory_records)
        if original_action_count > self._MAX_RAW_ACTION_RECORDS:
            removed = original_action_count - self._MAX_RAW_ACTION_RECORDS
            self.action_memory_records = self.action_memory_records[-self._MAX_RAW_ACTION_RECORDS :]
            for level_index, offset in list(self._attempt_action_offsets_by_level.items()):
                self._attempt_action_offsets_by_level[level_index] = max(0, int(offset) - removed)

        if len(self.events) > self._MAX_RAW_EVENTS:
            self.events = self.events[-self._MAX_RAW_EVENTS :]
        retained_event_ids = {event.event_id for event in self.events}
        self.failed_events = [
            event for event in self.failed_events
            if event.event_id in retained_event_ids
        ][-self._MAX_RAW_EVENTS :]
        self.irrelevant_events = [
            event for event in self.irrelevant_events
            if event.event_id in retained_event_ids
        ][-self._MAX_RAW_EVENTS :]
        self.action_surface_memory_records = self.action_surface_memory_records[-self._MAX_ACTION_SURFACE_RECORDS :]
        self.object_applicability_memory = self.object_applicability_memory[-self._MAX_OBJECT_APPLICABILITY_RECORDS :]
        self.trajectory_evaluations = self.trajectory_evaluations[-self._MAX_TRAJECTORY_EVALUATIONS :]

        kept_reversed: list[dict[str, Any]] = []
        retained_by_level: dict[int, int] = {}
        for record in reversed(self.level_attempt_records):
            record_level = _safe_int(record.get("level_index"), -1)
            limit = (
                self._MAX_FAILED_ATTEMPTS_ACTIVE_LEVEL
                if record_level == int(active_level_index)
                else self._MAX_FAILED_ATTEMPTS_OTHER_LEVEL
            )
            retained = retained_by_level.get(record_level, 0)
            if retained >= limit:
                continue
            retained_by_level[record_level] = retained + 1
            kept_reversed.append(record)
        self.level_attempt_records = list(reversed(kept_reversed))

    # ------------------------- persistent object identity -------------------------
    def assign_object_tracks(self, level_index: int, step_index: int, objects: tuple[ObjectRecord, ...]) -> tuple[ObjectRecord, ...]:
        tracks = self._tracks_by_level.setdefault(level_index, {})
        unmatched_tracks = set(tracks)
        assignments: dict[int, str] = {}
        pair_scores: list[tuple[float, str, int]] = []

        for idx, obj in enumerate(objects):
            for track_id, track in tracks.items():
                score = _track_match_score(obj, track)
                if score is not None:
                    pair_scores.append((score, track_id, idx))
        for score, track_id, idx in sorted(pair_scores, key=lambda item: (item[0], item[1], item[2])):
            if idx in assignments or track_id not in unmatched_tracks:
                continue
            assignments[idx] = track_id
            unmatched_tracks.remove(track_id)

        result: list[ObjectRecord] = []
        for idx, obj in enumerate(objects):
            track_id = assignments.get(idx)
            if track_id is None:
                sequence = self._next_track_index_by_level.get(level_index, 0)
                self._next_track_index_by_level[level_index] = sequence + 1
                track_id = stable_hash((self._current_game_id, level_index, obj.shape_signature, tuple(obj.colors), sequence), "trk_")
            frame_id = obj.frame_object_id or obj.object_id
            tracked = replace(obj, object_id=track_id, track_id=track_id, frame_object_id=frame_id)
            result.append(tracked)
            tracks[track_id] = {
                "shape_signature": obj.shape_signature,
                "colors": tuple(obj.colors),
                "area": obj.area,
                "centroid_rc": tuple(obj.centroid_rc),
                "bbox_rc": tuple(obj.bbox_rc),
                "topology_signature": obj.topology_signature,
                "last_step": step_index,
            }
        return tuple(result)

    # ----------------------------- attempt accounting -----------------------------
    def has_attempted_simple_action(self, level_index: int, action_id: str) -> bool:
        return self.simple_action_attempts_by_level.get(level_index, {}).get(action_id, 0) > 0

    def simple_action_attempt_count(self, level_index: int, action_id: str) -> int:
        return self.simple_action_attempts_by_level.get(level_index, {}).get(action_id, 0)

    def mark_simple_action_attempted(self, level_index: int, action_id: str) -> None:
        bucket = self.simple_action_attempts_by_level.setdefault(level_index, {})
        bucket[action_id] = bucket.get(action_id, 0) + 1

    def required_action_research_ids(self, snapshot: Any) -> list[str]:
        """Return the gameplay actions exposed by the current official frame."""
        return list(dict.fromkeys(
            str(action_id)
            for action_id in (getattr(snapshot, "available_actions", ()) or ())
            if str(action_id).upper() not in {"RESET", "RESTART"}
        ))

    def action_research_status(self, snapshot: Any) -> dict[str, list[str]]:
        required = self.required_action_research_ids(snapshot)
        coordinate = set(str(action_id) for action_id in (getattr(snapshot, "coordinate_action_ids", ()) or ()))
        undo = set(str(action_id) for action_id in (getattr(snapshot, "undo_action_ids", ()) or ()))
        observed = {
            str(record.action_id)
            for record in self.action_effects.values()
            if float(getattr(record, "confidence", 0.0) or 0.0) >= 0.45
        }
        # Undo semantics are supplied by the environment contract. Executing undo as
        # an exploratory action would spend a move and destroy the probe chronology.
        researched_set = observed | undo
        researched = [action_id for action_id in required if action_id in researched_set]
        missing = [action_id for action_id in required if action_id not in researched_set]
        return {
            "required_action_ids": required,
            "researched_action_ids": researched,
            "missing_action_ids": missing,
            "missing_simple_action_ids": [action_id for action_id in missing if action_id not in coordinate],
            "missing_coordinate_action_ids": [action_id for action_id in missing if action_id in coordinate],
            "intrinsically_known_undo_action_ids": [action_id for action_id in required if action_id in undo],
        }

    def unprobed_action_effect_ids(self, snapshot: Any, config: V8Config) -> list[str]:
        del config  # The official action surface is already bounded; every missing action is mandatory.
        return self.action_research_status(snapshot)["missing_simple_action_ids"]

    def researched_simple_action_ids(self, snapshot: Any) -> list[str]:
        status = self.action_research_status(snapshot)
        coordinate = set(str(action_id) for action_id in (getattr(snapshot, "coordinate_action_ids", ()) or ()))
        undo = set(str(action_id) for action_id in (getattr(snapshot, "undo_action_ids", ()) or ()))
        return [
            action_id
            for action_id in status["researched_action_ids"]
            if action_id not in coordinate
            and action_id not in undo
            and action_id.upper() not in {"RESET", "RESTART"}
        ]

    def action_effect_probe_complete(self, snapshot: Any, config: V8Config) -> bool:
        del config
        return not self.action_research_status(snapshot)["missing_action_ids"]

    def coordinate_action_research_needed(self, snapshot: Any) -> bool:
        return bool(self.action_research_status(snapshot)["missing_coordinate_action_ids"])

    @staticmethod
    def state_scoped_action_signature(suppression_signature: str, state_signature: str | None) -> str:
        if not state_signature:
            return suppression_signature
        return stable_hash((suppression_signature, state_signature), "attempt_")

    def mark_emitted_action(
        self,
        level_index: int,
        action_id: str,
        suppression_signature: str,
        *,
        state_signature: str | None,
        is_coordinate: bool,
        is_coordinate_research: bool = False,
        coordinate_candidate_id: str | None = None,
        coordinate_x: int | None = None,
        coordinate_y: int | None = None,
    ) -> None:
        scoped = self.state_scoped_action_signature(suppression_signature, state_signature)
        self.action_attempts_by_signature[scoped] = self.action_attempts_by_signature.get(scoped, 0) + 1
        if is_coordinate and is_coordinate_research:
            self.coordinate_probe_counts_by_level[level_index] = self.coordinate_probe_counts_by_level.get(level_index, 0) + 1
            self.coordinate_probe_signature_counts[scoped] = self.coordinate_probe_signature_counts.get(scoped, 0) + 1
            self.coordinate_candidates_clicked_this_attempt.update(
                self.coordinate_attempt_keys(action_id, coordinate_candidate_id, coordinate_x, coordinate_y)
            )
        elif not is_coordinate:
            self.mark_simple_action_attempted(level_index, action_id)

    def action_attempt_count(self, suppression_signature: str, state_signature: str | None = None) -> int:
        scoped = self.state_scoped_action_signature(suppression_signature, state_signature)
        return self.action_attempts_by_signature.get(scoped, 0)

    def mark_coordinate_probe(self, level_index: int, signature: str) -> None:
        # Compatibility API. New code accounts centrally in GameSession.
        self.coordinate_probe_counts_by_level[level_index] = self.coordinate_probe_counts_by_level.get(level_index, 0) + 1
        self.coordinate_probe_signature_counts[signature] = self.coordinate_probe_signature_counts.get(signature, 0) + 1

    def coordinate_probe_count(self, level_index: int) -> int:
        return self.coordinate_probe_counts_by_level.get(level_index, 0)

    def coordinate_signature_count(self, signature: str) -> int:
        return self.coordinate_probe_signature_counts.get(signature, 0)

    @staticmethod
    def coordinate_attempt_key(action_id: str, candidate_id: str | None, x: int | None, y: int | None) -> str:
        target = f"candidate:{candidate_id}" if candidate_id else f"xy:{x},{y}"
        return f"{action_id}:{target}"

    @staticmethod
    def coordinate_attempt_keys(action_id: str, candidate_id: str | None, x: int | None, y: int | None) -> set[str]:
        keys = {f"{action_id}:xy:{x},{y}"}
        if candidate_id:
            keys.add(f"{action_id}:candidate:{candidate_id}")
        return keys

    def coordinate_candidate_clicked_this_attempt(
        self,
        action_id: str,
        candidate_id: str | None,
        x: int | None,
        y: int | None,
    ) -> bool:
        return bool(
            self.coordinate_attempt_keys(action_id, candidate_id, x, y)
            & self.coordinate_candidates_clicked_this_attempt
        )

    # ---------------------------- semantic questions ----------------------------
    def ensure_question(self, question_id: str, question_type: SemanticQuestionType, target_signature: str) -> SemanticQuestion:
        question = self.semantic_questions.get(question_id)
        if question is None:
            question = SemanticQuestion(
                question_id=question_id,
                question_type=question_type,
                domain=default_question_domain(question_type),
                target_signature=target_signature,
            )
            self.semantic_questions[question_id] = question
        return question

    def resolve_question(self, question_id: str | None, question_type: SemanticQuestionType | None, target_signature: str | None, outcome: str) -> float:
        if not question_id or question_type is None:
            return 0.0
        question = self.ensure_question(question_id, question_type, target_signature or "unknown")
        gain, updated = resolve_question(question, outcome)
        self.semantic_questions[question_id] = updated
        return gain

    # -------------------------------- judgments --------------------------------
    def add_judgment(self, judgment: Judgment) -> None:
        idx = self._next_event_sequence
        self._next_event_sequence += 1
        event = MemoryEvent(
            event_id=f"evt_{idx:05d}",
            level_index=_safe_int(judgment.observed_delta.get("level_index"), 0),
            step_index=_safe_int(judgment.observed_delta.get("step_index"), idx),
            event_type="transition_judgment",
            before_hash=judgment.before_hash,
            action=judgment.action.to_arc_action(),
            after_hash=judgment.after_hash,
            hypothesis_id=judgment.hypothesis_id,
            truth=judgment.truth,
            relevance=judgment.relevance,
            validity=judgment.validity,
            progress=judgment.progress,
            attribution=judgment.attribution,
            reason_code=judgment.reason_code,
            summary=_summary(judgment),
            contract_kind=judgment.contract_kind.value if judgment.contract_kind else None,
            information_gain=judgment.observed_information_gain,
        )
        self.events.append(event)
        if judgment.validity is Validity.INVALID or judgment.truth is TriTruth.FALSE or judgment.progress is Progress.NEGATIVE:
            self.failed_events.append(event)
        elif judgment.relevance is Relevance.IRRELEVANT:
            self.irrelevant_events.append(event)

        center = judgment.observed_delta.get("changed_center_xy")
        if isinstance(center, (tuple, list)) and len(center) == 2:
            self._last_changed_center = (int(center[0]), int(center[1]))

        action = judgment.action
        target_signature = None
        if action.verification_contract is not None:
            target_signature = action.verification_contract.target_signature
        outcome = _effect_outcome(judgment)
        effect_key = (action.action_id, target_signature)
        self.action_effects[effect_key] = merge_effect_record(
            self.action_effects.get(effect_key),
            action_id=action.action_id,
            target_signature=target_signature,
            outcome=outcome,
            level_index=event.level_index,
            step_index=event.step_index,
        )
        applicability = self._object_applicability_from_judgment(event, judgment, target_signature, outcome)
        if applicability is not None:
            self.object_applicability_memory.append(applicability)
        action_memory = self._action_memory_from_judgment(event, judgment, outcome)
        if action_memory is not None:
            self.action_memory_records.append(action_memory)
        surface_memory = self._action_surface_memory_from_judgment(event, judgment)
        if surface_memory is not None:
            self.action_surface_memory_records.append(surface_memory)

        if action.x is not None and action.y is not None:
            sig = self.coordinate_repeat_signature(action.action_id, action.coordinate_candidate_id, action.x, action.y, judgment.before_hash)
            self.coordinate_effects.append(CoordinateEffectRecord(
                coordinate_action_id=action.action_id,
                candidate_target_id=action.coordinate_candidate_id,
                x=action.x,
                y=action.y,
                level_index=event.level_index,
                step_index=event.step_index,
                object_id=(action.verification_contract.target_object_ids[0] if action.verification_contract and action.verification_contract.target_object_ids else None),
                region_signature=str(action.coordinate_candidate_id or f"xy_{action.x}_{action.y}"),
                observed_effect=judgment.reason_code,
                truth=judgment.truth,
                relevance=judgment.relevance,
                progress=judgment.progress,
                attribution=judgment.attribution,
                state_signature=judgment.before_hash,
                repeat_suppression_signature=sig,
            ))
        if judgment.truth is TriTruth.FALSE and judgment.progress is Progress.NEGATIVE:
            self.confirmed_rule_contradicted = True
        for observation in derive_invariant_observations(judgment):
            self._record_invariant_observation(observation)

    def record_semantic_binding(self, binding: SemanticBindingResult | None, snapshot: Any | None = None) -> None:
        if binding is None:
            return
        self.semantic_bindings[binding.binding_id] = binding
        if snapshot is None:
            return
        objects = {str(item.object_id): item for item in getattr(snapshot, "objects", ())}
        relations = {str(item.relation_id): item for item in getattr(snapshot, "relations", ())}
        object_ids = _dedupe_strings((
            *binding.source_object_ids,
            *binding.reference_object_ids,
            *binding.inferred_object_ids,
        ))
        self.semantic_binding_contexts[binding.binding_id] = {
            "state_signature": binding.state_signature,
            "objects": {
                object_id: _binding_object_fingerprint(objects[object_id])
                for object_id in object_ids
                if object_id in objects
            },
            "relations": {
                relation_id: _binding_relation_fingerprint(relations[relation_id])
                for relation_id in _dedupe_strings((*binding.relation_ids, *binding.inferred_relation_ids))
                if relation_id in relations
            },
        }

    def resolve_semantic_binding_references(self, binding: SemanticBindingResult, snapshot: Any) -> dict[str, Any]:
        """Resolve attempt-local tracked IDs onto the current level snapshot."""
        current_objects = {str(item.object_id): item for item in getattr(snapshot, "objects", ())}
        current_relations = {str(item.relation_id): item for item in getattr(snapshot, "relations", ())}
        context = self.semantic_binding_contexts.get(binding.binding_id, {})
        fingerprints = context.get("objects", {}) if isinstance(context, dict) else {}
        ordered_object_ids = _dedupe_strings((
            *binding.source_object_ids,
            *binding.reference_object_ids,
            *binding.inferred_object_ids,
        ))
        object_map: dict[str, str] = {}
        used_current_ids: set[str] = set()
        for object_id in ordered_object_ids:
            if object_id in current_objects:
                object_map[object_id] = object_id
                used_current_ids.add(object_id)
        for object_id in ordered_object_ids:
            if object_id in object_map:
                continue
            fingerprint = fingerprints.get(object_id) if isinstance(fingerprints, dict) else None
            if not isinstance(fingerprint, dict):
                continue
            candidates = [
                item for current_id, item in current_objects.items()
                if current_id not in used_current_ids
            ]
            ranked = sorted(
                (
                    (_binding_object_match_score(fingerprint, item), str(item.object_id))
                    for item in candidates
                ),
                key=lambda item: (-item[0], item[1]),
            )
            if ranked and ranked[0][0] >= 70.0:
                object_map[object_id] = ranked[0][1]
                used_current_ids.add(ranked[0][1])

        relation_contexts = context.get("relations", {}) if isinstance(context, dict) else {}
        relation_map: dict[str, str] = {}
        for relation_id in _dedupe_strings((*binding.relation_ids, *binding.inferred_relation_ids)):
            if relation_id in current_relations:
                relation_map[relation_id] = relation_id
                continue
            fingerprint = relation_contexts.get(relation_id) if isinstance(relation_contexts, dict) else None
            if not isinstance(fingerprint, dict):
                continue
            mapped_a = object_map.get(str(fingerprint.get("a") or ""))
            mapped_b = object_map.get(str(fingerprint.get("b") or ""))
            if not mapped_a or not mapped_b:
                continue
            endpoints = {mapped_a, mapped_b}
            matches = [
                item for item in current_relations.values()
                if str(item.relation_type) == str(fingerprint.get("relation_type") or "")
                and {str(item.a), str(item.b)} == endpoints
                and str(item.metric_name or "") == str(fingerprint.get("metric_name") or "")
            ]
            if matches:
                relation_map[relation_id] = sorted(matches, key=lambda item: str(item.relation_id))[0].relation_id

        source_ids = tuple(object_map[item] for item in binding.source_object_ids if item in object_map)
        reference_ids = tuple(object_map[item] for item in binding.reference_object_ids if item in object_map)
        inferred_ids = tuple(object_map[item] for item in binding.inferred_object_ids if item in object_map)
        relation_ids = tuple(relation_map[item] for item in binding.relation_ids if item in relation_map)
        complete = (
            len(source_ids) == len(binding.source_object_ids)
            and len(reference_ids) == len(binding.reference_object_ids)
            and len(relation_ids) == len(binding.relation_ids)
        )
        return {
            "source_object_ids": source_ids,
            "reference_object_ids": reference_ids,
            "inferred_object_ids": inferred_ids,
            "relation_ids": relation_ids,
            "object_map": object_map,
            "relation_map": relation_map,
            "entity_references_still_current": complete,
            "rebound_after_reset": any(old != new for old, new in object_map.items()) or any(
                old != new for old, new in relation_map.items()
            ),
            "unresolved_object_count": len(ordered_object_ids) - len(object_map),
            "unresolved_relation_count": len(binding.relation_ids) - len(relation_ids),
        }

    def semantic_object_rebindings_for_snapshot(self, snapshot: Any) -> dict[str, str]:
        level_index = int(getattr(snapshot, "level_index", 0))
        out: dict[str, str] = {}
        for binding in self.semantic_bindings.values():
            if binding.level_index != level_index:
                continue
            resolved = self.resolve_semantic_binding_references(binding, snapshot)
            out.update({str(old): str(new) for old, new in resolved["object_map"].items()})
        return out

    def add_trajectory_evaluation(self, evaluation: TrajectoryEvaluation | None) -> None:
        if evaluation is None:
            return
        if any(item.evaluation_id == evaluation.evaluation_id for item in self.trajectory_evaluations):
            return
        self.trajectory_evaluations.append(evaluation)

    def _record_invariant_observation(self, observation: dict[str, Any]) -> None:
        base_key = str(observation.get("base_key") or "")
        observation_key = str(observation.get("observation_key") or "")
        if not base_key or not observation_key:
            return
        record = self.semantic_invariants.setdefault(base_key, {
            "base_key": base_key,
            "predicate": observation.get("predicate"),
            "action_id": observation.get("action_id"),
            "subject_object_ids": list(observation.get("subject_object_ids") or []),
            "control_context": observation.get("control_context"),
            "authority": EvidenceAuthority.OFFICIAL_OBSERVATION.value,
            "variants": {},
        })
        variants = record["variants"]
        variant = variants.setdefault(observation_key, {
            "observation_key": observation_key,
            "parameters": observation.get("parameters") or {},
            "count": 0,
            "evidence_refs": [],
        })
        variant["count"] = int(variant.get("count") or 0) + 1
        refs = variant.setdefault("evidence_refs", [])
        for ref in observation.get("evidence_refs") or []:
            if ref not in refs:
                refs.append(ref)
        variant["evidence_refs"] = refs[-8:]

    def semantic_feedback_for_qwen(self, snapshot: Any, config: V8Config) -> dict[str, Any]:
        valid_objects = {str(obj.object_id) for obj in getattr(snapshot, "objects", ())}
        valid_relations = {str(rel.relation_id) for rel in getattr(snapshot, "relations", ())}
        valid_actions = set(getattr(snapshot, "possible_actions", ()) or getattr(snapshot, "available_actions", ()))
        level_index = int(getattr(snapshot, "level_index", 0))

        bindings = []
        for binding in reversed(list(self.semantic_bindings.values())):
            if binding.level_index != level_index:
                continue
            resolved = self.resolve_semantic_binding_references(binding, snapshot)
            source_ids = list(resolved["source_object_ids"])
            reference_ids = list(resolved["reference_object_ids"])
            relation_ids = list(resolved["relation_ids"])
            metric = binding.metric_spec
            bindings.append({
                "binding_id": binding.binding_id,
                "hypothesis_id": binding.hypothesis_id,
                "status": binding.status.value,
                "objective_kind": binding.objective.kind,
                "goal_operator": binding.goal_operator.value,
                "source_object_ids": source_ids,
                "reference_object_ids": reference_ids,
                "relation_ids": relation_ids,
                "inferred_object_ids": list(resolved["inferred_object_ids"]),
                "metric": None if metric is None else {
                    "name": metric.name,
                    "direction": metric.direction.value,
                    "target_value": metric.target_value,
                    "baseline_value": binding.baseline_value,
                    "epsilon": metric.epsilon,
                },
                "reason_code": binding.reason_code,
                "entity_references_still_current": bool(resolved["entity_references_still_current"]),
                "rebound_after_reset": bool(resolved["rebound_after_reset"]),
                "unresolved_object_count": int(resolved["unresolved_object_count"]),
                "unresolved_relation_count": int(resolved["unresolved_relation_count"]),
                "authority": EvidenceAuthority.DETERMINISTIC_BINDER.value,
                "evidence_refs": _dedupe_strings(binding.evidence_refs)[:8],
            })
            if len(bindings) >= config.max_semantic_feedback_bindings:
                break

        trajectories = []
        for evaluation in reversed(self.trajectory_evaluations):
            if evaluation.level_index != level_index:
                continue
            binding = self.semantic_bindings.get(str(evaluation.binding_id or ""))
            resolved = self.resolve_semantic_binding_references(binding, snapshot) if binding is not None else None
            source_ids = (
                list(resolved["source_object_ids"])
                if resolved is not None
                else [item for item in evaluation.source_object_ids if item in valid_objects]
            )
            reference_ids = (
                list(resolved["reference_object_ids"])
                if resolved is not None
                else [item for item in evaluation.reference_object_ids if item in valid_objects]
            )
            relation_ids = (
                list(resolved["relation_ids"])
                if resolved is not None
                else [item for item in evaluation.relation_ids if item in valid_relations]
            )
            references_current = (
                bool(resolved["entity_references_still_current"])
                if resolved is not None
                else (
                    set(evaluation.source_object_ids).union(evaluation.reference_object_ids).issubset(valid_objects)
                    and set(evaluation.relation_ids).issubset(valid_relations)
                )
            )
            trajectories.append({
                "evaluation_id": evaluation.evaluation_id,
                "hypothesis_id": evaluation.hypothesis_id,
                "binding_id": evaluation.binding_id,
                "executed_action_runs": _compact_action_id_runs(evaluation.executed_action_ids),
                "executed_step_count": len(evaluation.executed_action_ids),
                "mechanic_result": evaluation.mechanic_result.value,
                "goal_progress": evaluation.goal_progress.value,
                "semantic_judgment": evaluation.semantic_judgment.value,
                "reason_code": evaluation.reason_code,
                "error_before": evaluation.error_before,
                "error_after": evaluation.error_after,
                "error_delta": evaluation.error_delta,
                "first_divergence_step": evaluation.first_divergence_step,
                "source_object_ids": source_ids,
                "reference_object_ids": reference_ids,
                "relation_ids": relation_ids,
                "entity_references_still_current": references_current,
                "rebound_after_reset": bool(resolved and resolved["rebound_after_reset"]),
                "authority": EvidenceAuthority.DETERMINISTIC_VERIFIER.value,
                "evidence_refs": _dedupe_strings(evaluation.evidence_refs)[:8],
            })
            if len(trajectories) >= config.max_semantic_feedback_trajectories:
                break

        invariants = []
        for record in reversed(list(self.semantic_invariants.values())):
            action_id = str(record.get("action_id") or "")
            subjects = [str(item) for item in record.get("subject_object_ids") or []]
            if action_id and action_id not in valid_actions:
                continue
            if subjects and not set(subjects).issubset(valid_objects):
                continue
            variants = sorted(
                (item for item in (record.get("variants") or {}).values() if isinstance(item, dict)),
                key=lambda item: (-int(item.get("count") or 0), str(item.get("observation_key") or "")),
            )
            if not variants:
                continue
            best = variants[0]
            if len(variants) > 1:
                status = EvidenceStatus.CONTRADICTED
            elif int(best.get("count") or 0) >= config.semantic_invariant_confirmation_count:
                status = EvidenceStatus.CONFIRMED
            else:
                status = EvidenceStatus.OBSERVED_ONCE
            invariants.append({
                "invariant_id": record.get("base_key"),
                "predicate": record.get("predicate"),
                "action_id": action_id,
                "subject_object_ids": subjects,
                "parameters": best.get("parameters") or {},
                "observation_count": int(best.get("count") or 0),
                "competing_variant_count": max(0, len(variants) - 1),
                "status": status.value,
                "authority": EvidenceAuthority.OFFICIAL_OBSERVATION.value,
                "control_context": record.get("control_context"),
                "evidence_refs": _dedupe_strings(best.get("evidence_refs") or [])[:8],
            })
            if len(invariants) >= config.max_semantic_feedback_invariants:
                break

        return {
            "schema_version": "v9.reverse_semantic_feedback",
            "authority_rule": "Only OFFICIAL_OBSERVATION and deterministic binder/verifier records are evidence. Qwen hypotheses remain proposals until observed.",
            "bindings": bindings,
            "trajectory_evaluations": trajectories,
            "invariants": invariants,
        }

    def coordinate_repeat_signature(self, action_id: str, candidate_id: str | None, x: int, y: int, state_signature: str) -> str:
        return stable_hash((action_id, candidate_id or f"xy:{x},{y}", state_signature), "coord_noeff_")

    def is_coordinate_no_effect_suppressed(self, action_id: str, candidate_id: str | None, x: int, y: int, state_signature: str) -> bool:
        sig = self.coordinate_repeat_signature(action_id, candidate_id, x, y, state_signature)
        return any(
            rec.repeat_suppression_signature == sig
            and rec.relevance is Relevance.IRRELEVANT
            and rec.progress is not Progress.POSITIVE
            for rec in self.coordinate_effects
        )

    def should_suppress_action(self, suppression_signature: str, state_signature: str) -> bool:
        if self.action_attempt_count(suppression_signature, state_signature) <= 0:
            return False
        for event in reversed(self.failed_events[-20:] + self.irrelevant_events[-20:]):
            if event.before_hash != state_signature:
                continue
            action_reasoning = ((event.action or {}).get("reasoning", {}) or {})
            if action_reasoning.get("verification_contract_id") and action_reasoning.get("verification_contract_id") in suppression_signature:
                return True
        return False

    def recent_progress_positive(self, window: int = 4) -> bool:
        return any(ev.progress is Progress.POSITIVE for ev in self.events[-window:])

    def recent_unknown_or_irrelevant_count(self, window: int = 6) -> int:
        return sum(
            1
            for ev in self.events[-window:]
            if ev.truth is TriTruth.UNKNOWN or ev.relevance is Relevance.IRRELEVANT
        )

    def coordinate_research_needed(self, level_index: int) -> bool:
        # Coordinate action semantics are game-scoped; only the per-level probe cap
        # is level-scoped. The memory object is reset between games.
        effects = [e for e in self.coordinate_effects if e.state_signature]
        if not effects:
            return True
        recent = effects[-8:]
        return not any(e.progress is Progress.POSITIVE for e in recent) and self.coordinate_probe_count(level_index) < 24

    def last_changed_region_center(self) -> tuple[int, int] | None:
        return self._last_changed_center

    def summarize_for_qwen(self, role: QwenRole, config: V8Config, snapshot: Any | None = None) -> dict[str, Any]:
        recent = self.events[-config.max_recent_transitions_in_packet :]
        failed = self.failed_events[-config.max_memory_notes_in_packet :]
        irrelevant = self.irrelevant_events[-config.max_memory_notes_in_packet :]
        feedback_events = _dedupe_events([
            *failed,
            *irrelevant,
            *(e for e in recent if e.progress is not Progress.POSITIVE),
        ])[-config.max_memory_notes_in_packet :]
        effects = sorted(self.action_effects.values(), key=lambda item: (-item.confidence, -item.last_step))[: config.max_memory_notes_in_packet]
        unresolved_questions = [
            {
                "question_id": q.question_id,
                "question_type": q.question_type.value,
                "target_signature": q.target_signature,
                "resolved_outcome": q.resolved_outcome,
                "observations": dict(q.observations),
            }
            for q in self.semantic_questions.values()
            if q.resolved_outcome is None
        ][: config.max_memory_notes_in_packet]
        return {
            "recent_transitions": [event_to_dict(e) for e in recent],
            "confirmed_hypotheses": [event_to_dict(e) for e in recent if e.truth is TriTruth.TRUE and e.progress is Progress.POSITIVE],
            "rejected_hypotheses": [event_to_dict(e) for e in failed],
            "unknown_hypotheses": [event_to_dict(e) for e in recent if e.truth is TriTruth.UNKNOWN],
            "irrelevant_attempts": [event_to_dict(e) for e in irrelevant],
            "do_not_repeat": [e.summary for e in (failed + irrelevant)[-config.max_memory_notes_in_packet :]],
            "previous_attempts_feedback": [event_to_feedback(e) for e in feedback_events],
            "prior_coordinate_probes": [coord_to_dict(e) for e in self.coordinate_effects[-config.max_memory_notes_in_packet :]],
            "coordinate_no_effect_memory": [coord_to_dict(e) for e in self.coordinate_effects if e.relevance is Relevance.IRRELEVANT][-config.max_memory_notes_in_packet :],
            "known_action_effects": [asdict(e) for e in effects],
            "action_memory": self.action_memory_for_qwen(config),
            "action_surface_memory": self.action_surface_memory_for_qwen(config),
            "object_applicability_memory": self.object_applicability_for_qwen(snapshot, config),
            "unresolved_semantic_questions": unresolved_questions,
        }

    def action_memory_for_qwen(self, config: V8Config) -> dict[str, Any]:
        records = self.action_memory_records[-config.max_action_memory_records_in_packet :]
        by_action: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_action.setdefault(str(record.get("action_id")), []).append(record)
        return {
            "purpose": "Compact action-effect table. Each record says what changed after executing an action from the official action surface; no historical frames are included here.",
            "probe_policy": "Initial unknown actions are tried once. A newly available action seen after a known control exists is researched as control-before, new-action, the same control-after. Coordinate actions require grounded coordinate candidates.",
            "records": records,
            "by_action": by_action,
        }

    def action_surface_memory_for_qwen(self, config: V8Config) -> dict[str, Any]:
        return {
            "purpose": "Chronological action-surface evidence. Compare action_id, grid/object delta, before/after current actions, and before/after planning actions to distinguish controllable-object switches from environment affordance changes.",
            "records": self.action_surface_memory_records[-config.max_action_memory_records_in_packet :],
        }

    def object_applicability_for_qwen(self, snapshot: Any | None, config: V8Config) -> dict[str, Any]:
        recent = self.object_applicability_memory[-config.max_memory_notes_in_packet :]
        current_candidates: list[dict[str, Any]] = []
        current_objects: list[dict[str, Any]] = []
        if snapshot is not None:
            objects = {getattr(o, "object_id", None): o for o in getattr(snapshot, "objects", ())}
            for obj in getattr(snapshot, "objects", ())[: config.max_objects_in_packet]:
                brief = object_record_brief(obj)
                brief["target_priority"] = _target_priority(obj, snapshot)
                current_objects.append(brief)
            for candidate in getattr(snapshot, "coordinate_targets", ())[: config.max_coordinate_candidates_in_packet]:
                obj = objects.get(getattr(candidate, "object_id", None))
                current_candidates.append({
                    "candidate_id": getattr(candidate, "candidate_id", None),
                    "object_id": getattr(candidate, "object_id", None),
                    "relation_id": getattr(candidate, "relation_id", None),
                    "source": getattr(candidate, "source", None),
                    "target_signature": getattr(candidate, "target_signature", None),
                    "region_signature": getattr(candidate, "region_signature", None),
                    "reason": getattr(candidate, "reason", None),
                    "object": object_record_brief(obj) if obj is not None else None,
                })
        return {
            "purpose": "Map actions to object traits and target candidates that previously produced effect/no_effect evidence. Prefer candidates with matching traits; avoid repeating no_effect on equivalent targets.",
            "recent_records": recent,
            "current_object_descriptors": current_objects,
            "current_candidate_descriptors": current_candidates,
        }

    def _object_applicability_from_judgment(self, event: MemoryEvent, judgment: Judgment, target_signature: str | None, outcome: str) -> dict[str, Any] | None:
        action = judgment.action
        contract = action.verification_contract
        objects_before = judgment.observed_delta.get("target_objects_before") or []
        objects_after = judgment.observed_delta.get("target_objects_after") or []
        if not objects_before and not action.coordinate_candidate_id and not target_signature:
            return None
        return {
            "event_id": event.event_id,
            "level_index": event.level_index,
            "level_index_before": judgment.observed_delta.get("level_index_before"),
            "step_index": event.step_index,
            "state_signature": event.before_hash,
            "grid_hash_before": judgment.observed_delta.get("grid_hash_before"),
            "grid_hash_after": judgment.observed_delta.get("grid_hash_after"),
            "action_id": action.action_id,
            "coordinate_candidate_id": action.coordinate_candidate_id,
            "target_signature": target_signature,
            "target_object_ids": list(contract.target_object_ids) if contract else [],
            "target_relation_ids": list(contract.target_relation_ids) if contract else [],
            "contract_kind": judgment.contract_kind.value if judgment.contract_kind else None,
            "outcome": outcome,
            "truth": judgment.truth.value,
            "relevance": judgment.relevance.value,
            "progress": judgment.progress.value,
            "reason_code": judgment.reason_code,
            "information_gain": judgment.observed_information_gain,
            "objects_before": objects_before,
            "objects_after": objects_after,
        }

    def _action_memory_from_judgment(self, event: MemoryEvent, judgment: Judgment, outcome: str) -> dict[str, Any] | None:
        action = judgment.action
        if action.action_id.upper() in {"RESET", "RESTART"}:
            return None
        delta = judgment.observed_delta
        object_deltas = _compact_object_deltas(delta.get("object_deltas") or [])
        return {
            "event_id": event.event_id,
            "level_index": event.level_index,
            "level_index_before": delta.get("level_index_before"),
            "step_index": event.step_index,
            "state_signature": event.before_hash,
            "grid_hash_before": delta.get("grid_hash_before"),
            "grid_hash_after": delta.get("grid_hash_after"),
            "action_id": action.action_id,
            "source": action.source,
            "hypothesis_id": action.hypothesis_id,
            "hypothesis_claim": str(action.reason)[:560] if action.hypothesis_id else None,
            "coordinate_candidate_id": action.coordinate_candidate_id,
            "coordinate_xy": [int(action.x), int(action.y)] if action.x is not None and action.y is not None else None,
            "coordinate_cell_before": delta.get("coordinate_cell_before"),
            "coordinate_cell_after": delta.get("coordinate_cell_after"),
            "contract_kind": judgment.contract_kind.value if judgment.contract_kind else None,
            "effect_outcome": outcome,
            "truth": judgment.truth.value,
            "relevance": judgment.relevance.value,
            "progress": judgment.progress.value,
            "mechanic_result": judgment.mechanic_result.value,
            "semantic_judgment": judgment.semantic_judgment.value,
            "semantic_binding_id": judgment.semantic_binding_id,
            "attribution": judgment.attribution.value,
            "reason_code": judgment.reason_code,
            "changed_cell_count": delta.get("changed_cell_count"),
            "changed_cells_xy": delta.get("changed_cells_xy"),
            "changed_cell_runs": delta.get("changed_cell_runs"),
            "changed_center_xy": delta.get("changed_center_xy"),
            "changed_bbox_rc": delta.get("changed_bbox_rc"),
            "changed_color_delta": delta.get("changed_color_delta"),
            "raw_visual_changes": delta.get("raw_visual_changes"),
            "action_effect_summary": _action_effect_summary(action.action_id, outcome, delta, object_deltas),
            "observed_motion_vectors": _motion_vectors(object_deltas),
            "object_deltas": object_deltas,
            "levels_completed_delta": delta.get("levels_completed_delta"),
            "level_index_delta": delta.get("level_index_delta"),
            "game_over_delta": delta.get("game_over_delta"),
            "terminal_delta": delta.get("terminal_delta"),
            "score_delta": judgment.score_delta,
            "available_actions_before": delta.get("available_actions_before"),
            "available_actions_after": delta.get("available_actions_after"),
            "planning_action_ids_before": delta.get("planning_action_ids_before"),
            "planning_action_ids_after": delta.get("planning_action_ids_after"),
            "undo_action_ids_before": delta.get("undo_action_ids_before"),
            "undo_action_ids_after": delta.get("undo_action_ids_after"),
            "action_surface_added": delta.get("action_surface_added"),
            "action_surface_removed": delta.get("action_surface_removed"),
            "planning_action_surface_added": delta.get("planning_action_surface_added"),
            "planning_action_surface_removed": delta.get("planning_action_surface_removed"),
            "affected_object_ids": list(judgment.affected_objects),
            "affected_relation_ids": list(judgment.affected_relations),
            "target_object_ids": delta.get("target_object_ids", []),
            "target_relation_ids": delta.get("target_relation_ids", []),
            "target_signature": delta.get("target_signature"),
            "target_coordinate_candidate_id": delta.get("target_coordinate_candidate_id"),
        }

    def _action_surface_memory_from_judgment(self, event: MemoryEvent, judgment: Judgment) -> dict[str, Any] | None:
        delta = judgment.observed_delta
        before = tuple(str(v) for v in (delta.get("available_actions_before") or ()))
        after = tuple(str(v) for v in (delta.get("available_actions_after") or ()))
        planning_before = tuple(str(v) for v in (delta.get("planning_action_ids_before") or ()))
        planning_after = tuple(str(v) for v in (delta.get("planning_action_ids_after") or ()))
        if not before and not after and not planning_before and not planning_after:
            return None
        return {
            "event_id": event.event_id,
            "level_index": event.level_index,
            "level_index_before": judgment.observed_delta.get("level_index_before"),
            "step_index": event.step_index,
            "state_signature": event.before_hash,
            "action_id": judgment.action.action_id,
            "source": judgment.action.source,
            "hypothesis_id": judgment.action.hypothesis_id,
            "reason_code": judgment.reason_code,
            "progress": judgment.progress.value,
            "changed_cell_count": delta.get("changed_cell_count"),
            "changed_bbox_rc": delta.get("changed_bbox_rc"),
            "changed_center_xy": delta.get("changed_center_xy"),
            "raw_visual_changes": delta.get("raw_visual_changes"),
            "affected_object_ids": list(judgment.affected_objects),
            "object_deltas": _compact_object_deltas(delta.get("object_deltas") or []),
            "available_actions_before": list(before),
            "available_actions_after": list(after),
            "planning_action_ids_before": list(planning_before),
            "planning_action_ids_after": list(planning_after),
            "undo_action_ids_before": delta.get("undo_action_ids_before"),
            "undo_action_ids_after": delta.get("undo_action_ids_after"),
            "action_surface_added": delta.get("action_surface_added"),
            "action_surface_removed": delta.get("action_surface_removed"),
            "planning_action_surface_added": delta.get("planning_action_surface_added"),
            "planning_action_surface_removed": delta.get("planning_action_surface_removed"),
        }


def summarize_for_qwen(memory: GameMemory, snapshot: Any, role: QwenRole, config: V8Config) -> dict[str, Any]:
    return memory.summarize_for_qwen(role, config, snapshot)


def _compact_attempt_action_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for record in records[-64:]:
        action_id = str(record.get("action_id") or "")
        if not action_id:
            continue
        source = str(record.get("source") or "unknown")
        hypothesis_id = record.get("hypothesis_id")
        coordinate_candidate_id = record.get("coordinate_candidate_id")
        coordinate_xy = record.get("coordinate_xy")
        if (
            runs
            and coordinate_candidate_id is None
            and runs[-1]["action_id"] == action_id
            and runs[-1]["source"] == source
            and runs[-1].get("hypothesis_id") == hypothesis_id
            and runs[-1].get("coordinate_candidate_id") is None
        ):
            runs[-1]["count"] += 1
        else:
            item = {
                "action_id": action_id,
                "count": 1,
                "source": source,
                "hypothesis_id": hypothesis_id,
            }
            if coordinate_candidate_id is not None:
                item["coordinate_candidate_id"] = coordinate_candidate_id
            if coordinate_xy is not None:
                item["coordinate_xy"] = coordinate_xy
            runs.append(item)
    return runs


def _attempt_total_execution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe one continuous environment trajectory without phase boundaries."""
    ordered_trajectory: list[dict[str, Any]] = []
    aggregate_transitions: dict[str, int] = {}
    coordinate_cell_transitions: dict[str, int] = {}
    visible_effect_steps = 0
    target_cell_change_steps = 0
    coordinate_steps = 0
    total_changed_cell_events = 0
    level_progress_observed = False
    game_over_observed = False

    for ordinal, record in enumerate(records, start=1):
        action_id = str(record.get("action_id") or "")
        if not action_id:
            continue
        changed_cell_count = int(record.get("changed_cell_count") or 0)
        target_cell_changed = _target_cell_changed(record)
        step_level_progress = bool(
            record.get("terminal_delta")
            or record.get("levels_completed_delta")
            or record.get("level_index_delta")
        )
        step_game_over = bool(record.get("game_over_delta"))
        before = record.get("coordinate_cell_before")
        after = record.get("coordinate_cell_after")
        candidate_id = record.get("coordinate_candidate_id")

        step = {
            "ordinal": ordinal,
            "step_index": record.get("step_index"),
            "action_id": action_id,
            "coordinate_candidate_id": candidate_id,
            "coordinate_xy": record.get("coordinate_xy"),
            "clicked_cell_before": before,
            "clicked_cell_after": after,
            "visible_effect_observed": bool(changed_cell_count > 0),
            "target_cell_changed": target_cell_changed,
            "observed_effect_class": _observed_effect_class(
                visible_effect_observed=changed_cell_count > 0,
                target_cell_changed=target_cell_changed,
                level_progress_observed=step_level_progress,
                game_over_observed=step_game_over,
            ),
            "level_progress_observed": step_level_progress,
            "game_over_observed": step_game_over,
        }
        ordered_trajectory.append({
            key: value
            for key, value in step.items()
            if value not in (None, [], {}) or isinstance(value, bool)
        })

        visible_effect_steps += int(changed_cell_count > 0)
        total_changed_cell_events += changed_cell_count
        level_progress_observed = level_progress_observed or step_level_progress
        game_over_observed = game_over_observed or step_game_over
        if candidate_id is not None:
            coordinate_steps += 1
            target_cell_change_steps += int(target_cell_changed)
            if before is not None and after is not None:
                transition = f"{before}->{after}"
                coordinate_cell_transitions[transition] = coordinate_cell_transitions.get(transition, 0) + 1

        color_delta = record.get("changed_color_delta")
        transitions = color_delta.get("transitions") if isinstance(color_delta, dict) else None
        if isinstance(transitions, dict):
            for transition, count in transitions.items():
                try:
                    numeric_count = int(count)
                except (TypeError, ValueError):
                    continue
                key = str(transition)
                aggregate_transitions[key] = aggregate_transitions.get(key, 0) + numeric_count

    if level_progress_observed:
        outcome = "LEVEL_PROGRESS_OBSERVED"
    elif game_over_observed:
        outcome = "GAME_OVER_WITHOUT_LEVEL_PROGRESS"
    else:
        outcome = "NO_LEVEL_PROGRESS_AFTER_FULL_TRAJECTORY"

    return {
        "scope": "ALL_ENVIRONMENT_ACTIONS_EXECUTED_FROM_ATTEMPT_ENTRY_UNTIL_FAILURE_RESET",
        "state_continuity": (
            "NO_RESET_BETWEEN_ORDERED_STEPS; research and goal labels are provenance only and do not isolate environment state"
        ),
        "ordered_trajectory": ordered_trajectory,
        "observed_result": {
            "executed_step_count": len(ordered_trajectory),
            "visible_effect_step_count": visible_effect_steps,
            "coordinate_target_step_count": coordinate_steps,
            "coordinate_target_changed_step_count": target_cell_change_steps,
            "all_coordinate_targets_changed": bool(coordinate_steps) and target_cell_change_steps == coordinate_steps,
            "coordinate_target_cell_transitions": coordinate_cell_transitions,
            "total_changed_cell_events": total_changed_cell_events,
            "aggregate_changed_color_transitions": aggregate_transitions,
            "level_progress_observed": level_progress_observed,
            "game_over_observed": game_over_observed,
        },
        "outcome": outcome,
        "exact_replay_forbidden": bool(ordered_trajectory) and not level_progress_observed,
    }


def _attempt_observed_step(record: dict[str, Any]) -> dict[str, Any]:
    color_delta = record.get("changed_color_delta") if isinstance(record.get("changed_color_delta"), dict) else {}
    changed_cell_count = int(record.get("changed_cell_count") or 0)
    target_cell_changed = _target_cell_changed(record)
    level_progress_observed = bool(
        record.get("terminal_delta")
        or record.get("levels_completed_delta")
        or record.get("level_index_delta")
    )
    object_changes = []
    for change in record.get("object_deltas") or []:
        if not isinstance(change, dict):
            continue
        item = {
            "object_id": change.get("object_id"),
            "motion_direction": change.get("motion_direction"),
            "delta_centroid_rc": change.get("delta_centroid_rc"),
            "before_colors": change.get("before_colors"),
            "after_colors": change.get("after_colors"),
            "shape_changed": change.get("shape_changed"),
            "palette_changed": change.get("palette_changed"),
            "area_delta": change.get("area_delta"),
        }
        object_changes.append({key: value for key, value in item.items() if value not in (None, [], {})})
    item = {
        "step_index": record.get("step_index"),
        "action_id": record.get("action_id"),
        "coordinate_candidate_id": record.get("coordinate_candidate_id"),
        "coordinate_xy": record.get("coordinate_xy"),
        "clicked_cell_before": record.get("coordinate_cell_before"),
        "clicked_cell_after": record.get("coordinate_cell_after"),
        "verifier_effect_outcome": record.get("effect_outcome"),
        "visible_effect_observed": bool(changed_cell_count > 0),
        "target_cell_changed": target_cell_changed,
        "observed_effect_class": _observed_effect_class(
            visible_effect_observed=changed_cell_count > 0,
            target_cell_changed=target_cell_changed,
            level_progress_observed=level_progress_observed,
            game_over_observed=bool(record.get("game_over_delta")),
        ),
        "changed_cell_count": changed_cell_count,
        "changed_color_transitions": color_delta.get("transitions") or {},
        "observed_motion_vectors": record.get("observed_motion_vectors") or [],
        "object_changes": object_changes[:8],
        "available_actions_before": record.get("available_actions_before") or [],
        "available_actions_after": record.get("available_actions_after") or [],
        "action_surface_added": record.get("planning_action_surface_added") or record.get("action_surface_added") or [],
        "action_surface_removed": record.get("planning_action_surface_removed") or record.get("action_surface_removed") or [],
        "reason_code": record.get("reason_code"),
        "progress": record.get("progress"),
        "level_progress_observed": level_progress_observed,
        "game_over_observed": bool(record.get("game_over_delta")),
    }
    return {key: value for key, value in item.items() if value not in (None, [], {}) or isinstance(value, bool)}


def _target_cell_changed(record: dict[str, Any]) -> bool:
    before = record.get("coordinate_cell_before")
    after = record.get("coordinate_cell_after")
    return before is not None and after is not None and before != after


def _observed_effect_class(
    *,
    visible_effect_observed: bool,
    target_cell_changed: bool,
    level_progress_observed: bool,
    game_over_observed: bool,
) -> str:
    if level_progress_observed:
        return "LEVEL_PROGRESS_OBSERVED"
    if game_over_observed:
        return "GAME_OVER_AFTER_ACTION"
    if target_cell_changed:
        return "TARGET_STATE_CHANGED_WITHOUT_LEVEL_PROGRESS"
    if visible_effect_observed:
        return "VISIBLE_CHANGE_WITHOUT_LEVEL_PROGRESS"
    return "NO_VISIBLE_CHANGE"


def _dedupe_events(events: list[MemoryEvent]) -> list[MemoryEvent]:
    seen: set[str] = set()
    out: list[MemoryEvent] = []
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        out.append(event)
    return sorted(out, key=lambda event: (event.level_index, event.step_index, event.event_id))


def _compact_object_deltas(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:12]:
        if not isinstance(item, dict):
            continue
        compact = {
            "object_id": item.get("object_id"),
            "lifecycle": item.get("lifecycle"),
            "before_bbox_rc": item.get("before_bbox_rc"),
            "after_bbox_rc": item.get("after_bbox_rc"),
            "before_centroid_rc": _round_number_list(item.get("before_centroid_rc"), 2),
            "after_centroid_rc": _round_number_list(item.get("after_centroid_rc"), 2),
            "delta_centroid_rc": _round_number_list(item.get("delta_centroid_rc"), 2),
            "motion_direction": item.get("motion_direction"),
            "before_colors": item.get("before_colors"),
            "after_colors": item.get("after_colors"),
            "before_area": item.get("before_area"),
            "after_area": item.get("after_area"),
            "before_shape_signature": item.get("before_shape_signature"),
            "after_shape_signature": item.get("after_shape_signature"),
            "shape_changed": item.get("shape_changed"),
            "palette_changed": item.get("palette_changed"),
            "area_delta": item.get("area_delta"),
            "changed_cells_in_before_bbox": item.get("changed_cells_in_before_bbox"),
            "changed_cells_in_after_bbox": item.get("changed_cells_in_after_bbox"),
            "tags": item.get("tags"),
        }
        out.append({key: value for key, value in compact.items() if value not in (None, [], {})})
    return out


def _motion_vectors(object_deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    for item in object_deltas:
        delta = item.get("delta_centroid_rc")
        direction = str(item.get("motion_direction") or "")
        if direction and direction != "stationary" and isinstance(delta, list):
            vectors.append({
                "object_id": item.get("object_id"),
                "delta_centroid_rc": delta,
                "direction": direction,
            })
    return vectors[:8]


def _action_effect_summary(action_id: str, outcome: str, delta: dict[str, Any], object_deltas: list[dict[str, Any]]) -> str:
    moving = _motion_vectors(object_deltas)
    if moving:
        parts = [
            f"{item.get('object_id')} {item.get('direction')} {item.get('delta_centroid_rc')}"
            for item in moving[:4]
        ]
        motion = "moves " + "; ".join(parts)
    elif object_deltas:
        touched = [str(item.get("object_id")) for item in object_deltas[:4] if item.get("object_id")]
        motion = "tracked objects stationary; local/color changes on " + ",".join(touched)
    else:
        motion = "no tracked object delta"
    surface_added = delta.get("planning_action_surface_added") or delta.get("action_surface_added") or []
    surface_removed = delta.get("planning_action_surface_removed") or delta.get("action_surface_removed") or []
    if surface_added or surface_removed:
        surface = f"surface +{surface_added} -{surface_removed}"
    else:
        surface = "surface unchanged"
    terminal = "terminal/level progress" if delta.get("terminal_delta") or delta.get("levels_completed_delta") else "no terminal progress"
    return f"{action_id}: {outcome}; {motion}; {surface}; {terminal}"


def _round_number_list(value: Any, digits: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(round(float(item), int(digits)))
        except (TypeError, ValueError):
            return None
    return out


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    for value in values or ():
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _compact_action_id_runs(action_ids: Any) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for value in action_ids or ():
        action_id = str(value or "")
        if not action_id:
            continue
        if runs and runs[-1]["action_id"] == action_id:
            runs[-1]["repeat"] += 1
        else:
            runs.append({"action_id": action_id, "repeat": 1})
    return runs


def _binding_object_fingerprint(obj: ObjectRecord) -> dict[str, Any]:
    return {
        "stable_hash": str(obj.stable_hash),
        "bbox_rc": list(obj.bbox_rc),
        "centroid_rc": [round(float(value), 3) for value in obj.centroid_rc],
        "area": int(obj.area),
        "colors": list(obj.colors),
        "shape_signature": str(obj.shape_signature),
        "local_mask_hex_rows": list(obj.local_mask_hex_rows),
        "topology_signature": str(obj.topology_signature),
    }


def _binding_relation_fingerprint(relation: Any) -> dict[str, Any]:
    return {
        "relation_type": str(relation.relation_type),
        "a": str(relation.a),
        "b": str(relation.b),
        "metric_name": str(relation.metric_name or ""),
    }


def _binding_object_match_score(fingerprint: dict[str, Any], obj: ObjectRecord) -> float:
    score = 0.0
    if str(fingerprint.get("stable_hash") or "") == str(obj.stable_hash):
        score += 120.0
    if tuple(fingerprint.get("bbox_rc") or ()) == tuple(obj.bbox_rc):
        score += 35.0
    old_centroid = fingerprint.get("centroid_rc")
    if isinstance(old_centroid, (list, tuple)) and len(old_centroid) == 2:
        distance = hypot(
            float(old_centroid[0]) - float(obj.centroid_rc[0]),
            float(old_centroid[1]) - float(obj.centroid_rc[1]),
        )
        score += max(0.0, 20.0 - distance)
    if str(fingerprint.get("shape_signature") or "") == str(obj.shape_signature):
        score += 30.0
    if tuple(fingerprint.get("local_mask_hex_rows") or ()) == tuple(obj.local_mask_hex_rows):
        score += 25.0
    if tuple(fingerprint.get("colors") or ()) == tuple(obj.colors):
        score += 20.0
    if int(fingerprint.get("area") or -1) == int(obj.area):
        score += 15.0
    old_topology = str(fingerprint.get("topology_signature") or "")
    if old_topology and old_topology == str(obj.topology_signature):
        score += 10.0
    return score


def event_to_dict(ev: MemoryEvent) -> dict[str, Any]:
    out = asdict(ev)
    for key in ("truth", "relevance", "validity", "progress", "attribution"):
        value = out.get(key)
        out[key] = value.value if hasattr(value, "value") else value
    return out


def event_to_feedback(ev: MemoryEvent) -> dict[str, Any]:
    return {
        "attempt_id": ev.event_id,
        "hypothesis_id": ev.hypothesis_id,
        "action": ev.action,
        "verifier_truth": ev.truth.value,
        "semantic_relevance": ev.relevance.value,
        "progress": ev.progress.value,
        "reason_code": ev.reason_code,
        "contract_kind": ev.contract_kind,
        "instruction": "Do not repeat an equivalent action from the same semantic state unless target, evidence, or action surface changes.",
    }


def coord_to_dict(rec: CoordinateEffectRecord) -> dict[str, Any]:
    out = asdict(rec)
    for key in ("truth", "relevance", "progress", "attribution"):
        value = out.get(key)
        out[key] = value.value if hasattr(value, "value") else value
    return out


def object_record_brief(obj: ObjectRecord) -> dict[str, Any]:
    return {
        "object_id": obj.object_id,
        "stable_hash": obj.stable_hash,
        "shape_signature": obj.shape_signature,
        "topology_signature": obj.topology_signature,
        "colors": list(obj.colors),
        "area": obj.area,
        "bbox_rc": list(obj.bbox_rc),
        "centroid_rc": list(obj.centroid_rc),
        "tags": list(obj.tags),
        "border_touching": list(obj.border_touching),
    }


def _target_priority(obj: ObjectRecord, snapshot: Any) -> str:
    tags = set(obj.tags)
    if "frame_like" not in tags:
        return "normal"
    borders = set(obj.border_touching)
    height = max(1, int(getattr(snapshot, "height", 0) or 1))
    width = max(1, int(getattr(snapshot, "width", 0) or 1))
    r0, c0, r1, c1 = obj.bbox_rc
    spans_grid = (r1 - r0 + 1) >= 0.80 * height and (c1 - c0 + 1) >= 0.80 * width
    touches_all_edges = {"top", "bottom", "left", "right"}.issubset(borders)
    bbox_is_global = r0 <= 0 and c0 <= 0 and r1 >= height - 1 and c1 >= width - 1
    if "border_touching" in tags and (touches_all_edges or bbox_is_global or spans_grid):
        return "deprioritize_global_frame_container"
    return "normal_internal_frame_like_object"


def _summary(j: Judgment) -> str:
    contract = j.contract_kind.value if j.contract_kind else "UNBOUND"
    return (
        f"{j.action.action_id} contract={contract}: truth={j.truth.value}, relevance={j.relevance.value}, "
        f"mechanic={j.mechanic_result.value}, progress={j.progress.value}, semantic={j.semantic_judgment.value}, "
        f"reason={j.reason_code}, ig={j.observed_information_gain:.3f}"
    )


def _effect_outcome(judgment: Judgment) -> str:
    if judgment.reason_code == "no_op_contradicted_by_action_effect":
        return "effect"
    if judgment.reason_code == "no_op_confirmed":
        return "no_effect"
    if judgment.reason_code == "target_object_not_displaced":
        return "not_moved"
    if judgment.reason_code in {"target_local_no_effect", "action_surface_unchanged"}:
        return "no_effect"
    if judgment.reason_code == "relation_error_unchanged":
        return "unchanged"
    if judgment.reason_code == "no_score_or_terminal_progress":
        return "no_progress"
    if judgment.mechanic_result.value == "MATCH":
        if judgment.attribution is Attribution.NO_VISIBLE_CHANGE:
            return "no_effect"
        return "effect"
    if judgment.mechanic_result.value == "MISMATCH":
        if judgment.attribution is Attribution.NO_VISIBLE_CHANGE:
            return "no_effect"
        return "negative_effect"
    if judgment.progress is Progress.NEGATIVE:
        return "negative_effect"
    if judgment.progress is Progress.POSITIVE or judgment.truth is TriTruth.TRUE:
        return "effect"
    if judgment.attribution is Attribution.NO_VISIBLE_CHANGE:
        return "no_effect"
    return "ambiguous"


def _track_match_score(obj: ObjectRecord, track: dict[str, Any]) -> float | None:
    track_colors = tuple(track.get("colors", ()))
    exact_shape = obj.shape_signature == track.get("shape_signature")
    shares_color = bool(set(obj.colors).intersection(track_colors))
    if not shares_color and not exact_shape:
        return None
    old_area = max(1, int(track.get("area", obj.area)))
    area_ratio = abs(obj.area - old_area) / old_area
    if not exact_shape and area_ratio > 0.55:
        return None
    old_centroid = track.get("centroid_rc", obj.centroid_rc)
    distance = hypot(obj.centroid_rc[0] - old_centroid[0], obj.centroid_rc[1] - old_centroid[1])
    color_penalty = 0.0 if tuple(obj.colors) == track_colors else (2.0 if shares_color else 3.0)
    shape_penalty = 0.0 if exact_shape else 4.0
    topology_penalty = 0.0 if obj.topology_signature == track.get("topology_signature") else 1.0
    return distance + 8.0 * area_ratio + color_penalty + shape_penalty + topology_penalty


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
