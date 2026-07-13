from __future__ import annotations

from dataclasses import asdict, replace
from math import hypot
from typing import Any

from .action_effects import default_question_domain, merge_effect_record, resolve_question
from .config import V8Config
from .observe import stable_hash
from .types import (
    ActionEffectRecord,
    Attribution,
    CoordinateEffectRecord,
    Judgment,
    MemoryEvent,
    ObjectRecord,
    Progress,
    QwenRole,
    Relevance,
    SemanticQuestion,
    SemanticQuestionType,
    TriTruth,
    Validity,
)


ACTION_EFFECT_PROBE_IDS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}


class GameMemory:
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

    def begin_level_attempt(self, level_index: int, attempt_index: int, *, retry: bool) -> None:
        self._current_attempt_by_level[level_index] = attempt_index
        self._attempt_action_offsets_by_level[level_index] = len(self.action_memory_records)
        self.coordinate_candidates_clicked_this_attempt.clear()
        if retry:
            # Mechanics remain game-scoped. Execution suppression is attempt-scoped.
            self.action_attempts_by_signature.clear()
            self.coordinate_probe_counts_by_level[level_index] = 0
            self.coordinate_probe_signature_counts.clear()

    def current_attempt_index(self, level_index: int) -> int:
        return self._current_attempt_by_level.get(level_index, 0)

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
        failures = []
        for record in records:
            game_over = bool(record.get("game_over_delta"))
            progress = str(record.get("progress") or "")
            if str(record.get("source") or "") == "initial_action_probe" and not game_over:
                continue
            if not game_over and progress == "POSITIVE":
                continue
            failures.append({
                "action_id": record.get("action_id"),
                "source": record.get("source"),
                "hypothesis_id": record.get("hypothesis_id"),
                "semantic_hypothesis": record.get("hypothesis_claim"),
                "reason_code": record.get("reason_code"),
                "effect_outcome": record.get("effect_outcome"),
                "progress": record.get("progress"),
                "game_over": game_over,
            })
        executed_hypotheses = []
        hypothesis_ids = list(dict.fromkeys(
            str(record.get("hypothesis_id"))
            for record in records
            if record.get("hypothesis_id")
        ))
        for hypothesis_id in hypothesis_ids:
            hypothesis_records = [record for record in records if str(record.get("hypothesis_id") or "") == hypothesis_id]
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
                "semantic_hypothesis": claim,
                "action_runs": _compact_attempt_action_runs(hypothesis_records),
                "executed_step_count": len(hypothesis_records),
                "outcome": outcome,
                "last_transition_reason": hypothesis_records[-1].get("reason_code") if hypothesis_records else None,
            })
        item = {
            "level_index": level_index,
            "attempt_index": attempt_index,
            "outcome": "FAILED_RESET_REQUESTED",
            "reset_trigger": reset_trigger,
            "step_index": step_index,
            "qwen_calls": qwen_calls,
            "action_runs": _compact_attempt_action_runs(records),
            "executed_hypotheses": executed_hypotheses,
            "execution_failures": failures[-12:],
            "verifier_feedback": verifier_feedback,
        }
        self.level_attempt_records.append(item)
        return item

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

    def unprobed_action_effect_ids(self, snapshot: Any, config: V8Config) -> list[str]:
        available = [
            str(action_id)
            for action_id in (getattr(snapshot, "planning_action_ids", ()) or getattr(snapshot, "available_actions", ()))
            if str(action_id) in ACTION_EFFECT_PROBE_IDS and str(action_id) not in set(getattr(snapshot, "coordinate_action_ids", ()))
        ]
        available = sorted(set(available))
        remaining = [
            action_id
            for action_id in available
            if self.simple_action_attempt_count(int(getattr(snapshot, "level_index", 0)), action_id) <= 0
        ]
        return remaining[: max(0, int(config.max_simple_action_probes_per_level))]

    def action_effect_probe_complete(self, snapshot: Any, config: V8Config) -> bool:
        return not self.unprobed_action_effect_ids(snapshot, config)

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
        coordinate_candidate_id: str | None = None,
        coordinate_x: int | None = None,
        coordinate_y: int | None = None,
    ) -> None:
        scoped = self.state_scoped_action_signature(suppression_signature, state_signature)
        self.action_attempts_by_signature[scoped] = self.action_attempts_by_signature.get(scoped, 0) + 1
        if is_coordinate:
            self.coordinate_probe_counts_by_level[level_index] = self.coordinate_probe_counts_by_level.get(level_index, 0) + 1
            self.coordinate_probe_signature_counts[scoped] = self.coordinate_probe_signature_counts.get(scoped, 0) + 1
            self.coordinate_candidates_clicked_this_attempt.update(
                self.coordinate_attempt_keys(action_id, coordinate_candidate_id, coordinate_x, coordinate_y)
            )
        else:
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
        idx = len(self.events) + 1
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
            "probe_policy": "At a fresh level, only currently available non-coordinate, non-undo simple actions are tried once. Coordinate actions require grounded coordinate candidates.",
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
        if action.action_id not in ACTION_EFFECT_PROBE_IDS and action.action_id != "ACTION6":
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
            "contract_kind": judgment.contract_kind.value if judgment.contract_kind else None,
            "effect_outcome": outcome,
            "truth": judgment.truth.value,
            "relevance": judgment.relevance.value,
            "progress": judgment.progress.value,
            "attribution": judgment.attribution.value,
            "reason_code": judgment.reason_code,
            "changed_cell_count": delta.get("changed_cell_count"),
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
        if (
            runs
            and runs[-1]["action_id"] == action_id
            and runs[-1]["source"] == source
            and runs[-1].get("hypothesis_id") == hypothesis_id
        ):
            runs[-1]["count"] += 1
        else:
            runs.append({
                "action_id": action_id,
                "count": 1,
                "source": source,
                "hypothesis_id": hypothesis_id,
            })
    return runs


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
        f"progress={j.progress.value}, reason={j.reason_code}, ig={j.observed_information_gain:.3f}"
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
