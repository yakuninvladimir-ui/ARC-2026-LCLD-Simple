from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .config import V8Config
from .observe import stable_hash
from .qwen_packet import translate_model_ids_to_internal
from .reverse_semantics import bind_semantic_objective, evaluate_trajectory
from .types import (
    ARGALiteSnapshot,
    BindingStatus,
    CandidateAction,
    HypothesisItem,
    Judgment,
    MechanicResult,
    Progress,
    QwenRole,
    Relevance,
    SemanticObjective,
    TestStep,
    TrajectoryEvaluation,
    TriTruth,
    Validity,
    VerificationContractKind,
)
from .verification import VerificationBinder


class HypothesisBank:
    def __init__(self) -> None:
        self.confirmed_rules: list[HypothesisItem] = []
        self.semantic_test_queue: list[HypothesisItem] = []
        self.coordinate_test_queue: list[HypothesisItem] = []
        self.fallback_exploration_queue: list[HypothesisItem] = []
        self.rejected: list[HypothesisItem] = []
        self.invalid_rejections: list[dict[str, Any]] = []
        self._binder = VerificationBinder()
        self._current_step = 0
        self._active_hypothesis_id: str | None = None
        self._pending_alternative_reset: dict[str, Any] | None = None

    def reset_level(self, level_index: int) -> None:
        # Concrete target IDs are level-local. Mechanic knowledge lives in GameMemory.
        self.confirmed_rules.clear()
        self.semantic_test_queue.clear()
        self.coordinate_test_queue.clear()
        self.fallback_exploration_queue.clear()
        self.rejected.clear()
        self.invalid_rejections.clear()
        self._active_hypothesis_id = None
        self._pending_alternative_reset = None

    def pending_alternative_reset(self) -> dict[str, Any] | None:
        if self._pending_alternative_reset is None:
            return None
        return dict(self._pending_alternative_reset)

    def acknowledge_alternative_reset(self) -> dict[str, Any] | None:
        request = self.pending_alternative_reset()
        self._pending_alternative_reset = None
        return request

    def rebind_pending_alternatives(self, snapshot: ARGALiteSnapshot, memory: Any) -> list[Any]:
        request = self._pending_alternative_reset
        if request is None:
            return []
        batch_id = str(request.get("proposal_batch_id") or "")
        rebound_bindings = []
        for item in self.semantic_test_queue:
            if item.proposal_batch_id != batch_id or not item.has_next_step():
                continue
            binding = item.semantic_binding
            objective = item.semantic_objective
            if binding is None or objective is None:
                continue
            resolved = memory.resolve_semantic_binding_references(binding, snapshot)
            if not resolved.get("entity_references_still_current"):
                item.validity = Validity.INVALID
                self.invalid_rejections.append({
                    "reason": "alternative_binding_unresolved_after_reset",
                    "hypothesis_id": item.hypothesis_id,
                    "unresolved_object_count": resolved.get("unresolved_object_count"),
                    "unresolved_relation_count": resolved.get("unresolved_relation_count"),
                })
                continue
            object_map = dict(resolved.get("object_map") or {})
            relation_map = dict(resolved.get("relation_map") or {})
            rebound_objective = replace(
                objective,
                source_object_ids=tuple(resolved.get("source_object_ids") or ()),
                reference_object_ids=tuple(resolved.get("reference_object_ids") or ()),
                relation_ids=tuple(resolved.get("relation_ids") or ()),
            )
            rebound_binding = bind_semantic_objective(rebound_objective, snapshot, item.hypothesis_id)
            if rebound_binding.status is BindingStatus.REJECTED:
                item.validity = Validity.INVALID
                self.invalid_rejections.append({
                    "reason": rebound_binding.reason_code,
                    "hypothesis_id": item.hypothesis_id,
                    "after_alternative_reset": True,
                })
                continue
            item.semantic_objective = rebound_objective
            item.semantic_binding = rebound_binding
            item.created_state_signature = snapshot.semantic_state_signature
            item.evidence_refs = tuple(relation_map.get(value, value) for value in item.evidence_refs)
            item.test_plan = tuple(
                replace(
                    step,
                    target_object_id=object_map.get(step.target_object_id, step.target_object_id),
                    target_relation_id=relation_map.get(step.target_relation_id, step.target_relation_id),
                    target_object_ids=tuple(object_map.get(value, value) for value in step.target_object_ids),
                    target_relation_ids=tuple(relation_map.get(value, value) for value in step.target_relation_ids),
                    semantic_binding=rebound_binding,
                )
                for step in item.test_plan
            )
            rebound_bindings.append(rebound_binding)
        self._purge_consumed_and_invalid(snapshot.step_index)
        self._sort()
        return rebound_bindings

    def attempt_feedback(self, limit: int = 12) -> dict[str, Any]:
        hypotheses = []
        seen: set[str] = set()
        for item in self.rejected + self.semantic_test_queue + self.coordinate_test_queue:
            if item.hypothesis_id in seen:
                continue
            seen.add(item.hypothesis_id)
            hypotheses.append({
                "hypothesis_id": item.hypothesis_id,
                "claim": item.claim[:560],
                "source": item.source,
                "actions": [step.action_id for step in item.test_plan],
                "executed_step_count": min(item.cursor, len(item.test_plan)),
                "total_step_count": len(item.test_plan),
                "truth": item.truth.value,
                "validity": item.validity.value,
                "relevance": item.relevance.value,
                "progress": item.progress.value,
            })
        return {
            "hypotheses": hypotheses[-limit:],
            "rejections": [
                _compact_attempt_rejection(item)
                for item in self.invalid_rejections[-limit:]
                if isinstance(item, dict)
            ],
        }

    def has_valid_action_candidates(self, snapshot: ARGALiteSnapshot | None = None) -> bool:
        if snapshot is None:
            return any(self._usable(h, self._current_step) for h in self.confirmed_rules + self.coordinate_test_queue + self.semantic_test_queue + self.fallback_exploration_queue)
        return self.has_executable_candidate(snapshot)

    def has_executable_candidate(self, snapshot: ARGALiteSnapshot) -> bool:
        return self.next_candidate_action(snapshot, dry_run=True) is not None

    def has_executable_coordinate_candidate(self, snapshot: ARGALiteSnapshot) -> bool:
        return self.next_candidate_action(snapshot, "coordinate", dry_run=True) is not None

    def add_qwen_output(self, role: QwenRole, output: dict[str, Any] | None, snapshot: ARGALiteSnapshot, config: V8Config, packet: dict[str, Any] | None = None) -> None:
        self._current_step = snapshot.step_index
        if not output:
            return
        if role is QwenRole.COORDINATE:
            self._add_coordinate_output(translate_model_ids_to_internal(output, snapshot, config), snapshot, config)
        else:
            self._add_semantic_output(role, output, snapshot, config, packet)
        self._sort()

    def _add_semantic_output(self, role: QwenRole, output: dict[str, Any], snapshot: ARGALiteSnapshot, config: V8Config, packet: dict[str, Any] | None = None) -> None:
        if output.get("schema_version") == "v8.7.semantic_trajectories":
            self._add_v87_semantic_trajectories(role, translate_model_ids_to_internal(output, snapshot, config), snapshot, config, model_output=output, packet=packet)
            return
        if output.get("schema_version") == "v8.6.semantic_trajectory_hypotheses":
            self._add_v86_semantic_hypotheses(role, translate_model_ids_to_internal(output, snapshot, config), snapshot, config, model_output=output, packet=packet)
            return
        if output.get("schema_version") == "v8.5.semantic_trajectory_hypotheses":
            self._add_semantic_hypotheses(role, output, snapshot, config)
            return
        if output.get("schema_version") == "v8.4.semantic_plan":
            self._add_semantic_plan(role, output, snapshot, config)
            return
        if output.get("schema_version") not in (None, "v8.2.semantic_output", "v8.3.semantic_output"):
            self.invalid_rejections.append({"reason": "bad_semantic_schema", "schema": output.get("schema_version")})
            return
        valid_obj = {o.object_id for o in snapshot.objects}
        valid_rel = {r.relation_id for r in snapshot.relations}
        valid_actions = set(snapshot.available_actions)
        undo_actions = set(getattr(snapshot, "undo_action_ids", ()) or ())
        if role is not QwenRole.RESERVE:
            valid_actions -= undo_actions
        valid_candidates = {c.candidate_id for c in snapshot.coordinate_targets}
        for idx, raw in enumerate(output.get("hypotheses", []) or []):
            if not isinstance(raw, dict):
                continue
            targets_obj = _semantic_target_object_ids(raw)
            targets_rel = _semantic_target_relation_ids(raw)
            if not set(targets_obj).issubset(valid_obj) or not set(targets_rel).issubset(valid_rel):
                self.invalid_rejections.append({"reason": "invented_target_id", "raw": raw})
                continue
            plan: list[TestStep] = []
            bad = False
            raw_steps = raw.get("trajectory")
            if not isinstance(raw_steps, list):
                raw_steps = raw.get("test_plan") or []
            for raw_step in raw_steps[: config.max_qwen_trajectory_steps]:
                if isinstance(raw_step, str):
                    raw_step = {"kind": "rule_application", "action_id": raw_step}
                if not isinstance(raw_step, dict):
                    continue
                action_id = _none_or_str(raw_step.get("action_id"))
                goal_ref = _object_goal_ref(raw, raw_step)
                target_object_id = (
                    _none_or_str(raw_step.get("target_object_id"))
                    or _none_or_str(goal_ref.get("source_object_id") or goal_ref.get("target_object_id"))
                    or (targets_obj[0] if targets_obj else None)
                )
                target_relation_id = (
                    _none_or_str(raw_step.get("target_relation_id"))
                    or _none_or_str(goal_ref.get("target_relation_id"))
                    or (targets_rel[0] if targets_rel else None)
                )
                candidate_id = _none_or_str(raw_step.get("coordinate_candidate_id"))
                if action_id is not None and action_id not in valid_actions:
                    bad = True
                    reason = "undo_requires_reserve_qwen" if action_id in undo_actions else "unknown_action_id"
                    self.invalid_rejections.append({"reason": reason, "action_id": action_id})
                    break
                if target_object_id is not None and target_object_id not in valid_obj:
                    bad = True
                    self.invalid_rejections.append({"reason": "invented_step_object_id", "target_object_id": target_object_id})
                    break
                if target_relation_id is not None and target_relation_id not in valid_rel:
                    bad = True
                    self.invalid_rejections.append({"reason": "invented_step_relation_id", "target_relation_id": target_relation_id})
                    break
                if candidate_id is not None and candidate_id not in valid_candidates:
                    bad = True
                    self.invalid_rejections.append({"reason": "invented_coordinate_candidate_id", "coordinate_candidate_id": candidate_id})
                    break
                contract_kind = _none_or_str(raw_step.get("contract_kind"))
                if contract_kind is not None and not _valid_contract_kind(contract_kind):
                    bad = True
                    self.invalid_rejections.append({"reason": "unknown_contract_kind", "contract_kind": contract_kind})
                    break
                if action_id is not None:
                    plan.append(TestStep(
                        kind=str(raw_step.get("kind") or "simple_action_probe"),
                        action_id=action_id,
                        target_object_id=target_object_id,
                        target_relation_id=target_relation_id,
                        coordinate_candidate_id=candidate_id,
                        expected_observation=_none_or_str(raw_step.get("expected_observation") or raw.get("expected_effect")),
                        contract_kind=contract_kind,
                        question_type=_none_or_str(raw_step.get("question_type")),
                    ))
            if bad or not plan:
                continue
            claim = _semantic_claim(raw)
            hid = stable_hash((role.value, snapshot.level_index, snapshot.step_index, idx, raw.get("hypothesis_id"), claim, [(s.kind, s.action_id, s.target_object_id, s.target_relation_id, s.coordinate_candidate_id, s.expected_observation, s.contract_kind) for s in plan]), "hyp_")
            item = HypothesisItem(
                hypothesis_id=hid,
                source=f"{role.value}_qwen",
                claim=claim,
                truth=_enum_or(raw.get("truth_prior"), TriTruth, TriTruth.UNKNOWN),
                relevance=_enum_or(raw.get("relevance_prior"), Relevance, Relevance.UNDECIDED),
                validity=Validity.UNCHECKED,
                progress=Progress.UNKNOWN,
                test_plan=tuple(plan),
                cursor=0,
                priority=_float(raw.get("priority"), 0.0),
                confidence=_float(raw.get("confidence"), 0.0),
                expiry_step=(snapshot.step_index + int(raw.get("expiry_steps") or 0)) if raw.get("expiry_steps") else None,
                evidence_refs=(),
                suppression_signature=stable_hash((raw.get("claim"), [(p.action_id, p.target_object_id, p.target_relation_id) for p in plan]), "sup_"),
                created_state_signature=snapshot.semantic_state_signature,
            )
            self.semantic_test_queue.append(item)

    def _add_v87_semantic_trajectories(
        self,
        role: QwenRole,
        output: dict[str, Any],
        snapshot: ARGALiteSnapshot,
        config: V8Config,
        *,
        model_output: dict[str, Any] | None = None,
        packet: dict[str, Any] | None = None,
    ) -> None:
        decision = str(output.get("decision") or "").upper()
        if decision == "ABSTAIN":
            if not _none_or_str(output.get("reason")):
                self.invalid_rejections.append({"reason": "abstain_missing_reason", "schema": output.get("schema_version")})
            return
        if decision != "PROPOSE":
            self.invalid_rejections.append({"reason": "bad_semantic_trajectory_decision", "decision": output.get("decision")})
            return
        hypotheses = output.get("hypotheses")
        model_hypotheses = (model_output or {}).get("hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            self.invalid_rejections.append({"reason": "empty_semantic_trajectories", "raw": output})
            return
        if len(hypotheses) > 3:
            self.invalid_rejections.append({"reason": "too_many_semantic_trajectories", "raw": output})
            return

        proposal_batch_id = stable_hash((
            role.value,
            snapshot.level_index,
            snapshot.step_index,
            output.get("schema_version"),
            [item.get("id") for item in hypotheses if isinstance(item, dict)],
        ), "batch_")
        valid_objects = {obj.object_id for obj in snapshot.objects}
        valid_relations = {relation.relation_id for relation in snapshot.relations}
        allowed_actions = set((packet or {}).get("execution_constraints", {}).get("allowed_action_ids") or snapshot.available_actions)
        if role is not QwenRole.RESERVE:
            allowed_actions -= set(snapshot.undo_action_ids)
        coordinate_action_ids = set(snapshot.coordinate_action_ids)
        valid_coordinate_candidates = {candidate.candidate_id for candidate in snapshot.coordinate_targets}
        added = 0
        for index, raw in enumerate(hypotheses):
            if not isinstance(raw, dict):
                self.invalid_rejections.append({"reason": "bad_semantic_trajectory_record", "raw": raw})
                continue
            model_raw = model_hypotheses[index] if isinstance(model_hypotheses, list) and index < len(model_hypotheses) and isinstance(model_hypotheses[index], dict) else raw
            if str(raw.get("status") or "") != "complete_candidate":
                self.invalid_rejections.append({"reason": "incomplete_semantic_trajectory_rejected", "raw": raw})
                continue
            objective = raw.get("objective")
            if not isinstance(objective, dict):
                self.invalid_rejections.append({"reason": "missing_semantic_objective", "raw": raw})
                continue
            sources = _string_list(objective.get("source_objects"))
            references = _string_list(objective.get("reference_objects"))
            relations = _string_list(raw.get("relations"))
            coordinate_candidates_in_trajectory = [
                _none_or_str(run.get("coordinate_candidate_id"))
                for run in (raw.get("action_runs") or [])
                if isinstance(run, dict) and _none_or_str(run.get("action_id")) in coordinate_action_ids
            ]
            if len(coordinate_candidates_in_trajectory) != len(set(coordinate_candidates_in_trajectory)):
                self.invalid_rejections.append({
                    "reason": "coordinate_candidate_repeated_in_trajectory",
                    "coordinate_candidate_ids": coordinate_candidates_in_trajectory,
                    "raw": raw,
                })
                continue
            coordinate_targets_by_id = {candidate.candidate_id: candidate for candidate in snapshot.coordinate_targets}
            coordinate_locations_in_trajectory = [
                (coordinate_targets_by_id[candidate_id].x, coordinate_targets_by_id[candidate_id].y)
                for candidate_id in coordinate_candidates_in_trajectory
                if candidate_id in coordinate_targets_by_id
            ]
            if len(coordinate_locations_in_trajectory) != len(set(coordinate_locations_in_trajectory)):
                self.invalid_rejections.append({
                    "reason": "coordinate_location_repeated_in_trajectory",
                    "coordinate_candidate_ids": coordinate_candidates_in_trajectory,
                    "coordinate_locations_xy": coordinate_locations_in_trajectory,
                    "raw": raw,
                })
                continue
            invalid_coordinate_repeat = next((
                run
                for run in (raw.get("action_runs") or [])
                if isinstance(run, dict)
                and _none_or_str(run.get("action_id")) in coordinate_action_ids
                and run.get("repeat") != 1
            ), None)
            if invalid_coordinate_repeat is not None:
                self.invalid_rejections.append({
                    "reason": "coordinate_action_repeat_must_equal_one",
                    "action_id": invalid_coordinate_repeat.get("action_id"),
                    "repeat": invalid_coordinate_repeat.get("repeat"),
                    "coordinate_candidate_id": invalid_coordinate_repeat.get("coordinate_candidate_id"),
                    "raw": raw,
                })
                continue
            action_run_steps = _expanded_action_runs(raw.get("action_runs"))
            actions = [action_id for action_id, _candidate_id in action_run_steps] or _action_sequence(raw.get("actions"))
            all_objects = list(dict.fromkeys([*sources, *references]))
            if not set(all_objects).issubset(valid_objects):
                self.invalid_rejections.append({"reason": "invented_object_id", "raw": raw})
                continue
            if not set(relations).issubset(valid_relations):
                self.invalid_rejections.append({"reason": "invented_relation_id", "raw": raw})
                continue
            if not actions or not set(actions).issubset(allowed_actions):
                self.invalid_rejections.append({"reason": "unknown_action_id", "raw": raw})
                continue
            if actions[0] not in set(snapshot.available_actions):
                self.invalid_rejections.append({"reason": "first_action_not_available_now", "action_id": actions[0], "raw": raw})
                continue
            if action_run_steps:
                bad_coordinate_binding = next((
                    (action_id, candidate_id)
                    for action_id, candidate_id in action_run_steps
                    if (
                        (action_id in coordinate_action_ids and candidate_id not in valid_coordinate_candidates)
                        or (action_id not in coordinate_action_ids and candidate_id is not None)
                    )
                ), None)
                if bad_coordinate_binding is not None:
                    action_id, candidate_id = bad_coordinate_binding
                    self.invalid_rejections.append({
                        "reason": "coordinate_candidate_required_or_invalid_for_action",
                        "action_id": action_id,
                        "coordinate_candidate_id": candidate_id,
                        "raw": raw,
                    })
                    continue
            elif any(action_id in coordinate_action_ids for action_id in actions):
                self.invalid_rejections.append({"reason": "coordinate_candidate_required_or_invalid_for_action", "raw": raw})
                continue
            count_mismatch = _v87_basis_action_count_mismatch(raw, actions)
            if count_mismatch:
                self.invalid_rejections.append({
                    "reason": "trajectory_repeat_count_contradicts_basis",
                    **count_mismatch,
                    "raw": raw,
                })
                continue
            objective_kind = str(objective.get("kind") or "other")
            if (
                objective_kind not in {"surface_change", "other"}
                and not all_objects
                and not _v87_has_bound_coordinate_action(model_raw, packet)
            ):
                self.invalid_rejections.append({"reason": "ungrounded_objective_without_objects", "raw": raw})
                continue
            repeated_failure = (
                _v87_matching_failed_trajectory(model_raw, packet)
                if config.reject_unchanged_failed_trajectories
                else None
            )
            if repeated_failure is not None:
                self.invalid_rejections.append({
                    "reason": "unchanged_failed_trajectory_repeat",
                    **repeated_failure,
                    "raw": raw,
                })
                continue
            layered_reachability_issue = (
                _v87_layered_trajectory_reachability_issue(model_raw, packet)
                if isinstance(packet, dict) and packet.get("schema_version") == "v8.8.layered_observation"
                else None
            )
            if layered_reachability_issue is not None:
                self.invalid_rejections.append({
                    "reason": "trajectory_action_unavailable_on_reached_surface",
                    **layered_reachability_issue,
                    "raw": raw,
                })
                continue
            if not _v87_trajectory_respects_control_context(model_raw, packet):
                self.invalid_rejections.append({"reason": "trajectory_uses_effect_from_wrong_control_context", "raw": raw})
                continue
            if _v87_has_unjustified_immediate_inverse_pair(model_raw, packet):
                self.invalid_rejections.append({"reason": "trajectory_contains_unjustified_immediate_inverse_pair", "raw": raw})
                continue
            if not _v87_trajectory_uses_observed_effect(model_raw, packet):
                self.invalid_rejections.append({"reason": "trajectory_not_grounded_in_observed_effect", "raw": raw})
                continue
            if not _v87_trajectory_has_progress(model_raw, packet):
                self.invalid_rejections.append({"reason": "trajectory_has_no_net_or_intermediate_effect", "raw": raw})
                continue
            incomplete_correspondence = _v87_incomplete_exact_correspondence(model_raw, packet)
            if incomplete_correspondence is not None:
                self.invalid_rejections.append({
                    "reason": "trajectory_does_not_complete_claimed_correspondence",
                    **incomplete_correspondence,
                    "raw": raw,
                })
                continue

            semantic_objective = SemanticObjective(
                kind=objective_kind,
                source_object_ids=tuple(sources),
                reference_object_ids=tuple(references),
                relation_ids=tuple(relations),
                description=_short_text(objective.get("description"), 320),
                family=_short_text(raw.get("family"), 80) or "other",
                basis=_short_text(raw.get("basis"), 320),
            )
            confidence = _float(raw.get("confidence"), 0.0)
            priority = _v87_priority(model_raw, confidence, packet, index)
            claim = _v87_hypothesis_claim(raw)
            hypothesis_id = stable_hash((
                role.value,
                snapshot.level_index,
                snapshot.step_index,
                raw.get("id"),
                claim,
                action_run_steps or actions,
                sources,
                references,
                relations,
                objective_kind,
            ), "hyp_")
            semantic_binding = bind_semantic_objective(semantic_objective, snapshot, hypothesis_id)
            if semantic_binding.status is BindingStatus.REJECTED:
                self.invalid_rejections.append({
                    "reason": semantic_binding.reason_code,
                    "hypothesis_id": hypothesis_id,
                    "raw": raw,
                })
                continue

            plan = []
            for action_index, action_id in enumerate(actions[: config.max_qwen_trajectory_steps]):
                contract_kind = _v87_contract_kind(model_raw, action_id, packet)
                target_ids = sources or all_objects
                coordinate_candidate_id = action_run_steps[action_index][1] if action_run_steps else None
                plan.append(TestStep(
                    kind="verified_effect_trajectory_step",
                    action_id=action_id,
                    target_object_id=target_ids[0] if target_ids else None,
                    target_relation_id=None,
                    target_object_ids=tuple(target_ids),
                    target_relation_ids=(),
                    coordinate_candidate_id=coordinate_candidate_id,
                    expected_observation=_v87_expected_effect_text(model_raw, action_id, packet),
                    contract_kind=contract_kind,
                    semantic_binding=semantic_binding,
                ))
            if not plan:
                self.invalid_rejections.append({"reason": "empty_semantic_trajectory", "raw": raw})
                continue
            self.semantic_test_queue.append(HypothesisItem(
                hypothesis_id=hypothesis_id,
                source=f"{role.value}_qwen_v87",
                claim=claim,
                truth=TriTruth.UNKNOWN,
                relevance=Relevance.UNDECIDED,
                validity=Validity.UNCHECKED,
                progress=Progress.UNKNOWN,
                test_plan=tuple(plan),
                cursor=0,
                priority=priority,
                confidence=confidence,
                expiry_step=None,
                evidence_refs=tuple(relations),
                suppression_signature=stable_hash((claim, action_run_steps or actions, sources, references), "sup_"),
                created_state_signature=snapshot.semantic_state_signature,
                proposal_batch_id=proposal_batch_id,
                semantic_objective=semantic_objective,
                semantic_binding=semantic_binding,
            ))
            added += 1
        if added <= 0:
            self.invalid_rejections.append({"reason": "no_valid_semantic_trajectories", "schema": output.get("schema_version")})

    def _add_v86_semantic_hypotheses(
        self,
        role: QwenRole,
        output: dict[str, Any],
        snapshot: ARGALiteSnapshot,
        config: V8Config,
        *,
        model_output: dict[str, Any] | None = None,
        packet: dict[str, Any] | None = None,
    ) -> None:
        decision = str(output.get("decision") or "").upper()
        if decision == "ABSTAIN":
            if not _none_or_str(output.get("reason")):
                self.invalid_rejections.append({"reason": "abstain_missing_reason", "schema": output.get("schema_version")})
            return
        if decision != "PROPOSE":
            self.invalid_rejections.append({"reason": "bad_semantic_hypotheses_decision", "decision": output.get("decision")})
            return
        raw_hypotheses = output.get("hypotheses")
        model_hypotheses = (model_output or {}).get("hypotheses")
        if not isinstance(raw_hypotheses, list) or not raw_hypotheses:
            self.invalid_rejections.append({"reason": "empty_semantic_hypotheses", "raw": output})
            return
        if len(raw_hypotheses) > 3:
            self.invalid_rejections.append({"reason": "too_many_semantic_hypotheses", "raw": output})
            return

        valid_obj = {o.object_id for o in snapshot.objects}
        valid_rel = {r.relation_id for r in snapshot.relations}
        valid_actions = set(snapshot.available_actions)
        undo_actions = set(getattr(snapshot, "undo_action_ids", ()) or ())
        if role is not QwenRole.RESERVE:
            valid_actions -= undo_actions
        added = 0
        for idx, raw in enumerate(raw_hypotheses):
            if not isinstance(raw, dict):
                self.invalid_rejections.append({"reason": "bad_semantic_hypothesis_record", "raw": raw})
                continue
            if any(key in raw for key in ("x", "y", "trajectory", "goal", "expected_progress")):
                self.invalid_rejections.append({"reason": "unsupported_v86_field", "raw": raw})
                continue
            objects = _string_list(raw.get("objects"))
            relations = _string_list(raw.get("relations"))
            actions = _action_sequence(raw.get("actions"))
            if not set(objects).issubset(valid_obj):
                self.invalid_rejections.append({"reason": "invented_expected_object_id", "raw": raw})
                continue
            if not set(relations).issubset(valid_rel):
                self.invalid_rejections.append({"reason": "invented_expected_relation_id", "raw": raw})
                continue
            if not actions or not set(actions).issubset(valid_actions):
                self.invalid_rejections.append({"reason": "unknown_action_id", "raw": raw})
                continue
            family = _none_or_str(raw.get("family")) or "other"
            if family == "same_shape_alignment":
                exact_relations = [relation_id for relation_id in relations if _is_exact_shape_relation(snapshot, relation_id)]
                if not exact_relations:
                    exact_relations = _exact_shape_relation_ids_for_objects(snapshot, objects)
                if not exact_relations:
                    self.invalid_rejections.append({"reason": "same_shape_family_without_same_geometry", "raw": raw})
                    continue
                relations = exact_relations
                objects = _relation_endpoint_object_ids(snapshot, relations) or objects
            model_raw = model_hypotheses[idx] if isinstance(model_hypotheses, list) and idx < len(model_hypotheses) and isinstance(model_hypotheses[idx], dict) else None
            if not _v86_trajectory_has_progress(model_raw or raw, packet):
                self.invalid_rejections.append({"reason": "trajectory_no_progress", "raw": raw})
                continue
            expected_type = "action_surface_change" if family == "action_surface_change" else ("object_move" if objects else "action_effect")
            expected_text = _v86_expected_observation_text(raw, expected_type, objects, relations)
            plan = [
                TestStep(
                    kind=expected_type,
                    action_id=action_id,
                    target_object_id=objects[0] if objects else None,
                    target_relation_id=relations[0] if relations else None,
                    target_object_ids=tuple(objects),
                    target_relation_ids=tuple(relations),
                    expected_observation=expected_text,
                    contract_kind=("ACTION_SURFACE_CHANGE" if family == "action_surface_change" else ("OBJECT_DISPLACEMENT" if objects else "ACTION_EFFECT_DISCOVERY")),
                )
                for action_id in actions[: config.max_qwen_trajectory_steps]
            ]
            if not plan:
                self.invalid_rejections.append({"reason": "empty_semantic_trajectory", "raw": raw})
                continue
            confidence = _float(raw.get("confidence"), 0.0)
            priority = _v86_priority(raw, confidence, packet, idx)
            claim = _v86_hypothesis_claim(raw)
            hid = stable_hash((role.value, snapshot.level_index, snapshot.step_index, raw.get("id"), claim, [(s.kind, s.action_id, s.target_object_ids, s.target_relation_ids, s.expected_observation) for s in plan]), "hyp_")
            self.semantic_test_queue.append(HypothesisItem(
                hypothesis_id=hid,
                source=f"{role.value}_qwen",
                claim=claim,
                truth=TriTruth.UNKNOWN,
                relevance=Relevance.UNDECIDED,
                validity=Validity.UNCHECKED,
                progress=Progress.UNKNOWN,
                test_plan=tuple(plan),
                cursor=0,
                priority=priority,
                confidence=confidence,
                expiry_step=None,
                evidence_refs=(),
                suppression_signature=stable_hash((claim, [(p.action_id, p.target_object_ids, p.target_relation_ids) for p in plan]), "sup_"),
                created_state_signature=snapshot.semantic_state_signature,
            ))
            added += 1
        if added <= 0:
            self.invalid_rejections.append({"reason": "no_valid_semantic_hypotheses", "schema": output.get("schema_version")})

    def _add_semantic_hypotheses(self, role: QwenRole, output: dict[str, Any], snapshot: ARGALiteSnapshot, config: V8Config) -> None:
        decision = str(output.get("decision") or "").upper()
        if decision == "ABSTAIN":
            if not _none_or_str(output.get("abstain_reason")):
                self.invalid_rejections.append({"reason": "abstain_missing_reason", "schema": output.get("schema_version")})
            return
        if decision != "PROPOSE":
            self.invalid_rejections.append({"reason": "bad_semantic_hypotheses_decision", "decision": output.get("decision")})
            return
        if any(key in output for key in ("x", "y")):
            self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "schema": output.get("schema_version")})
            return
        raw_hypotheses = output.get("hypotheses")
        if not isinstance(raw_hypotheses, list) or not raw_hypotheses:
            self.invalid_rejections.append({"reason": "empty_semantic_hypotheses", "raw": output})
            return

        valid_obj = {o.object_id for o in snapshot.objects}
        valid_rel = {r.relation_id for r in snapshot.relations}
        valid_actions = set(snapshot.available_actions)
        undo_actions = set(getattr(snapshot, "undo_action_ids", ()) or ())
        if role is not QwenRole.RESERVE:
            valid_actions -= undo_actions
        valid_candidates = {c.candidate_id for c in snapshot.coordinate_targets}
        added = 0

        for raw in raw_hypotheses[:3]:
            if not isinstance(raw, dict):
                self.invalid_rejections.append({"reason": "bad_semantic_hypothesis_record", "raw": raw})
                continue
            if any(key in raw for key in ("x", "y")):
                self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "raw": raw})
                continue

            goal = raw.get("goal") if isinstance(raw.get("goal"), dict) else {}
            goal_objects = _string_list(goal.get("object_ids"))
            goal_relations = _string_list(goal.get("relation_ids"))
            if not set(goal_objects).issubset(valid_obj) or not set(goal_relations).issubset(valid_rel):
                self.invalid_rejections.append({"reason": "invented_goal_id", "raw": raw})
                continue

            evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
            supporting_actions = _string_list(evidence.get("supporting_action_ids"))
            if not set(supporting_actions).issubset(valid_actions | undo_actions):
                self.invalid_rejections.append({"reason": "invented_evidence_action_id", "raw": raw})
                continue
            expected = raw.get("expected_progress") if isinstance(raw.get("expected_progress"), dict) else {}
            tracked_objects = _string_list(expected.get("tracked_object_ids")) or goal_objects
            tracked_relations = _string_list(expected.get("tracked_relation_ids")) or goal_relations
            if not set(tracked_objects).issubset(valid_obj):
                self.invalid_rejections.append({"reason": "invented_expected_object_id", "raw": raw})
                continue
            if not set(tracked_relations).issubset(valid_rel):
                self.invalid_rejections.append({"reason": "invented_expected_relation_id", "raw": raw})
                continue
            tracked_relations = _v85_relation_targets(raw, snapshot, goal_objects, goal_relations, tracked_relations)
            if tracked_relations is None:
                self.invalid_rejections.append({"reason": "same_shape_alignment_without_exact_shape_relation", "raw": raw})
                continue
            tracked_objects = _v85_tracked_objects(raw, snapshot, tracked_objects, tracked_relations)

            trajectory = raw.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                self.invalid_rejections.append({"reason": "empty_semantic_trajectory", "raw": raw})
                continue
            plan: list[TestStep] = []
            bad = False
            expected_text = _v85_expected_observation_text(raw, tracked_objects, tracked_relations)
            expected_type = _expected_observation_type_from_text(expected_text) or "rule_application"
            for raw_step in trajectory[: config.max_qwen_trajectory_steps]:
                if not isinstance(raw_step, dict):
                    self.invalid_rejections.append({"reason": "bad_step_record", "raw": raw_step})
                    bad = True
                    break
                if any(key in raw_step for key in ("x", "y")):
                    self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "raw": raw_step})
                    bad = True
                    break
                action_id = _none_or_str(raw_step.get("action_id"))
                if action_id is None or action_id not in valid_actions:
                    reason = "undo_requires_reserve_qwen" if action_id in undo_actions else "unknown_action_id"
                    self.invalid_rejections.append({"reason": reason, "action_id": action_id})
                    bad = True
                    break
                candidate_id = _none_or_str(raw_step.get("coordinate_candidate_id"))
                if candidate_id is not None and candidate_id not in valid_candidates:
                    self.invalid_rejections.append({"reason": "invented_coordinate_candidate_id", "coordinate_candidate_id": candidate_id})
                    bad = True
                    break
                plan.append(TestStep(
                    kind=expected_type,
                    action_id=action_id,
                    target_object_id=tracked_objects[0] if tracked_objects else None,
                    target_relation_id=tracked_relations[0] if tracked_relations else None,
                    target_object_ids=tuple(tracked_objects),
                    target_relation_ids=tuple(tracked_relations),
                    coordinate_candidate_id=candidate_id,
                    expected_observation=expected_text,
                    contract_kind=None,
                ))
            if bad or not plan:
                continue

            claim = _semantic_hypothesis_claim(raw)
            rank = max(1, min(3, int(_float(raw.get("rank"), 3))))
            confidence = _float(raw.get("confidence"), 0.0)
            priority = max(0.0, confidence + max(0, 3 - rank) * 0.05)
            hid = stable_hash((role.value, snapshot.level_index, snapshot.step_index, raw.get("hypothesis_id"), claim, [(s.kind, s.action_id, s.target_object_ids, s.target_relation_ids, s.coordinate_candidate_id, s.expected_observation) for s in plan]), "hyp_")
            self.semantic_test_queue.append(HypothesisItem(
                hypothesis_id=hid,
                source=f"{role.value}_qwen",
                claim=claim,
                truth=TriTruth.UNKNOWN,
                relevance=Relevance.UNDECIDED,
                validity=Validity.UNCHECKED,
                progress=Progress.UNKNOWN,
                test_plan=tuple(plan),
                cursor=0,
                priority=priority,
                confidence=confidence,
                expiry_step=None,
                evidence_refs=(),
                suppression_signature=stable_hash((claim, [(p.action_id, p.target_object_ids, p.target_relation_ids, p.coordinate_candidate_id) for p in plan]), "sup_"),
                created_state_signature=snapshot.semantic_state_signature,
            ))
            added += 1
        if added <= 0:
            self.invalid_rejections.append({"reason": "no_valid_semantic_hypotheses", "schema": output.get("schema_version")})

    def _add_semantic_plan(self, role: QwenRole, output: dict[str, Any], snapshot: ARGALiteSnapshot, config: V8Config) -> None:
        decision = str(output.get("decision") or "").upper()
        if decision == "ABSTAIN":
            if not _none_or_str(output.get("abstain_reason")):
                self.invalid_rejections.append({"reason": "abstain_missing_reason", "schema": output.get("schema_version")})
            return
        if decision != "PLAN":
            self.invalid_rejections.append({"reason": "bad_semantic_plan_decision", "decision": output.get("decision")})
            return
        if any(key in output for key in ("x", "y")):
            self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "schema": output.get("schema_version")})
            return
        valid_obj = {o.object_id for o in snapshot.objects}
        valid_rel = {r.relation_id for r in snapshot.relations}
        valid_actions = set(snapshot.available_actions)
        undo_actions = set(getattr(snapshot, "undo_action_ids", ()) or ())
        if role is not QwenRole.RESERVE:
            valid_actions -= undo_actions
        valid_candidates = {c.candidate_id for c in snapshot.coordinate_targets}
        goal = output.get("goal") if isinstance(output.get("goal"), dict) else {}
        goal_objects = _string_list(goal.get("object_ids"))
        goal_relations = _string_list(goal.get("relation_ids"))
        if not set(goal_objects).issubset(valid_obj) or not set(goal_relations).issubset(valid_rel):
            self.invalid_rejections.append({"reason": "invented_goal_id", "raw": output})
            return
        raw_steps = output.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            self.invalid_rejections.append({"reason": "empty_semantic_plan", "raw": output})
            return
        plan: list[TestStep] = []
        for raw_step in raw_steps[: config.max_qwen_trajectory_steps]:
            if not isinstance(raw_step, dict):
                self.invalid_rejections.append({"reason": "bad_step_record", "raw": raw_step})
                return
            if any(key in raw_step for key in ("x", "y")):
                self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "raw": raw_step})
                return
            action_id = _none_or_str(raw_step.get("action_id"))
            if action_id is None or action_id not in valid_actions:
                reason = "undo_requires_reserve_qwen" if action_id in undo_actions else "unknown_action_id"
                self.invalid_rejections.append({"reason": reason, "action_id": action_id})
                return
            candidate_id = _none_or_str(raw_step.get("coordinate_candidate_id"))
            if candidate_id is not None and candidate_id not in valid_candidates:
                self.invalid_rejections.append({"reason": "invented_coordinate_candidate_id", "coordinate_candidate_id": candidate_id})
                return
            target_object_ids = _string_list(raw_step.get("target_object_ids"))
            target_relation_ids = _string_list(raw_step.get("target_relation_ids"))
            if not set(target_object_ids).issubset(valid_obj):
                self.invalid_rejections.append({"reason": "invented_step_object_id", "target_object_ids": target_object_ids})
                return
            if not set(target_relation_ids).issubset(valid_rel):
                self.invalid_rejections.append({"reason": "invented_step_relation_id", "target_relation_ids": target_relation_ids})
                return
            expected = _expected_observation_text(raw_step.get("expected_observation"))
            plan.append(TestStep(
                kind=_expected_observation_type(raw_step.get("expected_observation")) or "rule_application",
                action_id=action_id,
                target_object_id=target_object_ids[0] if target_object_ids else None,
                target_relation_id=target_relation_ids[0] if target_relation_ids else None,
                target_object_ids=tuple(target_object_ids),
                target_relation_ids=tuple(target_relation_ids),
                coordinate_candidate_id=candidate_id,
                expected_observation=expected,
                contract_kind=None,
            ))
        if not plan:
            self.invalid_rejections.append({"reason": "empty_semantic_plan", "raw": output})
            return
        claim = _semantic_plan_claim(output)
        hid = stable_hash((role.value, snapshot.level_index, snapshot.step_index, claim, [(s.kind, s.action_id, s.target_object_id, s.target_relation_id, s.target_object_ids, s.target_relation_ids, s.coordinate_candidate_id, s.expected_observation) for s in plan]), "hyp_")
        self.semantic_test_queue.append(HypothesisItem(
            hypothesis_id=hid,
            source=f"{role.value}_qwen",
            claim=claim,
            truth=TriTruth.UNKNOWN,
            relevance=Relevance.UNDECIDED,
            validity=Validity.UNCHECKED,
            progress=Progress.UNKNOWN,
            test_plan=tuple(plan),
            cursor=0,
            priority=max(0.0, _float(output.get("confidence"), 0.0)),
            confidence=_float(output.get("confidence"), 0.0),
            expiry_step=None,
            evidence_refs=(),
            suppression_signature=stable_hash((claim, [(p.action_id, p.target_object_id, p.target_relation_id, p.target_object_ids, p.target_relation_ids, p.coordinate_candidate_id) for p in plan]), "sup_"),
            created_state_signature=snapshot.semantic_state_signature,
        ))

    def _add_coordinate_output(self, output: dict[str, Any], snapshot: ARGALiteSnapshot, config: V8Config) -> None:
        if output.get("schema_version") == "v8.4.coordinate_plan":
            self._add_coordinate_plan(output, snapshot, config)
            return
        if output.get("schema_version") not in (None, "v8.2.coordinate_output", "v8.3.coordinate_output"):
            self.invalid_rejections.append({"reason": "bad_coordinate_schema", "schema": output.get("schema_version")})
            return
        valid_actions = set(snapshot.coordinate_action_ids)
        candidates = {c.candidate_id: c for c in snapshot.coordinate_targets}
        for idx, raw in enumerate(output.get("coordinate_hypotheses", []) or []):
            if not isinstance(raw, dict):
                continue
            if not config.allow_qwen_raw_coordinates and ("x" in raw or "y" in raw):
                self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "raw": raw})
                continue
            action_id = str(raw.get("coordinate_action_id") or "")
            if action_id not in valid_actions:
                self.invalid_rejections.append({"reason": "unknown_coordinate_action_id", "raw": raw})
                continue
            contract_kind = _none_or_str(raw.get("contract_kind"))
            if contract_kind is not None and not _valid_contract_kind(contract_kind):
                self.invalid_rejections.append({"reason": "unknown_contract_kind", "contract_kind": contract_kind, "raw": raw})
                continue
            cids = [str(cid) for cid in (raw.get("candidate_target_ids") or [])]
            if config.require_coordinate_candidate_id and not cids:
                self.invalid_rejections.append({"reason": "missing_candidate_target_ids", "raw": raw})
                continue
            if not set(cids).issubset(candidates):
                self.invalid_rejections.append({"reason": "invented_coordinate_candidate_id", "raw": raw})
                continue
            steps: list[TestStep] = []
            for cid in cids[: config.max_qwen_trajectory_steps]:
                candidate = candidates[cid]
                steps.append(TestStep(
                    "coordinate_probe",
                    action_id,
                    target_object_id=candidate.object_id,
                    target_relation_id=candidate.relation_id,
                    coordinate_candidate_id=cid,
                    expected_observation=_none_or_str(raw.get("expected_effect")),
                    contract_kind=contract_kind,
                ))
            if not steps:
                continue
            hid = stable_hash(("coordinate", snapshot.level_index, snapshot.step_index, idx, raw.get("claim"), cids), "hyp_")
            self.coordinate_test_queue.append(HypothesisItem(
                hypothesis_id=hid,
                source="coordinate_qwen",
                claim=str(raw.get("claim") or "coordinate hypothesis"),
                truth=TriTruth.UNKNOWN,
                relevance=Relevance.UNDECIDED,
                validity=Validity.UNCHECKED,
                progress=Progress.UNKNOWN,
                test_plan=tuple(steps),
                cursor=0,
                priority=_float(raw.get("probe_priority"), 0.0),
                confidence=_float(raw.get("confidence"), 0.0),
                expiry_step=(snapshot.step_index + int(raw.get("expiry_steps") or 8)),
                evidence_refs=(),
                suppression_signature=stable_hash((action_id, cids), "sup_"),
                created_state_signature=snapshot.semantic_state_signature,
            ))

    def _add_coordinate_plan(self, output: dict[str, Any], snapshot: ARGALiteSnapshot, config: V8Config) -> None:
        decision = str(output.get("decision") or "").upper()
        if decision != "PLAN":
            self.invalid_rejections.append({"reason": "bad_coordinate_plan_decision", "decision": output.get("decision")})
            return
        if any(key in output for key in ("x", "y")):
            self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "schema": output.get("schema_version")})
            return
        action_id = _none_or_str(output.get("coordinate_action_id"))
        if action_id is None or action_id not in set(snapshot.coordinate_action_ids):
            self.invalid_rejections.append({"reason": "unknown_coordinate_action_id", "action_id": action_id})
            return
        candidates = {c.candidate_id: c for c in snapshot.coordinate_targets}
        valid_obj = {o.object_id for o in snapshot.objects}
        valid_rel = {r.relation_id for r in snapshot.relations}
        sequence = output.get("candidate_sequence")
        if not isinstance(sequence, list) or not sequence:
            self.invalid_rejections.append({"reason": "empty_coordinate_plan", "raw": output})
            return
        steps: list[TestStep] = []
        seen_candidate_ids: set[str] = set()
        seen_locations_xy: set[tuple[int, int]] = set()
        for raw_step in sequence[: config.max_qwen_trajectory_steps]:
            if not isinstance(raw_step, dict):
                self.invalid_rejections.append({"reason": "bad_step_record", "raw": raw_step})
                return
            if any(key in raw_step for key in ("x", "y")):
                self.invalid_rejections.append({"reason": "raw_coordinates_rejected", "raw": raw_step})
                return
            cid = _none_or_str(raw_step.get("coordinate_candidate_id"))
            if cid is None or cid not in candidates:
                self.invalid_rejections.append({"reason": "invented_coordinate_candidate_id", "coordinate_candidate_id": cid})
                return
            if cid in seen_candidate_ids:
                continue
            seen_candidate_ids.add(cid)
            candidate = candidates[cid]
            location_xy = (candidate.x, candidate.y)
            if location_xy in seen_locations_xy:
                continue
            seen_locations_xy.add(location_xy)
            target_object_ids = _string_list(raw_step.get("target_object_ids"))
            target_relation_ids = _string_list(raw_step.get("target_relation_ids"))
            if not set(target_object_ids).issubset(valid_obj):
                self.invalid_rejections.append({"reason": "invented_step_object_id", "target_object_ids": target_object_ids})
                return
            if not set(target_relation_ids).issubset(valid_rel):
                self.invalid_rejections.append({"reason": "invented_step_relation_id", "target_relation_ids": target_relation_ids})
                return
            expected = _expected_observation_text(raw_step.get("expected_observation"))
            steps.append(TestStep(
                _expected_observation_type(raw_step.get("expected_observation")) or "coordinate_probe",
                action_id,
                target_object_id=(target_object_ids[0] if target_object_ids else candidate.object_id),
                target_relation_id=(target_relation_ids[0] if target_relation_ids else candidate.relation_id),
                target_object_ids=tuple(target_object_ids),
                target_relation_ids=tuple(target_relation_ids),
                coordinate_candidate_id=cid,
                expected_observation=expected,
                contract_kind=None,
            ))
        if not steps:
            self.invalid_rejections.append({"reason": "empty_coordinate_plan", "raw": output})
            return
        claim = _coordinate_plan_claim(output)
        hid = stable_hash(("coordinate", snapshot.level_index, snapshot.step_index, claim, [s.coordinate_candidate_id for s in steps]), "hyp_")
        self.coordinate_test_queue.append(HypothesisItem(
            hypothesis_id=hid,
            source="coordinate_qwen",
            claim=claim,
            truth=TriTruth.UNKNOWN,
            relevance=Relevance.UNDECIDED,
            validity=Validity.UNCHECKED,
            progress=Progress.UNKNOWN,
            test_plan=tuple(steps),
            cursor=0,
            priority=max(0.0, _float(output.get("confidence"), 0.0)),
            confidence=_float(output.get("confidence"), 0.0),
            expiry_step=None,
            evidence_refs=(),
            suppression_signature=stable_hash((action_id, [s.coordinate_candidate_id for s in steps]), "sup_"),
            created_state_signature=snapshot.semantic_state_signature,
        ))

    def next_candidate_action(self, snapshot: ARGALiteSnapshot, queue_name: str | None = None, *, dry_run: bool = False) -> CandidateAction | None:
        self._current_step = snapshot.step_index
        queues: list[list[HypothesisItem]]
        if queue_name == "confirmed":
            queues = [self.confirmed_rules]
        elif queue_name == "coordinate":
            queues = [self.coordinate_test_queue]
        elif queue_name == "semantic":
            queues = [self.semantic_test_queue]
        else:
            queues = [self.confirmed_rules, self.coordinate_test_queue, self.semantic_test_queue, self.fallback_exploration_queue]
        active = self._find(self._active_hypothesis_id)
        if active is not None and self._usable(active, snapshot.step_index):
            if not any(active in queue for queue in queues):
                return None
            active_step = active.next_step()
            if active_step is None or active_step.action_id not in snapshot.available_actions:
                if dry_run:
                    return None
                active.validity = Validity.INVALID
                self.invalid_rejections.append({
                    "reason": "active_trajectory_action_surface_diverged",
                    "hypothesis_id": active.hypothesis_id,
                    "action_id": active_step.action_id if active_step else None,
                })
                self._active_hypothesis_id = None
                self._purge_consumed_and_invalid(snapshot.step_index)
            else:
                queues = [[active]]
        elif self._active_hypothesis_id is not None:
            self._active_hypothesis_id = None
        for queue in queues:
            for item in sorted(queue, key=lambda h: (-h.priority, -h.confidence, h.hypothesis_id)):
                if not self._usable(item, snapshot.step_index):
                    continue
                step = item.next_step()
                if step is None or step.action_id is None or step.action_id not in snapshot.available_actions:
                    continue
                contract = self._binder.bind(step, snapshot, item.hypothesis_id)
                if contract is None:
                    if not dry_run:
                        item.validity = Validity.INVALID
                        self.invalid_rejections.append({"reason": "contract_binding_failed_or_target_stale", "hypothesis_id": item.hypothesis_id})
                    continue
                if step.coordinate_candidate_id:
                    cand = next((c for c in snapshot.coordinate_targets if c.candidate_id == step.coordinate_candidate_id), None)
                    if cand is None:
                        if not dry_run:
                            item.validity = Validity.INVALID
                            self.invalid_rejections.append({"reason": "coordinate_target_stale", "hypothesis_id": item.hypothesis_id, "candidate_id": step.coordinate_candidate_id})
                        continue
                    if not dry_run:
                        self._active_hypothesis_id = item.hypothesis_id
                        if item.trajectory_start_snapshot is None:
                            item.trajectory_start_snapshot = snapshot
                    return CandidateAction(step.action_id, x=cand.x, y=cand.y, coordinate_candidate_id=cand.candidate_id, hypothesis_id=item.hypothesis_id, reason=item.claim, source=item.source, verification_contract=contract)
                if not dry_run:
                    self._active_hypothesis_id = item.hypothesis_id
                    if item.trajectory_start_snapshot is None:
                        item.trajectory_start_snapshot = snapshot
                return CandidateAction(step.action_id, hypothesis_id=item.hypothesis_id, reason=item.claim, source=item.source, verification_contract=contract)
        if not dry_run:
            self._purge_consumed_and_invalid(snapshot.step_index)
        return None

    def reject_candidate(self, hypothesis_id: str | None, reason: str, snapshot: ARGALiteSnapshot) -> None:
        item = self._find(hypothesis_id)
        if item is None:
            return
        item.validity = Validity.INVALID
        if self._active_hypothesis_id == item.hypothesis_id:
            self._active_hypothesis_id = None
        self.invalid_rejections.append({"reason": reason, "hypothesis_id": hypothesis_id})
        if item not in self.rejected:
            self.rejected.append(item)
        self._purge_consumed_and_invalid(snapshot.step_index)
        self._sort()

    def update(self, judgment: Judgment, after_snapshot: ARGALiteSnapshot | None = None) -> TrajectoryEvaluation | None:
        item = self._find(judgment.hypothesis_id)
        if item is None:
            return None
        item.truth = judgment.truth
        item.relevance = judgment.relevance
        item.validity = judgment.validity
        item.progress = judgment.progress
        item.trajectory_judgments.append(judgment)
        item.executed_action_ids.append(judgment.action.action_id)
        mechanics_mismatch = judgment.mechanic_result is MechanicResult.MISMATCH or judgment.reason_code in {
            "target_object_not_displaced",
            "target_object_tracking_unavailable",
            "target_local_no_effect",
            "action_surface_unchanged",
            "typed_no_effect_observed",
            "game_over_after_action",
        }
        if item.has_next_step():
            item.cursor += 1
        trajectory_finished = mechanics_mismatch or not item.has_next_step() or judgment.terminal_delta
        evaluation = None
        if trajectory_finished and item.trajectory_start_snapshot is not None and after_snapshot is not None:
            evaluation = evaluate_trajectory(
                item.semantic_binding,
                item.trajectory_start_snapshot,
                after_snapshot,
                item.trajectory_judgments,
                item.executed_action_ids,
            )
            item.progress = evaluation.goal_progress
            if evaluation.reason_code == "trajectory_completed_level":
                item.truth = TriTruth.TRUE
                item.relevance = Relevance.RELEVANT
            elif evaluation.semantic_judgment.value == "FORBIDDEN":
                item.truth = TriTruth.FALSE
                item.relevance = Relevance.RELEVANT
            elif evaluation.semantic_judgment.value == "IRRELEVANT":
                item.truth = TriTruth.UNKNOWN
                item.relevance = Relevance.IRRELEVANT
        if mechanics_mismatch and "qwen" in item.source:
            item.validity = Validity.INVALID
            if item not in self.rejected:
                self.rejected.append(item)
            self.invalid_rejections.append({
                "reason": "active_trajectory_effect_mismatch",
                "hypothesis_id": item.hypothesis_id,
                "transition_reason": judgment.reason_code,
                "action_id": judgment.action.action_id,
            })
            if self._active_hypothesis_id == item.hypothesis_id:
                self._active_hypothesis_id = None
        if item.truth is TriTruth.TRUE and item.progress is Progress.POSITIVE:
            item.priority += 2.0
            item.confidence = min(1.0, item.confidence + 0.25)
            # A concrete plan is not a reusable mechanic. Keep only remaining steps.
            if item.has_next_step() and "qwen" not in item.source and item not in self.confirmed_rules:
                self.confirmed_rules.append(item)
        elif item.truth is TriTruth.FALSE or judgment.validity is Validity.INVALID:
            item.validity = Validity.INVALID
            if item not in self.rejected:
                self.rejected.append(item)
        elif judgment.relevance is Relevance.IRRELEVANT:
            item.priority -= 1.0
        else:
            item.priority -= 0.25
        if not item.has_next_step() and self._active_hypothesis_id == item.hypothesis_id:
            self._active_hypothesis_id = None
        self._purge_consumed_and_invalid(self._current_step)
        self._sort()
        if (
            trajectory_finished
            and evaluation is not None
            and item.proposal_batch_id
            and not judgment.terminal_delta
            and after_snapshot is not None
            and evaluation.reason_code != "trajectory_completed_level"
        ):
            alternatives = [
                candidate
                for candidate in self.semantic_test_queue
                if candidate.hypothesis_id != item.hypothesis_id
                and candidate.proposal_batch_id == item.proposal_batch_id
                and self._usable(candidate, after_snapshot.step_index)
            ]
            if alternatives:
                self._pending_alternative_reset = {
                    "level_index": after_snapshot.level_index,
                    "proposal_batch_id": item.proposal_batch_id,
                    "completed_hypothesis_id": item.hypothesis_id,
                    "remaining_hypothesis_ids": [candidate.hypothesis_id for candidate in alternatives],
                }
        return evaluation

    def current_questions(self, config: V8Config, snapshot: ARGALiteSnapshot | None = None) -> list[str]:
        questions: list[str] = []
        if snapshot is not None:
            executable = self.has_executable_candidate(snapshot)
        else:
            executable = self.has_valid_action_candidates()
        if not executable:
            questions.append("No executable grounded hypothesis remains; propose 1-3 hypotheses for what must be done to advance to the next level, each with a complete verifier-gated action sequence from the current state using only valid ids.")
        questions.extend([f"Unresolved: {h.claim}" for h in self.semantic_test_queue[: config.max_memory_notes_in_packet] if h.truth is TriTruth.UNKNOWN])
        return questions[: config.max_memory_notes_in_packet]

    def current_coordinate_questions(self, config: V8Config) -> list[str]:
        return [f"Coordinate unresolved: {h.claim}" for h in self.coordinate_test_queue[: config.max_memory_notes_in_packet]] or ["Rank existing candidate target ids for coordinate probing; never emit raw coordinates."]

    def _find(self, hypothesis_id: str | None) -> HypothesisItem | None:
        if not hypothesis_id:
            return None
        for item in self.confirmed_rules + self.semantic_test_queue + self.coordinate_test_queue + self.fallback_exploration_queue:
            if item.hypothesis_id == hypothesis_id:
                return item
        return None

    def _usable(self, item: HypothesisItem, current_step: int) -> bool:
        if item.expiry_step is not None and current_step > item.expiry_step:
            item.validity = Validity.INVALID
        return item.validity is not Validity.INVALID and item.has_next_step() and item.truth is not TriTruth.FALSE

    def _purge_consumed_and_invalid(self, current_step: int) -> None:
        self.semantic_test_queue = [h for h in self.semantic_test_queue if self._usable(h, current_step)]
        self.coordinate_test_queue = [h for h in self.coordinate_test_queue if self._usable(h, current_step)]
        self.fallback_exploration_queue = [h for h in self.fallback_exploration_queue if self._usable(h, current_step)]
        self.confirmed_rules = [h for h in self.confirmed_rules if self._usable(h, current_step)]
        if self._active_hypothesis_id is not None and self._find(self._active_hypothesis_id) is None:
            self._active_hypothesis_id = None

    def _sort(self) -> None:
        for queue in (self.confirmed_rules, self.semantic_test_queue, self.coordinate_test_queue, self.fallback_exploration_queue):
            queue.sort(key=lambda h: (-h.priority, -h.confidence, h.hypothesis_id))


def _none_or_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _string_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_items(value):
        text = _none_or_str(item)
        if text:
            out.append(text)
    return list(dict.fromkeys(out))


def _action_sequence(value: Any) -> list[str]:
    """Normalize an ordered trajectory without removing repeated actions."""
    out: list[str] = []
    for item in _as_items(value):
        text = _none_or_str(item)
        if text:
            out.append(text)
    return out


def _expanded_action_runs(value: Any) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    if not isinstance(value, list):
        return out
    for run in value:
        if not isinstance(run, dict):
            return []
        action_id = _none_or_str(run.get("action_id"))
        repeat = run.get("repeat")
        candidate_id = _none_or_str(run.get("coordinate_candidate_id"))
        if action_id is None or not isinstance(repeat, int) or isinstance(repeat, bool) or repeat <= 0:
            return []
        out.extend((action_id, candidate_id) for _ in range(repeat))
        if len(out) > 50:
            return []
    return out


def _v87_matching_failed_trajectory(raw: dict[str, Any], packet: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(packet, dict):
        return None
    current = _normalized_recorded_trajectory(raw.get("action_runs"))
    if not current:
        return None
    memory = packet.get("memory") or {}
    attempts = (
        ((memory.get("attempts") or {}).get("previous_failed_attempts") or [])
        or ((memory.get("level_attempts") or {}).get("previous_failed_attempts") or [])
    )
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        for rejection in reversed(attempt.get("rejected_proposals") or []):
            if not isinstance(rejection, dict):
                continue
            previous = _normalized_rejected_trajectory(rejection)
            if previous != current:
                continue
            return {
                "previous_attempt_index": attempt.get("attempt_index"),
                "previous_hypothesis_id": rejection.get("hypothesis_id"),
                "previous_outcome": "VERIFIER_REJECTED",
                "previous_rejection_reason": rejection.get("reason"),
                "trajectory": [
                    {
                        "action_id": action_id,
                        **({"coordinate_candidate_id": candidate_id} if candidate_id is not None else {}),
                    }
                    for action_id, candidate_id in current
                ],
            }
        for hypothesis in reversed(attempt.get("executed_hypotheses") or []):
            if not isinstance(hypothesis, dict):
                continue
            outcome = str(hypothesis.get("outcome") or "")
            if outcome == "LEVEL_PROGRESS_OBSERVED":
                continue
            recorded_runs = hypothesis.get("action_runs") or []
            sources = {
                str(source)
                for source in [
                    hypothesis.get("source"),
                    *(run.get("source") for run in recorded_runs if isinstance(run, dict)),
                ]
                if source
            }
            if sources and not any(source.startswith(("primary", "reserve")) for source in sources):
                continue
            previous = _normalized_recorded_trajectory(recorded_runs)
            if previous != current:
                continue
            return {
                "previous_attempt_index": attempt.get("attempt_index"),
                "previous_hypothesis_id": hypothesis.get("hypothesis_id"),
                "previous_outcome": outcome or "FAILED_WITHOUT_LEVEL_PROGRESS",
                "trajectory": [
                    {
                        "action_id": action_id,
                        **({"coordinate_candidate_id": candidate_id} if candidate_id is not None else {}),
                    }
                    for action_id, candidate_id in current
                ],
            }
    return None


def _normalized_recorded_trajectory(value: Any) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    if not isinstance(value, list):
        return out
    for run in value:
        if not isinstance(run, dict):
            return []
        action_id = _none_or_str(run.get("action_id"))
        count = run.get("repeat", run.get("count"))
        candidate_id = _none_or_str(run.get("coordinate_candidate_id"))
        if action_id is None or not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            return []
        out.extend((action_id, candidate_id) for _ in range(count))
        if len(out) > 50:
            return []
    return out


def _normalized_rejected_trajectory(value: dict[str, Any]) -> list[tuple[str, str | None]]:
    runs = _normalized_recorded_trajectory(value.get("action_runs"))
    if runs:
        return runs
    actions = value.get("actions")
    if not isinstance(actions, list) or not actions or len(actions) > 50:
        return []
    candidate_id = _none_or_str(value.get("coordinate_candidate_id"))
    out = []
    for action_id in actions:
        canonical = _none_or_str(action_id)
        if canonical is None:
            return []
        out.append((canonical, candidate_id))
    return out


def _contiguous_step_indices(raw_steps: list[Any]) -> bool:
    indices = []
    for item in raw_steps:
        if not isinstance(item, dict):
            return False
        try:
            indices.append(int(item.get("step_index")))
        except Exception:
            return False
    return indices == list(range(len(indices)))


def _expected_observation_type(value: Any) -> str | None:
    if isinstance(value, dict):
        text = _none_or_str(value.get("type"))
        return text.lower() if text else None
    return None


def _expected_observation_text(value: Any) -> str | None:
    if not isinstance(value, dict):
        return _none_or_str(value)
    obs_type = _none_or_str(value.get("type"))
    description = _none_or_str(value.get("description"))
    targets = _string_list(value.get("target_ids"))
    parts = []
    if obs_type:
        parts.append(f"expected_type={obs_type}")
    if targets:
        parts.append(f"target_ids={','.join(targets)}")
    if description:
        parts.append(description)
    return "; ".join(parts) if parts else None


def _semantic_plan_claim(raw: dict[str, Any]) -> str:
    goal = raw.get("goal") if isinstance(raw.get("goal"), dict) else {}
    operation = _short_text(goal.get("operation"), 40)
    description = _short_text(goal.get("description"), 220)
    completion = _short_text(raw.get("completion_criterion"), 140)
    parts = [part for part in (f"operation: {operation}" if operation else "", description, f"completion: {completion}" if completion else "") if part]
    return "; ".join(parts)[:520] or "qwen semantic plan"


def _semantic_hypothesis_claim(raw: dict[str, Any]) -> str:
    goal = raw.get("goal") if isinstance(raw.get("goal"), dict) else {}
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    family = _short_text(raw.get("family"), 60)
    operation = _short_text(goal.get("operation"), 40)
    description = _short_text(goal.get("description"), 180)
    summary = _short_text(evidence.get("summary"), 140)
    uncertainty = _short_text(raw.get("main_uncertainty"), 120)
    completion = _short_text(raw.get("completion_criterion"), 120)
    parts = [
        f"family: {family}" if family else "",
        f"operation: {operation}" if operation else "",
        description,
        f"evidence: {summary}" if summary else "",
        f"uncertainty: {uncertainty}" if uncertainty else "",
        f"completion: {completion}" if completion else "",
    ]
    return "; ".join(part for part in parts if part)[:620] or "qwen semantic hypothesis"


def _v86_hypothesis_claim(raw: dict[str, Any]) -> str:
    family = _short_text(raw.get("family"), 60)
    basis = _short_text(raw.get("basis"), 160)
    uncertainty = _short_text(raw.get("uncertainty"), 120)
    status = _short_text(raw.get("status"), 40)
    parts = [
        f"family: {family}" if family else "",
        f"basis: {basis}" if basis else "",
        f"status: {status}" if status else "",
        f"uncertainty: {uncertainty}" if uncertainty else "",
    ]
    return "; ".join(part for part in parts if part)[:420] or "qwen v8.6 semantic hypothesis"


def _v87_hypothesis_claim(raw: dict[str, Any]) -> str:
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    parts = [
        f"family: {_short_text(raw.get('family'), 48)}",
        f"objective: {_short_text(objective.get('kind'), 40)}",
        _short_text(objective.get("description"), 180),
        f"basis: {_short_text(raw.get('basis'), 160)}",
        f"status: {_short_text(raw.get('status'), 40)}",
    ]
    return "; ".join(part for part in parts if part and not part.endswith(": "))[:560] or "qwen v8.7 semantic trajectory"


def _layered_action_diffs(packet: dict[str, Any], action_id: str) -> list[dict[str, Any]]:
    return [
        item
        for item in packet.get("action_diffs") or []
        if isinstance(item, dict) and str(item.get("action_id") or "") == str(action_id)
    ]


def _layered_surface_changed(diff: dict[str, Any]) -> bool:
    before = set(str(value) for value in (diff.get("before") or {}).get("available_action_ids") or [])
    after = set(str(value) for value in (diff.get("after") or {}).get("available_action_ids") or [])
    return before != after


def _layered_object_changed(change: dict[str, Any]) -> bool:
    return (change.get("before") or {}) != (change.get("after") or {}) or str(change.get("lifecycle") or "") != "persisted"


def _layered_centroid_changed(change: dict[str, Any]) -> bool:
    before = (change.get("before") or {}).get("centroid_xy")
    after = (change.get("after") or {}).get("centroid_xy")
    return isinstance(before, list) and isinstance(after, list) and before != after


def _layered_diff_has_effect(diff: dict[str, Any]) -> bool:
    changed_cells = int((diff.get("pixel_diff") or {}).get("changed_cell_count") or 0)
    object_change = any(
        _layered_object_changed(item)
        for item in diff.get("object_changes") or []
        if isinstance(item, dict)
    )
    result = diff.get("level_result") or {}
    return bool(
        changed_cells
        or object_change
        or _layered_surface_changed(diff)
        or result.get("levels_completed_delta")
        or result.get("level_index_delta")
        or result.get("game_over")
        or result.get("terminal")
    )


def _layered_historical_effect_action_ids(packet: dict[str, Any]) -> set[str]:
    memory = packet.get("memory") if isinstance(packet.get("memory"), dict) else {}
    action_ids = {
        str(item.get("action_id"))
        for item in memory.get("confirmed_effects") or []
        if isinstance(item, dict)
        and item.get("action_id")
        and str(item.get("observed_outcome") or "").lower() not in {"", "no_effect", "none", "unchanged"}
    }
    for level in memory.get("completed_levels") or []:
        if not isinstance(level, dict) or str(level.get("outcome") or "") != "LEVEL_COMPLETED":
            continue
        result = level.get("verified_result") if isinstance(level.get("verified_result"), dict) else {}
        if str(result.get("mechanic_result") or "") != "MATCH":
            continue
        for run in level.get("verified_action_runs") or []:
            if isinstance(run, dict) and run.get("action_id"):
                action_ids.add(str(run.get("action_id")))
    return action_ids


def _layered_action_has_effect_evidence(packet: dict[str, Any], action_id: str) -> bool:
    current_diffs = _layered_action_diffs(packet, action_id)
    if current_diffs:
        # Current-level research overrides historical mechanics for this action.
        return any(_layered_diff_has_effect(diff) for diff in current_diffs)
    return action_id in _layered_historical_effect_action_ids(packet)


def _v87_layered_trajectory_reachability_issue(raw: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any] | None:
    actions = [action_id for action_id, _candidate_id in _expanded_action_runs(raw.get("action_runs"))] or _action_sequence(raw.get("actions"))
    if not actions:
        return {"detail": "empty_trajectory"}
    action_space = packet.get("action_space") if isinstance(packet.get("action_space"), dict) else {}
    current_surface = set(str(value) for value in action_space.get("current_available_action_ids") or [])
    allowed = set(str(value) for value in (packet.get("execution_constraints") or {}).get("allowed_action_ids") or [])
    if not allowed:
        allowed = set(str(value) for value in action_space.get("possible_action_ids") or current_surface)
    for action_index, action_id in enumerate(actions, start=1):
        if action_id not in allowed:
            return {
                "action_id": action_id,
                "trajectory_step_index_1_based": action_index,
                "available_action_ids_before_step": sorted(current_surface),
                "detail": "action_not_allowed_by_execution_constraints",
            }
        if current_surface and action_id not in current_surface:
            enabling_actions = []
            for diff in packet.get("action_diffs") or []:
                if not isinstance(diff, dict):
                    continue
                before_surface = set(
                    str(value)
                    for value in (diff.get("before") or {}).get("available_action_ids") or []
                )
                after_surface = set(
                    str(value)
                    for value in (diff.get("after") or {}).get("available_action_ids") or []
                )
                transition_action = str(diff.get("action_id") or "")
                if (
                    before_surface == current_surface
                    and action_id in after_surface
                    and transition_action in current_surface
                    and transition_action in allowed
                ):
                    enabling_actions.append(transition_action)
            return {
                "action_id": action_id,
                "trajectory_step_index_1_based": action_index,
                "available_action_ids_before_step": sorted(current_surface),
                "observed_enabling_action_ids": list(dict.fromkeys(enabling_actions)),
                "detail": "action_unavailable_before_trajectory_establishes_required_surface",
            }
        contextual_diffs = [
            diff
            for diff in _layered_action_diffs(packet, action_id)
            if set(str(value) for value in (diff.get("before") or {}).get("available_action_ids") or []) == current_surface
        ]
        if contextual_diffs:
            current_surface = set(
                str(value)
                for value in (contextual_diffs[-1].get("after") or {}).get("available_action_ids") or []
            )
    return None


def _v87_layered_trajectory_actions_reachable(raw: dict[str, Any], packet: dict[str, Any]) -> bool:
    return _v87_layered_trajectory_reachability_issue(raw, packet) is None


def _v87_trajectory_uses_observed_effect(raw: dict[str, Any], packet: dict[str, Any] | None) -> bool:
    if packet is None:
        return True
    # Coordinate effects are target-dependent. Evidence from one probe cannot
    # invalidate the same action at another verifier-approved candidate.
    if _v87_has_bound_coordinate_action(raw, packet):
        return True
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    source_ids = set(_string_list(objective.get("source_objects")))
    actions = [action_id for action_id, _candidate_id in _expanded_action_runs(raw.get("action_runs"))] or _action_sequence(raw.get("actions"))
    if packet.get("schema_version") == "v8.8.layered_observation":
        # Qwen owns the semantic object assignment. The verifier only requires
        # factual effect evidence for every distinct action. A current-level
        # diff takes precedence; otherwise game-scoped confirmed mechanics and
        # verified completed-level trajectories are admissible evidence.
        return bool(actions) and all(
            _layered_action_has_effect_evidence(packet, action_id)
            for action_id in dict.fromkeys(actions)
        )
    action_models = ((packet.get("action_model") or {}).get("actions") or {}) if isinstance(packet, dict) else {}
    if str(objective.get("kind") or "") == "surface_change":
        return any(
            isinstance(action_models.get(action_id), dict)
            and bool(action_models[action_id].get("surface_added") or action_models[action_id].get("surface_removed"))
            for action_id in actions
        )
    saw_observed_action = False
    for action_id in actions:
        action = action_models.get(action_id)
        if not isinstance(action, dict) or action.get("observed") is False:
            continue
        saw_observed_action = True
        effects = [effect for effect in action.get("effects") or [] if isinstance(effect, dict)]
        if source_ids and any(str(effect.get("object_id")) in source_ids for effect in effects):
            return True
        if not source_ids and effects:
            return True
    return saw_observed_action and str(objective.get("kind") or "") == "other"


def _v87_trajectory_respects_control_context(raw: dict[str, Any], packet: dict[str, Any] | None) -> bool:
    if not isinstance(packet, dict):
        return True
    if packet.get("schema_version") == "v8.8.layered_observation":
        return _v87_layered_trajectory_actions_reachable(raw, packet)
    candidates = ((packet.get("scene") or {}).get("control_state_transition_candidates") or [])
    current = next((
        item for item in reversed(candidates)
        if isinstance(item, dict)
        and item.get("type") == "SUPPORTED_CONTROL_GROUP_SWITCH"
        and item.get("current_inferred_control_group_id")
    ), None)
    if current is None:
        return True
    before_group = str(current.get("before_control_group_id"))
    after_group = str(current.get("after_control_group_id"))
    active_group = str(current.get("current_inferred_control_group_id"))
    trigger_action_id = str(current.get("trigger_action_id"))
    action_models = ((packet.get("action_model") or {}).get("actions") or {})
    for action_id in _action_sequence(raw.get("actions")):
        if action_id == trigger_action_id:
            if active_group == after_group:
                active_group = before_group
            elif active_group == before_group:
                active_group = after_group
            else:
                return False
            continue
        action = action_models.get(action_id)
        if not isinstance(action, dict):
            continue
        observed_groups = set(str(value) for value in action.get("observed_control_group_ids") or [])
        if observed_groups and active_group not in observed_groups:
            return False
    return True


def _v87_contract_kind(raw: dict[str, Any], action_id: str, packet: dict[str, Any] | None) -> str:
    if packet is None:
        return "ACTION_EFFECT_DISCOVERY"
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    sources = set(_string_list(objective.get("source_objects")))
    if packet.get("schema_version") == "v8.8.layered_observation":
        diffs = _layered_action_diffs(packet, action_id)
        if str(objective.get("kind") or "") == "surface_change" and any(_layered_surface_changed(diff) for diff in diffs):
            return "ACTION_SURFACE_CHANGE"
        relevant = [
            change
            for diff in diffs
            for change in diff.get("object_changes") or []
            if isinstance(change, dict) and (not sources or str(change.get("object_id")) in sources)
        ]
        if any(_layered_centroid_changed(change) for change in relevant):
            return "OBJECT_DISPLACEMENT"
        if relevant and sources:
            return "LOCAL_TARGET_CHANGE"
        return "ACTION_EFFECT_DISCOVERY"
    action = ((packet.get("action_model") or {}).get("actions") or {}).get(action_id) or {}
    if str(objective.get("kind") or "") == "surface_change" and (action.get("surface_added") or action.get("surface_removed")):
        return "ACTION_SURFACE_CHANGE"
    relevant = [
        effect for effect in action.get("effects") or []
        if isinstance(effect, dict) and (not sources or str(effect.get("object_id")) in sources)
    ]
    if any(str(effect.get("kind")) == "translation" for effect in relevant):
        return "OBJECT_DISPLACEMENT"
    if relevant and sources:
        return "LOCAL_TARGET_CHANGE"
    return "ACTION_EFFECT_DISCOVERY"


def _v87_expected_effect_text(raw: dict[str, Any], action_id: str, packet: dict[str, Any] | None) -> str:
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    sources = set(_string_list(objective.get("source_objects")))
    if isinstance(packet, dict) and packet.get("schema_version") == "v8.8.layered_observation":
        summaries = []
        for diff in _layered_action_diffs(packet, action_id)[-2:]:
            changed_cells = int((diff.get("pixel_diff") or {}).get("changed_cell_count") or 0)
            object_ids = [
                str(change.get("object_id"))
                for change in diff.get("object_changes") or []
                if isinstance(change, dict) and (not sources or str(change.get("object_id")) in sources)
            ]
            surface = "available_actions_changed" if _layered_surface_changed(diff) else "available_actions_unchanged"
            summaries.append(f"cells={changed_cells};objects={','.join(object_ids) or 'none'};{surface}")
        return f"observed factual diff for {action_id}; " + (" | ".join(summaries) if summaries else "no matching diff")
    action = (((packet or {}).get("action_model") or {}).get("actions") or {}).get(action_id) or {}
    effects = [
        effect for effect in action.get("effects") or []
        if isinstance(effect, dict) and (not sources or str(effect.get("object_id")) in sources)
    ]
    compact = []
    for effect in effects[:6]:
        compact.append(
            f"{effect.get('object_id')}:{effect.get('kind')}:{effect.get('delta_xy') or effect.get('direction') or ''}"
        )
    return f"verifier-observed effect for {action_id}; " + (", ".join(compact) if compact else "action-linked change")


def _v87_priority(raw: dict[str, Any], confidence: float, packet: dict[str, Any] | None, index: int) -> float:
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    source_ids = set(_string_list(objective.get("source_objects")))
    reference_ids = set(_string_list(objective.get("reference_objects")))
    score = max(0.0, confidence) * 5.0
    if raw.get("status") == "complete_candidate":
        score += 0.75
    if source_ids:
        score += 0.5
    scene = (packet or {}).get("scene") or {}
    for fact in scene.get("priority_facts_not_goals") or []:
        if not isinstance(fact, dict):
            continue
        fact_ids = set(_string_list(fact.get("object_ids")))
        if source_ids and reference_ids and source_ids.issubset(fact_ids) and reference_ids.issubset(fact_ids):
            score += 3.0
            break
    score += max(0, 2 - index) * 0.1
    return score


def _v86_expected_observation_text(raw: dict[str, Any], expected_type: str, objects: list[str], relations: list[str]) -> str:
    parts = [f"expected_type={expected_type}"]
    if objects:
        parts.append(f"tracked_object_ids={','.join(objects)}")
    if relations:
        parts.append(f"tracked_relation_ids={','.join(relations)}")
    basis = _short_text(raw.get("basis"), 120)
    if basis:
        parts.append(basis)
    status = _short_text(raw.get("status"), 60)
    if status:
        parts.append(f"status={status}")
    return "; ".join(parts)


def _unsupported_effect_claim(raw: dict[str, Any]) -> bool:
    text = " ".join(str(raw.get(key) or "") for key in ("basis", "uncertainty", "family")).lower()
    unsupported_tokens = ("rotate", "rotation", "recolor", "colour", "color swap", "reshape", "shape change", "geometry group change", "change geometry")
    if not any(token in text for token in unsupported_tokens):
        return False
    return str(raw.get("family") or "") not in {"recolor_or_state_match", "action_surface_change"}


def _v86_trajectory_has_progress(raw: dict[str, Any], packet: dict[str, Any] | None) -> bool:
    if packet is None:
        return True
    selected_objects = set(_string_list(raw.get("objects")))
    selected_relations = set(_string_list(raw.get("relations")))
    actions = [action_id for action_id, _candidate_id in _expanded_action_runs(raw.get("action_runs"))] or _action_sequence(raw.get("actions"))
    if not actions:
        return False
    control_actions = ((packet.get("action_model") or packet.get("control_model") or {}).get("actions") or {}) if isinstance(packet, dict) else {}
    scene_relations = ((packet.get("scene") or {}).get("relations") or []) if isinstance(packet, dict) else []
    relations = {item.get("id"): item for item in (scene_relations or packet.get("neutral_relations") or packet.get("derived_features", {}).get("relations") or []) if isinstance(item, dict)}
    relation_objects: set[str] = set()
    for relation_id in selected_relations:
        relation = relations.get(relation_id)
        if relation is not None:
            relation_objects.update(str(v) for v in relation.get("object_ids", []) if v is not None)
    relevant_objects = selected_objects | relation_objects
    family = str(raw.get("family") or "")
    net_motion: dict[str, list[float]] = {object_id: [0.0, 0.0] for object_id in relevant_objects}
    saw_relevant_effect = False
    saw_nontranslation_effect = False
    saw_surface_change = False
    for action_id in actions:
        action = control_actions.get(action_id)
        if not isinstance(action, dict):
            continue
        if action.get("surface_added") or action.get("surface_removed"):
            saw_surface_change = True
        effects = action.get("effects")
        if isinstance(effects, list):
            for effect in effects:
                if not isinstance(effect, dict):
                    continue
                object_id = str(effect.get("object_id"))
                if relevant_objects and object_id not in relevant_objects:
                    continue
                delta_xy = effect.get("delta_xy")
                saw_relevant_effect = True
                effect_kind = str(effect.get("kind") or "")
                if effect_kind and effect_kind != "translation":
                    saw_nontranslation_effect = True
                if _nonzero_delta(delta_xy):
                    dx, dy = _delta_xy_pair(delta_xy)
                    net_motion.setdefault(object_id, [0.0, 0.0])
                    net_motion[object_id][0] += dx
                    net_motion[object_id][1] += dy
        motions = action.get("motions_xy")
        if isinstance(motions, dict):
            for object_id, delta_xy in motions.items():
                object_key = str(object_id)
                if object_key not in relevant_objects or not _nonzero_delta(delta_xy):
                    continue
                if isinstance(effects, list) and any(isinstance(effect, dict) and str(effect.get("object_id")) == object_key for effect in effects):
                    continue
                saw_relevant_effect = True
                dx, dy = _delta_xy_pair(delta_xy)
                net_motion.setdefault(object_key, [0.0, 0.0])
                net_motion[object_key][0] += dx
                net_motion[object_key][1] += dy
    if not saw_relevant_effect:
        return saw_surface_change and family == "action_surface_change"
    if saw_nontranslation_effect or saw_surface_change:
        return True
    if family in {"action_surface_change", "recolor_or_state_match"}:
        return True
    return any(_nonzero_delta(delta_xy) for delta_xy in net_motion.values())


def _v87_trajectory_has_progress(raw: dict[str, Any], packet: dict[str, Any] | None) -> bool:
    if packet is None:
        return True
    if _v87_has_bound_coordinate_action(raw, packet):
        return True
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    involved_ids = set(_string_list(objective.get("source_objects"))) | set(_string_list(objective.get("reference_objects")))
    actions = [action_id for action_id, _candidate_id in _expanded_action_runs(raw.get("action_runs"))] or _action_sequence(raw.get("actions"))
    if packet.get("schema_version") == "v8.8.layered_observation":
        return bool(actions) and any(
            _layered_action_has_effect_evidence(packet, action_id)
            for action_id in dict.fromkeys(actions)
        )
    action_models = ((packet.get("action_model") or {}).get("actions") or {}) if isinstance(packet, dict) else {}
    objective_kind = str(objective.get("kind") or "other")
    if objective_kind == "surface_change":
        return any(
            isinstance(action_models.get(action_id), dict)
            and bool(action_models[action_id].get("surface_added") or action_models[action_id].get("surface_removed"))
            for action_id in actions
        )
    net_motion: dict[str, list[float]] = {object_id: [0.0, 0.0] for object_id in involved_ids}
    saw_relevant_effect = False
    saw_nontranslation_effect = False
    saw_surface_change = False
    for action_id in actions:
        action = action_models.get(action_id)
        if not isinstance(action, dict):
            continue
        if action.get("surface_added") or action.get("surface_removed"):
            saw_surface_change = True
        for effect in action.get("effects") or []:
            if not isinstance(effect, dict):
                continue
            object_id = str(effect.get("object_id"))
            if involved_ids and object_id not in involved_ids:
                continue
            saw_relevant_effect = True
            effect_kind = str(effect.get("kind") or "")
            if effect_kind and effect_kind != "translation":
                saw_nontranslation_effect = True
            delta_xy = effect.get("delta_xy")
            if _nonzero_delta(delta_xy):
                dx, dy = _delta_xy_pair(delta_xy)
                net_motion.setdefault(object_id, [0.0, 0.0])
                net_motion[object_id][0] += dx
                net_motion[object_id][1] += dy
    if saw_nontranslation_effect or saw_surface_change:
        return True
    family = str(raw.get("family") or "other")
    if objective_kind in {"select_or_activate", "pattern_or_state", "surface_change", "other"}:
        return saw_relevant_effect
    if family in {"interaction_sequence", "pattern_transformation", "action_surface_change"}:
        return saw_relevant_effect
    return any(_nonzero_delta(delta_xy) for delta_xy in net_motion.values())


def _v87_has_bound_coordinate_action(raw: dict[str, Any], packet: dict[str, Any] | None) -> bool:
    if not isinstance(packet, dict):
        return False
    coordinate_action_ids = {
        str(item.get("id"))
        for item in (
            ((packet.get("action_space") or {}).get("actions") or [])
            or ((packet.get("action_surface") or {}).get("actions") or [])
        )
        if isinstance(item, dict) and str(item.get("kind") or "") == "coordinate"
    }
    allowed_candidate_ids = {
        str(candidate_id)
        for candidate_id in ((packet.get("execution_constraints") or {}).get("allowed_coordinate_candidate_ids") or [])
    }
    if not coordinate_action_ids or not allowed_candidate_ids:
        return False
    return any(
        isinstance(run, dict)
        and str(run.get("action_id") or "") in coordinate_action_ids
        and str(run.get("coordinate_candidate_id") or "") in allowed_candidate_ids
        for run in (raw.get("action_runs") or [])
    )


def _v87_has_unjustified_immediate_inverse_pair(raw: dict[str, Any], packet: dict[str, Any] | None) -> bool:
    if packet is None:
        return False
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    family = str(raw.get("family") or "other")
    objective_kind = str(objective.get("kind") or "other")
    if family in {"interaction_sequence", "pattern_transformation", "action_surface_change"}:
        return False
    if objective_kind in {"select_or_activate", "pattern_or_state", "surface_change", "other"}:
        return False
    source_ids = set(_string_list(objective.get("source_objects")))
    if not source_ids:
        return False
    action_models = ((packet.get("action_model") or {}).get("actions") or {}) if isinstance(packet, dict) else {}
    actions = _action_sequence(raw.get("actions"))

    def pure_vectors(action_id: str) -> dict[str, tuple[float, float]] | None:
        action = action_models.get(action_id)
        if not isinstance(action, dict):
            return None
        if action.get("surface_added") or action.get("surface_removed"):
            return None
        if action.get("current_control_context_status") == "SUPPORTED_CONTROL_GROUP_SWITCH_TRIGGER":
            return None
        vectors: dict[str, tuple[float, float]] = {}
        for effect in action.get("effects") or []:
            if not isinstance(effect, dict):
                continue
            object_id = str(effect.get("object_id"))
            if object_id not in source_ids:
                continue
            if str(effect.get("kind") or "") != "translation":
                return None
            delta_xy = effect.get("delta_xy")
            if _nonzero_delta(delta_xy):
                vectors[object_id] = _delta_xy_pair(delta_xy)
        return vectors or None

    for first_id, second_id in zip(actions, actions[1:]):
        first = pure_vectors(first_id)
        second = pure_vectors(second_id)
        if not first or not second or set(first) != set(second):
            continue
        if all(abs(first[object_id][0] + second[object_id][0]) <= 1e-9 and abs(first[object_id][1] + second[object_id][1]) <= 1e-9 for object_id in first):
            return True
    return False


def _v87_incomplete_exact_correspondence(raw: dict[str, Any], packet: dict[str, Any] | None) -> dict[str, Any] | None:
    if packet is None or str(raw.get("status") or "") != "complete_candidate":
        return None
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    if str(objective.get("kind") or "") != "match_or_overlap":
        return None
    sources = _string_list(objective.get("source_objects"))
    references = _string_list(objective.get("reference_objects"))
    if len(sources) != 1 or len(references) != 1:
        return None
    source, reference = sources[0], references[0]
    facts = ((packet.get("scene") or {}).get("priority_facts_not_goals") or []) if isinstance(packet, dict) else []
    fact = next((
        item for item in facts
        if isinstance(item, dict)
        and item.get("type") == "MOVABLE_REFERENCE_EXACT_GEOMETRY_CORRESPONDENCE"
        and str(item.get("movable_object_id")) == source
        and str(item.get("reference_object_id")) == reference
    ), None)
    if fact is None:
        return None
    required_delta = fact.get("source_to_reference_delta_xy")
    if not isinstance(required_delta, (list, tuple)) or len(required_delta) < 2:
        return None
    try:
        required_x, required_y = float(required_delta[0]), float(required_delta[1])
    except (TypeError, ValueError):
        return None
    action_models = ((packet.get("action_model") or {}).get("actions") or {}) if isinstance(packet, dict) else {}
    cumulative_x = 0.0
    cumulative_y = 0.0
    actions = _action_sequence(raw.get("actions"))
    for action_id in actions:
        action = action_models.get(action_id)
        effects = action.get("effects") if isinstance(action, dict) else []
        translation = next((
            effect for effect in (effects or [])
            if isinstance(effect, dict)
            and effect.get("kind") == "translation"
            and str(effect.get("object_id")) == source
        ), None)
        delta_xy = translation.get("delta_xy") if isinstance(translation, dict) else None
        if not isinstance(delta_xy, (list, tuple)) or len(delta_xy) < 2:
            surface_changed = bool(
                isinstance(action, dict)
                and (action.get("surface_added") or action.get("surface_removed"))
            )
            switches_control = bool(
                isinstance(action, dict)
                and action.get("current_control_context_status") == "SUPPORTED_CONTROL_GROUP_SWITCH_TRIGGER"
            )
            if surface_changed or switches_control:
                continue
            return None
        try:
            cumulative_x += float(delta_xy[0])
            cumulative_y += float(delta_xy[1])
        except (TypeError, ValueError):
            return None
    residual_x = required_x - cumulative_x
    residual_y = required_y - cumulative_y
    if residual_x * residual_x + residual_y * residual_y <= 1e-9:
        return None
    return {
        "required_delta_xy": [required_x, required_y],
        "planned_delta_xy": [cumulative_x, cumulative_y],
        "residual_delta_xy": [residual_x, residual_y],
    }


def _v86_priority(raw: dict[str, Any], confidence: float, packet: dict[str, Any] | None, index: int) -> float:
    score = max(0.0, confidence)
    if raw.get("relations"):
        score += 0.2
    if _v86_trajectory_has_progress(raw, packet):
        score += 0.2
    score += max(0, 2 - index) * 0.03
    return score


def _v85_relation_targets(
    raw: dict[str, Any],
    snapshot: ARGALiteSnapshot,
    goal_objects: list[str],
    goal_relations: list[str],
    tracked_relations: list[str],
) -> list[str] | None:
    family = _none_or_str(raw.get("family"))
    if family != "same_shape_alignment":
        return tracked_relations
    relation_ids = list(dict.fromkeys([*goal_relations, *tracked_relations]))
    exact_relation_ids = [relation_id for relation_id in relation_ids if _is_exact_shape_relation(snapshot, relation_id)]
    if exact_relation_ids:
        return exact_relation_ids
    inferred = _exact_shape_relation_ids_for_objects(snapshot, goal_objects)
    if inferred:
        return inferred
    return None


def _v85_tracked_objects(raw: dict[str, Any], snapshot: ARGALiteSnapshot, tracked_objects: list[str], tracked_relations: list[str]) -> list[str]:
    family = _none_or_str(raw.get("family"))
    if family != "same_shape_alignment":
        return tracked_objects
    exact_endpoints = _relation_endpoint_object_ids(snapshot, tracked_relations)
    return exact_endpoints or tracked_objects


def _is_exact_shape_relation(snapshot: ARGALiteSnapshot, relation_id: str) -> bool:
    relation = _snapshot_relation_by_id(snapshot, relation_id)
    return relation is not None and getattr(relation, "relation_type", None) in {"same_shape", "translated_shape", "unique_symbol_pair"}


def _relation_endpoint_object_ids(snapshot: ARGALiteSnapshot, relation_ids: list[str]) -> list[str]:
    out: list[str] = []
    for relation_id in relation_ids:
        relation = _snapshot_relation_by_id(snapshot, relation_id)
        if relation is None:
            continue
        for object_id in (getattr(relation, "a", None), getattr(relation, "b", None)):
            text = _none_or_str(object_id)
            if text:
                out.append(text)
    return list(dict.fromkeys(out))


def _exact_shape_relation_ids_for_objects(snapshot: ARGALiteSnapshot, object_ids: list[str]) -> list[str]:
    allowed = set(object_ids)
    if len(allowed) < 2:
        return []
    out = []
    for relation in getattr(snapshot, "relations", ()) or ():
        if getattr(relation, "relation_type", None) not in {"same_shape", "translated_shape", "unique_symbol_pair"}:
            continue
        if getattr(relation, "a", None) in allowed and getattr(relation, "b", None) in allowed:
            out.append(str(relation.relation_id))
    return list(dict.fromkeys(out))


def _snapshot_relation_by_id(snapshot: ARGALiteSnapshot, relation_id: str) -> Any | None:
    for relation in getattr(snapshot, "relations", ()) or ():
        if getattr(relation, "relation_id", None) == relation_id:
            return relation
    return None


def _v85_expected_observation_text(raw: dict[str, Any], tracked_objects: list[str], tracked_relations: list[str]) -> str:
    expected = raw.get("expected_progress") if isinstance(raw.get("expected_progress"), dict) else {}
    description = _short_text(expected.get("description"), 180)
    if tracked_relations:
        obs_type = "relation_improvement"
    elif tracked_objects:
        obs_type = "object_move"
    else:
        obs_type = "target_change"
    parts = [f"expected_type={obs_type}"]
    if tracked_objects:
        parts.append(f"tracked_object_ids={','.join(tracked_objects)}")
    if tracked_relations:
        parts.append(f"tracked_relation_ids={','.join(tracked_relations)}")
    if description:
        parts.append(description)
    status = _short_text(raw.get("trajectory_status"), 80)
    if status:
        parts.append(f"trajectory_status={status}")
    return "; ".join(parts)


def _expected_observation_type_from_text(text: str | None) -> str | None:
    if not text:
        return None
    marker = "expected_type="
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1]
    token = tail.split(";", 1)[0].strip().lower()
    return token or None


def _coordinate_plan_claim(raw: dict[str, Any]) -> str:
    mechanism = _short_text(raw.get("mechanism_hypothesis"), 240)
    completion = _short_text(raw.get("completion_criterion"), 180)
    action_id = _short_text(raw.get("coordinate_action_id"), 80)
    return "; ".join(part for part in (mechanism, f"coordinate action: {action_id}" if action_id else "", completion) if part) or "qwen coordinate plan"


def _semantic_target_object_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in _as_items(raw.get("target_object_ids")):
        text = _none_or_str(item)
        if text:
            values.append(text)
    object_goal = raw.get("object_goal")
    if isinstance(object_goal, dict):
        for key in ("source_object_id", "target_object_id"):
            text = _none_or_str(object_goal.get(key))
            if text:
                values.append(text)
    for item in _as_items(raw.get("object_goals")):
        if not isinstance(item, dict):
            continue
        for key in ("source_object_id", "target_object_id"):
            text = _none_or_str(item.get(key))
            if text:
                values.append(text)
    return tuple(dict.fromkeys(values))


def _semantic_target_relation_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in _as_items(raw.get("target_relation_ids")):
        text = _none_or_str(item)
        if text:
            values.append(text)
    object_goal = raw.get("object_goal")
    if isinstance(object_goal, dict):
        text = _none_or_str(object_goal.get("target_relation_id"))
        if text:
            values.append(text)
    for item in _as_items(raw.get("object_goals")):
        if not isinstance(item, dict):
            continue
        text = _none_or_str(item.get("target_relation_id"))
        if text:
            values.append(text)
    return tuple(dict.fromkeys(values))


def _semantic_claim(raw: dict[str, Any]) -> str:
    claim = _short_text(raw.get("claim"), 220)
    goal = _short_text(raw.get("level_transition_goal"), 220)
    mechanism = _short_text(raw.get("mechanism_hypothesis"), 180)
    completion = _short_text(raw.get("completion_criterion"), 140)
    object_goal = raw.get("object_goal")
    object_summary = ""
    if isinstance(object_goal, dict):
        op = _short_text(object_goal.get("operation"), 50)
        source = _short_text(object_goal.get("source_object_id"), 80)
        target = _short_text(object_goal.get("target_object_id") or object_goal.get("target_relation_id"), 80)
        desired = _short_text(object_goal.get("desired_relation"), 100)
        object_summary = " ".join(part for part in (op, source, "->" if source or target else "", target, desired) if part)
    object_goals = []
    for item in _as_items(raw.get("object_goals"))[:6]:
        if not isinstance(item, dict):
            continue
        stage = _short_text(item.get("stage"), 20)
        op = _short_text(item.get("operation"), 30)
        source = _short_text(item.get("source_object_id"), 50)
        target = _short_text(item.get("target_object_id") or item.get("target_relation_id"), 50)
        if op or source or target:
            object_goals.append(" ".join(part for part in (stage, op, source, "->" if source or target else "", target) if part))
    multi_summary = " | ".join(object_goals)
    parts = [part for part in (claim, f"goal: {goal}" if goal else "", f"object_goal: {object_summary}" if object_summary else "", f"object_goals: {multi_summary}" if multi_summary else "", f"mechanism: {mechanism}" if mechanism else "", f"completion: {completion}" if completion else "") if part]
    return "; ".join(parts)[:520] or "qwen semantic hypothesis"


def _object_goal_ref(raw: dict[str, Any], raw_step: dict[str, Any]) -> dict[str, Any]:
    index_value = raw_step.get("object_goal_index")
    if index_value in (None, ""):
        return {}
    try:
        index = int(index_value)
    except (TypeError, ValueError):
        return {}
    goals = [item for item in _as_items(raw.get("object_goals")) if isinstance(item, dict)]
    if 0 <= index < len(goals):
        return goals[index]
    return {}


def _short_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text[: max(0, int(limit))]


def _as_items(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _nonzero_delta(delta: Any) -> bool:
    if not isinstance(delta, (list, tuple)):
        return False
    for value in delta:
        try:
            if abs(float(value)) > 1e-6:
                return True
        except Exception:
            return False
    return False


def _compact_attempt_rejection(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    objective = raw.get("objective") if isinstance(raw.get("objective"), dict) else {}
    actions = []
    first_action = raw.get("first_action")
    if first_action:
        actions.append(str(first_action))
    actions.extend(str(action_id) for action_id in (raw.get("remaining_actions") or [])[:63])
    for run in (raw.get("action_runs") or [])[:50]:
        if not isinstance(run, dict):
            continue
        action_id = str(run.get("action_id") or "")
        repeat = run.get("repeat")
        if action_id and isinstance(repeat, int) and not isinstance(repeat, bool) and repeat > 0:
            actions.extend([action_id] * min(repeat, 50 - len(actions)))
    compact = {
        "reason": str(item.get("reason") or "invalid_qwen_candidate")[:180],
        "hypothesis_id": item.get("hypothesis_id") or raw.get("id") or raw.get("hypothesis_id"),
        "action_id": item.get("action_id"),
        "coordinate_candidate_id": item.get("coordinate_candidate_id"),
        "trajectory_step_index_1_based": item.get("trajectory_step_index_1_based"),
        "available_action_ids_before_step": item.get("available_action_ids_before_step") or [],
        "observed_enabling_action_ids": item.get("observed_enabling_action_ids") or [],
        "detail": item.get("detail"),
        "previous_attempt_index": item.get("previous_attempt_index"),
        "previous_hypothesis_id": item.get("previous_hypothesis_id"),
        "previous_outcome": item.get("previous_outcome"),
        "previous_rejection_reason": item.get("previous_rejection_reason"),
    }
    claim = objective.get("description") or raw.get("claim") or raw.get("basis")
    if claim:
        compact["claim"] = str(claim)[:360]
    if actions:
        compact["actions"] = actions
        compact["exact_replay_forbidden"] = True
    return compact


def _v87_basis_action_count_mismatch(raw: dict[str, Any], actions: list[str]) -> dict[str, Any] | None:
    basis = str(raw.get("basis") or "")
    matches = re.findall(r"\b(ACTION[1-7])\s*(?:(?:[xXyY*]|times?)\s*)?(\d+)\b", basis, flags=re.IGNORECASE)
    if not matches:
        return None
    declared: dict[str, int] = {}
    for action_id, count in matches:
        canonical = action_id.upper()
        declared[canonical] = declared.get(canonical, 0) + int(count)
    actual = {action_id: actions.count(action_id) for action_id in declared}
    if all(actual[action_id] == count for action_id, count in declared.items()):
        return None
    return {"declared_action_counts": declared, "actual_action_counts": actual}


def _delta_xy_pair(delta: Any) -> tuple[float, float]:
    if not isinstance(delta, (list, tuple)) or len(delta) < 2:
        return (0.0, 0.0)
    try:
        return (float(delta[0]), float(delta[1]))
    except Exception:
        return (0.0, 0.0)


def _enum_or(value: Any, enum_cls: Any, default: Any) -> Any:
    try:
        return enum_cls(str(value))
    except Exception:
        return default


def _valid_contract_kind(value: str) -> bool:
    try:
        VerificationContractKind(str(value).upper())
        return True
    except ValueError:
        return False
