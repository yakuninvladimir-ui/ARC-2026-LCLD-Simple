from __future__ import annotations

from .config import V8Config
from .types import ARGALiteSnapshot, CandidateAction, TestStep
from .verification import VerificationBinder


class ActionExplorer:
    def __init__(self) -> None:
        self._binder = VerificationBinder()

    def simple_probe(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", config: V8Config) -> CandidateAction | None:
        for action_id in memory.unprobed_action_effect_ids(snapshot, config):
            step = TestStep("simple_action_probe", action_id, expected_observation="Discover the typed local action effect.", contract_kind="ACTION_EFFECT_DISCOVERY")
            contract = self._binder.bind(step, snapshot, None)
            return CandidateAction(action_id, reason="one-shot current-action effect probe before Qwen planning", source="initial_action_probe", verification_contract=contract)
        return None

    def coordinate_probe(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", config: V8Config) -> CandidateAction | None:
        if not snapshot.coordinate_action_ids or memory.coordinate_probe_count(snapshot.level_index) >= config.max_coordinate_probes_per_level:
            return None
        action_id = snapshot.coordinate_action_ids[0]
        for target in snapshot.coordinate_targets:
            step = TestStep(
                "coordinate_probe",
                action_id,
                target_object_id=target.object_id,
                target_relation_id=target.relation_id,
                coordinate_candidate_id=target.candidate_id,
                expected_observation="Observe a target-local affordance effect.",
            )
            contract = self._binder.bind(step, snapshot, None)
            if contract is None:
                continue
            candidate = CandidateAction(action_id, x=target.x, y=target.y, coordinate_candidate_id=target.candidate_id, reason=target.reason, source="deterministic_coordinate_explorer", verification_contract=contract)
            if memory.action_attempt_count(candidate.suppression_signature, snapshot.semantic_state_signature) >= config.max_coordinate_probe_repeats_per_signature:
                continue
            return candidate
        return None

    def exhaustion_revisit(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", config: V8Config) -> CandidateAction:
        """Return a legal least-used action after all informative candidates are exhausted.

        This is an internal liveness path, not a semantic confirmation path. It keeps
        the competition loop alive until the outer timeout/action budget rather than
        crashing when every available action has already been classified as irrelevant.
        """
        simple = [a for a in snapshot.available_actions if a not in snapshot.coordinate_action_ids and a != "RESET"]
        if simple:
            action_id = min(simple, key=lambda aid: (memory.simple_action_attempt_count(snapshot.level_index, aid), aid))
            step = TestStep("exhaustion_revisit", action_id, expected_observation="Revisit least-used legal action after epistemic exhaustion.", contract_kind="ACTION_EFFECT_DISCOVERY")
            contract = self._binder.bind(step, snapshot, None)
            return CandidateAction(
                action_id,
                reason="bounded liveness revisit after all informative candidates were exhausted",
                source="exhaustion_revisit",
                verification_contract=contract,
                allow_exhaustion_revisit=True,
            )
        if snapshot.coordinate_action_ids and snapshot.coordinate_targets:
            action_id = snapshot.coordinate_action_ids[0]
            def key(target):
                step = TestStep("exhaustion_coordinate_revisit", action_id, target_object_id=target.object_id, target_relation_id=target.relation_id, coordinate_candidate_id=target.candidate_id, expected_observation="Revisit least-used coordinate target after epistemic exhaustion.")
                contract = self._binder.bind(step, snapshot, None)
                candidate = CandidateAction(action_id, x=target.x, y=target.y, coordinate_candidate_id=target.candidate_id, verification_contract=contract)
                return (memory.action_attempt_count(candidate.suppression_signature, snapshot.semantic_state_signature), -target.salience_score, target.candidate_id)
            target = min(snapshot.coordinate_targets, key=key)
            step = TestStep("exhaustion_coordinate_revisit", action_id, target_object_id=target.object_id, target_relation_id=target.relation_id, coordinate_candidate_id=target.candidate_id, expected_observation="Revisit least-used coordinate target after epistemic exhaustion.")
            contract = self._binder.bind(step, snapshot, None)
            return CandidateAction(
                action_id,
                x=target.x,
                y=target.y,
                coordinate_candidate_id=target.candidate_id,
                reason="bounded coordinate liveness revisit after all informative candidates were exhausted",
                source="exhaustion_revisit",
                verification_contract=contract,
                allow_exhaustion_revisit=True,
            )
        if snapshot.game_over or snapshot.state_name in {"NOT_STARTED", "NOT_PLAYED"}:
            return CandidateAction("RESET", reason="state requires reset", source="state_reset")
        return CandidateAction("RESET", reason="no legal action surface", source="exhaustion_revisit", allow_exhaustion_revisit=True)

    def safe_fallback(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", config: V8Config) -> CandidateAction:
        simple = [a for a in snapshot.available_actions if a not in snapshot.coordinate_action_ids and a != "RESET"]
        simple.sort(key=lambda action_id: (memory.simple_action_attempt_count(snapshot.level_index, action_id), action_id))
        for action_id in simple:
            step = TestStep("fallback_action_effect_probe", action_id, expected_observation="Resolve remaining action-effect uncertainty.", contract_kind="ACTION_EFFECT_DISCOVERY")
            contract = self._binder.bind(step, snapshot, None)
            candidate = CandidateAction(action_id, reason="least-tried legal action with typed effect question", source="memory_aware_fallback", verification_contract=contract)
            if memory.action_attempt_count(candidate.suppression_signature, snapshot.semantic_state_signature) < config.max_same_state_action_repeats:
                return candidate
        if snapshot.coordinate_action_ids:
            action_id = snapshot.coordinate_action_ids[0]
            for target in snapshot.coordinate_targets:
                step = TestStep("fallback_coordinate_probe", action_id, target_object_id=target.object_id, target_relation_id=target.relation_id, coordinate_candidate_id=target.candidate_id, expected_observation="Resolve remaining coordinate affordance uncertainty.")
                contract = self._binder.bind(step, snapshot, None)
                if contract is None:
                    continue
                candidate = CandidateAction(action_id, x=target.x, y=target.y, coordinate_candidate_id=target.candidate_id, reason="least-tried coordinate target", source="memory_aware_fallback", verification_contract=contract)
                if memory.action_attempt_count(candidate.suppression_signature, snapshot.semantic_state_signature) < config.max_same_state_action_repeats:
                    return candidate
        if snapshot.game_over or snapshot.state_name in {"NOT_STARTED", "NOT_PLAYED"}:
            return CandidateAction("RESET", reason="reset required by environment state", source="state_reset")
        return self.exhaustion_revisit(snapshot, memory, config)
