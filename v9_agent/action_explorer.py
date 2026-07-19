from __future__ import annotations

from .config import V8Config
from .types import ARGALiteSnapshot, CandidateAction, TestStep
from .verification import VerificationBinder


class ActionExplorer:
    def __init__(self) -> None:
        self._binder = VerificationBinder()
        self._first_seen_controls: dict[tuple[str, int, str], tuple[str, ...]] = {}
        self._active_control_cycle: dict[str, object] | None = None

    @staticmethod
    def _context_key(snapshot: ARGALiteSnapshot, action_id: str) -> tuple[str, int, str]:
        return (str(snapshot.game_id), int(snapshot.level_index), str(action_id))

    def _drop_stale_cycle(self, snapshot: ARGALiteSnapshot) -> None:
        cycle = self._active_control_cycle
        if cycle is None:
            return
        if (
            cycle.get("game_id") != str(snapshot.game_id)
            or int(cycle.get("level_index", -1)) != int(snapshot.level_index)
        ):
            self._active_control_cycle = None

    def has_pending_control_cycle(self, snapshot: ARGALiteSnapshot) -> bool:
        """Return whether primary planning must wait for a control-new-control probe."""
        self._drop_stale_cycle(snapshot)
        cycle = self._active_control_cycle
        if cycle is None:
            return False
        available = set(snapshot.available_actions)
        phase = str(cycle.get("phase") or "")
        required_action = (
            str(cycle.get("new_action_id"))
            if phase == "new"
            else str(cycle.get("control_action_id"))
        )
        if required_action not in available:
            # The surface transition itself is sufficient evidence when the next
            # comparison action is no longer legal.
            self._active_control_cycle = None
            return False
        return True

    def _probe_candidate(
        self,
        snapshot: ARGALiteSnapshot,
        action_id: str,
        *,
        source: str,
        reason: str,
        step_kind: str,
    ) -> CandidateAction:
        step = TestStep(
            step_kind,
            action_id,
            expected_observation="Compare the typed action effect and action surface in a controlled probe sequence.",
            contract_kind="ACTION_EFFECT_DISCOVERY",
        )
        contract = self._binder.bind(step, snapshot, None)
        return CandidateAction(
            action_id,
            reason=reason,
            source=source,
            verification_contract=contract,
        )

    def simple_probe(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", config: V8Config) -> CandidateAction | None:
        self._drop_stale_cycle(snapshot)
        missing = memory.unprobed_action_effect_ids(snapshot, config)
        researched_controls = tuple(memory.researched_simple_action_ids(snapshot))

        # Register the whole newly observed surface at once. Actions co-visible on
        # the first unknown surface therefore remain ordinary one-shot probes;
        # actions appearing later can use controls known before they appeared.
        for action_id in missing:
            key = self._context_key(snapshot, action_id)
            self._first_seen_controls.setdefault(
                key,
                tuple(control for control in researched_controls if control != action_id),
            )

        cycle = self._active_control_cycle
        if cycle is not None:
            phase = str(cycle.get("phase") or "")
            control_action_id = str(cycle.get("control_action_id"))
            new_action_id = str(cycle.get("new_action_id"))
            if phase == "new" and new_action_id in snapshot.available_actions:
                cycle["phase"] = "after"
                return self._probe_candidate(
                    snapshot,
                    new_action_id,
                    source="action_probe_new_action",
                    reason=f"controlled probe new action between two {control_action_id} observations",
                    step_kind="controlled_new_action_probe",
                )
            if phase == "after" and control_action_id in snapshot.available_actions:
                self._active_control_cycle = None
                return self._probe_candidate(
                    snapshot,
                    control_action_id,
                    source="action_probe_control_after",
                    reason=f"control-after probe for {new_action_id}; compare with the same control before it",
                    step_kind="action_probe_control_after",
                )
            self._active_control_cycle = None

        available = set(snapshot.available_actions)
        for action_id in missing:
            controls = [
                control
                for control in self._first_seen_controls.get(self._context_key(snapshot, action_id), ())
                if control in available
            ]
            if controls:
                control_action_id = min(
                    controls,
                    key=lambda control: (
                        memory.simple_action_attempt_count(snapshot.level_index, control),
                        control,
                    ),
                )
                self._active_control_cycle = {
                    "game_id": str(snapshot.game_id),
                    "level_index": int(snapshot.level_index),
                    "new_action_id": action_id,
                    "control_action_id": control_action_id,
                    "phase": "new",
                }
                return self._probe_candidate(
                    snapshot,
                    control_action_id,
                    source="action_probe_control_before",
                    reason=f"control-before probe for newly available {action_id}",
                    step_kind="action_probe_control_before",
                )
            return self._probe_candidate(
                snapshot,
                action_id,
                source="initial_action_probe",
                reason="one-shot current-action effect probe before Qwen planning",
                step_kind="simple_action_probe",
            )
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
