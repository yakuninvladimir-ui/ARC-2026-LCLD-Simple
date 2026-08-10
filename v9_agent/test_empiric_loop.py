"""Unit tests for the empiric-loop fixes (2026-08-07).

Covers three contract points:
1. Binder: explicit contract kinds are honored only when measurable targets
   exist (targetless OBJECT_DISPLACEMENT degrades to ACTION_EFFECT_DISCOVERY
   instead of producing a guaranteed tracking MISMATCH).
2. Preflight: an exact same-state repeat is suppressed only when the last
   identical action had no visible effect (or worse); repeating a visibly
   effective action is confirmed-effect exploitation.
3. Bank: a finished trajectory with confirmed visible effect enqueues a
   confirmed_continuation; a no-effect trajectory does not.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from v9_agent.config import config_from_mapping
from v9_agent.hypothesis_bank import HypothesisBank
from v9_agent.judge import PreflightJudge
from v9_agent.types import (
    Attribution,
    CandidateAction,
    HypothesisItem,
    Judgment,
    MechanicResult,
    MemoryEvent,
    Progress,
    Relevance,
    SemanticJudgment,
    TestStep,
    TriTruth,
    Validity,
    VerificationContractKind,
)
from v9_agent.verification import VerificationBinder


def _obj(object_id: str, bbox=(2, 1, 3, 2)):
    return SimpleNamespace(
        object_id=object_id,
        bbox_rc=bbox,
        centroid_rc=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        stable_hash=f"sh_{object_id}",
    )


def _snapshot(objects=("o1",), actions=("ACTION1", "ACTION2")):
    return SimpleNamespace(
        game_id="g",
        level_index=0,
        step_index=0,
        objects=tuple(_obj(o) for o in objects),
        relations=(),
        coordinate_targets=(),
        coordinate_action_ids=(),
        available_actions=tuple(actions),
        semantic_state_signature="sem_1",
        grid_hash="g_1",
        width=8,
        height=8,
        state_name="PLAYING",
        game_over=False,
    )


class BinderTargetGuardTest(unittest.TestCase):
    def test_explicit_displacement_without_targets_degrades(self):
        binder = VerificationBinder()
        step = TestStep("verified_effect_trajectory_step", "ACTION1", contract_kind="OBJECT_DISPLACEMENT")
        contract = binder.bind(step, _snapshot(), "h1")
        self.assertIsNotNone(contract)
        self.assertEqual(contract.kind, VerificationContractKind.ACTION_EFFECT_DISCOVERY)

    def test_explicit_displacement_with_target_preserved(self):
        binder = VerificationBinder()
        step = TestStep(
            "verified_effect_trajectory_step",
            "ACTION1",
            target_object_id="o1",
            contract_kind="OBJECT_DISPLACEMENT",
        )
        contract = binder.bind(step, _snapshot(), "h1")
        self.assertIsNotNone(contract)
        self.assertEqual(contract.kind, VerificationContractKind.OBJECT_DISPLACEMENT)
        self.assertEqual(contract.target_object_ids, ("o1",))

    def test_expected_type_marker_guarded(self):
        binder = VerificationBinder()
        step = TestStep(
            "rule_application",
            "ACTION1",
            expected_observation="expected_type=object_move; something moves",
        )
        contract = binder.bind(step, _snapshot(), "h1")
        self.assertEqual(contract.kind, VerificationContractKind.ACTION_EFFECT_DISCOVERY)


class PreflightRepeatSuppressionTest(unittest.TestCase):
    def _memory(self, attribution, progress=Progress.NEUTRAL):
        event = MemoryEvent(
            event_id="evt_1",
            level_index=0,
            step_index=0,
            event_type="transition_judgment",
            before_hash="g_1",
            action={"id": "ACTION1"},
            after_hash="g_2",
            hypothesis_id=None,
            truth=TriTruth.UNKNOWN,
            relevance=Relevance.RELEVANT,
            validity=Validity.VALID,
            progress=progress,
            attribution=attribution,
            reason_code="typed_action_effect_observed",
            summary="",
        )
        return SimpleNamespace(
            action_attempt_count=lambda signature, state_signature=None: 1,
            events=[event],
        )

    def _candidate(self):
        return CandidateAction("ACTION1", reason="t", source="primary_qwen_v87", hypothesis_id="h1")

    def test_repeat_after_visible_effect_allowed(self):
        judge = PreflightJudge()
        result = judge.validate(
            self._candidate(),
            _snapshot(),
            self._memory(Attribution.ACTION_LINKED),
            config_from_mapping({}),
        )
        self.assertTrue(result.valid, result.reason_code)

    def test_repeat_after_no_effect_suppressed(self):
        judge = PreflightJudge()
        result = judge.validate(
            self._candidate(),
            _snapshot(),
            self._memory(Attribution.NO_VISIBLE_CHANGE),
            config_from_mapping({}),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason_code, "same_candidate_repeat_suppressed")


class ConfirmedContinuationTest(unittest.TestCase):
    def _bank_with_item(self):
        bank = HypothesisBank()
        item = HypothesisItem(
            hypothesis_id="hyp_src",
            source="primary_qwen_v87",
            claim="ACTION1 moves the block",
            truth=TriTruth.UNKNOWN,
            relevance=Relevance.UNDECIDED,
            validity=Validity.UNCHECKED,
            progress=Progress.UNKNOWN,
            test_plan=(TestStep("verified_effect_trajectory_step", "ACTION1"),),
            cursor=0,
            priority=1.0,
            confidence=0.5,
            expiry_step=None,
            evidence_refs=(),
            suppression_signature="sup_src",
            created_state_signature="sem_0",
            trajectory_start_snapshot=_snapshot(),
        )
        bank.verified_semantic.append(item)
        return bank, item

    def _judgment(self, item, attribution, mechanic, progress=Progress.NEUTRAL,
                  reason="typed_action_effect_observed"):
        return Judgment(
            truth=TriTruth.UNKNOWN,
            relevance=Relevance.RELEVANT,
            validity=Validity.VALID,
            progress=progress,
            attribution=attribution,
            reason_code=reason,
            observed_delta={"level_index": 0, "step_index": 1},
            affected_objects=(),
            affected_relations=(),
            score_delta=None,
            terminal_delta=False,
            action=CandidateAction("ACTION1", reason="t", source=item.source, hypothesis_id=item.hypothesis_id),
            hypothesis_id=item.hypothesis_id,
            before_hash="g_1",
            after_hash="g_2",
            mechanic_result=mechanic,
            semantic_judgment=SemanticJudgment.UNRESOLVED,
        )

    def test_confirmed_effect_enqueues_continuation(self):
        bank, item = self._bank_with_item()
        bank.update(
            self._judgment(item, Attribution.ACTION_LINKED, MechanicResult.MATCH),
            _snapshot(),
        )
        continuations = [h for h in bank.confirmed_rules if h.source == "confirmed_continuation"]
        self.assertEqual(len(continuations), 1)
        continuation = continuations[0]
        self.assertEqual([s.action_id for s in continuation.test_plan], ["ACTION1"])
        self.assertEqual(continuation.proposal_batch_id, "")
        self.assertNotEqual(continuation.hypothesis_id, item.hypothesis_id)

    def test_no_effect_does_not_continue(self):
        bank, item = self._bank_with_item()
        bank.update(
            self._judgment(
                item,
                Attribution.NO_VISIBLE_CHANGE,
                MechanicResult.MATCH,
                reason="typed_no_effect_observed",
            ),
            _snapshot(),
        )
        continuations = [h for h in bank.confirmed_rules if h.source == "confirmed_continuation"]
        self.assertEqual(continuations, [])

    def test_evaluation_attributed_to_executing_item(self):
        bank, item = self._bank_with_item()
        evaluation = bank.update(
            self._judgment(item, Attribution.ACTION_LINKED, MechanicResult.MATCH),
            _snapshot(),
        )
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation.hypothesis_id, "hyp_src")


if __name__ == "__main__":
    unittest.main()
