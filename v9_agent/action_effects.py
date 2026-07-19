from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

from .types import ActionEffectRecord, SemanticQuestion, SemanticQuestionType


def normalized_entropy(probabilities: Iterable[float]) -> float:
    values = [max(0.0, float(p)) for p in probabilities]
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    probs = [p / total for p in values]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy / math.log(len(probs))


def posterior_for(question: SemanticQuestion) -> dict[str, float]:
    domain = question.domain
    if not domain:
        return {}
    if question.resolved_outcome in domain:
        if len(domain) == 1:
            return {domain[0]: 1.0}
        tail = 0.08 / (len(domain) - 1)
        return {outcome: (0.92 if outcome == question.resolved_outcome else tail) for outcome in domain}
    return {outcome: 1.0 / len(domain) for outcome in domain}


def resolve_question(question: SemanticQuestion, outcome: str) -> tuple[float, SemanticQuestion]:
    before = posterior_for(question)
    h_before = normalized_entropy(before.values())
    if outcome not in question.domain:
        return 0.0, question
    observations = dict(question.observations)
    observations[outcome] = observations.get(outcome, 0) + 1
    if question.resolved_outcome is None:
        resolved = outcome
    elif question.resolved_outcome == outcome:
        resolved = outcome
    else:
        # Contradictory typed evidence reopens the question instead of forcing certainty.
        resolved = None
    updated = replace(question, resolved_outcome=resolved, observations=observations)
    h_after = normalized_entropy(posterior_for(updated).values())
    return max(0.0, h_before - h_after), updated


def default_question_domain(question_type: SemanticQuestionType) -> tuple[str, ...]:
    if question_type in {SemanticQuestionType.ACTION_EFFECT, SemanticQuestionType.AFFORDANCE}:
        return ("effect", "no_effect", "negative_effect")
    if question_type is SemanticQuestionType.CONTROLLABILITY:
        return ("moved", "not_moved", "ambiguous")
    if question_type is SemanticQuestionType.RELATION_RELEVANCE:
        return ("error_decreased", "unchanged", "error_increased")
    if question_type is SemanticQuestionType.ACTION_SURFACE_CHANGE:
        return ("changed", "unchanged")
    return ("progress", "no_progress", "negative")


def merge_effect_record(
    existing: ActionEffectRecord | None,
    *,
    action_id: str,
    target_signature: str | None,
    outcome: str,
    level_index: int,
    step_index: int,
) -> ActionEffectRecord:
    count = 1 if existing is None else existing.evidence_count + 1
    confidence = min(0.98, 0.45 + 0.12 * count)
    return ActionEffectRecord(
        action_id=action_id,
        target_signature=target_signature,
        outcome=outcome,
        evidence_count=count,
        confidence=confidence,
        level_index=level_index,
        last_step=step_index,
    )
