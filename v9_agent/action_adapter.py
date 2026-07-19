from __future__ import annotations

from typing import Any

from .types import CandidateAction


def to_action_dict(candidate: CandidateAction) -> dict[str, Any]:
    return candidate.to_arc_action()


def from_action_dict(payload: dict[str, Any]) -> CandidateAction:
    action_id = str(payload.get("action_id") or payload.get("id") or "ACTION1")
    data = dict(payload.get("data", {}) or {})
    return CandidateAction(action_id, x=data.get("x"), y=data.get("y"), reason="from external action dict", source="action_adapter")
