from __future__ import annotations

from .config import V8Config
from .qwen_roles import can_call_qwen_role
from .types import ARGALiteSnapshot, QwenBudgetState, QwenRole, WorldState


def choose_qwen_role(
    state: WorldState,
    snapshot: ARGALiteSnapshot,
    memory: "GameMemory",
    bank: "HypothesisBank",
    budget: QwenBudgetState,
    config: V8Config,
    *,
    is_new_level: bool = False,
) -> QwenRole | None:
    if not config.enable_qwen or config.qwen_backend == "disabled":
        return None
    research = memory.action_research_status(snapshot)
    if research["missing_simple_action_ids"]:
        return None
    if bank.has_executable_coordinate_candidate(snapshot):
        return None
    has_candidate = bank.has_executable_candidate(snapshot)
    urgent_no_candidate = not has_candidate
    if research["missing_coordinate_action_ids"]:
        if can_call_qwen_role(QwenRole.COORDINATE, state.level_index, state.step_index, budget, config, ignore_spacing=urgent_no_candidate):
            return QwenRole.COORDINATE
        return None
    if research["missing_action_ids"]:
        return None
    if can_call_qwen_role(QwenRole.PRIMARY, state.level_index, state.step_index, budget, config, ignore_spacing=urgent_no_candidate):
        return QwenRole.PRIMARY
    return None
