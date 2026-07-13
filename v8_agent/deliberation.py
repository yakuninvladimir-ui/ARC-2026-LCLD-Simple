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
    if not memory.action_effect_probe_complete(snapshot, config):
        return None
    has_candidate = bank.has_executable_candidate(snapshot)
    urgent_no_candidate = not has_candidate
    if snapshot.coordinate_action_ids and memory.coordinate_research_needed(state.level_index):
        if can_call_qwen_role(QwenRole.COORDINATE, state.level_index, state.step_index, budget, config, ignore_spacing=urgent_no_candidate):
            return QwenRole.COORDINATE
    if can_call_qwen_role(QwenRole.PRIMARY, state.level_index, state.step_index, budget, config, ignore_spacing=urgent_no_candidate):
        return QwenRole.PRIMARY
    return None
