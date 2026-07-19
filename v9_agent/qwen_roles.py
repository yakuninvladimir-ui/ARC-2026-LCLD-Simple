from __future__ import annotations

from .config import V8Config
from .types import QwenBudgetState, QwenRole


def can_call_qwen_role(role: QwenRole, level_index: int, step_index: int, budget: QwenBudgetState, config: V8Config, *, ignore_spacing: bool = False) -> bool:
    if not config.enable_qwen or config.qwen_backend == "disabled":
        return False
    if budget.calls_this_game >= config.max_qwen_calls_per_game:
        return False
    if budget.total_calls_by_level.get(level_index, 0) >= config.max_total_qwen_calls_per_level:
        return False
    if not ignore_spacing and step_index - budget.last_qwen_step < config.min_steps_between_qwen_calls:
        return False
    if role is QwenRole.PRIMARY:
        return budget.primary_calls_by_level.get(level_index, 0) < config.max_primary_qwen_calls_per_level
    if role is QwenRole.RESERVE:
        return budget.reserve_calls_by_level.get(level_index, 0) < config.max_reserve_qwen_calls_per_level
    if role is QwenRole.COORDINATE:
        return budget.coordinate_calls_by_level.get(level_index, 0) < config.max_coordinate_qwen_calls_per_level
    return False


def record_qwen_call(role: QwenRole, level_index: int, step_index: int, budget: QwenBudgetState) -> None:
    budget.calls_this_game += 1
    budget.total_calls_by_level[level_index] = budget.total_calls_by_level.get(level_index, 0) + 1
    budget.last_qwen_step = step_index
    if role is QwenRole.PRIMARY:
        budget.primary_calls_by_level[level_index] = budget.primary_calls_by_level.get(level_index, 0) + 1
    elif role is QwenRole.RESERVE:
        budget.reserve_calls_by_level[level_index] = budget.reserve_calls_by_level.get(level_index, 0) + 1
    elif role is QwenRole.COORDINATE:
        budget.coordinate_calls_by_level[level_index] = budget.coordinate_calls_by_level.get(level_index, 0) + 1
