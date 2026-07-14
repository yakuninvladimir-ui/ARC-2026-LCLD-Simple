from v8_agent.config import V8Config, config_from_mapping
from v8_agent.qwen_roles import can_call_qwen_role, record_qwen_call
from v8_agent.types import QwenBudgetState, QwenRole


def test_default_qwen_envelope_matches_competition_profile():
    cfg = V8Config()
    assert cfg.qwen_context_tokens == 98304
    assert cfg.qwen_max_input_tokens == 65536
    assert cfg.qwen_max_output_tokens == 12288
    assert cfg.qwen_timeout_seconds == 500
    assert cfg.max_primary_qwen_calls_per_level == 1
    assert cfg.max_coordinate_qwen_calls_per_level == 1
    assert cfg.max_reserve_qwen_calls_per_level == 0
    assert cfg.max_total_qwen_calls_per_level == 2


def test_role_budget_permits_coordinate_then_primary_per_attempt():
    cfg = V8Config(min_steps_between_qwen_calls=0)
    b = QwenBudgetState()
    assert can_call_qwen_role(QwenRole.COORDINATE, 0, 0, b, cfg)
    record_qwen_call(QwenRole.COORDINATE, 0, 0, b)
    assert can_call_qwen_role(QwenRole.PRIMARY, 0, 1, b, cfg)
    record_qwen_call(QwenRole.PRIMARY, 0, 1, b)
    assert not can_call_qwen_role(QwenRole.COORDINATE, 0, 2, b, cfg)
    assert not can_call_qwen_role(QwenRole.RESERVE, 0, 2, b, cfg)
    assert not can_call_qwen_role(QwenRole.PRIMARY, 0, 3, b, cfg)
    assert b.calls_this_game == 2
    assert b.primary_calls_by_level[0] == 1
    assert b.coordinate_calls_by_level[0] == 1
    assert not b.reserve_calls_by_level


def test_game_budget_stops_at_20_and_spacing_enforced():
    cfg = V8Config(max_qwen_calls_per_game=20, min_steps_between_qwen_calls=3)
    b = QwenBudgetState(calls_this_game=20)
    assert not can_call_qwen_role(QwenRole.PRIMARY, 0, 100, b, cfg)
    b = QwenBudgetState(last_qwen_step=10)
    assert not can_call_qwen_role(QwenRole.PRIMARY, 0, 12, b, cfg)
    assert can_call_qwen_role(QwenRole.PRIMARY, 0, 13, b, cfg)


def test_config_forces_thinking_disabled(monkeypatch):
    monkeypatch.setenv("ARC_QWEN_ENABLE_THINKING", "1")
    monkeypatch.setenv("ARC_QWEN_REASONING_MODE", "on")
    monkeypatch.setenv("ARC_QWEN_REASONING_BUDGET_TOKENS", "2048")
    cfg = config_from_mapping({"qwen_enable_thinking": True, "qwen_reasoning_mode": "on", "qwen_reasoning_budget_tokens": 1024})
    assert cfg.qwen_enable_thinking is False
    assert cfg.qwen_reasoning_mode == "off"
    assert cfg.qwen_reasoning_budget_tokens == 0
