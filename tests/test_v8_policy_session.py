import pytest

from v8_agent.config import V8Config
from v8_agent.session import GameSession


def test_session_smoke_emits_one_action_and_judges_next_transition():
    cfg=V8Config(qwen_backend="fake", min_steps_between_qwen_calls=0)
    session=GameSession(cfg)
    obs={"grid":[[0,0,0],[0,1,0],[0,0,0]], "metadata":{"game_id":"g","available_actions":["ACTION1","ACTION6"],"frame_index":0}}
    action1=session.act(obs)
    assert isinstance(action1, dict)
    assert "action_id" in action1
    obs2={"grid":[[0,0,0],[0,1,0],[0,0,2]], "metadata":{"game_id":"g","available_actions":["ACTION1","ACTION6"],"frame_index":1}}
    action2=session.act(obs2)
    assert isinstance(action2, dict)
    assert len(session.memory.events) == 1


def test_coordinate_action_requires_qwen_candidate_when_fallback_disabled():
    cfg=V8Config(qwen_backend="disabled", require_coordinate_candidate_id=False, min_steps_between_qwen_calls=0)
    session=GameSession(cfg)
    obs={"grid":[[0,0,0],[0,1,0],[0,0,0]], "metadata":{"game_id":"g","available_actions":["ACTION6"],"frame_index":0}}
    with pytest.raises(RuntimeError, match="deterministic fallback is disabled"):
        session.act(obs)
