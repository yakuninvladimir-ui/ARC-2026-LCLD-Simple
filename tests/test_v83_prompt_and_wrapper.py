import importlib.util
from pathlib import Path

from v8_agent.config import V8Config
from v8_agent.llm import _prompt, compact_packet
from v8_agent.qwen_packet import QwenPacketBuilder
from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.game_adapter import GameAdapter
from v8_agent.hypothesis_bank import HypothesisBank
from v8_agent.memory import GameMemory
from v8_agent.types import ActionEffectRecord, QwenRole


def _packet():
    memory = GameMemory()
    memory.reset_game("g")
    obs = {"grid": [[0, 0, 0], [0, 1, 0], [0, 0, 2]], "metadata": {"game_id": "g", "available_actions": ["ACTION1", "ACTION6"], "frame_index": 0}}
    snapshot = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), memory, V8Config())
    for action_id in snapshot.planning_action_ids or snapshot.available_actions:
        memory.action_effects[(action_id, None)] = ActionEffectRecord(action_id, None, "effect", 1, 0.57, snapshot.level_index, snapshot.step_index)
    return QwenPacketBuilder().build_semantic_packet(snapshot, memory, HypothesisBank(), QwenRole.PRIMARY, V8Config())


def test_prompt_protocol_has_single_packet_and_schema_without_thinking_or_clues():
    text = _prompt(QwenRole.PRIMARY, _packet(), V8Config())
    assert "/think" not in text
    assert "/no_think" not in text
    assert text.count("OBSERVATION_PACKET_JSON=") == 1
    assert text.count("ALLOWED_OUTPUT_JSON_SCHEMA=") == 1
    assert "clue_3x3" not in text
    assert "verification_contracts" not in text
    assert "hidden reasoning" not in text
    assert "confirmed_mechanics" not in text
    assert "v8.7.semantic_trajectories" in text
    assert "1-3 genuinely distinct" in text
    assert "only semantic model call" in text
    assert "do not return a prefix" in text
    assert "prefix_until_reobservation" not in text
    assert "do not claim rotation, recoloring, reshaping" in text
    assert "correct solution" not in text
    assert "You may propose action IDs" in text
    assert "You do not execute or authorize actions" in text
    assert text.endswith("RETURN_JSON_ONLY")
    order = [text.index("OBSERVATION_PACKET_JSON="), text.index("ALLOWED_OUTPUT_JSON_SCHEMA="), text.index("RETURN_JSON_ONLY")]
    assert order == sorted(order)


def test_minimal_compaction_preserves_described_allowed_ids():
    minimal = compact_packet(_packet(), "minimal")
    for key in ("state", "scene", "current_frame", "action_model", "action_surface", "memory", "execution_constraints"):
        assert key in minimal
    constraints = minimal["execution_constraints"]
    scene = minimal["scene"]
    assert set(constraints["allowed_object_ids"]) == {obj["id"] for obj in scene["objects"]}
    assert set(constraints["allowed_relation_ids"]) == {rel["id"] for rel in scene["relations"]}
    assert set(constraints["allowed_coordinate_candidate_ids"]) == {cand["id"] for cand in scene["coordinate_candidates"]}


def test_generated_competition_shim_compiles_and_contains_real_commit_contract():
    path = Path(__file__).resolve().parents[1] / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("v83_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    source = module._generated_kaggle_agent_source()
    compile(source, "<generated_kaggle_agent_v83>", "exec")
    assert "self._session.observe_action_result" in source
    assert "pending_official_transition" in source
    assert "self._session.update_runtime_config" in source


def test_production_notebook_uses_tufa_derived_competition_lifecycle_without_my_agent():
    root = Path(__file__).resolve().parents[1]
    path = root / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("v83_working_wrapper_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    notebook = module.build()
    common = notebook["cells"][3]["source"]
    phase_b = notebook["cells"][4]["source"]
    unpack = notebook["cells"][2]["source"]

    assert "operation_mode=arc_agi.OperationMode.COMPETITION" in phase_b
    assert "scorecard_id = arcade.create_scorecard()" in phase_b
    assert "arcade.make(game_id, scorecard_id=scorecard_id)" in phase_b
    assert "arcade.close_scorecard(scorecard_id)" in phase_b
    assert "competition_scorecard_closed" in phase_b
    assert "if state_name == 'GAME_OVER':" in phase_b
    assert "delegate.reset_after_game_over" in phase_b
    assert "'OPERATION_MODE': 'competition'" in common
    assert "IS_PHASE_B_CANDIDATE = RERUN_ENV_TRUE" in common
    assert "config.old" not in unpack
    assert "agent/my_agent.py" not in unpack
    assert "code_dir / 'agent'" not in unpack


def test_runtime_has_no_game_specific_solver_branch():
    runtime = Path(__file__).resolve().parents[1] / "v8_agent"
    source = "\n".join(path.read_text(encoding="utf-8") for path in runtime.glob("*.py"))

    assert "ar25" not in source.lower()


def test_generated_shim_executes_pending_commit_without_arcengine(monkeypatch):
    import sys
    import types

    path = Path(__file__).resolve().parents[1] / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("v83_builder_exec", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    submission = types.ModuleType("submission")
    submission.default_config = lambda: {"qwen_backend": "disabled", "enable_qwen": False}
    submission._state_name = lambda value: str(value).upper()
    monkeypatch.setitem(sys.modules, "submission", submission)
    namespace = {}
    exec(module._generated_kaggle_agent_source(), namespace)
    delegate = namespace["ARC_AGI_Agent"]()
    before = {"grid": [[0, 0], [0, 1]], "metadata": {"game_id": "g", "available_actions": ["ACTION1"], "frame_index": 0, "state": "PLAYING"}}
    after = {"grid": [[0, 0], [0, 1]], "metadata": {"game_id": "g", "available_actions": ["ACTION1"], "frame_index": 1, "state": "PLAYING"}}
    action = delegate.act(before)
    if isinstance(action, dict):
        assert action["id"] == "ACTION1"
    else:
        assert namespace["_action_name"](getattr(action, "id", action)) == "ACTION1"
    assert delegate.harness_telemetry()["pending_official_transition"] is True
    assert delegate.observe_action_result(after) is True
    telemetry = delegate.harness_telemetry()
    assert telemetry["pending_official_transition"] is False
    assert telemetry["observed_transition_ingestions"] == 1


def test_generated_shim_commits_previous_frame_in_callback_free_working_loop(monkeypatch):
    import sys
    import types

    path = Path(__file__).resolve().parents[1] / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("v83_builder_callback_free", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    submission = types.ModuleType("submission")
    submission.default_config = lambda: {"qwen_backend": "disabled", "enable_qwen": False}
    submission._state_name = lambda value: str(value).upper()
    monkeypatch.setitem(sys.modules, "submission", submission)
    namespace = {}
    exec(module._generated_kaggle_agent_source(), namespace)
    delegate = namespace["ARC_AGI_Agent"]()

    before = {"grid": [[0, 0], [0, 1]], "metadata": {"game_id": "g", "available_actions": ["ACTION1"], "frame_index": 0, "state": "PLAYING"}}
    after = {"grid": [[0, 0], [1, 0]], "metadata": {"game_id": "g", "available_actions": ["ACTION1"], "frame_index": 1, "state": "PLAYING"}}
    delegate.act(before)
    assert delegate.harness_telemetry()["pending_official_transition"] is True

    try:
        delegate.act(after)
    except RuntimeError as exc:
        # Qwen is disabled in this structural test, so no next verified action
        # is expected after the transition itself has been committed.
        assert "no verifier-authorized action" in str(exc)
    telemetry = delegate.harness_telemetry()
    assert telemetry["observed_transition_ingestions"] == 1
    assert telemetry["pending_official_transition"] is False
