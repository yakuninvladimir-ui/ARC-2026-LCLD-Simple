import json
from pathlib import Path
from types import SimpleNamespace

from v8_agent.arga_lite import ARGALiteBuilder
from v8_agent.config import V8Config
from v8_agent.game_adapter import GameAdapter
from v8_agent.hypothesis_bank import HypothesisBank
import v8_agent.llm as llm
from v8_agent.llm import FakeQwenClient, QwenClient
from v8_agent.memory import GameMemory
from v8_agent.qwen_packet import QwenPacketBuilder, _action_effect_table, _annotate_control_context, _annotate_object_geometry, _control_groups, _control_state_transition_candidates, _objects, _regions, _scene_entities, _typed_action_effects
from v8_agent.session import GameSession
from v8_agent.types import ActionEffectRecord, Attribution, CandidateAction, Judgment, Progress, QwenRole, Relevance, TriTruth, Validity, VerificationContract, VerificationContractKind


def _snapshot():
    grid = [[0,0,0],[0,1,0],[0,0,2]]
    obs = {"grid": grid, "metadata": {"game_id":"g", "available_actions":["ACTION1","ACTION6"], "frame_index":0}}
    return ARGALiteBuilder().build(GameAdapter().to_world_state(obs), GameMemory(), V8Config())


def _ready_memory(snap):
    memory = GameMemory()
    memory.reset_game("g")
    obj = snap.objects[0] if snap.objects else None
    for action_id in snap.planning_action_ids or snap.available_actions:
        if action_id in snap.undo_action_ids:
            continue
        memory.action_effects[(action_id, None)] = ActionEffectRecord(action_id, None, "effect", 1, 0.57, snap.level_index, snap.step_index)
        if obj is not None:
            memory.action_memory_records.append({
                "action_id": action_id,
                "changed_cell_count": 1,
                "object_deltas": [{
                    "object_id": obj.object_id,
                    "before_bbox_rc": list(obj.bbox_rc),
                    "after_bbox_rc": list(obj.bbox_rc),
                    "before_centroid_rc": list(obj.centroid_rc),
                    "after_centroid_rc": [float(obj.centroid_rc[0]), float(obj.centroid_rc[1]) + 1.0],
                    "delta_centroid_rc": [0.0, 1.0],
                    "motion_direction": "right",
                }],
            })
    return memory


def test_semantic_packet_includes_required_fields():
    snap = _snapshot()
    mem = _ready_memory(snap)
    packet = QwenPacketBuilder().build_semantic_packet(snap, mem, HypothesisBank(), QwenRole.PRIMARY, V8Config())
    assert packet["schema_version"] == "v8.7.semantic_observation"
    assert packet["current_frame"]["hex_rows"] == list(snap.full_grid_hex_rows)
    assert packet["scene"]["objects"]
    assert "scene_entities" not in packet["scene"]
    assert "relations" in packet["scene"]
    assert "confirmed_mechanics" not in packet
    assert "action_model" in packet
    assert packet["action_model"]["control_mapping_status"] in {"KNOWN", "PARTIALLY_KNOWN"}
    assert "action_effects" not in packet["action_model"]
    assert packet["current_frame"]["grid_source"] == "official_observation_grid"
    assert {item["id"] for item in packet["action_surface"]["actions"]} == set(snap.available_actions)
    assert "clue_3x3_patterns" not in str(packet)
    assert "goal_error" not in str(packet)
    assert packet["execution_constraints"]["allowed_action_ids"]
    assert "semantic_digest" not in packet
    assert "canonical_objects" not in packet
    assert all(item["id"].startswith("o") for item in packet["scene"]["objects"])
    assert all(not item["id"].startswith("trk_") for item in packet["scene"]["objects"])
    assert "delta_xy_contract" in packet["action_model"]
    assert packet["action_model"]["motion_invariants"]["translation_preserves_orientation"] is True


def test_coordinate_packet_includes_targets_and_no_effect_memory():
    snap = _snapshot()
    mem = GameMemory()
    packet = QwenPacketBuilder().build_coordinate_packet(snap, mem, HypothesisBank(), V8Config())
    assert packet["schema_version"] == "v8.7.semantic_observation"
    assert packet["scene"]["coordinate_candidates"]
    assert packet["execution_constraints"]["allowed_coordinate_candidate_ids"]
    assert "coordinate_no_effect_memory" not in packet
    assert all(len(item["location_xy"]) == 2 for item in packet["scene"]["coordinate_candidates"])
    assert all(item["cell_value"] in "0123456789ABCDEF" for item in packet["scene"]["coordinate_candidates"])


def test_coordinate_model_aliases_are_translated_before_verification():
    snap = _snapshot()
    packet = QwenPacketBuilder().build_coordinate_packet(snap, GameMemory(), HypothesisBank(), V8Config())
    output = FakeQwenClient().call(QwenRole.COORDINATE, packet, V8Config(qwen_backend="fake"))
    bank = HypothesisBank()

    bank.add_qwen_output(QwenRole.COORDINATE, output, snap, V8Config(), packet=packet)

    assert bank.coordinate_test_queue
    assert not any(item["reason"] == "invented_coordinate_candidate_id" for item in bank.invalid_rejections)


def test_semantic_packet_exposes_dynamic_action_surface_chronology():
    obs = {
        "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "metadata": {
            "game_id": "g",
            "level_index": 2,
            "available_actions": ["ACTION3", "ACTION4", "ACTION5", "ACTION7"],
            "possible_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"],
        },
    }
    memory = GameMemory()
    snap = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), memory, V8Config())
    for action_id in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
        memory.action_effects[(action_id, None)] = ActionEffectRecord(action_id, None, "effect", 1, 0.57, 2, 0)
    memory.action_surface_memory_records.append({
        "level_index": 2,
        "level_index_before": 2,
        "step_index": 4,
        "action_id": "ACTION5",
        "planning_action_ids_before": ["ACTION3", "ACTION4", "ACTION5"],
        "planning_action_ids_after": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
        "changed_cell_count": 5,
        "raw_visual_changes": {
            "changed_cell_count": 5,
            "repeated_isolated_interior_change": True,
            "isolated_center_cell_transition_groups": [{
                "before_value": "0",
                "after_value": "5",
                "occurrence_count": 4,
                "interior_location_count": 4,
                "edge_location_count": 0,
                "locations_xy": [[1, 1], [4, 1], [1, 4], [4, 4]],
            }],
            "local_3x3_transition_groups": [],
        },
    })

    packet = QwenPacketBuilder().build_semantic_packet(snap, memory, HypothesisBank(), QwenRole.PRIMARY, V8Config())
    actions = {item["id"]: item for item in packet["action_surface"]["actions"]}
    transition = packet["action_surface"]["observed_transitions"][0]

    assert actions["ACTION1"]["available_now"] is False
    assert actions["ACTION1"]["planning_allowed"] is True
    assert actions["ACTION7"]["undo"] is True
    assert actions["ACTION7"]["planning_allowed"] is False
    assert transition["trigger_action_id"] == "ACTION5"
    assert transition["added"] == ["ACTION1", "ACTION2"]
    assert transition["forward_source_matches_current"] is True
    assert transition["current_surface_position"] == "OBSERVED_BEFORE"
    assert transition["visual_and_surface_changed_same_transition"] is True
    assert transition["simultaneous_raw_visual_changes"]["repeated_isolated_interior_change"] is True


def test_surface_transition_is_not_presented_as_replayable_from_another_surface():
    obs = {
        "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "metadata": {
            "game_id": "g",
            "level_index": 2,
            "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION5", "ACTION7"],
            "possible_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"],
        },
    }
    memory = GameMemory()
    snap = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), memory, V8Config())
    for action_id in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
        memory.action_effects[(action_id, None)] = ActionEffectRecord(action_id, None, "effect", 1, 0.57, 2, 0)
    memory.action_surface_memory_records.append({
        "level_index": 2,
        "level_index_before": 2,
        "step_index": 4,
        "action_id": "ACTION5",
        "planning_action_ids_before": ["ACTION3", "ACTION4", "ACTION5"],
        "planning_action_ids_after": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        "changed_cell_count": 5,
    })

    packet = QwenPacketBuilder().build_semantic_packet(snap, memory, HypothesisBank(), QwenRole.RESERVE, V8Config())
    transition = packet["action_surface"]["observed_transitions"][0]
    action5 = packet["action_model"]["actions"]["ACTION5"]

    assert transition["forward_source_matches_current"] is False
    assert transition["current_surface_position"] == "DIFFERENT_SURFACE"
    assert transition["replay_effect_status"].startswith("NOT_APPLICABLE")
    assert action5["available_now"] is True
    assert action5["surface_before_matches_current"] is False


def test_packet_marks_coupled_control_and_exact_geometry_as_facts_not_goals():
    grid = [[0 for _ in range(9)] for _ in range(9)]
    grid[1][1] = 1
    grid[4][4] = 2
    grid[7][7] = 3
    obs = {"grid": grid, "metadata": {"game_id": "g", "available_actions": ["ACTION1"], "level_index": 0}}
    memory = GameMemory()
    snap = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), memory, V8Config())
    first, second = snap.objects[:2]
    memory.action_effects[("ACTION1", None)] = ActionEffectRecord("ACTION1", None, "effect", 1, 0.57, 0, 0)
    memory.action_memory_records.append({
        "level_index": 0,
        "level_index_before": 0,
        "state_signature": snap.semantic_state_signature,
        "action_id": "ACTION1",
        "changed_cell_count": 4,
        "object_deltas": [
            {
                "object_id": first.object_id,
                "delta_centroid_rc": [1.0, 0.0],
                "motion_direction": "down",
                "before_bbox_rc": list(first.bbox_rc),
                "after_bbox_rc": [first.bbox_rc[0] + 1, first.bbox_rc[1], first.bbox_rc[2] + 1, first.bbox_rc[3]],
            },
            {
                "object_id": second.object_id,
                "delta_centroid_rc": [1.0, 0.0],
                "motion_direction": "down",
                "before_bbox_rc": list(second.bbox_rc),
                "after_bbox_rc": [second.bbox_rc[0] + 1, second.bbox_rc[1], second.bbox_rc[2] + 1, second.bbox_rc[3]],
            },
        ],
    })

    packet = QwenPacketBuilder().build_semantic_packet(snap, memory, HypothesisBank(), QwenRole.PRIMARY, V8Config())

    assert any(len(group["object_ids"]) == 2 for group in packet["scene"]["control_groups"])
    correspondence = next(fact for fact in packet["scene"]["priority_facts_not_goals"] if fact["type"] == "MOVABLE_REFERENCE_EXACT_GEOMETRY_CORRESPONDENCE")
    assert correspondence["movable_object_id"] in correspondence["object_ids"]
    assert correspondence["reference_object_id"] in correspondence["object_ids"]
    assert len(correspondence["source_to_reference_delta_xy"]) == 2
    assert correspondence["observed_source_translation_by_action"]
    source_effect = correspondence["observed_source_translation_by_action"][0]
    assert source_effect["available_now"] is True
    assert source_effect["planning_available_now"] is True
    assert "current_control_context_status" in source_effect
    assert "supported_control_group_switch_action_ids" in correspondence
    assert all("NOT_A_PRESELECTED_GOAL" in fact["goal_status"] for fact in packet["scene"]["priority_facts_not_goals"])


def test_control_groups_preserve_state_specific_simultaneous_pairs():
    objects = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    model = {
        "actions": {
            "ACTION1": {
                "observation_step_index": 1,
                "surface_before": ["ACTION1", "ACTION2"],
                "surface_after": ["ACTION1", "ACTION2"],
                "effects": [
                    {"kind": "translation", "object_id": "a", "delta_xy": [0, -1]},
                    {"kind": "translation", "object_id": "b", "delta_xy": [0, -1]},
                ],
            },
            "ACTION3": {
                "observation_step_index": 3,
                "surface_before": ["ACTION3", "ACTION4"],
                "surface_after": ["ACTION3", "ACTION4"],
                "effects": [
                    {"kind": "translation", "object_id": "b", "delta_xy": [-1, 0]},
                    {"kind": "translation", "object_id": "c", "delta_xy": [1, 0]},
                ],
            },
        },
    }

    groups = _control_groups(objects, model)

    assert {tuple(group["object_ids"]) for group in groups} == {("a", "b"), ("b", "c")}
    assert all(group["chronological_observations"] for group in groups)


def test_reciprocal_markers_and_coupled_groups_support_control_switch_inference():
    objects = [
        {"id": "controller_a", "bbox_xyxy": [0, 0, 2, 8]},
        {"id": "controller_b", "bbox_xyxy": [5, 0, 7, 5]},
        {"id": "payload", "bbox_xyxy": [10, 0, 12, 2]},
    ]
    groups = [
        {
            "id": "cg0",
            "object_ids": ["controller_a", "payload"],
            "observed_action_ids": ["ACTION3"],
            "chronological_observations": [{
                "surface_before": ["ACTION3", "ACTION4", "ACTION5"],
                "surface_after": ["ACTION3", "ACTION4", "ACTION5"],
            }],
        },
        {
            "id": "cg1",
            "object_ids": ["controller_b", "payload"],
            "observed_action_ids": ["ACTION1"],
            "chronological_observations": [{
                "surface_before": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                "surface_after": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            }],
        },
    ]
    transitions = [{
        "step_index": 5,
        "trigger_action_id": "ACTION5",
        "available_before": ["ACTION3", "ACTION4", "ACTION5"],
        "available_after": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        "surface_changed": True,
        "simultaneous_raw_visual_changes": {
            "reciprocal_isolated_interior_transition_pairs": [{
                "value_pair": ["0", "9"],
                "forward_locations_bbox_xyxy": [1, 1, 1, 7],
                "reverse_locations_bbox_xyxy": [6, 1, 6, 4],
            }],
        },
    }]

    candidates = _control_state_transition_candidates(objects, groups, transitions)
    model = {"actions": {"ACTION1": {}, "ACTION3": {}, "ACTION5": {}}}
    _annotate_control_context(model, groups, candidates)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["type"] == "SUPPORTED_CONTROL_GROUP_SWITCH"
    assert candidate["shared_translated_object_ids"] == ["payload"]
    assert candidate["current_inferred_control_group_id"] == "cg1"
    assert model["actions"]["ACTION1"]["current_control_context_status"] == "OBSERVED_IN_CURRENT_CONTROL_GROUP"
    assert model["actions"]["ACTION3"]["current_control_context_status"] == "OBSERVED_IN_NONCURRENT_CONTROL_GROUP"
    assert model["actions"]["ACTION5"]["current_control_context_status"] == "SUPPORTED_CONTROL_GROUP_SWITCH_TRIGGER"

    reverse_transitions = [*transitions, {
        "step_index": 6,
        "trigger_action_id": "ACTION5",
        "available_before": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        "available_after": ["ACTION3", "ACTION4"],
        "surface_changed": True,
        "simultaneous_raw_visual_changes": {
            "reciprocal_isolated_interior_transition_pairs": [{
                "value_pair": ["0", "9"],
                "forward_locations_bbox_xyxy": [6, 1, 6, 4],
                "reverse_locations_bbox_xyxy": [1, 1, 1, 7],
            }],
        },
    }]
    reversed_candidates = _control_state_transition_candidates(objects, groups, reverse_transitions)

    assert len(reversed_candidates) == 2
    assert reversed_candidates[-1]["before_control_group_id"] == "cg1"
    assert reversed_candidates[-1]["after_control_group_id"] == "cg0"
    assert reversed_candidates[-1]["after_surface_match_score"] == 20
    assert reversed_candidates[-1]["current_inferred_control_group_id"] == "cg0"


def test_focus_compaction_prunes_exact_geometry_references_to_omitted_objects():
    grid = [[0 for _ in range(32)] for _ in range(32)]
    for index in range(30):
        y = 1 + (index // 10) * 4
        x = 1 + (index % 10) * 3
        grid[y][x] = 1 + (index % 14)
    obs = {"grid": grid, "metadata": {"game_id": "g", "available_actions": ["ACTION1"]}}
    memory = GameMemory()
    snap = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), memory, V8Config())
    memory.action_effects[("ACTION1", None)] = ActionEffectRecord("ACTION1", None, "effect", 1, 0.57, 0, 0)

    packet = QwenPacketBuilder().build_semantic_packet(
        snap,
        memory,
        HypothesisBank(),
        QwenRole.PRIMARY,
        V8Config(max_semantic_objects_in_packet=12),
    )
    retained = {item["id"] for item in packet["scene"]["objects"]}

    assert len(retained) <= 12
    for item in packet["scene"]["objects"]:
        assert set(item["shape_geometry"]["same_exact_geometry_object_ids"]).issubset(retained)


def test_explicit_llama_completion_backend_uses_constrained_json(monkeypatch, tmp_path):
    model = tmp_path / "Qwen3.5-9B-Q4_K_M.gguf"
    completion = tmp_path / "llama-completion.exe"
    model.write_bytes(b"model")
    completion.write_text("", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        prompt_path = Path(command[command.index("-f") + 1])
        schema_path = Path(command[command.index("--json-schema-file") + 1])
        captured["prompt_path"] = prompt_path
        captured["schema_path"] = schema_path
        assert prompt_path.exists()
        assert schema_path.exists()
        assert b"\r\n" not in prompt_path.read_bytes()
        assert "RETURN_JSON_ONLY" in prompt_path.read_text(encoding="utf-8")
        assert prompt_path.read_text(encoding="utf-8").startswith("<|im_start|>user\n")
        assert prompt_path.read_text(encoding="utf-8").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["properties"]["decision"] == {"const": "PROPOSE"}
        hypothesis = schema["properties"]["hypotheses"]["items"]
        action_id = hypothesis["properties"]["action_runs"]["items"]["properties"]["action_id"]["enum"][0]
        output = {
            "schema_version": "v8.7.semantic_trajectories",
            "decision": "PROPOSE",
            "hypotheses": [{
                "id": "h1",
                "family": "other",
                "objective": {
                    "kind": "other",
                    "source_objects": [],
                    "reference_objects": [],
                    "description": "test hypothesis",
                },
                "relations": [],
                "basis": "grounded test",
                "action_runs": [{"action_id": action_id, "repeat": 1}],
                "status": "complete_candidate",
                "uncertainty": "",
                "confidence": 0.2,
            }],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(llm.os, "access", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(llm.subprocess, "run", fake_run)

    cfg = V8Config(
        qwen_backend="llama_cli",
        qwen_model_path=str(model),
        qwen_llama_cli_path=str(completion),
        qwen_llama_device="Vulkan0",
        qwen_require_runtime=True,
    )
    snap = _snapshot()
    result = QwenClient().call(QwenRole.PRIMARY, QwenPacketBuilder().build_semantic_packet(snap, _ready_memory(snap), HypothesisBank(), QwenRole.PRIMARY, cfg), cfg)

    assert result["schema_version"] == "v8.7.semantic_trajectories"
    assert result["decision"] == "PROPOSE"
    assert Path(captured["command"][0]) == completion
    assert "--prompt" not in captured["command"]
    assert "--no-conversation" in captured["command"]
    assert "--chat-template-kwargs" not in captured["command"]
    assert "-f" in captured["command"]
    assert not captured["prompt_path"].exists()
    assert not captured["schema_path"].exists()


def test_llama_cli_backend_falls_back_without_completion_binary(monkeypatch, tmp_path):
    model = tmp_path / "Qwen3.5-9B-Q4_K_M.gguf"
    cli = tmp_path / "llama-cli.exe"
    model.write_bytes(b"model")
    cli.write_text("", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        prompt_path = Path(command[command.index("-f") + 1])
        captured["prompt_path"] = prompt_path
        assert prompt_path.exists()
        return SimpleNamespace(returncode=0, stdout='{"schema_version":"v8.4.semantic_plan","decision":"ABSTAIN","abstain_reason":"test"}', stderr="")

    monkeypatch.setattr(llm.os, "access", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    cfg = V8Config(
        qwen_backend="llama_cli",
        qwen_model_path=str(model),
        qwen_llama_cli_path=str(cli),
        qwen_llama_device="CUDA0,CUDA1",
        qwen_split_mode="layer",
        qwen_tensor_split="1,1",
        qwen_require_runtime=True,
    )
    snap = _snapshot()

    result = QwenClient().call(QwenRole.PRIMARY, QwenPacketBuilder().build_semantic_packet(snap, _ready_memory(snap), HypothesisBank(), QwenRole.PRIMARY, cfg), cfg)

    assert result["schema_version"] == "v8.4.semantic_plan"
    assert Path(captured["command"][0]) == cli
    assert "--json-schema-file" not in captured["command"]
    assert "--single-turn" in captured["command"]
    assert "--chat-template-kwargs" in captured["command"]
    assert captured["command"][captured["command"].index("--device") + 1] == "CUDA0,CUDA1"
    assert captured["command"][captured["command"].index("--split-mode") + 1] == "layer"
    assert captured["command"][captured["command"].index("--tensor-split") + 1] == "1,1"
    assert not captured["prompt_path"].exists()


def test_v87_generation_schema_uses_ordered_run_length_trajectory():
    packet = {
        "execution_constraints": {
            "allowed_action_ids": ["ACTION1", "ACTION3", "ACTION5"],
            "allowed_object_ids": ["o0"],
            "allowed_relation_ids": [],
            "allowed_coordinate_candidate_ids": [],
            "max_plan_steps": 50,
        },
        "action_surface": {
            "actions": [
                {"id": "ACTION1", "available_now": False, "planning_allowed": True},
                {"id": "ACTION3", "available_now": True, "planning_allowed": True},
                {"id": "ACTION5", "available_now": False, "planning_allowed": True},
            ]
        },
    }

    schema = llm._allowed_output_schema(QwenRole.PRIMARY, packet)
    hypothesis = schema["properties"]["hypotheses"]["items"]

    run = hypothesis["properties"]["action_runs"]["items"]
    assert run["properties"]["action_id"]["enum"] == ["ACTION1", "ACTION3", "ACTION5"]
    repeat = run["properties"]["repeat"]
    assert {key: repeat[key] for key in ("type", "minimum", "maximum")} == {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
    }
    assert "coordinate action" in repeat["description"]
    assert "actions" not in hypothesis["properties"]
    assert "first_action" not in hypothesis["properties"]
    assert "remaining_actions" not in hypothesis["properties"]


def test_extract_json_retains_and_expands_v87_action_runs_without_off_by_one():
    raw = json.dumps({
        "schema_version": "v8.7.semantic_trajectories",
        "decision": "PROPOSE",
        "hypotheses": [{
            "id": "h1",
            "family": "other",
            "objective": {"kind": "other", "source_objects": [], "reference_objects": [], "description": "test"},
            "relations": [],
            "basis": "grounded test",
            "action_runs": [
                {"action_id": "ACTION3", "repeat": 5},
                {"action_id": "ACTION5", "repeat": 10},
            ],
            "status": "complete_candidate",
            "uncertainty": "unknown",
            "confidence": 0.2,
        }],
    })

    result = llm._extract_json(raw, role=QwenRole.PRIMARY)

    assert result["hypotheses"][0]["actions"] == ["ACTION3"] * 5 + ["ACTION5"] * 10
    assert result["hypotheses"][0]["action_runs"] == [
        {"action_id": "ACTION3", "repeat": 5},
        {"action_id": "ACTION5", "repeat": 10},
    ]


def test_extract_json_salvages_run_length_v87_hypothesis_from_broken_outer_json():
    hypothesis = {
        "id": "h1",
        "family": "other",
        "objective": {"kind": "other", "source_objects": [], "reference_objects": [], "description": "test"},
        "relations": [],
        "basis": "grounded test",
        "action_runs": [
            {"action_id": "ACTION3", "repeat": 1},
            {"action_id": "ACTION5", "repeat": 1},
        ],
        "status": "complete_candidate",
        "uncertainty": "unknown",
        "confidence": 0.2,
    }
    raw = (
        '{"schema_version":"v8.7.semantic_trajectories","decision":"PROPOSE","hypotheses":['
        + json.dumps(hypothesis)
        + ',{"id":"broken","basis":"unfinished\nstring"}]}'
    )

    result = llm._extract_json(raw, role=QwenRole.RESERVE)

    assert result["hypotheses"][0]["actions"] == ["ACTION3", "ACTION5"]


def test_extract_json_ignores_echoed_prompt_region():
    raw = (
        'OBSERVATION_PACKET_JSON=\n{"schema_version":"v8.4.observation_packet","derived_features":{"objects":[]}}\n'
        'RETURN_JSON_ONLY\n'
        '{"schema_version":"v8.4.semantic_plan","decision":"ABSTAIN","abstain_reason":"test"}\n'
    )
    result = llm._extract_json(raw, role=QwenRole.PRIMARY)

    assert result == {"schema_version": "v8.4.semantic_plan", "decision": "ABSTAIN", "abstain_reason": "test"}


def test_extract_json_does_not_accept_prompt_json_when_response_missing():
    raw = (
        'OBSERVATION_PACKET_JSON=\n{"schema_version":"v8.4.observation_packet","derived_features":{"objects":[]}}\n'
        'RETURN_JSON_ONLY\n'
        "model started but did not finish"
    )

    assert llm._extract_json(raw, role=QwenRole.PRIMARY) == {}


def test_extract_json_recovers_after_truncated_prompt_echo():
    raw = (
        'OBSERVATION_PACKET_JSON=\n{"schema_version":"v8.4.observation_packet","objects" ... (truncated)\n\n'
        '{"schema_version":"v8.4.semantic_plan","decision":"ABSTAIN","abstain_reason":"test"}\n'
    )

    assert llm._extract_json(raw, role=QwenRole.PRIMARY) == {"schema_version": "v8.4.semantic_plan", "decision": "ABSTAIN", "abstain_reason": "test"}


def test_extract_json_repairs_model_trailing_commas():
    raw = (
        'llama log\n'
        '{"schema_version":"v8.5.semantic_trajectory_hypotheses","decision":"PROPOSE","hypotheses":[{'
        '"hypothesis_id":"h1","rank":1,"family":"other",'
        '"goal":{"operation":"other","object_ids":[],},'
        '"evidence":{"supporting_fact_ids":[],"supporting_action_ids":[],"summary":"x"},'
        '"trajectory":[{"action_id":"ACTION1",}],'
        '"trajectory_status":"candidate_complete",'
        '"expected_progress":{"description":"x","tracked_object_ids":[],"tracked_relation_ids":[]},'
        '"completion_criterion":"x","main_uncertainty":"x","confidence":0.1'
        '}]}'
    )

    result = llm._extract_json(raw, role=QwenRole.PRIMARY)

    assert result["schema_version"] == "v8.5.semantic_trajectory_hypotheses"
    assert result["hypotheses"][0]["trajectory"][0]["action_id"] == "ACTION1"


def test_extract_json_rejects_v86_schema_invalid_output():
    raw = (
        '{"schema_version":"v8.6.semantic_trajectory_hypotheses","decision":"PROPOSE","hypotheses":[{'
        '"id":"h1","family":"other","objects":[],"relations":[],"basis":"x",'
        '"actions":[],"status":"prefix_until_reobservation","uncertainty":"x","confidence":0.1'
        '}]}'
    )

    assert llm._extract_json(raw, role=QwenRole.PRIMARY) == {}


def test_extract_json_accepts_v86_answer_with_descriptive_basis_after_llama_echo():
    basis = "x" * 220
    uncertainty = "y" * 170
    raw = (
        "Loading model...\n"
        "> You are a semantic trajectory proposer ... (truncated)\n"
        + json.dumps(
            {
                "schema_version": "v8.6.semantic_trajectory_hypotheses",
                "decision": "PROPOSE",
                "hypotheses": [
                    {
                        "id": "h1",
                        "family": "other",
                        "objects": [],
                        "relations": [],
                        "basis": basis,
                        "actions": ["ACTION1", "ACTION1"],
                        "status": "prefix_until_reobservation",
                        "uncertainty": uncertainty,
                        "confidence": 0.1,
                    }
                ],
            }
        )
        + "\n[ Prompt: 143.5 t/s | Generation: 8.5 t/s ]\n"
    )

    result = llm._extract_json(raw, role=QwenRole.PRIMARY)

    assert result["schema_version"] == "v8.6.semantic_trajectory_hypotheses"
    assert result["hypotheses"][0]["basis"] == basis
    assert result["hypotheses"][0]["actions"] == ["ACTION1", "ACTION1"]


def test_extract_json_salvages_valid_v87_hypotheses_without_an_extra_call():
    valid = {
        "id": "h1",
        "family": "spatial_configuration",
        "objective": {
            "kind": "match_or_overlap",
            "source_objects": ["o0"],
            "reference_objects": ["o1"],
            "description": "Move the source toward the reference.",
        },
        "relations": ["r0"],
        "basis": "The observed translation can reduce the supplied displacement.",
        "actions": ["ACTION1", "ACTION1"],
        "status": "complete_candidate",
        "uncertainty": "collision behavior unknown",
        "confidence": 0.7,
    }
    invalid = dict(valid)
    invalid["id"] = "h2"
    invalid["actions"] = []
    raw = "llama log\n" + json.dumps({
        "schema_version": "v8.7.semantic_trajectories",
        "decision": "PROPOSE",
        "hypotheses": [valid, invalid],
    })

    result = llm._extract_json(raw, role=QwenRole.PRIMARY)

    assert [item["id"] for item in result["hypotheses"]] == ["h1"]
    assert result["hypotheses"][0]["actions"] == ["ACTION1", "ACTION1"]


def test_extract_json_salvages_valid_v87_object_when_later_sibling_breaks_json():
    valid = {
        "id": "h1",
        "family": "object_correspondence",
        "objective": {
            "kind": "match_or_overlap",
            "source_objects": ["o0"],
            "reference_objects": ["o1"],
            "description": "Move the source toward the exact-shape reference.",
        },
        "relations": ["r0"],
        "basis": "ACTION3 is the observed current-context horizontal vector.",
        "actions": ["ACTION3", "ACTION3"],
        "status": "complete_candidate",
        "uncertainty": "Vertical movement needs another control context.",
        "confidence": 0.6,
    }
    raw = (
        '{"schema_version":"v8.7.semantic_trajectories","decision":"PROPOSE","hypotheses":['
        + json.dumps(valid)
        + ',{"id":"broken","family":"other","basis":"unfinished\nstring","actions":[]}]}'
    )

    result = llm._extract_json(raw, role=QwenRole.RESERVE)

    assert [item["id"] for item in result["hypotheses"]] == ["h1"]
    assert result["hypotheses"][0]["actions"] == ["ACTION3", "ACTION3"]


def test_first_no_effect_observation_is_tail_feedback_for_qwen():
    cfg = V8Config(enable_qwen=False, qwen_backend="disabled")
    session = GameSession(cfg)
    obs0 = {"grid": [[0, 0], [0, 1]], "metadata": {"game_id": "g", "level_index": 0, "frame_index": 0, "available_actions": ["ACTION1"], "state": "PLAYING"}}
    obs1 = {"grid": [[0, 0], [0, 1]], "metadata": {"game_id": "g", "level_index": 0, "frame_index": 1, "available_actions": ["ACTION1"], "state": "PLAYING"}}

    session.act(obs0)
    assert session.observe_action_result(obs1) is True
    for action_id in session._latest_snapshot.planning_action_ids or session._latest_snapshot.available_actions:
        if action_id in session._latest_snapshot.coordinate_action_ids:
            continue
        session.memory.action_effects[(action_id, None)] = ActionEffectRecord(action_id, None, "no_effect", 1, 0.57, 0, 1)
    packet = QwenPacketBuilder().build_semantic_packet(session._latest_snapshot, session.memory, session.bank, QwenRole.PRIMARY, cfg)

    assert packet["memory"]["recent_evidence"]["failed_or_irrelevant_steps"]
    assert packet["memory"]["recent_evidence"]["failed_or_irrelevant_steps"][-1]["outcome"] == "typed_no_effect_observed"


def test_successful_action_probe_is_mechanics_evidence_not_failed():
    cfg = V8Config(enable_qwen=False, qwen_backend="disabled")
    session = GameSession(cfg)
    obs0 = {"grid": [[0, 0], [0, 1]], "metadata": {"game_id": "g", "level_index": 0, "frame_index": 0, "available_actions": ["ACTION1"], "state": "PLAYING"}}
    obs1 = {"grid": [[0, 0], [1, 0]], "metadata": {"game_id": "g", "level_index": 0, "frame_index": 1, "available_actions": ["ACTION1"], "state": "PLAYING"}}

    session.act(obs0)
    assert session.observe_action_result(obs1) is True
    for action_id in session._latest_snapshot.planning_action_ids or session._latest_snapshot.available_actions:
        if action_id in session._latest_snapshot.coordinate_action_ids:
            continue
        session.memory.action_effects[(action_id, None)] = ActionEffectRecord(action_id, None, "effect", 1, 0.57, 0, 1)
    packet = QwenPacketBuilder().build_semantic_packet(session._latest_snapshot, session.memory, session.bank, QwenRole.PRIMARY, cfg)

    assert packet["memory"]["recent_evidence"]["mechanics_probe_summary"]
    assert packet["memory"]["recent_evidence"]["mechanics_probe_summary"][-1] == {
        "action_id": "ACTION1",
        "outcome": "CONFIRMED_MOTION_OR_EFFECT",
    }
    assert packet["memory"]["recent_evidence"]["failed_or_irrelevant_steps"] == []


def test_object_applicability_memory_is_in_qwen_packet():
    snap = _snapshot()
    target = snap.objects[0]
    contract = VerificationContract(
        contract_id="vc_obj",
        kind=VerificationContractKind.LOCAL_TARGET_CHANGE,
        target_object_ids=(target.object_id,),
        target_region_rc=target.bbox_rc,
        target_signature=target.stable_hash,
    )
    action = CandidateAction("ACTION1", verification_contract=contract)
    memory = GameMemory()
    memory.add_judgment(Judgment(
        truth=TriTruth.TRUE,
        relevance=Relevance.RELEVANT,
        validity=Validity.VALID,
        progress=Progress.POSITIVE,
        attribution=Attribution.ACTION_LINKED,
        reason_code="target_local_change_observed",
        observed_delta={
            "level_index": snap.level_index,
            "step_index": snap.step_index + 1,
            "target_objects_before": [{
                "object_id": target.object_id,
                "stable_hash": target.stable_hash,
                "shape_signature": target.shape_signature,
                "topology_signature": target.topology_signature,
                "colors": list(target.colors),
                "area": target.area,
                "bbox_rc": list(target.bbox_rc),
                "centroid_rc": list(target.centroid_rc),
                "tags": list(target.tags),
                "border_touching": list(target.border_touching),
            }],
            "target_objects_after": [],
        },
        affected_objects=(target.object_id,),
        affected_relations=(),
        score_delta=None,
        terminal_delta=False,
        action=action,
        hypothesis_id=None,
        before_hash=snap.grid_hash,
        after_hash="after",
        contract_kind=contract.kind,
        observed_information_gain=0.1,
    ))

    for action_id in snap.planning_action_ids or snap.available_actions:
        memory.action_effects[(action_id, None)] = ActionEffectRecord(action_id, None, "effect", 1, 0.57, 0, 1)
    packet = QwenPacketBuilder().build_semantic_packet(snap, memory, HypothesisBank(), QwenRole.PRIMARY, V8Config())
    evidence = packet["memory"]["recent_evidence"]

    assert evidence["successful_steps"]
    assert evidence["successful_steps"][-1]["action_id"] == "ACTION1"
    assert packet["scene"]["objects"]
    assert packet["scene"]["coordinate_candidates"]
    assert "ACTION6" in packet["execution_constraints"]["allowed_action_ids"]


def test_object_applicability_distinguishes_global_and_internal_frames():
    def obj(object_id, bbox, tags, borders):
        return SimpleNamespace(
            object_id=object_id,
            stable_hash=f"hash_{object_id}",
            shape_signature=f"shape_{object_id}",
            topology_signature=f"topo_{object_id}",
            colors=(1,),
            area=45,
            bbox_rc=bbox,
            centroid_rc=(float(bbox[0]), float(bbox[1])),
            tags=tuple(tags),
            border_touching=tuple(borders),
        )

    snapshot = SimpleNamespace(
        height=64,
        width=64,
        objects=(
            obj("global", (0, 0, 63, 63), ("frame_like", "border_touching"), ("top", "bottom", "left", "right")),
            obj("internal", (15, 15, 23, 23), ("frame_like",), ()),
        ),
        coordinate_targets=(),
    )

    applicability = GameMemory().object_applicability_for_qwen(snapshot, V8Config())
    priorities = {item["object_id"]: item["target_priority"] for item in applicability["current_object_descriptors"]}

    assert priorities["global"] == "deprioritize_global_frame_container"
    assert priorities["internal"] == "normal_internal_frame_like_object"


def test_full_grid_container_is_scene_entity_but_not_targetable_object():
    def obj(object_id, bbox, area):
        return SimpleNamespace(
            object_id=object_id,
            track_id=object_id,
            bbox_rc=bbox,
            centroid_rc=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            area=area,
            color_histogram={1: area},
            salience_score=1.0,
            shape_signature=f"shape_{object_id}",
            local_mask_hex_rows=["1"],
        )

    snapshot = SimpleNamespace(
        height=64,
        width=64,
        objects=(
            obj("container", (0, 0, 63, 63), 316),
            obj("tile", (15, 18, 23, 26), 45),
        ),
        full_grid_hex_rows=[],
        palette_histogram={},
    )

    objects = _objects(snapshot, V8Config())
    entities = _scene_entities(snapshot, objects, V8Config())

    assert [item["id"] for item in objects] == ["tile"]
    assert any(item["id"] == "scene_container" and item["linked_object_id"] is None for item in entities)


def test_action_effect_table_omits_container_object_motion():
    snapshot = SimpleNamespace(width=64, height=64)
    memory = SimpleNamespace(action_memory_records=[{
        "action_id": "ACTION1",
        "changed_cell_count": 12,
        "changed_bbox_rc": [0, 0, 63, 63],
        "object_deltas": [
            {"object_id": "container", "delta_centroid_rc": [0.0, 0.0], "motion_direction": "stationary"},
            {"object_id": "tile", "delta_centroid_rc": [-3.0, 0.0], "motion_direction": "up"},
        ],
    }])

    table = _action_effect_table(memory, snapshot, ["ACTION1"], {"tile"}, V8Config())

    assert [item["object_id"] for item in table[0]["object_motion"]] == ["tile"]
    assert table[0]["stationary_or_local_changes"] == []


def test_centroid_drift_from_shape_shrink_is_not_translation():
    snapshot = SimpleNamespace(width=64, height=64)
    memory = SimpleNamespace(action_memory_records=[{
        "action_id": "ACTION1",
        "changed_cell_count": 1,
        "object_deltas": [{
            "object_id": "edge_strip",
            "lifecycle": "persisted",
            "before_bbox_rc": [0, 63, 63, 63],
            "after_bbox_rc": [1, 63, 63, 63],
            "delta_centroid_rc": [0.5, 0.0],
            "motion_direction": "down",
            "before_shape_signature": "long_64",
            "after_shape_signature": "long_63",
            "shape_changed": True,
            "palette_changed": False,
            "area_delta": -1,
        }],
    }])

    row = _action_effect_table(memory, snapshot, ["ACTION1"], {"edge_strip"}, V8Config())[0]
    effects = _typed_action_effects(row)

    assert row["object_motion"] == []
    assert any(effect["kind"] == "shape_area_or_visibility_change" for effect in effects)


def test_same_color_recovery_keeps_l_object_when_touching_structural_strip():
    grid = [[0 for _ in range(12)] for _ in range(12)]
    for r in range(12):
        grid[r][5] = 3
    for r in range(2, 5):
        for c in range(6, 9):
            if r == 2 or c == 6:
                grid[r][c] = 4
    obs = {"grid": grid, "metadata": {"game_id": "g", "available_actions": ["ACTION1"], "frame_index": 0}}
    snap = ARGALiteBuilder().build(GameAdapter().to_world_state(obs), GameMemory(), V8Config())

    packet_objects = _objects(snap, V8Config())

    assert any(item["palette_histogram"] == {"4": 5} for item in packet_objects)
    assert any(item["palette_histogram"] == {"3": 12} for item in packet_objects)
    recovered = next(item for item in packet_objects if item["palette_histogram"] == {"4": 5})
    assert recovered["shape_profile"]["row_occupancy"] == [3, 1, 1]


def test_object_geometry_groups_distinguish_l_orientation():
    packet_objects = [
        {"id": "left_a", "shape_profile": {"bbox_hw": [3, 3]}, "local_hex_rows": ["111", "1..", "1.."]},
        {"id": "left_b", "shape_profile": {"bbox_hw": [3, 3]}, "local_hex_rows": ["222", "2..", "2.."]},
        {"id": "right", "shape_profile": {"bbox_hw": [3, 3]}, "local_hex_rows": ["333", "..3", "..3"]},
    ]

    groups = _annotate_object_geometry(packet_objects)

    assert len(groups) == 2
    assert any(set(group["object_ids"]) == {"left_a", "left_b"} and group["orientation_label"] == "top_left_corner_L" for group in groups)
    assert any(group["object_ids"] == ["right"] and group["orientation_label"] == "top_right_corner_L" for group in groups)
    assert packet_objects[0]["shape_geometry"]["same_exact_geometry_object_ids"] == ["left_a", "left_b"]
    assert packet_objects[2]["shape_geometry"]["same_exact_geometry_object_ids"] == ["right"]


def test_regions_keep_edge_geometry_neutral_and_skip_full_grid_container():
    def obj(object_id, bbox, color, area):
        return SimpleNamespace(
            object_id=object_id,
            track_id=object_id,
            bbox_rc=bbox,
            centroid_rc=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            area=area,
            color_histogram={color: area},
            salience_score=1.0,
            shape_signature=f"shape_{object_id}",
            local_mask_hex_rows=["1"],
        )

    snapshot = SimpleNamespace(
        height=64,
        width=64,
        objects=(
            obj("container", (0, 0, 63, 63), 5, 316),
            obj("right_a", (0, 63, 3, 63), 5, 4),
            obj("right_b", (4, 63, 63, 63), 11, 60),
            obj("bottom", (63, 0, 63, 62), 5, 63),
            obj("divider", (0, 30, 62, 32), 10, 189),
        ),
    )

    regions = _regions(snapshot)
    neutral = regions["large_or_edge_regions"]

    assert all(item["semantic_role"] == "UNKNOWN" for item in neutral)
    assert any(item["bbox_xyxy"] == [63, 0, 63, 3] for item in neutral)
    assert any(item["bbox_xyxy"] == [63, 4, 63, 63] for item in neutral)
    assert all(item["bbox_xyxy"] != [0, 0, 63, 63] for item in neutral)
