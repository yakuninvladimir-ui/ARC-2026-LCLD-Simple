from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import List, Optional

from .observe import stable_hash
from .types import ARGALiteSnapshot, HypothesisItem, Progress, TestStep


@dataclass
class TrajectoryStep:
    action_id: str
    coordinate_candidate_id: Optional[str] = None
    repeat: int = 1
    expected_state_hash: Optional[str] = None
    kind: str = "corrected_trajectory_step"
    target_object_id: Optional[str] = None
    target_relation_id: Optional[str] = None
    target_object_ids: tuple[str, ...] = ()
    target_relation_ids: tuple[str, ...] = ()
    expected_observation: Optional[str] = None
    contract_kind: Optional[str] = None
    question_type: Optional[str] = None

    def to_test_step(self, *, semantic_binding=None) -> TestStep:
        return TestStep(
            kind=self.kind or "corrected_trajectory_step",
            action_id=self.action_id,
            target_object_id=self.target_object_id,
            target_relation_id=self.target_relation_id,
            target_object_ids=tuple(self.target_object_ids or ()),
            target_relation_ids=tuple(self.target_relation_ids or ()),
            coordinate_candidate_id=self.coordinate_candidate_id,
            expected_observation=self.expected_observation,
            contract_kind=self.contract_kind,
            question_type=self.question_type,
            semantic_binding=semantic_binding,
        )


@dataclass
class TrajectoryPlan:
    hypothesis_id: str
    goal: dict | None
    steps: List[TrajectoryStep]
    total_steps: int
    expected_final_state_hash: str


@dataclass
class TrajectoryVerificationResult:
    status: str  # "ACCEPT", "REJECT", "CORRECTED", "PARTIAL", "PASSTHROUGH"
    plan: Optional[TrajectoryPlan] = None
    corrected_steps: Optional[List[TrajectoryStep]] = None
    reason: str = ""
    confidence: float = 0.0
    details: dict[str, object] | None = None


def resolve_goal_hash(item: HypothesisItem, snapshot: ARGALiteSnapshot, memory: object, simulated_final: str | None = None) -> tuple[str | None, str]:
    """Prefer model goal hash, then positive trajectory evidence, then simulated final."""
    if isinstance(item.goal_spec, dict):
        for key in ("expected_final_state_hash", "goal_signature", "final_grid_hash"):
            value = item.goal_spec.get(key)
            if isinstance(value, str) and value:
                return value, f"goal_spec.{key}"
    model_hash = getattr(item, "expected_final_state_hash", None)
    if isinstance(model_hash, str) and model_hash:
        return model_hash, "item.expected_final_state_hash"
    target_signatures = [
        ev.end_state_signature
        for ev in getattr(memory, "trajectory_evaluations", []) or []
        if getattr(ev, "hypothesis_id", None) == item.hypothesis_id
        and getattr(ev, "goal_progress", None) is Progress.POSITIVE
        and ev.end_state_signature
    ]
    if target_signatures:
        return str(target_signatures[-1]), "positive_trajectory_evaluation"
    if isinstance(simulated_final, str) and simulated_final:
        return simulated_final, "simulated_final_state"
    # Never invent a goal from the current frame. Unknown goal stays unresolved so
    # Contour B can PASSTHROUGH the model trajectory for empiric execution.
    return None, "unresolved"


def trajectory_steps_from_hypothesis(item: HypothesisItem) -> List[TrajectoryStep]:
    steps: List[TrajectoryStep] = []
    for step in (item.test_plan or ()):
        if getattr(step, "action_id", None) is None:
            continue
        steps.append(
            TrajectoryStep(
                action_id=step.action_id,
                coordinate_candidate_id=getattr(step, "coordinate_candidate_id", None),
                repeat=getattr(step, "repeat", 1) or 1,
                kind=str(getattr(step, "kind", None) or "corrected_trajectory_step"),
                target_object_id=getattr(step, "target_object_id", None),
                target_relation_id=getattr(step, "target_relation_id", None),
                target_object_ids=tuple(getattr(step, "target_object_ids", ()) or ()),
                target_relation_ids=tuple(getattr(step, "target_relation_ids", ()) or ()),
                expected_observation=getattr(step, "expected_observation", None),
                contract_kind=getattr(step, "contract_kind", None),
                question_type=getattr(step, "question_type", None),
            )
        )
    return steps


def rebind_plan_to_test_steps(
    steps: List[TrajectoryStep],
    snapshot: ARGALiteSnapshot,
    hypothesis_id: str | None,
    *,
    semantic_binding=None,
    original_plan: tuple = (),
) -> tuple:
    """Rebuild TestStep list and re-attach VerificationContract targets via VerificationBinder."""
    from .verification import VerificationBinder

    binder = VerificationBinder()
    original_by_index = list(original_plan or ())
    rebound: list[TestStep] = []
    for index, step in enumerate(steps):
        seed = step
        if index < len(original_by_index):
            original = original_by_index[index]
            # Prefer original semantic targeting when the action matches the model step.
            if getattr(original, "action_id", None) == step.action_id:
                seed = TrajectoryStep(
                    action_id=step.action_id,
                    coordinate_candidate_id=step.coordinate_candidate_id or getattr(original, "coordinate_candidate_id", None),
                    repeat=step.repeat,
                    expected_state_hash=step.expected_state_hash,
                    kind=str(getattr(original, "kind", None) or step.kind or "corrected_trajectory_step"),
                    target_object_id=getattr(original, "target_object_id", None) or step.target_object_id,
                    target_relation_id=getattr(original, "target_relation_id", None) or step.target_relation_id,
                    target_object_ids=tuple(getattr(original, "target_object_ids", ()) or step.target_object_ids or ()),
                    target_relation_ids=tuple(getattr(original, "target_relation_ids", ()) or step.target_relation_ids or ()),
                    expected_observation=getattr(original, "expected_observation", None) or step.expected_observation,
                    contract_kind=getattr(original, "contract_kind", None) or step.contract_kind,
                    question_type=getattr(original, "question_type", None) or step.question_type,
                )
        binding = semantic_binding or getattr(original_by_index[index], "semantic_binding", None) if index < len(original_by_index) else semantic_binding
        test_step = seed.to_test_step(semantic_binding=binding)
        # Force binder to capture before-metrics / contracts for the current snapshot.
        contract = binder.bind(test_step, snapshot, hypothesis_id)
        if contract is not None and binding is not None and test_step.semantic_binding is None:
            test_step = replace(test_step, semantic_binding=binding)
        # Even if contract is None (stale targets), keep the step so empiric execution can still try first action.
        rebound.append(test_step)
    return tuple(rebound)


def _objective_is_grounded(item: HypothesisItem) -> bool:
    binding = getattr(item, "semantic_binding", None)
    if binding is None:
        # Legacy hypotheses without binder still may be executable if first action is legal.
        return bool(item.test_plan)
    status = getattr(binding, "status", None)
    status_value = getattr(status, "value", status)
    return str(status_value).upper() in {"GROUNDED", "PARTIAL"}


def _first_action_legal(steps: List[TrajectoryStep], allowed_actions: set[str], allowed_candidates: set[str]) -> bool:
    if not steps:
        return False
    first = steps[0]
    if first.action_id not in allowed_actions:
        return False
    if first.coordinate_candidate_id is not None and allowed_candidates and first.coordinate_candidate_id not in allowed_candidates:
        return False
    return True



class HexStateComparator:
    """Comparator for hex-based states with enriched color comparison and mask-based matching.

    Provides:
    - signature_from_snapshot(snapshot): stable compact signature of a snapshot's hex rows.
    - compare_hex(expected_rows, observed_rows): low-level hex diff between two row-sequences.
    - compare_snapshots(expected, observed): a structured delta mapping
      expected_object_id -> {matched_obs_id, dx, dy, color_from, color_to, color_histogram_from, color_histogram_to, color_histogram_iou, appeared, vanished}
      plus overall appeared/vanished lists and hex-level delta summary.

    Matching heuristic (when IDs differ) uses:
    - local_mask hex similarity
    - centroid proximity
    - color histogram overlap

    The function is conservative: only creates a match when combined score exceeds a threshold.
    """

    @staticmethod
    def signature_from_snapshot(snapshot: ARGALiteSnapshot) -> str:
        return stable_hash((tuple(snapshot.full_grid_hex_rows), snapshot.semantic_state_signature), "hex_")

    @staticmethod
    def _dominant_color(obj) -> int | None:
        if not obj:
            return None
        try:
            if getattr(obj, "colors", None):
                colors = getattr(obj, "colors")
                if isinstance(colors, (list, tuple)) and len(colors) > 0:
                    return colors[0]
            hist = getattr(obj, "color_histogram", None)
            if isinstance(hist, dict) and hist:
                return max(hist.items(), key=lambda kv: kv[1])[0]
        except Exception:
            return None
        return None

    @staticmethod
    def _hist_iou(hist_a: dict | None, hist_b: dict | None) -> float:
        if not hist_a or not hist_b:
            return 0.0
        keys = set(hist_a) | set(hist_b)
        inter = 0
        union = 0
        for k in keys:
            a = hist_a.get(k, 0)
            b = hist_b.get(k, 0)
            inter += min(a, b)
            union += max(a, b)
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _mask_similarity(mask_a: tuple[str, ...] | list[str] | None, mask_b: tuple[str, ...] | list[str] | None) -> float:
        """Compute a simple normalized similarity between two local mask hex rows.

        Score = matching_nonempty_cells / max(total_nonempty_cells_union, 1)
        Returns in [0,1].
        """
        if not mask_a or not mask_b:
            return 0.0
        a = list(mask_a)
        b = list(mask_b)
        max_h = max(len(a), len(b))
        a += [""] * (max_h - len(a))
        b += [""] * (max_h - len(b))
        matches = 0
        total_nonempty = 0
        for row_a, row_b in zip(a, b):
            max_w = max(len(row_a), len(row_b))
            ra = row_a.ljust(max_w, " ")
            rb = row_b.ljust(max_w, " ")
            for ca, cb in zip(ra, rb):
                nonempty = (ca != " " ) or (cb != " ")
                if nonempty:
                    total_nonempty += 1
                    if ca == cb and ca != " ":
                        matches += 1
        if total_nonempty == 0:
            return 0.0
        return matches / total_nonempty

    @staticmethod
    def compare_hex(expected_rows: tuple[str, ...] | list[str], observed_rows: tuple[str, ...] | list[str]) -> dict:
        if expected_rows is None or observed_rows is None:
            return {"changed_cell_count": 0, "changed_bbox": None, "changed_cells": []}
        er = list(expected_rows)
        orr = list(observed_rows)
        max_h = max(len(er), len(orr))
        er += [""] * (max_h - len(er))
        orr += [""] * (max_h - len(orr))
        changed = []
        minx = miny = None
        maxx = maxy = None
        for y, (row_e, row_o) in enumerate(zip(er, orr)):
            max_w = max(len(row_e), len(row_o))
            re = row_e.ljust(max_w, " ")
            ro = row_o.ljust(max_w, " ")
            for x, (ce, co) in enumerate(zip(re, ro)):
                if ce != co:
                    changed.append((x, y, ce, co))
                    if minx is None:
                        minx = maxx = x
                        miny = maxy = y
                    else:
                        minx = min(minx, x)
                        miny = min(miny, y)
                        maxx = max(maxx, x)
                        maxy = max(maxy, y)
        bbox = (minx, miny, maxx, maxy) if minx is not None else None
        return {"changed_cell_count": len(changed), "changed_bbox": bbox, "changed_cells": changed}

    @staticmethod
    def compare_snapshots(expected: ARGALiteSnapshot, observed: ARGALiteSnapshot) -> dict:
        exp_objs = {o.object_id: o for o in (expected.objects or ())}
        obs_objs = {o.object_id: o for o in (observed.objects or ())}
        # first pass: direct id matches
        matched_expected: dict[str, str] = {}
        matched_observed: set[str] = set()
        for oid in exp_objs:
            if oid in obs_objs:
                matched_expected[oid] = oid
                matched_observed.add(oid)

        # Build lists of unmatched
        unmatched_exp = [o for k, o in exp_objs.items() if k not in matched_expected]
        unmatched_obs = [o for k, o in obs_objs.items() if k not in matched_observed]

        width = max(getattr(expected, "width", 0) or 0, getattr(observed, "width", 0) or 0)
        height = max(getattr(expected, "height", 0) or 0, getattr(observed, "height", 0) or 0)
        diag = max(1.0, (width ** 2 + height ** 2) ** 0.5)

        # match remaining by heuristic scores
        for e in unmatched_exp:
            best = None
            best_score = 0.0
            for o in unmatched_obs:
                if o.object_id in matched_observed:
                    continue
                # mask similarity
                mask_sim = HexStateComparator._mask_similarity(getattr(e, "local_mask_hex_rows", None), getattr(o, "local_mask_hex_rows", None))
                # centroid proximity
                try:
                    ddx = o.centroid_rc[1] - e.centroid_rc[1]
                    ddy = o.centroid_rc[0] - e.centroid_rc[0]
                    dist = (ddx ** 2 + ddy ** 2) ** 0.5
                    centroid_score = max(0.0, 1.0 - (dist / diag))
                except Exception:
                    centroid_score = 0.0
                # color histogram IOU
                hist_iou = HexStateComparator._hist_iou(getattr(e, "color_histogram", None), getattr(o, "color_histogram", None))
                # combine weights
                score = 0.5 * mask_sim + 0.3 * centroid_score + 0.2 * hist_iou
                if score > best_score:
                    best_score = score
                    best = o
            # accept match if above threshold
            if best is not None and best_score >= 0.35:
                matched_expected[e.object_id] = best.object_id
                matched_observed.add(best.object_id)

        object_deltas: dict[str, dict] = {}
        appeared = []
        vanished = []

        # Build deltas for expected objects (matched or vanished)
        for exp_id, exp_obj in exp_objs.items():
            if exp_id not in matched_expected:
                # vanished
                object_deltas[exp_id] = {
                    "matched_obs_id": None,
                    "dx": None,
                    "dy": None,
                    "color_from": HexStateComparator._dominant_color(exp_obj),
                    "color_to": None,
                    "color_histogram_from": getattr(exp_obj, "color_histogram", None),
                    "color_histogram_to": None,
                    "color_histogram_iou": 0.0,
                    "appeared": False,
                    "vanished": True,
                }
                vanished.append(exp_id)
                continue
            obs_id = matched_expected[exp_id]
            obs_obj = obs_objs.get(obs_id)
            if obs_obj is None:
                object_deltas[exp_id] = {
                    "matched_obs_id": None,
                    "dx": None,
                    "dy": None,
                    "color_from": HexStateComparator._dominant_color(exp_obj),
                    "color_to": None,
                    "color_histogram_from": getattr(exp_obj, "color_histogram", None),
                    "color_histogram_to": None,
                    "color_histogram_iou": 0.0,
                    "appeared": False,
                    "vanished": True,
                }
                vanished.append(exp_id)
                continue
            # compute dx/dy and color comparisons
            try:
                dx = obs_obj.centroid_rc[1] - exp_obj.centroid_rc[1]
                dy = obs_obj.centroid_rc[0] - exp_obj.centroid_rc[0]
            except Exception:
                dx = None
                dy = None
            color_from = HexStateComparator._dominant_color(exp_obj)
            color_to = HexStateComparator._dominant_color(obs_obj)
            hist_iou = HexStateComparator._hist_iou(getattr(exp_obj, "color_histogram", None), getattr(obs_obj, "color_histogram", None))
            object_deltas[exp_id] = {
                "matched_obs_id": obs_id,
                "dx": dx,
                "dy": dy,
                "color_from": color_from,
                "color_to": color_to,
                "color_histogram_from": getattr(exp_obj, "color_histogram", None),
                "color_histogram_to": getattr(obs_obj, "color_histogram", None),
                "color_histogram_iou": hist_iou,
                "appeared": False,
                "vanished": False,
            }

        # remaining observed objects that were not matched are appeared
        for obs_id, obs_obj in obs_objs.items():
            if obs_id in matched_observed:
                continue
            object_deltas[obs_id] = {
                "matched_obs_id": None,
                "dx": None,
                "dy": None,
                "color_from": None,
                "color_to": HexStateComparator._dominant_color(obs_obj),
                "color_histogram_from": None,
                "color_histogram_to": getattr(obs_obj, "color_histogram", None),
                "color_histogram_iou": 0.0,
                "appeared": True,
                "vanished": False,
            }
            appeared.append(obs_id)

        hex_delta = HexStateComparator.compare_hex(getattr(expected, "full_grid_hex_rows", ()), getattr(observed, "full_grid_hex_rows", ()))
        return {
            "object_deltas": object_deltas,
            "appeared": appeared,
            "vanished": vanished,
            "hex_delta": hex_delta,
        }


class TrajectoryVerifier:
    """Lightweight verifier which checks syntactic reachability and attempts
    correction via TrajectoryPlanner using recorded memory transitions.
    """

    def __init__(self, config=None):
        self.config = config

    @staticmethod
    def _goal_snapshot_from_spec(snapshot: ARGALiteSnapshot, goal_spec: dict[str, object] | None) -> ARGALiteSnapshot | None:
        if not goal_spec or not isinstance(goal_spec, dict):
            return None
        target_xy = goal_spec.get("target_xy")
        object_id = goal_spec.get("object_id") if isinstance(goal_spec.get("object_id"), str) else None
        grid_rows = goal_spec.get("expected_grid_hex_rows")
        if target_xy and object_id and isinstance(target_xy, (list, tuple)) and len(target_xy) == 2:
            try:
                tx = float(target_xy[0])
                ty = float(target_xy[1])
            except (TypeError, ValueError):
                tx = ty = None
            if tx is not None and ty is not None:
                objs = []
                for obj in snapshot.objects:
                    if obj.object_id == object_id:
                        bbox_rc = obj.bbox_rc
                        try:
                            dx = int(round(tx - obj.centroid_rc[1]))
                            dy = int(round(ty - obj.centroid_rc[0]))
                        except Exception:
                            dx = dy = 0
                        new_bbox = (bbox_rc[0] + dy, bbox_rc[1] + dx, bbox_rc[2] + dy, bbox_rc[3] + dx)
                        objs.append(replace(obj, centroid_rc=(ty, tx), bbox_rc=new_bbox))
                    else:
                        objs.append(obj)
                rows = tuple(grid_rows) if isinstance(grid_rows, (list, tuple)) else snapshot.full_grid_hex_rows
                return replace(snapshot, objects=tuple(objs), full_grid_hex_rows=rows)
        if isinstance(grid_rows, (list, tuple)) and grid_rows:
            rows = tuple(grid_rows)
            return replace(snapshot, full_grid_hex_rows=rows)
        return None

    @staticmethod
    def _format_comparison_details(comparison: dict[str, object]) -> dict[str, object]:
        object_reasons: list[str] = []
        for oid, obj_delta in comparison.get("object_deltas", {}).items():
            if obj_delta.get("appeared"):
                object_reasons.append(f"unexpected object {oid} appeared")
            elif obj_delta.get("vanished"):
                object_reasons.append(f"expected object {oid} vanished")
            else:
                dx = obj_delta.get("dx")
                dy = obj_delta.get("dy")
                if dx not in (None, 0) or dy not in (None, 0):
                    object_reasons.append(f"object {oid} moved dx={dx},dy={dy}")
                color_from = obj_delta.get("color_from")
                color_to = obj_delta.get("color_to")
                if color_from is not None and color_to is not None and color_from != color_to:
                    object_reasons.append(f"object {oid} color changed from {color_from} to {color_to}")
                if obj_delta.get("matched_obs_id") and obj_delta.get("matched_obs_id") != oid:
                    object_reasons.append(f"object {oid} matched observed object {obj_delta.get('matched_obs_id')}")
        if object_reasons:
            return {"human_readable": "; ".join(object_reasons)}
        hex_delta = comparison.get("hex_delta", {})
        return {"human_readable": f"state mismatch with {hex_delta.get('changed_cell_count', 0)} changed cells"}

    def verify_hypothesis(
        self,
        item: HypothesisItem,
        snapshot: ARGALiteSnapshot,
        memory: object,
        config: object,
        verifier_packet: dict[str, object] | None = None,
    ) -> TrajectoryVerificationResult:
        steps = trajectory_steps_from_hypothesis(item)

        packet = verifier_packet if isinstance(verifier_packet, dict) else {}
        constraints = packet.get("execution_constraints") if isinstance(packet.get("execution_constraints"), dict) else {}
        packet_allowed_actions = {
            str(value)
            for value in (constraints.get("allowed_action_ids") or getattr(snapshot, "available_actions", ()) or ())
        }
        packet_allowed_candidates = {
            str(value) for value in (constraints.get("allowed_coordinate_candidate_ids") or [])
        }
        packet_grid_hash = None
        if isinstance(packet.get("state"), dict):
            packet_grid_hash = packet["state"].get("grid_hash")
        packet_hex_rows = packet.get("full_grid_hex_rows")
        if isinstance(packet_hex_rows, list) and packet_hex_rows and packet_grid_hash is None:
            packet_grid_hash = stable_hash(tuple(str(row) for row in packet_hex_rows), "grid_")

        allowed_actions = packet_allowed_actions or set(snapshot.available_actions)
        unavailable_action = None
        invalid_candidate = None
        illegal_step_index: int | None = None
        for idx, s in enumerate(steps):
            if s.action_id not in allowed_actions:
                unavailable_action = s.action_id
                illegal_step_index = idx
                break
            if (
                s.coordinate_candidate_id is not None
                and packet_allowed_candidates
                and s.coordinate_candidate_id not in packet_allowed_candidates
            ):
                invalid_candidate = s.coordinate_candidate_id
                illegal_step_index = idx
                break

        transitions: dict[tuple[str, str, str | None], list[str]] = {}
        probe_rows = list(getattr(memory, "action_memory_records", []) or [])
        for rec in packet.get("action_probe_summaries") or []:
            if isinstance(rec, dict):
                probe_rows.append(rec)
        for rec in probe_rows:
            if not isinstance(rec, dict):
                continue
            before = rec.get("grid_hash_before") or rec.get("state_signature")
            after = rec.get("grid_hash_after")
            action_id = rec.get("action_id")
            cand = rec.get("coordinate_candidate_id")
            if not before or not after or not action_id:
                continue
            key = (before, action_id, cand)
            transitions.setdefault(key, []).append(after)
            if cand is not None:
                transitions.setdefault((before, action_id, None), []).append(after)

        current_sig = (
            packet_grid_hash
            or getattr(snapshot, "grid_hash", None)
            or stable_hash(tuple(snapshot.full_grid_hex_rows), "grid_")
        )
        simulated_states = [current_sig]
        failed_at_step = None
        for idx, step in enumerate(steps):
            key = (current_sig, step.action_id, step.coordinate_candidate_id)
            next_states = transitions.get(key) or transitions.get((current_sig, step.action_id, None)) or []
            if not next_states:
                failed_at_step = idx
                break
            current_sig = next_states[-1]
            simulated_states.append(current_sig)
            step.expected_state_hash = current_sig

        shared_details: dict[str, object] = {
            "verifier_packet_fingerprint": packet.get("packet_fingerprint"),
            "planning_object_count": len(packet.get("planning_objects") or []),
            "probe_summary_count": len(packet.get("action_probe_summaries") or []),
            "hex_row_count": len(packet.get("full_grid_hex_rows") or []),
            "dual_view_contract": packet.get("dual_view_contract"),
            "objective_grounded": _objective_is_grounded(item),
            "first_action_legal": _first_action_legal(steps, allowed_actions, packet_allowed_candidates),
        }

        # Empty trajectory is always rejected; no plan to simulate or correct.
        if not steps:
            return TrajectoryVerificationResult(status="REJECT", reason="empty_trajectory", details=shared_details)

        # Full simulation success.
        if failed_at_step is None:
            final_sig = simulated_states[-1]
            goal_hash, goal_source = resolve_goal_hash(item, snapshot, memory, simulated_final=final_sig)
            plan = TrajectoryPlan(
                hypothesis_id=item.hypothesis_id,
                goal={"expected_final_state_hash": goal_hash, "goal_source": goal_source},
                steps=steps,
                total_steps=len(steps),
                expected_final_state_hash=goal_hash or final_sig,
            )
            return TrajectoryVerificationResult(
                status="ACCEPT",
                plan=plan,
                reason="simulated_from_memory",
                confidence=0.9,
                details={
                    **shared_details,
                    "simulated_final_state_hash": final_sig,
                    "simulated_steps": len(steps),
                    "goal_hash_source": goal_source,
                },
            )

        # Build planner goal with explicit hash resolution.
        goal_hash, goal_source = resolve_goal_hash(item, snapshot, memory, simulated_final=None)
        
        # If no explicit goal can be resolved, reject the trajectory — there is no
        # target to plan toward and no empiric execution can validate progress.
        if goal_hash is None:
            return TrajectoryVerificationResult(
                status="REJECT",
                reason="no_resolved_goal_hash",
                details={**shared_details, "goal_source": goal_source},
            )
        
        planner_goal: dict[str, object] = {}
        if isinstance(item.goal_spec, dict):
            planner_goal = {
                key: item.goal_spec[key]
                for key in ("expected_final_state_hash", "goal_signature", "target_xy", "object_id", "expected_grid_hex_rows")
                if key in item.goal_spec and item.goal_spec[key] is not None
            }
        if goal_hash and "expected_final_state_hash" not in planner_goal:
            planner_goal["expected_final_state_hash"] = goal_hash
        target_signatures = [
            ev.end_state_signature
            for ev in getattr(memory, "trajectory_evaluations", []) or []
            if getattr(ev, "hypothesis_id", None) == item.hypothesis_id
            and getattr(ev, "goal_progress", None) is Progress.POSITIVE
            and ev.end_state_signature
        ]
        if target_signatures and "target_signatures" not in planner_goal:
            planner_goal["target_signatures"] = list(dict.fromkeys(target_signatures))
        if not planner_goal:
            planner_goal = {"goal_signature": getattr(item, "created_state_signature", None)}

        planner = TrajectoryPlanner(config)
        alt_plan = planner.plan(planner_goal, snapshot, memory, config)
        if alt_plan is not None:
            details = {
                **shared_details,
                "found_alternative_steps": len(alt_plan.steps),
                "goal": planner_goal,
                "goal_hash_source": goal_source,
                "failed_at_step": failed_at_step,
            }
            if item.goal_spec:
                goal_snapshot = self._goal_snapshot_from_spec(snapshot, item.goal_spec)
                if goal_snapshot is not None:
                    comparison = HexStateComparator.compare_snapshots(snapshot, goal_snapshot)
                    details["goal_comparison"] = comparison
                    details.update(self._format_comparison_details(comparison))
            # Preserve targeting metadata from the original hypothesis where actions align.
            enriched_steps: List[TrajectoryStep] = []
            for index, step in enumerate(alt_plan.steps):
                seed = step
                if index < len(steps) and steps[index].action_id == step.action_id:
                    original = steps[index]
                    seed = TrajectoryStep(
                        action_id=step.action_id,
                        coordinate_candidate_id=step.coordinate_candidate_id or original.coordinate_candidate_id,
                        repeat=step.repeat,
                        expected_state_hash=step.expected_state_hash,
                        kind=original.kind,
                        target_object_id=original.target_object_id,
                        target_relation_id=original.target_relation_id,
                        target_object_ids=original.target_object_ids,
                        target_relation_ids=original.target_relation_ids,
                        expected_observation=original.expected_observation,
                        contract_kind=original.contract_kind,
                        question_type=original.question_type,
                    )
                enriched_steps.append(seed)
            alt_plan.steps = enriched_steps
            alt_plan.total_steps = len(enriched_steps)
            if goal_hash:
                alt_plan.expected_final_state_hash = goal_hash
            # Empty alternative means matched/empty goal — not a repair of the model plan.
            if not alt_plan.steps:
                return TrajectoryVerificationResult(
                    status="PASSTHROUGH",
                    plan=None,
                    reason="planner_empty_alternative_keep_model_plan",
                    confidence=0.2,
                    details=details,
                )
            # Validate planner steps against current action surface before accepting.
            # An illegal alternative is not a hard reject: the original plan's
            # legal simulated prefix may still be salvageable below.
            planner_surface_ok = True
            for s in alt_plan.steps:
                if s.action_id not in allowed_actions:
                    planner_surface_ok = False
                    details["planner_rejected_unavailable_action"] = s.action_id
                    break
                if (
                    s.coordinate_candidate_id is not None
                    and packet_allowed_candidates
                    and s.coordinate_candidate_id not in packet_allowed_candidates
                ):
                    planner_surface_ok = False
                    details["planner_rejected_invalid_candidate"] = s.coordinate_candidate_id
                    break
            if planner_surface_ok:
                return TrajectoryVerificationResult(
                    status="CORRECTED",
                    plan=alt_plan,
                    corrected_steps=list(alt_plan.steps),
                    reason="planner_found_alternative",
                    confidence=0.75,
                    details=details,
                )

        # Prefix salvage before any hard reject: a prefix that is both
        # memory-supported and surface-legal executes, the remainder is
        # reobserved. The prefix is clipped at the earlier of the first
        # unknown transition and the first surface-illegal step.
        prefix_limit = failed_at_step if isinstance(failed_at_step, int) else 0
        if illegal_step_index is not None:
            prefix_limit = min(prefix_limit, illegal_step_index)
        if prefix_limit > 0:
            prefix = steps[:prefix_limit]
            prefix_final = simulated_states[prefix_limit]
            partial_goal, partial_source = resolve_goal_hash(item, snapshot, memory, simulated_final=prefix_final)
            clipped_illegal = illegal_step_index is not None and illegal_step_index <= (failed_at_step or 0)
            plan = TrajectoryPlan(
                hypothesis_id=item.hypothesis_id,
                goal={"expected_final_state_hash": partial_goal, "goal_source": partial_source, "partial": True},
                steps=prefix,
                total_steps=len(prefix),
                expected_final_state_hash=partial_goal or prefix_final,
            )
            return TrajectoryVerificationResult(
                status="PARTIAL",
                plan=plan,
                reason=(
                    "prefix_simulated_clipped_before_illegal_step"
                    if clipped_illegal
                    else "prefix_simulated_until_unknown_transition"
                ),
                confidence=0.55,
                details={
                    **shared_details,
                    "failed_at_step": failed_at_step,
                    "prefix_steps": len(prefix),
                    "original_steps": len(steps),
                    "prefix_final_state_hash": prefix_final,
                    "goal_hash_source": partial_source,
                    "remaining_action_ids": [s.action_id for s in steps[prefix_limit:]],
                    **({"illegal_step_index": illegal_step_index} if illegal_step_index is not None else {}),
                    **({"unavailable_action": unavailable_action} if unavailable_action is not None else {}),
                    **({"invalid_coordinate_candidate_id": invalid_candidate} if invalid_candidate is not None else {}),
                },
            )

        # Hard reject for illegal surface only when no alternative plan and no
        # executable prefix exist (illegality at the very first step).
        if unavailable_action is not None:
            return TrajectoryVerificationResult(
                status="REJECT",
                reason=f"unknown_action:{unavailable_action}",
                details={**shared_details, "unavailable_action": unavailable_action, "failed_at_step": failed_at_step},
            )
        if invalid_candidate is not None:
            return TrajectoryVerificationResult(
                status="REJECT",
                reason=f"invalid_coordinate_candidate:{invalid_candidate}",
                details={**shared_details, "invalid_coordinate_candidate_id": invalid_candidate, "failed_at_step": failed_at_step},
            )

        # Hybrid: unknown transition, but first action is legal and objective is grounded.
        # Do not reject — let the empiric TransitionJudge decide after environment execution.
        if _first_action_legal(steps, allowed_actions, packet_allowed_candidates) and _objective_is_grounded(item):
            hybrid_goal, hybrid_source = resolve_goal_hash(item, snapshot, memory, simulated_final=None)
            plan = TrajectoryPlan(
                hypothesis_id=item.hypothesis_id,
                goal={"expected_final_state_hash": hybrid_goal, "goal_source": hybrid_source, "hybrid": True},
                steps=steps,
                total_steps=len(steps),
                expected_final_state_hash=hybrid_goal or (getattr(snapshot, "grid_hash", None) or ""),
            )
            return TrajectoryVerificationResult(
                status="ACCEPT",
                plan=plan,
                reason="hybrid_empiric_execution",
                confidence=0.4,
                details={
                    **shared_details,
                    "failed_at_step": failed_at_step,
                    "goal": planner_goal,
                    "goal_hash_source": hybrid_source,
                    "hybrid_policy": (
                        "first action legal and objective grounded; "
                        "unknown memory transition is not sufficient for offline reject"
                    ),
                },
            )

        details = {
            **shared_details,
            "failed_at_step": failed_at_step,
            "goal": planner_goal,
            "goal_hash_source": goal_source,
        }
        if item.goal_spec:
            goal_snapshot = self._goal_snapshot_from_spec(snapshot, item.goal_spec)
            if goal_snapshot is not None:
                comparison = HexStateComparator.compare_snapshots(snapshot, goal_snapshot)
                details["goal_comparison"] = comparison
                details.update(self._format_comparison_details(comparison))
        # Repair contour, not admission gate: keep the model plan executable.
        # Empiric TransitionJudge remains the authority after environment step.
        return TrajectoryVerificationResult(
            status="PASSTHROUGH",
            plan=None,
            reason="no_simulation_passthrough_model_plan",
            confidence=0.25,
            details=details,
        )


class TrajectoryPlanner:
    """A* planner that searches for a sequence of actions leading to a goal signature.

    Notes:
    - The planner uses recorded transitions from memory.action_memory_records when available.
    - Supports goal dict keys: expected_final_state_hash, goal_signature, final_grid_hash,
      and target_signatures (a list of acceptable end-state hashes).
    """

    def __init__(self, config=None):
        self.config = config

    def _heuristic_from_state_sig(self, state_sig: str, goal: dict | None, start_snapshot: ARGALiteSnapshot) -> float:
        """Compute heuristic from current state signature to goal.
        
        Since we only have state_sig (not full snapshot), we use reverse distance
        if available, or fall back to goal signature matching.
        """
        if not goal:
            return 0.0
        goal_sig = goal.get("expected_final_state_hash") or goal.get("goal_signature") or goal.get("final_grid_hash")
        target_signatures = goal.get("target_signatures")
        
        # If current state matches goal, heuristic is 0
        if goal_sig and state_sig == goal_sig:
            return 0.0
        if target_signatures and state_sig in target_signatures:
            return 0.0
        
        # Fallback: use Manhattan distance in state space (admissible)
        # This is a simple proxy - actual distance depends on action graph
        return 1.0  # Each step costs 1, so minimum remaining steps >= 0

    def _heuristic(self, snapshot: ARGALiteSnapshot, goal: dict | None, action_seq: tuple) -> float:
        # Heuristic must estimate distance from CURRENT state to goal.
        # Using snapshot (current state in A* expansion) instead of start_snapshot.
        if not goal:
            return 0.0
        target_xy = goal.get("target_xy") if isinstance(goal, dict) else None
        object_id = goal.get("object_id") if isinstance(goal, dict) else None
        if target_xy and object_id:
            objs = {o.object_id: o for o in snapshot.objects}
            obj = objs.get(object_id)
            if obj is None:
                return 0.0
            cx, cy = obj.centroid_rc[1], obj.centroid_rc[0]
            try:
                tx, ty = float(target_xy[0]), float(target_xy[1])
            except (TypeError, ValueError):
                return 0.0
            return abs(cx - tx) + abs(cy - ty)
        # Fallback: use goal signature match distance if available
        goal_sig = goal.get("expected_final_state_hash") or goal.get("goal_signature") or goal.get("final_grid_hash")
        if goal_sig:
            # If current snapshot matches goal, heuristic is 0
            current_sig = HexStateComparator.signature_from_snapshot(snapshot)
            if current_sig == goal_sig:
                return 0.0
        return 0.0

    def _build_transition_indexes(self, memory: object) -> tuple[dict[tuple[str, str, str | None], list[str]], dict[str, list[tuple[str, str | None, str]]]]:
        transitions: dict[tuple[str, str, str | None], list[str]] = {}
        transitions_by_source: dict[str, list[tuple[str, str | None, str]]] = {}
        for rec in getattr(memory, "action_memory_records", []) or []:
            before = rec.get("grid_hash_before") or rec.get("state_signature")
            after = rec.get("grid_hash_after")
            action_id = rec.get("action_id")
            cand = rec.get("coordinate_candidate_id")
            if not before or not after or not action_id:
                continue
            key = (before, action_id, cand)
            transitions.setdefault(key, []).append(after)
            transitions_by_source.setdefault(before, []).append((action_id, cand, after))
            if cand is not None:
                # allow fallback for actions that ignore precise coordinate candidate
                transitions.setdefault((before, action_id, None), []).append(after)
        return transitions, transitions_by_source

    def _action_effect_outcome(self, action_id: str, candidate_id: str | None, memory: object) -> str | None:
        effects = getattr(memory, "action_effects", {}) or {}
        record = effects.get((action_id, candidate_id)) or effects.get((action_id, None))
        if record is None:
            return None
        return getattr(record, "outcome", None)
 
    def _apply_action(self, state_sig: str, action_id: str, candidate_id: str | None, transitions: dict[tuple[str, str, str | None], list[str]], memory: object) -> list[str]:
        next_states = transitions.get((state_sig, action_id, candidate_id), []) or transitions.get((state_sig, action_id, None), [])
        if next_states:
            return next_states
        outcome = self._action_effect_outcome(action_id, candidate_id, memory)
        if outcome == "no_effect":
            return [state_sig]
        return []
 
    def _build_reverse_transition_indexes(self, transitions: dict[tuple[str, str, str | None], list[str]]) -> dict[str, list[str]]:
        reverse_indexes: dict[str, list[str]] = {}
        for (before_sig, action_id, candidate_id), after_states in transitions.items():
            for after_sig in after_states:
                reverse_indexes.setdefault(after_sig, []).append(before_sig)
        return reverse_indexes
 
    def _compute_reverse_distances(self, goal_signatures: Iterable[str], reverse_indexes: dict[str, list[str]], max_steps: int) -> dict[str, int]:
        distances: dict[str, int] = {}
        queue = deque()
        for sig in goal_signatures:
            distances[sig] = 0
            queue.append(sig)
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            if current_distance >= max_steps:
                continue
            for predecessor in reverse_indexes.get(current, []):
                if predecessor in distances:
                    continue
                distances[predecessor] = current_distance + 1
                queue.append(predecessor)
        return distances

    def plan(self, goal: dict | None, start_snapshot: ARGALiteSnapshot, memory: object, config: object) -> TrajectoryPlan | None:
        """Return a TrajectoryPlan if found, otherwise None.

        This planner uses recorded action transitions from memory.action_memory_records
        to simulate actual observed effects. Nodes are state signatures (grid_hash),
        and edges are recorded transitions (state_signature, action_id, coordinate_candidate_id) -> next_state_signature.
        If the memory contains matching transitions, the A* search explores the known
        state graph instead of an abstract hash-composition model.
        """
        max_steps = int(getattr(config, "max_qwen_trajectory_steps", 20)) if config is not None else 20
        start_sig = getattr(start_snapshot, "grid_hash", None) or stable_hash(tuple(start_snapshot.full_grid_hex_rows), "grid_")
        goal_sig = None
        target_signatures = None
        if isinstance(goal, dict):
            goal_sig = goal.get("expected_final_state_hash") or goal.get("goal_signature") or goal.get("final_grid_hash")
            target_signatures = goal.get("target_signatures")

        transitions, transitions_by_source = self._build_transition_indexes(memory)
        reverse_indexes = self._build_reverse_transition_indexes(transitions)
        snapshot_id = getattr(start_snapshot, "snapshot_id", start_sig)
        goal_signatures: list[str] = []
        if goal_sig is not None:
            goal_signatures.append(goal_sig)
        if target_signatures:
            goal_signatures.extend(target_signatures)
 
        reverse_distances = self._compute_reverse_distances(goal_signatures, reverse_indexes, max_steps) if goal_signatures else {}
 
        if (goal_sig is not None and start_sig == goal_sig) or (target_signatures and start_sig in target_signatures):
            return TrajectoryPlan(
                hypothesis_id=stable_hash((snapshot_id, start_sig), "plan_"),
                goal=goal,
                steps=[],
                total_steps=0,
                expected_final_state_hash=start_sig,
            )

        # Quick path: if goal is a final grid hash and direct match exists via recorded single-step.
        if goal_sig is not None or target_signatures:
            for action_id in list(start_snapshot.available_actions):
                if action_id in start_snapshot.coordinate_action_ids:
                    for c in (getattr(start_snapshot, "coordinate_targets", ()) or []):
                        next_states = self._apply_action(start_sig, action_id, getattr(c, "candidate_id", None), transitions, memory)
                        if goal_sig is not None and goal_sig in next_states:
                            plan = TrajectoryPlan(
                                hypothesis_id=stable_hash((snapshot_id, goal_sig), "plan_"),
                                goal=goal,
                                steps=[TrajectoryStep(action_id=action_id, coordinate_candidate_id=getattr(c, "candidate_id", None))],
                                total_steps=1,
                                expected_final_state_hash=goal_sig,
                            )
                            return plan
                        if target_signatures and any(s in next_states for s in target_signatures):
                            matched = next(s for s in next_states if s in target_signatures)
                            plan = TrajectoryPlan(
                                hypothesis_id=stable_hash((snapshot_id, matched), "plan_"),
                                goal=goal,
                                steps=[TrajectoryStep(action_id=action_id, coordinate_candidate_id=getattr(c, "candidate_id", None))],
                                total_steps=1,
                                expected_final_state_hash=matched,
                            )
                            return plan
                else:
                    next_states = self._apply_action(start_sig, action_id, None, transitions, memory)
                    if goal_sig is not None and goal_sig in next_states:
                        plan = TrajectoryPlan(
                            hypothesis_id=stable_hash((snapshot_id, goal_sig), "plan_"),
                            goal=goal,
                            steps=[TrajectoryStep(action_id=action_id)],
                            total_steps=1,
                            expected_final_state_hash=goal_sig,
                        )
                        return plan
                    if target_signatures and any(s in next_states for s in target_signatures):
                        matched = next(s for s in next_states if s in target_signatures)
                        plan = TrajectoryPlan(
                            hypothesis_id=stable_hash((snapshot_id, matched), "plan_"),
                            goal=goal,
                            steps=[TrajectoryStep(action_id=action_id)],
                            total_steps=1,
                            expected_final_state_hash=matched,
                        )
                        return plan

        import heapq
 
        def neighbors(state_sig: str):
            return transitions_by_source.get(state_sig, [])
 
        def heuristic(state_sig: str) -> float:
            if reverse_distances:
                return float(reverse_distances.get(state_sig, 0))
            # Compute heuristic from current state (state_sig), not start_snapshot.
            # We need to reconstruct a minimal snapshot-like object for the current state.
            # Since we only have state_sig, use goal-based distance estimation.
            return self._heuristic_from_state_sig(state_sig, goal, start_snapshot)
 
        open_heap = []  # (f, g, state_sig, action_seq)
        heapq.heappush(open_heap, (heuristic(start_sig), 0, start_sig, tuple()))
        seen_costs: dict[str, int] = {start_sig: 0}

        while open_heap:
            f, g, state_sig, seq = heapq.heappop(open_heap)
            if g > max_steps:
                continue
            if seq:
                if goal_sig is not None and state_sig == goal_sig:
                    steps = [TrajectoryStep(action_id=a, coordinate_candidate_id=cid, repeat=r) for (a, cid, r) in seq]
                    return TrajectoryPlan(
                        hypothesis_id=stable_hash((snapshot_id, state_sig), "plan_"),
                        goal=goal,
                        steps=steps,
                        total_steps=len(steps),
                        expected_final_state_hash=state_sig,
                    )
                if target_signatures and state_sig in target_signatures:
                    steps = [TrajectoryStep(action_id=a, coordinate_candidate_id=cid, repeat=r) for (a, cid, r) in seq]
                    return TrajectoryPlan(
                        hypothesis_id=stable_hash((snapshot_id, state_sig), "plan_"),
                        goal=goal,
                        steps=steps,
                        total_steps=len(steps),
                        expected_final_state_hash=state_sig,
                    )
            if g >= max_steps:
                continue
            for action_id, candidate_id, after_sig in neighbors(state_sig):
                new_g = g + 1
                prev = seen_costs.get(after_sig)
                if prev is not None and new_g >= prev:
                    continue
                seen_costs[after_sig] = new_g
                new_seq = tuple(list(seq) + [(action_id, candidate_id, 1)])
                h = heuristic(after_sig)
                heapq.heappush(open_heap, (new_g + h, new_g, after_sig, new_seq))

        return None

