from __future__ import annotations

from .config import V8Config
from .types import ARGALiteSnapshot, CandidateAction


class Policy:
    def choose_action(
        self,
        snapshot: ARGALiteSnapshot,
        memory: "GameMemory",
        bank: "HypothesisBank",
        explorer: "ActionExplorer",
        config: V8Config,
        *,
        allow_semantic: bool = True,
    ) -> CandidateAction | None:
        # Must include verified_semantic: Contour B moves ACCEPT/CORRECTED/PARTIAL
        # (including hybrid empiric) out of semantic_test_queue into verified_semantic.
        # Omitting it black-holes every offline-accepted Qwen plan.
        queues = ["confirmed"]
        if allow_semantic:
            queues.extend(["verified_semantic", "semantic"])
        queues.append("coordinate")
        for queue_name in queues:
            candidate = bank.next_candidate_action(snapshot, queue_name)
            if candidate is not None:
                return candidate
        candidate = explorer.simple_probe(snapshot, memory, config)
        if candidate is not None:
            return candidate
        return None

    def safe_fallback(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", config: V8Config) -> CandidateAction:
        from .action_explorer import ActionExplorer
        return ActionExplorer().safe_fallback(snapshot, memory, config)
