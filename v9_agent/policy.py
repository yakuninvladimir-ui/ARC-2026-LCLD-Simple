from __future__ import annotations

from .config import V8Config
from .types import ARGALiteSnapshot, CandidateAction


class Policy:
    def choose_action(self, snapshot: ARGALiteSnapshot, memory: "GameMemory", bank: "HypothesisBank", explorer: "ActionExplorer", config: V8Config) -> CandidateAction | None:
        for queue_name in ("confirmed", "coordinate", "semantic"):
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
