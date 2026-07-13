# V8.3 Changelog

## 2026-07-12 semantic/runtime freeze

- Replaced the overloaded semantic packet with exact geometry, state-scoped action effects, current/noncurrent control groups, synchronized raw visual transitions, and explicit action-surface chronology.
- Kept trajectory generation in Qwen; verifier now checks IDs, current first action, observed effects, control context, correspondence progress, boundaries, and unjustified immediate inverse pairs.
- Added split first/remaining action response fields with canonical parser normalization and malformed-response salvage.
- Kept `ACTION7` out of probing and fallback; reserve may use its hard undo meaning only.
- Removed automatic discovery of `llama-completion`; standard local and competition profiles both use the explicitly configured `llama-cli`.
- Froze context/input/output at `98304 / 65536 / 4096` and Qwen/game limits at `350 / 5000` seconds, with no per-level or global competition deadline.
- Set the competition notebook to two T4 GPUs, explicit layer split `1,1`, and a short Phase-A model-load smoke.
- Made Kaggle publish metadata always attach the Qwen runtime dataset and ARC competition source; missing mounts now fail immediately with an explicit diagnostic.

## Correctness-critical changes

- Replaced the V8.2 rule “action-linked visible change implies confirmation” with contract-specific verification.
- Added seven fixed verification contracts and a small grounding binder.
- Added explicit `observe_action_result()` commit to the generated competition shim.
- Added true pending-transition telemetry and duplicate-ingestion protection.
- Propagated dynamic Qwen timeout values into the live `GameSession`.
- Changed `GAME_OVER` from terminal handling to unlimited reset/replay until the outer budget expires.

## Perception and graph

- Added persistent object track IDs independent of bbox position.
- Added stable relation IDs independent of current metric value.
- Added holes, topology, multicolor component support, and additional compact relations.
- Made coordinate candidate identity depend on stable target semantics rather than current coordinates.

## Reasoning and memory

- Added typed verifier-side semantic questions and entropy-reduction information gain.
- Split failed and irrelevant memories.
- Added action-effect records and structured previous-attempt feedback.
- Made candidate repeat suppression state-scoped.
- Added an explicit least-used `exhaustion_revisit` liveness path so fully explored no-effect states do not crash before the outer timeout.
- Added current-snapshot executability checks for queued hypotheses.

## Qwen

- Updated packet/output schemas to V8.3.
- Added contract menu to every proposal packet.
- Enforced tail order: feedback, candidate menu, questions, schema, instruction.
- Added normal/aggressive/minimal packet compaction.
- Added balanced JSON extraction and one compact retry.
- Added `--no-display-prompt` and `--simple-io` to `llama-cli`.

## Testing

- Expanded the suite from 18 to 34 tests.
- Added tests for contract judgment, persistent identity, stale targets, information gain, explicit transition commit, replayable `GAME_OVER`, prompt-tail order, compaction, and generated shim structure.
