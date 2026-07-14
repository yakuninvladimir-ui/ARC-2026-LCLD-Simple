# LCLD Qwen competition wrapper

The gameplay lifecycle is derived from the working Tufa-style competition
wrapper. The model runtime is taken from the Tufa Qwen Kaggle setup:

- `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`
- `driessmit1/arc3-vllm-h100-wheelhouse-v3`

Cell 2 is generated from the current `v8_agent` tree plus generated
`kaggle_agent.py` and `submission.py` compatibility modules. The starter
`agent/` directory and `my_agent.py` are intentionally not embedded.

The active limits are 98304 model context, 65536 maximum input, 12288 maximum
output, 500 seconds per Qwen call, 6000 seconds and 200 accepted actions per
game, and four attempts per level. There is no per-level action limit. Phase A
is static; Phase B starts one persistent vLLM server, validates it before
opening the scorecard, then uses the direct Arcade loop with an initial RESET.
