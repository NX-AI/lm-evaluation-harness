#!/usr/bin/env bash
# BFCL-v3 (Berkeley Function Calling Leaderboard) with the official bfcl-eval package against a running
# OpenAI-compatible server (e.g. `vllm serve`), then the v3 overall score.
#
#   scripts/bfcl_v3/run_bfcl_v3.sh MODEL_ID LOCAL_MODEL_PATH OUT_DIR
#
# MODEL_ID          a bfcl-eval model id that selects the prompt/tool format, e.g.
#                   Qwen/Qwen3-30B-A3B-Instruct-2507-FC (Qwen3 chat format with <tool_call> blocks)
# LOCAL_MODEL_PATH  the model directory; bfcl loads its tokenizer and config from it and sends it as the
#                   request's `model`, so the server must serve it under this name (--served-model-name)
# OUT_DIR           becomes BFCL_PROJECT_ROOT: result/, score/ and bfcl_v3_summary.json land here
#
# Environment: BFCL_BIN (directory with the `bfcl` and `python` of a venv with bfcl-eval; default: PATH),
# LOCAL_SERVER_ENDPOINT / LOCAL_SERVER_PORT (default localhost:1053), BFCL_NUM_THREADS (default 32).
# Sampling is bfcl's default (temperature 0.001); requests go to /v1/completions with at most 4,096 new
# tokens, so the server's context must cover the longest prompt plus 4,096.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
model_id=$1; model_path=$2; out=$(realpath -m "$3")
bin=${BFCL_BIN:+$BFCL_BIN/}
cats=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance
cats+=,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance
cats+=,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context
mkdir -p "$out"
export BFCL_PROJECT_ROOT=$out
"${bin}bfcl" version || true
"${bin}bfcl" generate --model "$model_id" --local-model-path "$model_path" --test-category "$cats" \
  --skip-server-setup --num-threads "${BFCL_NUM_THREADS:-32}"
"${bin}bfcl" evaluate --model "$model_id" --test-category "$cats"
"${bin}python" "$here/bfcl_v3_overall.py" "$out/score" --out "$out/bfcl_v3_summary.json"
