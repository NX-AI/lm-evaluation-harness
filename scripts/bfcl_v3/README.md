# BFCL-v3 (Berkeley Function Calling Leaderboard)

BFCL scores tool calling: 17 categories in v3, 4,441 entries. Its multi-turn third needs an
execution loop: the model calls simulated APIs, gets their results and continues for up to 20
steps per turn. lm-eval's single-call generation cannot express that loop, so BFCL is not an
lm-eval task. These scripts run the official
[bfcl-eval](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
package against an OpenAI-compatible server of the model and compute the v3 overall score.

| Group | Categories | Entries | Aggregation |
| --- | --- | ---: | --- |
| Non-Live | simple_python, simple_java, simple_javascript (averaged as "simple"), multiple, parallel, parallel_multiple, irrelevance | 1,390 | unweighted mean of 5 |
| Live | live_simple, live_multiple, live_parallel, live_parallel_multiple, live_irrelevance, live_relevance | 2,251 | weighted by entry count |
| Multi-Turn | multi_turn_base, multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context | 800 | unweighted mean of 4 |

The v3 overall is the unweighted mean of the three groups (`eval_runner_helper.py` at gorilla
`65a1c94`). The current package (v4) adds agentic categories and reweights its "Overall Acc",
so `bfcl_v3_overall.py` recomputes the v3 number from the per-category score files.

## Usage

```bash
pip install bfcl-eval==2026.3.23 soundfile   # in its own venv: it pins numpy 1.26
vllm serve MODEL_DIR --served-model-name MODEL_DIR --max-model-len 120000 --port 1053 &
BFCL_BIN=/path/to/bfcl-venv/bin LOCAL_SERVER_PORT=1053 \
  scripts/bfcl_v3/run_bfcl_v3.sh Qwen/Qwen3-30B-A3B-Instruct-2507-FC MODEL_DIR OUT
```

* `MODEL_ID` selects bfcl's prompt and tool-call format. `Qwen/...-FC` builds the Qwen3 chat
  prompt with a `<tools>` system block and parses `<tool_call>` JSON from the reply. Use it for
  any model with Qwen3's chat format, including distilled students.
* bfcl loads the tokenizer and config from `MODEL_DIR` (with `trust_remote_code`) and sends
  `MODEL_DIR` as the request's model name, so serve it under that name.
* Sampling is bfcl's default, temperature 0.001. Each request allows up to 4,096 new tokens,
  so the server's context has to cover the longest prompt plus 4,096.
* `qwen-agent`, which bfcl imports, needs `soundfile`; bfcl-eval 2026.3.23 does not declare it.
* `OUT` becomes `BFCL_PROJECT_ROOT` (`result/`, `score/`, `bfcl_v3_summary.json`).

Reference: the BFCL leaderboard lists Qwen3-30B-A3B-Instruct-2507 (FC) with Non-Live AST 85.77,
Live 77.94 and Multi-Turn 30.00 (v4 group definitions). Qwen's model card reports BFCL-v3 65.1.
