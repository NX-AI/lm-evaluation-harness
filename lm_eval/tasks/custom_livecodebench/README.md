# LiveCodeBench (v6) — lm-eval integration

This task implements **LiveCodeBench v6** code generation evaluation (subset: `code_generation_lite`) in `lm-evaluation-harness`, reusing the same prompt constants + dataset cleaning conventions as NVIDIA NeMo Skills, and running the official `livecodebench.evaluate.evaluate(...)` inside a **local sandbox server** (no Docker).

## What this task does

- Loads `livecodebench/code_generation_lite` (`revision=refs/pr/7`, `release_v6`) and applies NeMo-aligned cleaning:
  - `task_id = question_id`
  - sets `formatting_message` + `starter_code` exactly like the official LCB runner prompt
  - adds `release_version="v6"` and `subset_for_metrics=difficulty`
  - filters by contest month range (default: `2024-08` … `2025-05`)
- Prompts with the official LCB codegen prompt.
- Extracts code from generations (last fenced block if present; otherwise uses the raw string).
- Runs the official LCB evaluator **in a sandbox** and reports:
  - `pass@k` (NeMo/AIME-style estimator from all samples)
  - `majority@k` (vote over the first *k* samples)
  - `no_answer_rate` / `no_answer_count` / `num_generations`

## Task variants (recommended)

- `livecodebench`: greedy, `repeats=1`, metrics at `k=1`
- `livecodebench_agg8`: sampling, `repeats=8`, metrics at `k∈{1,2,4,8}`
- `livecodebench_agg32`: sampling, `repeats=32`, metrics at `k∈{1,2,4,8,16,32}`

All task variants set `max_gen_toks=15360` by default to support long-context / thinking models.

Note: the task YAMLs explicitly set `generation_kwargs.until: []` so lm-eval does **not** default to stopping at `"\n\n"` (the `fewshot_delimiter`). This avoids truncating generations after short preambles and instead stops only at EOS (or `max_gen_toks`).

## Requirements

1. `livecodebench` must be importable in the environment that runs the sandbox server.
   - Recommended install (pin it yourself for reproducibility):
     - `pip install git+https://github.com/wasiahmad/livecodebench.git`
2. A local sandbox server must be running and reachable at `NEMO_SKILLS_SANDBOX_HOST:NEMO_SKILLS_SANDBOX_PORT`.

## Starting the sandbox server (sidecar)

Use the helper script:

- `python scripts/livecodebench_local_sandbox_server.py`

Environment variables:

- `NEMO_SKILLS_SANDBOX_BIND` (default: `127.0.0.1`)
- `NEMO_SKILLS_SANDBOX_PORT` (default: `6000`)
- `NEMO_SKILLS_SANDBOX_MEM_LIMIT` (bytes, default: `53687091200` = 50 GiB)
- `NEMO_SKILLS_SANDBOX_MAX_OUTPUT_CHARACTERS` (default: `100000`) — truncation limit for stdout/stderr in `/execute`
- `NEMO_SKILLS_SANDBOX_LOG_REQUESTS=1` — enable per-request HTTP logs (off by default)

## Running the task

Useful env vars:

- `LMEVAL_LCB_NUM_PROCESSES` (default: `4`) — evaluator parallelism
- `LMEVAL_LCB_TIMEOUT` (default: `6`) — per-sample timeout (seconds)
- `LMEVAL_LCB_TIMEOUT_BUFFER` (default: `60`) — added to total timeout (scaled by total candidates / processes)
- `LMEVAL_LCB_NUM_RETRIES` (default: `3`) — sandbox HTTP retries on transient network errors
- `LMEVAL_LCB_INTERPRETER` (default: current `sys.executable`) — interpreter used inside the sandbox
- `LMEVAL_LCB_SAVE_FILES=1` — save evaluator artifacts (predictions + eval results + stdout/stderr) to:
  - `LMEVAL_LCB_ARTIFACT_DIR` / `LMEVAL_LCB_SAVE_DIR` (if set), else
  - Hydra `hydra.runtime.output_dir` (if running under Hydra), else
  - `cwd`
- `LMEVAL_LCB_ARTIFACT_DIR` / `LMEVAL_LCB_SAVE_DIR` — output directory used when `LMEVAL_LCB_SAVE_FILES=1`
- `LMEVAL_LCB_SAVE_FULL_GENERATIONS=1` — save **raw model outputs** (incl. thinking traces) plus extracted code to gzipped JSONL:
  - per-rank shards: `livecodebench_full_generations_<release>_<lang>_n<n>_rank<rank>_pid<pid>.jsonl.gz`
  - merged file (rank 0): `livecodebench_full_generations_<release>_<lang>_n<n>.jsonl.gz`
  - manifest of shards: `livecodebench_full_generations_<release>_<lang>_n<n>.manifest.txt`
- `LMEVAL_LCB_MERGE_FULL_GENERATIONS=0/1` (default: `1`) — disable/enable merging shards on rank 0
- `LMEVAL_LCB_FULL_GENERATIONS_MAX_CHARS` — optional per-field truncation when writing full generations (unset = no truncation)

Safety switches:

- `LMEVAL_LCB_AUTO_INSTALL=1` — auto-install `livecodebench` inside the sandbox (needs network access)
- `LMEVAL_LCB_PIP_SPEC` (default: `git+https://github.com/wasiahmad/livecodebench.git`) — pip spec used when auto-installing
- `LMEVAL_LCB_ALLOW_LOCAL_EVAL=1` — run evaluator locally if sandbox is down (**unsafe**, not recommended)

Dataset overrides (via `--metadata` / TaskManager metadata):

- `start_date` / `end_date` (default: `2024-08` … `2025-05`)
- `revision` (default: `refs/pr/7`)

## Metrics details

### Choosing which `k` to report

`livecodebench_agg8` / `livecodebench_agg32` set a constant `metric_ks` column via `dataset_kwargs.metric_ks`. The task’s `process_results` uses that to emit only the requested `pass@k` / `majority@k` keys.

If you make your own variant, ensure `max(metric_ks) <= repeats` so that you actually sample enough candidates.

### `no_answer_*`

Computed from the **post-processed** code strings:

- A generation counts as “no answer” if the extracted code is empty (`""`), e.g. due to:
  - an unterminated code fence
  - a `<think>`/`<analysis>` tag without a closing tag (strictly treated as invalid)

### `pass@k`

This follows the same estimator used in the AIME task integration:

- Let `n = repeats` and `c = #correct` among all `n` candidates for a problem.
- `pass@k = 1 - C(n-c, k) / C(n, k)` (with `k = min(k, n)`).

The official LCB evaluator is run once to obtain a boolean `graded_list` for each candidate; all `pass@k` values reuse that output.

### `majority@k`

Mode vote among the **first** `k` candidates (empty candidates are excluded from voting). Ties are averaged.

## Configuration notes (Slurm / accelerate)

- The LCB evaluator is file-mutating and expensive; metric aggregation runs after all ranks’ payloads are gathered, so the sandbox evaluator only runs once.
- Keep `LMEVAL_LCB_NUM_PROCESSES` conservative on shared CPU jobs (e.g., 4–8) to avoid timeouts caused by contention.
- Prefer node-local scratch for temps (`SLURM_TMPDIR` / `TMPDIR`) to avoid NFS hot-spots.

## “Gold solution” smoketest note

The `livecodebench/code_generation_lite` dataset does **not** include reference solutions (only test cases), so a true “gold solution” eval is not available from the dataset alone.

For a setup smoketest, run a tiny subset with a trivial fast program (expected to fail most problems) and just verify:

- sandbox `/health` is reachable
- the evaluator runs to completion and produces `_eval_results.json`
