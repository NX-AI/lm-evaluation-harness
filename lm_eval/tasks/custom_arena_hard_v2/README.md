# Arena-Hard v2.0

Arena-Hard-Auto v2.0 ([lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto),
[paper](https://arxiv.org/abs/2406.11939)): 750 real-world prompts from Chatbot Arena, 500
`hard_prompt` (software engineering, math and other hard queries) and 250 `creative_writing`.
A judge model compares each answer with a fixed baseline answer: `o3-mini-2025-01-31` for hard
prompts, `gemini-2.0-flash-001` for creative writing.

The benchmark has three steps; only the first runs in lm-eval.

1. **Generate answers** with the `custom_arena_hard_v2` task (this directory). The prompts are loaded
   from the Hugging Face dataset `lmarena-ai/arena-hard-auto` at a pinned revision. The prompt is
   sent as one user turn, so run it with `--apply_chat_template`, and keep the samples
   (`--log_samples`). The task's per-sample metrics only describe the answers
   (`answered`, `answer_chars`).
2. **Judge** with `scripts/arena_hard_v2/judge.py`: two games per prompt (baseline as assistant A,
   then as B), with the official system prompts, user template and verdict parsing. Any
   OpenAI-compatible endpoint works.
3. **Score** with `scripts/arena_hard_v2/score.py`, the official `show_result.py` scoring:
   plain win rate against the baseline, or Bradley-Terry with style control
   (`--style-control`, length and markdown features). Categories are scored separately.

## Defaults and comparability

* Generation: Qwen3-2507 instruct sampling (temperature 0.7, top-p 0.8, top-k 20) and
  16,384 new tokens. Override with lm-eval's `--gen_kwargs` as needed; arena-hard-auto sets
  these per model.
* Only the answer is judged: text after the last `</think>` (as arena-hard-auto does for
  reasoning models).
* The official judges are GPT-4.1 (temperature 0, 16,000 tokens) and Gemini-2.5. Scores from
  another judge are only comparable among models judged by it.
* Style-controlled scores come from a joint fit over every model whose judgments are loaded.
  Pass the same judgment directories when comparing models.

`score.py` reproduces the published leaderboard: on the 27 GPT-4.1 judgment files of
`lmarena-ai/arena-hard-auto` (revision `15f3746e`), its hard-prompt style-controlled scores
are within 0.4 points of the README table at arena-hard-auto commit `48e797c1`. The difference
is bootstrap noise. `judge.py` reproduces the stored style metadata of all official answers
exactly.

## Usage

```bash
# 1. answers (any lm-eval model backend)
lm_eval --model vllm --model_args pretrained=MODEL --tasks custom_arena_hard_v2 --apply_chat_template \
  --log_samples --output_path OUT
# 2. judgments; ARENA = data/arena-hard-v2.0 of the HF dataset or GitHub repo (questions, baselines)
python scripts/arena_hard_v2/judge.py --samples OUT/*/samples_custom_arena_hard_v2_*.jsonl --model NAME \
  --upstream ARENA --out-dir RUN --judge-model gpt-4.1 --api-key-env OPENAI_API_KEY
# 3. scores
python scripts/arena_hard_v2/score.py --judgments RUN/model_judgment/gpt-4.1 --category hard_prompt
python scripts/arena_hard_v2/score.py --judgments RUN/model_judgment/gpt-4.1 \
  --answers RUN/model_answer ARENA/model_answer --category hard_prompt --style-control
```

`judge.py` resumes where it stopped. Use `--base-url` and `--judge-model` for other
OpenAI-compatible judges. A reasoning judge needs a larger `--max-tokens` (the official
Gemini-2.5 setting is 32,000).

Judging costs about 1,500 judge calls per model (1,000 for hard prompts only).

## Citation

```
@article{li2024crowdsourced,
  title={From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline},
  author={Li, Tianle and Chiang, Wei-Lin and Frick, Evan and Dunlap, Lisa and Wu, Tianhao and Zhu, Banghua and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2406.11939},
  year={2024}
}
```

The judge prompts and scoring code are adapted from arena-hard-auto (Apache License 2.0).
