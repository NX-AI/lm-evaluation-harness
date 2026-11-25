"""Pairwise Arena-Hard v2.0 judgments of lm-eval ``custom_arena_hard_v2`` samples against the official baselines.

    python scripts/arena_hard_v2/judge.py --samples OUT/samples_custom_arena_hard_v2_*.jsonl --model NAME \
        --upstream ARENA_HARD_DATA_DIR --out-dir RUN_DIR \
        --judge-model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY

Writes, in arena-hard-auto's formats,
* ``RUN_DIR/model_answer/NAME.jsonl``: the model's answers with style metadata (``add_markdown_info.py``),
* ``RUN_DIR/model_judgment/JUDGE/NAME.jsonl``: one judgment record per question (two games),
which ``score.py`` reads (pass ``--answers RUN_DIR/model_answer ARENA_HARD_DATA_DIR/model_answer``
so the baselines' metadata is available). Re-running resumes: judged questions are skipped.

``ARENA_HARD_DATA_DIR`` is ``data/arena-hard-v2.0`` of lmarena/arena-hard-auto (GitHub) or of the
Hugging Face dataset ``lmarena-ai/arena-hard-auto``; it provides ``question.jsonl`` and the baseline
answers ``model_answer/{o3-mini-2025-01-31,gemini-2.0-flash-001}.jsonl``.

The judging follows ``gen_judgment.py`` (commit 48e797c1, Apache License 2.0): the category's system
prompt, the user prompt template, game 1 with the baseline as assistant A and the model as B, game 2
swapped, and the verdict parsed from the upper-cased reply with ``\\[\\[([AB<>=]+)\\]\\]`` (then
``\\[([AB<>=]+)\\]``), last match. Only the answer (text after ``</think>``) is judged. The official
judge is ``gpt-4.1`` at temperature 0 with 16,000 max tokens; scores from another judge are not
comparable to published ones.
"""

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path

OG_ARENA_HARD_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
CREATIVE_WRITING_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nWhen evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
JUDGE_SETTINGS = {
    "hard_prompt": {"baseline": "o3-mini-2025-01-31", "system_prompt": OG_ARENA_HARD_PROMPT},
    "coding": {"baseline": "o3-mini-2025-01-31", "system_prompt": OG_ARENA_HARD_PROMPT},
    "math": {"baseline": "o3-mini-2025-01-31", "system_prompt": OG_ARENA_HARD_PROMPT},
    "creative_writing": {"baseline": "gemini-2.0-flash-001", "system_prompt": CREATIVE_WRITING_PROMPT},
}
PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n"
    "<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n"
    "<|The End of Assistant B's Answer|>"
)
REGEX_PATTERNS = (r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]")
THINK_END = "</think>"


def get_score(judgment, patterns=REGEX_PATTERNS):
    for pattern in patterns:
        matches = [m for m in re.compile(pattern).findall(judgment.upper()) if m != ""]
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def final_answer(text):
    return text.rsplit(THINK_END, 1)[1].strip() if THINK_END in text else text.strip()


# --------------------------------------------------------------- style metadata (add_markdown_info.py)
def count_markdown_elements(markdown_text, suffix=""):
    return {
        f"header_count{suffix}": {
            f"h{i}": len(re.findall(rf"^#{{{i}}}\s", markdown_text, re.MULTILINE)) for i in range(1, 7)
        },
        f"list_count{suffix}": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count{suffix}": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
    }


def style_metadata(answer, encoder):
    code = re.compile("```([^`]*)```")
    text = answer
    for block in code.findall(answer):
        text = text.replace(block, "")
    return {"token_len": len(encoder.encode(answer, disallowed_special=()))} | count_markdown_elements(text)


# --------------------------------------------------------------- inputs
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def model_answers_from_samples(samples_path, model, encoder):
    """lm-eval samples (``--log_samples``) -> arena-hard-auto model_answer records."""
    answers = {}
    for row in read_jsonl(samples_path):
        doc = row["doc"]
        resps = row["resps"][0] if isinstance(row["resps"][0], list) else row["resps"]
        answer = final_answer(resps[0])
        answers[doc["uid"]] = {
            "uid": doc["uid"],
            "ans_id": f"lm-eval-{row['doc_id']}",
            "model": model,
            "messages": [
                {"role": "user", "content": doc["prompt"]},
                {"role": "assistant", "content": {"answer": answer}},
            ],
            "tstamp": time.time(),
            "metadata": style_metadata(answer, encoder),
        }
    return answers


# --------------------------------------------------------------- judging
class Judge:
    def __init__(self, model, base_url, api_key, temperature, max_tokens, retries=6):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
        self.model, self.temperature, self.max_tokens, self.retries = model, temperature, max_tokens, retries

    def __call__(self, messages):
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
                )
                msg = r.choices[0].message
                usage = r.usage.model_dump() if r.usage else None
                return {"answer": msg.content or "", "served_model": r.model, "finish_reason": r.choices[0].finish_reason,
                        "usage": usage}
            except Exception as error:  # rate limits, timeouts, transient 5xx
                if attempt == self.retries - 1:
                    print(f"judge call failed after {self.retries} attempts: {type(error).__name__}: {error}")
                    return None
                time.sleep(min(60, 2 ** attempt * 2))


def pairwise(judge, question, answer_a, answer_b):
    messages = [
        {"role": "system", "content": JUDGE_SETTINGS[question["category"]]["system_prompt"]},
        {"role": "user", "content": PROMPT_TEMPLATE.format(
            QUESTION=question["prompt"],
            ANSWER_A=answer_a["messages"][-1]["content"]["answer"],
            ANSWER_B=answer_b["messages"][-1]["content"]["answer"],
        )},
    ]
    output = judge(messages)
    if output is None:
        return None
    return {"score": get_score(output["answer"]), "judgment": output, "prompt": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, help="lm-eval samples_custom_arena_hard_v2_*.jsonl")
    parser.add_argument("--model", required=True, help="name for the evaluated model")
    parser.add_argument("--upstream", required=True, help="arena-hard-auto data/arena-hard-v2.0 directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--judge-name", help="directory name for the judgments (default: --judge-model)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (default: OpenAI)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--categories", nargs="*", default=None, help="judge only these categories")
    parser.add_argument("--limit", type=int, default=None, help="judge at most this many new questions")
    args = parser.parse_args()

    import tiktoken

    encoder = tiktoken.encoding_for_model("gpt-4o")
    upstream = Path(args.upstream)
    questions = read_jsonl(upstream / "question.jsonl")
    baselines = {}
    for name in {s["baseline"] for s in JUDGE_SETTINGS.values()}:
        baselines[name] = {r["uid"]: r for r in read_jsonl(upstream / "model_answer" / f"{name}.jsonl")}

    out = Path(args.out_dir)
    answer_file = out / "model_answer" / f"{args.model}.jsonl"
    judgment_file = out / "model_judgment" / (args.judge_name or args.judge_model) / f"{args.model}.jsonl"
    answer_file.parent.mkdir(parents=True, exist_ok=True)
    judgment_file.parent.mkdir(parents=True, exist_ok=True)

    answers = model_answers_from_samples(args.samples, args.model, encoder)
    with open(answer_file, "w") as f:
        for q in questions:
            if q["uid"] in answers:
                f.write(json.dumps(answers[q["uid"]], ensure_ascii=False) + "\n")
    missing = [q["uid"] for q in questions if q["uid"] not in answers]
    print(f"{len(answers)} answers written to {answer_file}; {len(missing)} questions without an answer")

    done = set()
    if judgment_file.exists():
        done = {r["uid"] for r in read_jsonl(judgment_file)
                if r["games"][0] is not None and r["games"][1] is not None}
    todo = []
    for q in questions:
        if q["uid"] in done or q["uid"] not in answers:
            continue
        if args.categories and q["category"] not in args.categories:
            continue
        baseline = baselines[JUDGE_SETTINGS[q["category"]]["baseline"]].get(q["uid"])
        if baseline is None:
            print(f"warning: no baseline answer for {q['uid']} ({q['category']}); skipped")
            continue
        todo.append((q, baseline))
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(done)} questions already judged; judging {len(todo)} with {args.judge_model}")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env}")
    judge = Judge(args.judge_model, args.base_url, api_key, args.temperature, args.max_tokens)
    lock = threading.Lock()

    def run(item):
        q, baseline = item
        answer = answers[q["uid"]]
        record = {
            "uid": q["uid"], "category": q["category"], "judge": args.judge_name or args.judge_model,
            "model": args.model, "baseline": baseline["model"],
            "games": [pairwise(judge, q, baseline, answer), pairwise(judge, q, answer, baseline)],
        }
        with lock, open(judgment_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    unparsed = 0
    served = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for i, record in enumerate(pool.map(run, todo), 1):
            for game in record["games"]:
                if game is None or game["score"] is None:
                    unparsed += 1
                    continue
                served.add(game["judgment"].get("served_model"))
                for key in usage:
                    usage[key] += (game["judgment"].get("usage") or {}).get(key) or 0
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} judged; unparsed/failed games so far: {unparsed}; tokens {usage}", flush=True)
    print(f"done: {judgment_file}; served judge model(s): {sorted(s for s in served if s)}; "
          f"unparsed/failed games: {unparsed}; tokens: {usage}")


if __name__ == "__main__":
    main()
