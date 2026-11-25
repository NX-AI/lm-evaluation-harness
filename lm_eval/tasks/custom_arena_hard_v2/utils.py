"""Arena-Hard v2.0 answer generation: the per-sample metrics only describe the answers.

The benchmark score comes from pairwise judging against the official baselines
(scripts/arena_hard_v2/judge.py) and the official scoring (scripts/arena_hard_v2/score.py).
"""

THINK_END = "</think>"

QUESTIONS_REPO = "lmarena-ai/arena-hard-auto"
QUESTIONS_FILE = "data/arena-hard-v2.0/question.jsonl"
QUESTIONS_REVISION = "15f3746e21432264ce9b453999bde4f3c946d2e6"


def load_questions(questions_revision: str = QUESTIONS_REVISION, **_):
    """The 750 Arena-Hard v2.0 questions as a ``test`` split.

    The file is fetched with ``hf_hub_download`` and read as local JSON, so that a run with
    ``HF_HUB_OFFLINE=1`` finds it in the hub cache (a hub dataset loaded with ``data_files`` gets a
    different cache key offline and is not found). Task metadata arrives as keyword arguments.
    """
    import datasets
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(QUESTIONS_REPO, QUESTIONS_FILE, repo_type="dataset", revision=questions_revision)
    return datasets.load_dataset("json", data_files={"test": path})


def final_answer(text: str) -> str:
    """The answer the judge sees: text after the last ``</think>``, else the whole output.

    Mirrors arena-hard-auto's handling of reasoning models (``end_think_token``): only the answer
    is judged, and an unterminated thought is judged as the answer.
    """
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[1].strip()
    return text.strip()


def process_results(doc, results):
    answer = final_answer(results[0])
    return {"answered": float(bool(answer)), "answer_chars": len(answer)}
