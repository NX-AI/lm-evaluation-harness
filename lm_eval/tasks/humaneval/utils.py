import os
import uuid

import evaluate as hf_evaluate


# Use a per-process `experiment_id` so concurrent/sequential Hydra jobs don't race on
# the same cache file (`{HF_cache}/metrics/code_eval/default/{experiment_id}-1-0.arrow`).
# Without this, a second process hits FileNotFoundError inside `os.remove(file_path)`
# because the first process already removed the shared default file.
_EXPERIMENT_ID = f"lm_eval_{os.getpid()}_{uuid.uuid4().hex}"

try:
    compute_ = hf_evaluate.load("code_eval", experiment_id=_EXPERIMENT_ID)
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
