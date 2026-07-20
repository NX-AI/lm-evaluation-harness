import os
import evaluate as hf_evaluate


try:
    job = os.environ.get("SLURM_JOB_ID", "local")
    step = os.environ.get("SLURM_STEP_ID", "0")
    rank = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    pid = os.getpid()
    compute_ = hf_evaluate.load("code_eval", experiment_id=f"humaneval_{job}_{step}_{rank}_{pid}")
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