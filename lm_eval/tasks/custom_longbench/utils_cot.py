from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any


def _load_local_module(module_name: str):
    module_path = Path(__file__).with_name(f"{module_name}.py")
    qualified_name = f"lm_eval.tasks.custom_longbench.{module_name}"
    cached = sys.modules.get(qualified_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local module from {module_path}") from None

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


metrics = _load_local_module("metrics")


_FINAL_BLOCK_RE = re.compile(r"<final>(?P<body>.*?)</final>", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_ANALYSIS_BLOCK_RE = re.compile(
    r"<analysis>.*?</analysis>\s*", re.DOTALL | re.IGNORECASE
)
_FINAL_ANSWER_PATTERNS = (
    re.compile(
        r"(?:^|\n)\s*(?:\*\*|__)?final answer(?:\*\*|__)?\s*(?:is|:)?\s*(?P<body>[\s\S]+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*answer\s*:\s*(?P<body>[\s\S]+)$",
        re.IGNORECASE,
    ),
)


def _extract_final_answer(text: str) -> str:
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group("body").strip()
    return text.strip()


def strip_reasoning(text: str) -> str:
    """Strip reasoning traces and preserve the final answer when it is marked."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", "")

    finals = _FINAL_BLOCK_RE.findall(text)
    if finals:
        return finals[-1].strip()

    lowered = text.lower()
    if "<think>" in lowered and "</think>" not in lowered:
        return ""
    if "<analysis>" in lowered and "</analysis>" not in lowered:
        return ""

    text = re.sub(_THINK_BLOCK_RE, "", text)
    text = re.sub(_ANALYSIS_BLOCK_RE, "", text)

    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?analysis>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?final>", "", text, flags=re.IGNORECASE)

    return _extract_final_answer(text)


def filter_responses(
    resps: list[list[str]], docs: list[dict[str, Any]]
) -> list[list[str]]:
    del docs
    return [[strip_reasoning(resp) for resp in inst] for inst in resps]


def _wrap_metrics(fn):
    def _wrapped(doc: dict[str, Any], results: list[str], **kwargs):
        cleaned = strip_reasoning(results[0] if results else "")
        return fn(doc, [cleaned], **kwargs)

    return _wrapped


# Export the same function names as `metrics.py`, but strip reasoning tags first.
get_qa_f1_with_score = _wrap_metrics(metrics.get_qa_f1_with_score)
get_qa_f1_zh_with_score = _wrap_metrics(metrics.get_qa_f1_zh_with_score)
get_rouge_with_score = _wrap_metrics(metrics.get_rouge_with_score)
get_rouge_zh_with_score = _wrap_metrics(metrics.get_rouge_zh_with_score)
get_classification_with_score = _wrap_metrics(metrics.get_classification_with_score)
get_count_with_score = _wrap_metrics(metrics.get_count_with_score)
get_retrieval_with_score = _wrap_metrics(metrics.get_retrieval_with_score)
get_retrieval_zh_with_score = _wrap_metrics(metrics.get_retrieval_zh_with_score)
get_code_sim_with_score = _wrap_metrics(metrics.get_code_sim_with_score)
