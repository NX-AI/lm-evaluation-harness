from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import datasets
from math_verify import ExprExtractionConfig, LatexExtractionConfig, math_metric


QUERY_TEMPLATE = "{Question}"

# prompt sourced from https://arxiv.org/pdf/2505.23281
QUERY_TEMPLATE_INSTRUCT_BOXED_202502 = """Solve the following problem from the HMMT February 2025 competition.

Please reason step by step, and put your final answer within \\boxed{{}} on the last line.

Problem:
{Question}
"""

QUERY_TEMPLATE_INSTRUCT_BOXED_202511 = """Solve the following problem from the HMMT November 2025 competition.

Please reason step by step, and put your final answer within \\boxed{{}} on the last line.

Problem:
{Question}
"""

KS = (1, 2, 4, 8, 16, 32)
_INT_TOKEN_RE = re.compile(r"\b\d+\b")


def _last_five_agg(vals: List[float]) -> float:
    vals = vals[-5:]
    return max(vals) if vals else 0.0


hmmt_pred_extractors: Sequence[Any] = (
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
)

hmmt_verify_fn = math_metric(
    gold_extraction_target=(ExprExtractionConfig(),),
    pred_extraction_target=hmmt_pred_extractors,
    aggregation_function=_last_five_agg,
    precision=6,
)


def doc_to_text(doc: dict) -> str:
    q = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question") or ""
    return QUERY_TEMPLATE.format(Question=q)


def doc_to_text_instruct_202502(doc: dict) -> str:
    q = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question") or ""
    return QUERY_TEMPLATE_INSTRUCT_BOXED_202502.format(Question=q)


def doc_to_text_instruct_202511(doc: dict) -> str:
    q = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question") or ""
    return QUERY_TEMPLATE_INSTRUCT_BOXED_202511.format(Question=q)


def _normalize_hmmt_int(text: str) -> Optional[str]:
    if not text:
        return None
    for t in reversed(_INT_TOKEN_RE.findall(text)):
        try:
            v = int(t)
        except Exception:
            continue
        if 0 <= v <= 999:
            return str(v)
    return None


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        problem = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question")
        answer = doc.get("answer") or doc.get("Answer") or doc.get("orig_answer") or doc.get("orig_orig_answer")

        out_doc = {"problem": problem, "answer": answer}
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def _fallback_vote_key(text: str) -> str:
    v = _normalize_hmmt_int(text)
    return v if v is not None else ""


def _vote_key_from_extracted_preds(extracted_preds: Optional[Sequence[str]]) -> str:
    if not extracted_preds:
        return ""
    for item in reversed(extracted_preds):
        s = str(item).strip()
        if not s:
            continue
        v = _normalize_hmmt_int(s)
        return v if v is not None else s
    return ""


def _grade_and_extract(gold: str, pred: str) -> Tuple[float, Sequence[str]]:
    try:
        grade, extracted = hmmt_verify_fn([gold], [pred])
        extracted_preds: Sequence[str] = ()
        if isinstance(extracted, tuple) and len(extracted) == 2:
            extracted_preds = extracted[1] or ()
        return float(grade), extracted_preds
    except Exception:
        return 0.0, ()
    

def _flatten_results(results):
    # Flatten list-of-lists-of-... into a flat list of strings
    while results and isinstance(results[0], list):
        if len(results) == 1:
            results = results[0]
        else:
            results = [x for sub in results for x in sub]
    return results


def process_results(doc: dict, results: List[str]) -> Dict[str, Any]:
    if not results:
        out: Dict[str, Any] = {"exact_matches": 0.0}
        for k in KS:
            out[f"cov@{k}"] = 0
            out[f"maj@{k}"] = 0.0
        return out

    results = _flatten_results(results)

    gt = str(doc.get("answer", doc.get("Answer", "")))

    accs: List[float] = []
    vote_keys: List[str] = []

    for res in results:
        res = str(res)
        grade, extracted_preds = _grade_and_extract(gt, res)
        accs.append(float(grade))

        vk = _vote_key_from_extracted_preds(extracted_preds) or _fallback_vote_key(res)
        vote_keys.append(vk or "")

    out: Dict[str, Any] = {"exact_matches": (sum(accs) / len(accs)) if accs else 0.0}

    for k in KS:
        m = min(k, len(accs))
        out[f"cov@{k}"] = int(any(a == 1.0 for a in accs[:m])) if m else 0

        keys = [x for x in vote_keys[:m] if x]
        if keys:
            maj_key = Counter(keys).most_common(1)[0][0]
            maj_grade, _ = _grade_and_extract(gt, maj_key)
            out[f"maj@{k}"] = float(maj_grade)
        else:
            out[f"maj@{k}"] = 0.0

    return out