from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional
import datasets

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify


QUERY_TEMPLATE = "{Question}"

QUERY_TEMPLATE_INSTRUCT_BOXED_NO_CONTEXT = """Solve the following math problem.

Return your final answer as an integer from 0 to 999, and put it in \\boxed{{}} on the last line.

Problem:
{Question}
"""

QUERY_TEMPLATE_INSTRUCT_BOXED_2024 = """Solve the following AIME 2024 problem.

Return your final answer as an integer from 0 to 999, and put it in \\boxed{{}} on the last line.

Problem:
{Question}
"""

QUERY_TEMPLATE_INSTRUCT_BOXED_2025 = """Solve the following AIME 2025 problem.

Return your final answer as an integer from 0 to 999, and put it in \\boxed{{}} on the last line.

Problem:
{Question}
"""

def doc_to_text(doc: dict) -> str:
    q = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question") or ""
    return QUERY_TEMPLATE.format(Question=q)


def doc_to_text_instruct_no_context(doc: dict) -> str:
    q = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question") or ""
    return QUERY_TEMPLATE_INSTRUCT_BOXED_NO_CONTEXT.format(Question=q)


def doc_to_text_instruct_2024(doc: dict) -> str:
    q = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question") or ""
    return QUERY_TEMPLATE_INSTRUCT_BOXED_2024.format(Question=q)


def doc_to_text_instruct_2025(doc: dict) -> str:
    q = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question") or ""
    return QUERY_TEMPLATE_INSTRUCT_BOXED_2025.format(Question=q)

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution") or doc.get("Solution") or doc.get("orig_solution") or doc.get("orig_orig_solution")
        problem = doc.get("problem") or doc.get("Problem") or doc.get("question") or doc.get("Question")
        answer = doc.get("answer") or doc.get("Answer") or doc.get("orig_answer") or doc.get("orig_orig_answer")

        out_doc = {"problem": problem, "solution": solution, "answer": answer}
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


# NeMo-style sampling metrics (per-instance):
# - pass@k: combinatorial estimator for binary scores; otherwise max over first-k
# - majority@k: mode over (answer, score) pairs; ties averaged
# - no_answer@k: all first-k answers are None
KS = (1, 2, 4, 8, 16, 32)
LOG = logging.getLogger(__name__)


# -------------------------
# NeMo math_grader.py (integrated)
# -------------------------
def _additional_normalization(expr: str) -> str:
    # Remove % and \\% from the number
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    # Remove . corresponding to the end of sentence
    expr = expr.rstrip(".\\")
    return expr


def math_equal(
    gt_answer: Any,
    predicted_answer: Any,
    take_modulo: int | None = None,
    **kwargs,
) -> bool:
    """
    NeMo math_grader.math_equal:
      - optional modulo integer comparison
      - MCQ option comparison (A-J) if GT is a single option
      - latex string normalization fast-path
      - fallback to symbolic comparison using math_verify.parse/verify
    """
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    # If we are sure that gt is always integer
    if take_modulo is not None:
        gt_answer_int = int(gt_answer) % take_modulo
        try:
            predicted_int = int(predicted_answer) % take_modulo
        except Exception:
            predicted_int = None
        return predicted_int == gt_answer_int

    # Try to compare as MCQ options
    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()
    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)

    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    if is_mcq and verify(parsed_gt, parsed_pred):
        return verify(parsed_gt, parsed_pred)

    # Additional normalization step
    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    # Fast path: if normalized strings are equal, no need for symbolic comparison
    if is_normalized_equal:
        return True

    # For TEXT literals (not numeric), use direct string comparison
    text_literal_pattern = r"[a-zA-Z ,]+"
    is_text_literal = re.fullmatch(text_literal_pattern, normalized_gt) and re.fullmatch(
        text_literal_pattern, normalized_pred
    )
    if is_text_literal:
        return False  # Already checked is_normalized_equal above

    # Fallback to symbolic comparison via math_verify
    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer

    # math_verify.parse expects input to be in latex environment, e.g. $...$
    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()])
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()])

    return bool(verify(parsed_gt, parsed_pred, **kwargs))


def extract_answer(
    string: str,
    extract_from_boxed: bool = True,
    extract_regex: str = r"The final answer is (.+)$",
    relaxed: bool = False,
) -> Optional[str]:
    """NeMo math_grader.extract_answer

    If relaxed=True: try both methods, regex first.
    If relaxed=False: use only one method based on extract_from_boxed flag.
    """
    if relaxed:
        return search_regex(string, extract_regex) or search_boxed(string)

    if extract_from_boxed:
        return search_boxed(string)
    return search_regex(string, extract_regex)


def search_regex(string: str, regex: str) -> Optional[str]:
    match = re.findall(regex, string)
    if match:
        return match[-1]
    return None


def search_boxed(string: str) -> Optional[str]:
    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


# -------------------------
# Metrics (NeMo BaseMetrics-style formulations, flat output)
# -------------------------
def _flatten_results(results):
    # Flatten list-of-lists-of-... into a flat list of strings
    while results and isinstance(results[0], list):
        if len(results) == 1:
            results = results[0]
        else:
            results = [x for sub in results for x in sub]
    return results


def _is_binary_score(scores: List[float]) -> bool:
    return all(score in (0, 1, True, False) for score in scores)


def process_results(
    doc: dict,
    results: List[str],
    *,
    # NeMo MathEvaluatorConfig-like knobs (defaults match the Nemo code you pasted)
    extract_from_boxed: bool = True,
    extract_regex: str = r"The final answer is (.+)$",
    relaxed_extraction: bool = False,
    take_modulo: int | None = None,
    numeric_precision: int = 15,
    timeout_seconds: int = 10,
) -> Dict[str, Any]:
    """
    Outputs (flat dict):
      - avg_score
      - no_answer_rate, no_answer_count, num_generations
      - extracted_answers
      - pass@k for k in KS
      - majority@k for k in KS

    Extraction + symbolic checking:
      predicted_answer := Nemo extract_answer(generation, ...)
      score := Nemo math_equal(expected_answer, predicted_answer, ...)
    """
    if not results:
        out: Dict[str, Any] = {
            "avg_score": 0.0,
            "no_answer_rate": 1.0,
            "no_answer_count": 0,
            "num_generations": 0,
            # For per-sample logging/debugging (aggregate with `bypass`).
            "extracted_answers": [],
        }
        for k in KS:
            out[f"pass@{k}"] = 0.0
            out[f"majority@{k}"] = 0.0
        return out

    results = _flatten_results(results)

    # Accept either "answer"/"Answer" style or NeMo-style "expected_answer"
    gt = doc.get("expected_answer", doc.get("answer", doc.get("Answer", "")))
    gt = str(gt)

    predicted_answers: List[Optional[str]] = []
    scores: List[float] = []

    for gen in results:
        gen = str(gen)

        pred_ans = extract_answer(
            gen,
            extract_from_boxed=extract_from_boxed,
            extract_regex=extract_regex,
            relaxed=relaxed_extraction,
        )
        predicted_answers.append(pred_ans)

        correct = math_equal(
            gt,
            pred_ans,
            take_modulo=take_modulo,
            numeric_precision=numeric_precision,
            timeout_seconds=timeout_seconds,
        )
        scores.append(1.0 if correct else 0.0)

    n = len(scores)
    no_answer_count = sum(1 for a in predicted_answers if a is None)
    out: Dict[str, Any] = {
        "avg_score": float(sum(scores) / n) if n else 0.0,
        "no_answer_rate": float(no_answer_count / n) if n else 1.0,
        "no_answer_count": no_answer_count,
        "num_generations": n,
        # For per-sample logging/debugging (aggregate with `bypass`).
        "extracted_answers": predicted_answers,
    }

    binary = _is_binary_score(scores)
    if binary:
        total = n
        total_correct = int(sum(scores))
        total_incorrect = total - total_correct

    for k in KS:
        k_eff = min(k, n)

        # NOTE: We intentionally do not emit the `*_no_answer` metrics here to keep logs clean.
        # (Those metrics are typically redundant with `no_answer_rate` / `no_answer_count`.)
        # If you want them back, restore the NeMo-style keys:
        # no_answer = all(a is None for a in predicted_answers[:k_eff]) if k_eff else True
        # out[f"pass@{k}_no_answer"] = float(no_answer)
        # out[f"majority@{k}_no_answer"] = float(no_answer)

        # pass@k: NeMo formulation
        if k_eff == 0:
            out[f"pass@{k}"] = 0.0
        elif binary:
            if total_incorrect < k_eff:
                prob_all_incorrect = 0.0
            else:
                prob_all_incorrect = math.comb(total_incorrect, k_eff) / math.comb(total, k_eff)
            out[f"pass@{k}"] = float(1.0 - prob_all_incorrect)
        else:
            out[f"pass@{k}"] = float(max(scores[:k_eff]))

        # majority@k: NeMo BaseMetrics logic (mode over (answer, score) pairs; ties averaged)
        valid_answers_and_results = [
            (pred_answer, score)
            for pred_answer, score in zip(predicted_answers[:k_eff], scores[:k_eff])
            if pred_answer is not None
        ]
        if not valid_answers_and_results:
            out[f"majority@{k}"] = 0.0
        else:
            counter = Counter(valid_answers_and_results)
            majority_count = counter.most_common(1)[0][1]
            tied_pairs = [(ans, sc) for (ans, sc), cnt in counter.items() if cnt == majority_count]
            out[f"majority@{k}"] = float(sum(sc for _, sc in tied_pairs) / len(tied_pairs))

    return out
