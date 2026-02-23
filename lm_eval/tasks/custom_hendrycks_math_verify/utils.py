from typing import Dict, List
import datasets
import re

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify


def _additional_normalization(expr):
    # Remove % and \\% from the number
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    # Remove . corresponding to the end of sentence
    expr = expr.rstrip(".\\")
    return expr


def math_equal(gt_answer, predicted_answer, take_modulo: int | None = None, **kwargs):
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    # if we are sure that gt is always integer
    if take_modulo is not None:
        gt_answer = int(gt_answer) % take_modulo
        try:
            predicted_answer = int(predicted_answer) % take_modulo
        except Exception:
            predicted_answer = None
        # no need to simpy call in this case
        return predicted_answer == gt_answer

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

    # Try literal comparison
    literal_pattern = r"[a-zA-Z ,]+|[0-9 ]+"
    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_literal = re.fullmatch(literal_pattern, normalized_gt) and re.fullmatch(literal_pattern, normalized_pred)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    if is_literal or is_normalized_equal:
        return is_normalized_equal

    # Fallback to symbolic comparison
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

    return verify(parsed_gt, parsed_pred, **kwargs)


def extract_answer(string: str, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$"):
    """Extract Answer String from \\boxed expression or based on regex"""
    if not extract_from_boxed:
        match = re.search(extract_regex, string)
        if match:
            return match.group(1)
        return None

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

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc: dict) -> dict:
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": extract_answer(doc["solution"]),  # expects last \boxed{...}
        }

    return dataset.map(_process_doc)

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    pred = extract_answer(results[0])  # expects last \boxed{...} in model output
    gold = doc["answer"]
    is_equal = bool(math_equal(gold, pred))
    return {"exact_match": 1 if is_equal else 0}


def process_results_avg(doc: dict, results):
    """
    Average exact-match over multiple rollouts for a single question.

    `results` may be either [single_str] or [list_of_strs] if a filter
    (e.g. take_first_k) preserved multiple generations per document.
    """
    first = results[0]
    if isinstance(first, list):
        preds = [extract_answer(r) for r in first]
    else:
        preds = [extract_answer(first)]

    gold = doc["answer"]
    correct = [1 if math_equal(gold, p) else 0 for p in preds]
    avg = sum(correct) / len(correct) if len(correct) > 0 else 0.0
    return {"exact_match": avg}

