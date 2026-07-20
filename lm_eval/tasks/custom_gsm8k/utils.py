from typing import Dict, List, Optional
import datasets
import re

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify


# ---------- normalization + comparison ----------

def _additional_normalization(expr: str) -> str:
    # Strip '%' / '\%' if it's a pure percentage literal, and trailing sentence punctuation.
    m = re.fullmatch(r"^(\d+\.?\d*)(?:\\%|%)$", expr)
    if m:
        expr = m.group(1)
    return expr.rstrip(".\\")


def math_equal(
    gt_answer: str,
    predicted_answer: Optional[str],
    take_modulo: Optional[int] = None,
    **kwargs,
) -> bool:
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    # Optional integer-mod comparison
    if take_modulo is not None:
        try:
            return (int(gt_answer) % take_modulo) == (int(predicted_answer) % take_modulo)
        except Exception:
            return False

    # MCQ fallback (kept for compatibility)
    mcq_options = "ABCDEFGHIJ"
    if re.fullmatch("|".join(mcq_options), gt_answer.strip()):
        pg = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
        pp = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
        return verify(pg, pp)

    # Literal / normalized equality
    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    norm_gt = normalize_latex(gt_answer, NormalizationConfig)
    norm_pred = normalize_latex(predicted_answer, NormalizationConfig)
    if (
        re.fullmatch(r"[a-zA-Z ,]+|[0-9 ]+", norm_gt)
        and re.fullmatch(r"[a-zA-Z ,]+|[0-9 ]+", norm_pred)
    ):
        return norm_gt.replace(" ", "") == norm_pred.replace(" ", "")
    if norm_gt.replace(" ", "") == norm_pred.replace(" ", ""):
        return True

    # Symbolic verification via math_verify
    env_pat = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    gt = gt_answer if re.search(env_pat, gt_answer, re.DOTALL) else f"${gt_answer}$"
    pr = predicted_answer if re.search(env_pat, predicted_answer, re.DOTALL) else f"${predicted_answer}$"
    pg = parse(gt, [LatexExtractionConfig()])
    pp = parse(pr, [LatexExtractionConfig()])
    return verify(pg, pp, **kwargs)


# ---------- boxed extraction ----------

def extract_answer(text: str, extract_from_boxed: bool = True,
                   extract_regex: str = r"The final answer is (.+)$") -> Optional[str]:
    """
    Prefer the *last* \boxed{...} or \fbox{...} in the string.
    Falls back to a regex if extract_from_boxed=False.
    """
    if not extract_from_boxed:
        m = re.search(extract_regex, text)
        return m.group(1) if m else None

    if "\\boxed" not in text and "\\fbox" not in text:
        return None

    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None

    # Find matching '}' for the '{' after \boxed or \fbox
    i = idx
    opens = 0
    right = None
    while i < len(text):
        c = text[i]
        if c == "{":
            opens += 1
        elif c == "}":
            opens -= 1
            if opens == 0:
                right = i
                break
        i += 1
    if right is None:
        return None

    candidate = text[idx:right + 1]
    for pref in ("\\boxed{", "\\fbox{"):
        if candidate.startswith(pref) and candidate.endswith("}"):
            return candidate[len(pref):-1]
    return None


# ---------- dataset plumbing ----------

def _normalize_gold(raw: Optional[str]) -> str:
    """
    GSM8K 'answer' is rationale ending with '#### <gold>'.
    Return just the normalized numeric/string gold.
    """
    if not isinstance(raw, str):
        return ""
    if "####" in raw:
        return raw.split("####")[-1].strip()
    # If a solution with boxed gold is present (some variants), accept it.
    boxed = extract_answer(raw)
    if boxed is not None:
        return boxed.strip()
    return raw.strip()


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Produce a normalized 'answer' field used for scoring (not shown in prompt).
    Keep original text for debugging.
    """
    def _one(doc: dict) -> dict:
        q = doc.get("question") or doc.get("problem") or ""
        raw_answer = doc.get("answer") or doc.get("solution") or ""
        gold = _normalize_gold(raw_answer)
        return {
            "question": q,
            "answer": gold,          # normalized gold (numeric/string)
            "answer_text": raw_answer,  # original rationale if needed
        }
    return dataset.map(_one)


def doc_to_target(doc: dict) -> str:
    """
    IMPORTANT: This is used to *append few-shot targets to the prompt*.
    Behavior:
      - For YAML-provided few-shots, return the 'target' string verbatim
        (rationale + 'The final answer is ...' or boxed).
      - For dataset/eval items, return nothing; lm-eval won’t append it to the prompt anyway,
        and scoring is handled in process_results.
    """
    t = doc.get("target")
    if isinstance(t, str) and t.strip():
        return t.strip()

    # For dataset items, returning empty string avoids any chance of leakage.
    return ""


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """
    Score by extracting the model's final boxed value and verifying against gold via math_verify.
    """
    pred = extract_answer(results[0])
    gold = doc.get("answer", "")
    ok = bool(math_equal(gold, pred))
    return {"exact_match": 1 if ok else 0}