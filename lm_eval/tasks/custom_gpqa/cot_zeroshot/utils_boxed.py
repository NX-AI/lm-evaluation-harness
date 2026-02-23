from typing import Dict, List, Optional
import datasets
import re
import random

MCQ_OPTIONS = tuple("ABCDEFGHIJ")

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            # store bare letter, e.g. "A"
            "answer": chr(65 + correct_answer_index),
        }
        return out_doc

    return dataset.map(_process_doc)

# ----------------
# Robust extractor
# ----------------

def _find_last_boxed_span(s: str) -> Optional[tuple[int, int]]:
    """Return (start_idx, end_idx_exclusive) for the last \boxed{...} or \fbox{...} by brace matching."""
    if s is None:
        return None
    idx = s.rfind(r"\boxed")
    if idx < 0:
        idx = s.rfind(r"\fbox")
        if idx < 0:
            return None

    # move to first '{' after the command
    i = idx
    while i < len(s) and s[i] != "{":
        i += 1
    if i >= len(s) or s[i] != "{":
        return None

    depth = 0
    j = i
    while j < len(s):
        ch = s[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (idx, j + 1)
        j += 1
    return None

def extract_last_boxed_content(s: str) -> Optional[str]:
    """Inner payload of the final \boxed{...}/\fbox{...} without outer braces."""
    span = _find_last_boxed_span(s)
    if not span:
        return None
    start, end = span
    block = s[start:end]
    brace_idx = block.find("{")
    if brace_idx < 0 or not block.endswith("}"):
        return None
    return block[brace_idx + 1 : -1]

# Common single-argument TeX wrappers to peel off repeatedly
_WRAPPER_PATTERNS = [
    r"\\text\s*\{(?P<inner>.*)\}\s*$",
    r"\\mathrm\s*\{(?P<inner>.*)\}\s*$",
    r"\\mathbf\s*\{(?P<inner>.*)\}\s*$",
    r"\\mathsf\s*\{(?P<inner>.*)\}\s*$",
    r"\\textrm\s*\{(?P<inner>.*)\}\s*$",
    r"\\textbf\s*\{(?P<inner>.*)\}\s*$",
    r"\\emph\s*\{(?P<inner>.*)\}\s*$",
    r"\\underline\s*\{(?P<inner>.*)\}\s*$",
    r"^\$\s*(?P<inner>.*)\s*\$\s*$",
    r"^\\\(\s*(?P<inner>.*)\s*\\\)\s*$",
    r"^\\\[\s*(?P<inner>.*)\s*\\\]\s*$",
    r"^\(\s*(?P<inner>.*)\s*\)\s*$",
]

def _strip_tex_wrappers(s: str) -> str:
    """Recursively strip wrappers/parens around a single symbol."""
    if s is None:
        return ""
    prev = None
    cur = s.strip()
    for _ in range(10):  # defensive depth cap
        if prev == cur:
            break
        prev = cur
        for pat in _WRAPPER_PATTERNS:
            m = re.fullmatch(pat, cur, flags=re.DOTALL)
            if m:
                cur = m.group("inner").strip()
                break
    return cur

def normalize_mcq_label(s: Optional[str]) -> Optional[str]:
    """Map payload to a canonical MCQ label A-J if present; else None."""
    if s is None:
        return None
    t = _strip_tex_wrappers(s)

    # Prefer A-J letters; take the *last* one to be safe (e.g., "Option A")
    letters = re.findall(r"[A-Za-z]", t)
    letters = [L.upper() for L in letters if L.upper() in MCQ_OPTIONS]
    if letters:
        return letters[-1]

    # Optional: digits 1..10 -> A..J
    m = re.search(r"\b([1-9]|10)\b", t)
    if m:
        k = int(m.group(1)) - 1
        if 0 <= k < len(MCQ_OPTIONS):
            return MCQ_OPTIONS[k]
    return None

def extract_mcq_from_output(s: str) -> Optional[str]:
    """Primary: last \\boxed{...}. Fallbacks: 'The answer is A' or last '(A)'."""
    payload = extract_last_boxed_content(s)
    lab = normalize_mcq_label(payload)
    if lab:
        return lab

    # Fallback 1
    for pat in [
        r"The answer is[^A-Ja-j]*([A-Ja-j])",
        r"Answer\s*:\s*([A-Ja-j])",
        r"choice\s*[:=]?\s*([A-Ja-j])",
    ]:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Fallback 2
    candidates = list(re.finditer(r"\(([A-Ja-j])\)", s))
    if candidates:
        return candidates[-1].group(1).upper()

    return None

# ----------------
# Harness glue
# ----------------

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """results[0] is the full model generation."""
    pred = extract_mcq_from_output(results[0])
    gold = doc["answer"]  # e.g., "A"
    return {"exact_match": 1 if (pred is not None and pred == gold) else 0}