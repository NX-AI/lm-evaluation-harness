# cruxeval_lm_eval.py
import re
import io
import tokenize
from .utils_general import evaluate_doc_generations

# --- Regex helpers ---
_ANSWER_BLOCK_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL | re.IGNORECASE)
_ASSERT_F_RE = re.compile(
    r"^\s*assert\s+f\s*\((.*?)\)\s*==\s*([^\n\r]*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Common “reasoning wrappers”
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FINAL_BLOCK_RE = re.compile(r"<final>.*?</final>\s*", re.DOTALL | re.IGNORECASE)


def _strip_trailing_comment(expr: str) -> str:
    """
    Remove a trailing Python comment from an expression, without touching # inside strings.
    Example: '"a#b"  # ok' -> '"a#b"'
    """
    try:
        b = expr.encode("utf-8")
        tokens = tokenize.tokenize(io.BytesIO(b).readline)

        out = []
        for tok in tokens:
            if tok.type == tokenize.ENCODING:
                continue
            if tok.type == tokenize.COMMENT:
                break
            if tok.type == tokenize.ENDMARKER:
                break
            out.append(tok.string)

        return "".join(out).strip()

    except Exception:
        return re.sub(r"\s+#.*$", "", expr).strip()


def _clean_generation(text):
    """
    Cleans a single generation string:
    - pulls content out of markdown code fences
    - strips some common prefixes
    - removes obvious reasoning wrappers
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    # Extract code from Markdown blocks if present
    if "```" in text:
        pattern = r"```(?:python)?\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    # Strip common prefixes
    if text.lower().startswith("answer:"):
        text = text[7:].strip()

    # Remove reasoning wrappers if they exist (best-effort)
    text = re.sub(_THINK_BLOCK_RE, "", text).strip()
    text = re.sub(_FINAL_BLOCK_RE, "", text).strip()

    # Strip trailing "# done" if present
    text = re.sub(r"\s*#\s*done\s*$", "", text, flags=re.IGNORECASE).strip()

    return text


def _extract_answer_region(text: str) -> str:
    """
    Returns content inside [ANSWER]...[/ANSWER] if present.
    If only [ANSWER] is present (because [/ANSWER] got stopped/stripped), return everything after [ANSWER].
    If neither is present, return original text.
    """
    m = _ANSWER_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    # Handle partial tag cases
    idx = text.lower().find("[answer]")
    if idx != -1:
        after = text[idx + len("[answer]") :].strip()
        # If a closing tag exists later, cut before it (even if malformed spacing)
        j = after.lower().find("[/answer]")
        if j != -1:
            after = after[:j].strip()
        return after

    return text.strip()


def _extract_expected_expression(text: str, mode: str) -> str:
    """
    Extracts the expression we want to execute/compare:
      - Prefer [ANSWER] region
      - Accept either:
          * expression-only (preferred)
          * full assert line: assert f(<args>) == <rhs>
    """
    text = _extract_answer_region(text).strip()

    # If they returned a full assertion, extract args / rhs
    m2 = _ASSERT_F_RE.search(text)
    if not m2 and "assert" in text.lower() and "==" in text:
        # Fallback: split on first == (avoid greedy capture)
        left, rhs = text.split("==", 1)
        mleft = re.search(r"assert\s+f\s*\((.*?)\)\s*$", left.strip(), re.IGNORECASE | re.DOTALL)
        if mleft:
            args = mleft.group(1).strip()
            rhs = _strip_trailing_comment(rhs.strip())
            rhs = re.sub(r"\s*#\s*done\s*$", "", rhs, flags=re.IGNORECASE).strip()
            return args if mode == "input" else rhs

    if m2:
        args = m2.group(1).strip()
        rhs = _strip_trailing_comment(m2.group(2).strip())
        rhs = re.sub(r"\s*#\s*done\s*$", "", rhs, flags=re.IGNORECASE).strip()
        return args if mode == "input" else rhs

    # Otherwise, assume the whole thing is the expression
    return text.strip()


def _flatten_results(results):
    """
    Keep ALL rollouts.
    Common shapes:
      - results == [gen1, gen2, ...]
      - results == [[gen1, gen2, ...]]   (single doc, multiple samples)
      - results == [[gen1], [gen2], ...] (less common)
    """
    if not results:
        return []

    # If it's exactly [[str, str, ...]] treat inner list as the full set of rollouts
    if (
        len(results) == 1
        and isinstance(results[0], (list, tuple))
        and all(isinstance(x, str) for x in results[0])
    ):
        return list(results[0])

    # Otherwise flatten everything
    flat = []
    for r in results:
        if isinstance(r, (list, tuple)):
            flat.extend(list(r))
        else:
            flat.append(r)
    return flat


def _clean_generations(results, mode: str):
    """
    Unwrap lm-eval results structure, clean strings, then extract the expression to score.
    """
    flat_results = _flatten_results(results)
    cleaned = [_clean_generation(r) for r in flat_results]
    return [_extract_expected_expression(r, mode=mode) for r in cleaned]


def cruxeval_process_results_input(doc, results):
    """
    For CRUXEval-I (input prediction).
    """
    gens = _clean_generations(results, mode="input")
    code = doc["code"]
    inp = doc["input"]
    out = doc["output"]

    scores = evaluate_doc_generations(gens, code, inp, out, mode="input")
    return scores


def cruxeval_process_results_output(doc, results):
    """
    For CRUXEval-O (output prediction).
    """
    gens = _clean_generations(results, mode="output")
    code = doc["code"]
    inp = doc["input"]
    out = doc["output"]

    scores = evaluate_doc_generations(gens, code, inp, out, mode="output")
    return scores
