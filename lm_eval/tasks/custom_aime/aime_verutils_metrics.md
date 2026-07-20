# AIME NVIDIA Verifier Metrics (`aime_verutils_nvidia.py`)

This note documents exactly how `lm_eval/tasks/aime/aime_verutils.py` extracts answers, scores generations, and computes the summary metrics returned by `process_results(...)`.

## What `process_results` Consumes

`process_results(doc, results, ...)` receives:

- `doc`: the dataset row. The ground-truth is read from `doc["expected_answer"]`, falling back to `doc["answer"]` then `doc["Answer"]`.
- `results`: model generations. In sampling setups (e.g. `repeats: 8`), this is typically a list of multiple full generations for the same question. The function also tolerates nested list shapes and flattens them via `_flatten_results(...)`.

## Extraction: From Generation -> `predicted_answer`

For each generation string `gen`, the code calls:

`pred_ans = extract_answer(gen, extract_from_boxed=..., extract_regex=..., relaxed=...)`

Extraction behavior:

- Default path (`extract_from_boxed=True`, `relaxed=False`): use `search_boxed(gen)`.
- Regex path (`extract_from_boxed=False`, `relaxed=False`): use `search_regex(gen, extract_regex)`.
- Relaxed path (`relaxed=True`): try regex first; if it fails, try boxed.

### `search_boxed(...)`

Returns the content of the *last* `\\boxed{...}` (or `\\fbox{...}`) in the string:

- It finds the last occurrence via `rfind("\\boxed")` (or `\\fbox`).
- It then scans forward and balances braces until the matching closing `}` is found.
- If braces do not balance, or the slice is not exactly `\\boxed{...}`, it returns `None`.

### `search_regex(...)`

Applies `re.findall(regex, string)` and returns the last match (`match[-1]`) or `None`.

The default regex is `r"The final answer is (.+)$"`, so if your prompt format does not include that phrase, it will typically return `None` unless you enable boxed extraction.

## Scoring: `predicted_answer` vs Ground Truth

Each extracted answer is scored with:

`correct = math_equal(gt, pred_ans, take_modulo=..., numeric_precision=..., timeout_seconds=...)`

and turned into a binary float:

- `score = 1.0` if `correct` else `0.0`

Key `math_equal(...)` behavior:

- If `predicted_answer is None`: returns `False`.
- If `take_modulo` is set: compares `int(gt) % take_modulo` against `int(pred) % take_modulo`. If `pred` cannot be parsed as an int, it fails.
- MCQ shortcut (A-J): if the GT is a single option letter, it attempts parsing/verification as a string option.
- Normalization:
  - `_additional_normalization` removes a trailing `%` / `\\%` from a pure number like `50%`, and strips trailing `.` or `\\`.
  - `normalize_latex(...)` is applied to both sides. If the normalized strings match ignoring spaces, it returns `True` immediately.
- Text-literal guard:
  - If both sides look like plain text (letters/spaces/commas) and they were not equal after normalization, it returns `False` (no symbolic math).
- Symbolic fallback:
  - Wraps inputs in `$...$` if they are not already in a LaTeX environment.
  - Uses `math_verify.parse(..., LatexExtractionConfig())` and `math_verify.verify(...)`.
  - `timeout_seconds` and `numeric_precision` are forwarded to `verify(...)`.

## Returned Per-Example Fields

`process_results(...)` returns a *flat dict* of per-example metrics. The harness then aggregates these across the dataset according to the YAML `metric_list`.

### Debug / Logging Metric

- `extracted_answers`
  - Value: the per-generation extracted answers list (`List[Optional[str]]`), in the same order as `results` after flattening.
  - Intended aggregation: `bypass` (log it, don’t average it).
  - Notes:
    - `None` means extraction failed for that generation.
    - Depending on your logger, `None` may be stringified (e.g. `"None"`) during JSON sanitization.

### Missing-Answer Summary Metrics

Let `n = num_generations = len(results)` after flattening.

- `num_generations`
  - Value: `n`
- `no_answer_count`
  - Value: number of generations where extraction returned `None`.
- `no_answer_rate`
  - Value: `no_answer_count / n` (a float in `[0, 1]`).

### Accuracy / Sampling Metrics

Let `scores[i]` be the 0/1 score for generation `i`.

- `avg_score`
  - Value: mean of `scores` across all generations for that example.

- `pass@k` for `k in (1, 2, 4, 8, 16, 32)`
  - Uses `k_eff = min(k, n)`.
  - For binary scores (AIME here), it uses the NeMo-style combinatorial estimator:
    - `pass@k = 1 - C(total_incorrect, k_eff) / C(total, k_eff)`
    - where `total = n` and `total_incorrect = n - sum(scores)`.
  - Notes:
    - This estimator uses the counts over *all* `n` samples, not only the first `k`.
    - If `total_incorrect < k_eff`, then `pass@k = 1.0`.

- `majority@k` for `k in (1, 2, 4, 8, 16, 32)`
  - Uses `k_eff = min(k, n)`.
  - Builds a multiset over the first `k_eff` generations of `(predicted_answer, score)` pairs, dropping any where `predicted_answer is None`.
  - Takes the mode of that multiset. If multiple pairs tie for most frequent, it averages the tied pairs’ scores.
  - If there are no non-`None` extracted answers in the first `k_eff`, `majority@k = 0.0`.

## Edge Cases

- Empty `results`:
  - Returns:
    - `avg_score = 0.0`
    - `no_answer_rate = 1.0`
    - `no_answer_count = 0`
    - `num_generations = 0`
    - `extracted_answers = []`
    - `pass@k = 0.0` and `majority@k = 0.0` for all `k`

- Extraction failures (`predicted_answer is None`):
  - That generation is scored as incorrect (`score = 0.0`).
  - It contributes to `no_answer_count` / `no_answer_rate`.
  - It is excluded from `majority@k` voting (because the vote list filters out `None` answers).

- Multiple `\\boxed{...}` occurrences:
  - Only the *last* one is used.

- Unbalanced braces after `\\boxed` / malformed `\\boxed`:
  - Extraction returns `None`.

- Percentage strings like `50%`:
  - If the entire extracted answer matches the pattern, it is normalized to `50` before comparison.

## Note on `*_no_answer` Metrics

The file previously emitted `pass@{k}_no_answer` and `majority@{k}_no_answer` (a strict “all first-k are None” indicator). These are currently commented out/disabled to keep logs/results cleaner; the recommended replacement signal is `no_answer_rate` / `no_answer_count`.

