from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import datasets


DEFAULT_REPO_ID = "zai-org/LongBench-v2"
DEFAULT_SPLIT = "train"
DEFAULT_PROMPT_HEADER = "Please read the following text and answer the question below."
DEFAULT_BOXED_ANSWER_INSTRUCTION = (
    r"On a new line, output the best choice as \boxed{<LETTER>} where "
    r"<LETTER> is one of A, B, C or D."
)
DEFAULT_COT_PROMPT_SUFFIX = (
    "Let's think step by step. " + DEFAULT_BOXED_ANSWER_INSTRUCTION
)
DEFAULT_ANSWER_PREFIX = (
    DEFAULT_BOXED_ANSWER_INSTRUCTION
)
DEFAULT_MAX_GEN_TOKS = 512
DEFAULT_COT_MAX_GEN_TOKS = 512
CHOICE_LABELS = ("A", "B", "C", "D")
DEFAULT_PROMPT_STYLE = "default"
COT_PROMPT_STYLE = "cot"
PROMPT_STYLE_ALIASES = {
    "default": DEFAULT_PROMPT_STYLE,
    "base": DEFAULT_PROMPT_STYLE,
    "standard": DEFAULT_PROMPT_STYLE,
    "cot": COT_PROMPT_STYLE,
    "paper_cot": COT_PROMPT_STYLE,
    "longbench2_cot": COT_PROMPT_STYLE,
}
TOKEN_COUNT_FIELD = "qwen_4b_it_tokens"
CONTEXT_TOKEN_COUNT_FIELD = "qwen_4b_it_context_tokens"
TOKEN_LENGTH_BUCKET_FIELD = "qwen_4b_it_length_bucket"
DEFAULT_TOKEN_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parent / "qwen3_4b_it_longbench_v2_tokens.jsonl"
)
DATASET_LOAD_KWARGS = {
    "cache_dir",
    "data_dir",
    "data_files",
    "download_config",
    "download_mode",
    "features",
    "keep_in_memory",
    "num_proc",
    "revision",
    "storage_options",
    "streaming",
    "token",
    "trust_remote_code",
    "verification_mode",
}
TOKEN_LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0_16k", 0, 16384),
    ("16k_32k", 16384, 32768),
    ("32k_64k", 32768, 65536),
    ("64k_128k", 65536, 131072),
    ("128k_512k", 131072, 524288),
    ("512k_1024k", 524288, 1048576),
)
TOKEN_LENGTH_BUCKET_ALIASES = {
    "0_16k": "0_16k",
    "0-16k": "0_16k",
    "0k_16k": "0_16k",
    "0k-16k": "0_16k",
    "0_16384": "0_16k",
    "0-16384": "0_16k",
    "16k_32k": "16k_32k",
    "16384_32768": "16k_32k",
    "16384-32768": "16k_32k",
    "32k_64k": "32k_64k",
    "32768_65536": "32k_64k",
    "32768-65536": "32k_64k",
    "64k_128k": "64k_128k",
    "65536_131072": "64k_128k",
    "65536-131072": "64k_128k",
    "128k_512k": "128k_512k",
    "131072_524288": "128k_512k",
    "131072-524288": "128k_512k",
    "512k_1024k": "512k_1024k",
    "524288_1048576": "512k_1024k",
    "524288-1048576": "512k_1024k",
}

_EXPLICIT_ANSWER_PATTERNS = (
    re.compile(r"the correct answer is\s*\(?\s*([ABCD])\s*\)?", re.IGNORECASE),
    re.compile(r"\banswer\s*(?:is|:)\s*\(?\s*([ABCD])\s*\)?", re.IGNORECASE),
)
_STANDALONE_ANSWER_PATTERN = re.compile(r"^\s*\(?\s*([ABCD])\s*\)?[\s\.\)]*$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_BOXED_COMMAND_RE = re.compile(r"\\(?:boxed|fbox)\s*")
_WRAPPER_PATTERNS = (
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
)


def slugify_sub_domain(sub_domain: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", sub_domain.strip().lower())
    return slug.strip("_")


def normalize_token_length_bucket(bucket: str) -> str:
    normalized = bucket.strip().lower().replace(" ", "")
    canonical = TOKEN_LENGTH_BUCKET_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unsupported token length bucket {bucket!r}. Expected one of "
            f"{', '.join(label for label, _, _ in TOKEN_LENGTH_BUCKETS)}."
        )
    return canonical


def normalize_prompt_style(prompt_style: str | None) -> str:
    if prompt_style is None:
        return DEFAULT_PROMPT_STYLE

    normalized = prompt_style.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = PROMPT_STYLE_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unsupported LongBench v2 prompt style {prompt_style!r}. "
            f"Expected one of {', '.join(sorted(PROMPT_STYLE_ALIASES))}."
        )
    return canonical


def get_token_annotations_path(prompt_style: str | None = None) -> Path:
    # Bucketing is intentionally prompt-agnostic so different prompt strategies
    # evaluate the exact same document partitions.
    del prompt_style
    return DEFAULT_TOKEN_ANNOTATIONS_PATH


def bucket_for_token_count(token_count: int) -> str | None:
    for label, lower, upper in TOKEN_LENGTH_BUCKETS:
        if lower <= token_count < upper:
            return label
    if token_count == TOKEN_LENGTH_BUCKETS[-1][2]:
        return TOKEN_LENGTH_BUCKETS[-1][0]
    if token_count > TOKEN_LENGTH_BUCKETS[-1][2]:
        return TOKEN_LENGTH_BUCKETS[-1][0]
    return None


@cache
def load_token_annotations(
    annotations_path: str = str(DEFAULT_TOKEN_ANNOTATIONS_PATH),
) -> dict[str, dict[str, Any]]:
    path = Path(annotations_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing LongBench v2 token annotations at {path}. "
            "Generate them with qwen3_4b_it_tokenize_longbench_v2.py first."
        )

    annotations: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = str(record["_id"])
            annotations[doc_id] = record
    return annotations


def _augment_and_filter_dataset(
    dataset: datasets.Dataset,
    *,
    sub_domain: str | None = None,
    token_length_bucket: str | None = None,
    annotations_path: str | None = None,
    prompt_style: str | None = None,
) -> datasets.Dataset:
    filter_bucket = (
        normalize_token_length_bucket(token_length_bucket)
        if token_length_bucket is not None
        else None
    )
    should_use_annotations = annotations_path is not None or filter_bucket is not None
    annotations: dict[str, dict[str, Any]] = {}
    if should_use_annotations:
        resolved_annotations_path = (
            Path(annotations_path).resolve()
            if annotations_path is not None
            else get_token_annotations_path(prompt_style).resolve()
        )
        annotations = load_token_annotations(
            str(resolved_annotations_path)
        )

    rows: list[dict[str, Any]] = []
    missing_annotation_ids: list[str] = []
    for doc in dataset:
        row = dict(doc)
        if sub_domain is not None and str(row.get("sub_domain")) != sub_domain:
            continue

        annotation = annotations.get(str(row.get("_id"))) if should_use_annotations else None
        if annotation is not None:
            row[TOKEN_COUNT_FIELD] = int(annotation[TOKEN_COUNT_FIELD])
            row[CONTEXT_TOKEN_COUNT_FIELD] = int(annotation[CONTEXT_TOKEN_COUNT_FIELD])
            row[TOKEN_LENGTH_BUCKET_FIELD] = normalize_token_length_bucket(
                str(annotation[TOKEN_LENGTH_BUCKET_FIELD])
            )
        elif should_use_annotations and filter_bucket is not None:
            missing_annotation_ids.append(str(row.get("_id")))
            continue

        if filter_bucket is not None and row.get(TOKEN_LENGTH_BUCKET_FIELD) != filter_bucket:
            continue

        rows.append(row)

    if missing_annotation_ids and filter_bucket is not None:
        preview = ", ".join(missing_annotation_ids[:5])
        raise KeyError(
            "Missing token annotations for LongBench v2 docs while filtering by bucket: "
            f"{preview}"
        )

    return datasets.Dataset.from_list(rows)


def load_longbench2_dataset(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    split: str = DEFAULT_SPLIT,
    dataset_name: str | None = None,
    sub_domain: str | None = None,
    token_length_bucket: str | None = None,
    annotations_path: str | None = None,
    prompt_style: str | None = None,
    **dataset_kwargs: Any,
) -> datasets.DatasetDict:
    load_kwargs = {
        key: value for key, value in dict(dataset_kwargs).items() if key in DATASET_LOAD_KWARGS
    }
    if dataset_name is not None:
        load_kwargs["name"] = dataset_name

    dataset = datasets.load_dataset(path=repo_id, split=split, **load_kwargs)
    if isinstance(dataset, datasets.DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found in dataset {repo_id!r}.")
        dataset = dataset[split]

    dataset = _augment_and_filter_dataset(
        dataset,
        sub_domain=sub_domain,
        token_length_bucket=token_length_bucket,
        annotations_path=annotations_path,
        prompt_style=prompt_style,
    )

    return datasets.DatasetDict({split: dataset})


def normalize_answer_label(answer: Any) -> str:
    if answer is None:
        return ""

    text = str(answer).strip().upper()
    match = _STANDALONE_ANSWER_PATTERN.match(text)
    if match is not None:
        return match.group(1)

    for pattern in _EXPLICIT_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()

    word_match = re.search(r"\b([ABCD])\b", text)
    if word_match is not None:
        return word_match.group(1)

    return text


def strip_reasoning_blocks(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _choice_text(doc: dict[str, Any], label: str) -> str:
    return str(doc.get(f"choice_{label}", "")).strip()


def build_prompt(doc: dict[str, Any]) -> str:
    return build_prompt_for_style(doc, DEFAULT_PROMPT_STYLE)


def build_prompt_for_style(
    doc: dict[str, Any],
    prompt_style: str | None = None,
) -> str:
    normalized_prompt_style = normalize_prompt_style(prompt_style)
    if normalized_prompt_style == COT_PROMPT_STYLE:
        return build_cot_prompt(doc)

    return build_default_prompt(doc)


def build_default_prompt(doc: dict[str, Any]) -> str:
    task_description = str(doc.get("task_description", "")).strip()
    header = task_description or DEFAULT_PROMPT_HEADER
    context = str(doc.get("context", ""))
    question = str(doc.get("question", "")).strip()
    answer_prefix = str(doc.get("answer_prefix") or DEFAULT_ANSWER_PREFIX).strip()
    choice_block = "\n".join(
        f"({label}) {_choice_text(doc, label)}" for label in CHOICE_LABELS
    )

    return (
        f"{header}\n\n"
        f"<text>\n{context}\n</text>\n\n"
        f"What is the correct answer to this question: {question}\n"
        f"Choices:\n{choice_block}\n\n"
        f"{answer_prefix}"
    )


def build_cot_prompt(doc: dict[str, Any]) -> str:
    context = str(doc.get("context", ""))
    question = str(doc.get("question", "")).strip()
    choice_block = "\n".join(
        f"({label}) {_choice_text(doc, label)}" for label in CHOICE_LABELS
    )

    return (
        f"{DEFAULT_PROMPT_HEADER}\n"
        f"<text>\n{context}\n</text>\n"
        f"What is the correct answer to this question: {question}\n"
        f"Choices:\n{choice_block}\n"
        f"{DEFAULT_COT_PROMPT_SUFFIX}"
    )


def doc_to_text(doc: dict[str, Any]) -> str:
    return build_prompt(doc)


def doc_to_text_cot(doc: dict[str, Any]) -> str:
    return build_prompt_for_style(doc, COT_PROMPT_STYLE)


def doc_to_target(doc: dict[str, Any]) -> str:
    return rf"\boxed{{{normalize_answer_label(doc.get('answer'))}}}"


def extract_response_text(results: list[Any]) -> str:
    if not results:
        return ""

    response: Any = results[0]
    while isinstance(response, list):
        if not response:
            return ""
        response = response[0]

    if response is None:
        return ""
    return response if isinstance(response, str) else str(response)


def _find_last_boxed_span(text: str) -> tuple[int, int] | None:
    if not text:
        return None

    matches = list(_BOXED_COMMAND_RE.finditer(text))
    if not matches:
        return None

    start_idx = matches[-1].start()
    i = matches[-1].end()
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None

    depth = 0
    j = i
    while j < len(text):
        char = text[j]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return (start_idx, j + 1)
        j += 1
    return None


def extract_last_boxed_content(text: str) -> str | None:
    span = _find_last_boxed_span(text)
    if span is None:
        return None

    start, end = span
    block = text[start:end]
    brace_idx = block.find("{")
    if brace_idx < 0 or not block.endswith("}"):
        return None
    return block[brace_idx + 1 : -1]


def _strip_tex_wrappers(text: str) -> str:
    current = text.strip()
    previous = None
    for _ in range(10):
        if current == previous:
            break
        previous = current
        for pattern in _WRAPPER_PATTERNS:
            match = re.fullmatch(pattern, current, flags=re.DOTALL)
            if match is not None:
                current = match.group("inner").strip()
                break
    return current


def extract_boxed_answer_label(response: str) -> str | None:
    payload = extract_last_boxed_content(response)
    if payload is None:
        return None

    normalized = _strip_tex_wrappers(payload)
    letters = [
        letter.upper()
        for letter in re.findall(r"[A-Za-z]", normalized)
        if letter.upper() in CHOICE_LABELS
    ]
    if letters:
        return letters[-1]

    digit_match = re.search(r"\b([1-4])\b", normalized)
    if digit_match is not None:
        return CHOICE_LABELS[int(digit_match.group(1)) - 1]
    return None


def extract_answer_label(response: str) -> str | None:
    stripped = strip_reasoning_blocks(response)
    if not stripped:
        return None

    boxed = extract_boxed_answer_label(stripped)
    if boxed is not None:
        return boxed

    for pattern in _EXPLICIT_ANSWER_PATTERNS:
        matches = pattern.findall(stripped)
        if matches:
            return matches[-1].upper()

    match = _STANDALONE_ANSWER_PATTERN.match(stripped.upper())
    if match is not None:
        return match.group(1)

    bracketed = re.findall(r"\(([ABCD])\)", stripped, flags=re.IGNORECASE)
    if bracketed:
        return bracketed[-1].upper()

    if len(stripped) <= 8:
        loose = re.search(r"\b([ABCD])\b", stripped, flags=re.IGNORECASE)
        if loose is not None:
            return loose.group(1).upper()

    return None


def process_results(doc: dict[str, Any], results: list[Any]) -> dict[str, float]:
    gold = normalize_answer_label(doc.get("answer"))
    predicted = extract_answer_label(extract_response_text(results))
    return {"acc": float(predicted == gold and gold != "")}
