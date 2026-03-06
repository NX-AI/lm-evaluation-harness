from __future__ import annotations

import atexit
import gzip
import logging
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets


@dataclass(frozen=True)
class PromptConstants:
    """Prompt strings from LiveCodeBench's official runner.

    Source: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    """

    CODE_GENERATION_PROMPT: str = (
        "You are given a competitive programming problem. Solve it and output a Python program that reads from stdin and writes to stdout.\n\n"
        "{question_content}\n\n"
        "{formatting_message}\n\n"
        "{starter_code}"
    )
    FORMATTING_MESSAGE_WITH_STARTER_CODE: str = (
        "Use the provided starter code and write your solution in the `solve` function.\n"
        "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs).\n"
        "Enclose your code within delimiters as follows.\n"
        "```python\n"
        "# YOUR CODE HERE\n"
        "```\n"
        "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
    )
    FORMATTING_WITHOUT_STARTER_CODE: str = (
        "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs).\n"
        "Enclose your code within delimiters as follows.\n"
        "```python\n"
        "# YOUR CODE HERE\n"
        "```\n"
        "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
    )


LOG = logging.getLogger(__name__)

_RAW_LOGGERS: dict[tuple[str, str, int, int, int], "_JsonlGzipLogger"] = {}


def _parse_datetime(date_str: str) -> datetime:
    # Example: "2024-08-01T00:00:00"
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")


def _clean_doc(doc: dict[str, Any]) -> dict[str, Any]:
    starter_code = (doc.get("starter_code") or "").strip("\n")
    if starter_code:
        doc["formatting_message"] = PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE
        doc["starter_code"] = f"```python\n{starter_code}\n```"
    else:
        doc["formatting_message"] = PromptConstants.FORMATTING_WITHOUT_STARTER_CODE
        doc["starter_code"] = "```python\n# YOUR CODE HERE\n```"
    doc["task_id"] = doc.get("question_id", doc.get("task_id"))
    return doc


def load_livecodebench_codegen_dataset(
    *,
    # NeMo-aligned defaults for LCB v6.
    subset: str = "code_generation_lite",
    release_version: str = "v6",
    revision: str = "refs/pr/7",
    split: str = "test",
    start_date: str | None = "2024-08",
    end_date: str | None = "2025-05",
    keep_all_columns: bool = False,
    metric_ks: list[int] | tuple[int, ...] | None = None,
    # If provided, load a local JSON/JSONL file instead of HF datasets.
    data_files: str | dict[str, str] | None = None,
    **_: Any,
) -> datasets.DatasetDict:
    """Load LiveCodeBench code generation split, matching NeMo's prepare.py as closely as possible.

    If `data_files` is provided, loads via `datasets.load_dataset("json", data_files=...)`.
    """
    if metric_ks is not None:
        metric_ks_list = [int(k) for k in metric_ks]
    else:
        metric_ks_list = None

    if data_files is not None:
        ds = datasets.load_dataset("json", data_files=data_files)
        # If the user passes a single json/jsonl, HF datasets calls the split "train".
        if isinstance(ds, datasets.DatasetDict) and "test" not in ds and "train" in ds:
            ds = datasets.DatasetDict({"test": ds["train"]})
        if metric_ks_list is not None:
            ds = ds.map(lambda _: {"metric_ks": metric_ks_list})
        return ds

    release_version = _normalize_release_version_bare(release_version)
    ds = datasets.load_dataset(
        f"livecodebench/{subset}",
        name=f"release_{release_version}",
        revision=revision,
    )
    dataset = ds[split]

    # Mirror NeMo's prepare.py column drops. We pass these via `remove_columns` to
    # avoid rewriting very large string columns (public/private test cases) and
    # hitting pyarrow "offset overflow" limits during `.map(...)`.
    drop_cols = [
        "question_title",
        "contest_id",
        "metadata",
        "platform",
        "question_id",
        "public_test_cases",
        "private_test_cases",
    ]
    remove_cols = [c for c in drop_cols if c in dataset.column_names] if not keep_all_columns else None

    map_kwargs: dict[str, Any] = {}
    if keep_all_columns:
        # When keeping large string columns, write in small chunks to avoid
        # pyarrow string offset overflows.
        map_kwargs["writer_batch_size"] = 1

    dataset = dataset.map(_clean_doc, remove_columns=remove_cols, **map_kwargs)

    if start_date is not None and end_date is not None:
        start_ym = tuple(int(x) for x in start_date.split("-", 1))
        end_ym = tuple(int(x) for x in end_date.split("-", 1))

        def _in_range(x: dict[str, Any]) -> bool:
            contest_dt = _parse_datetime(x["contest_date"])
            contest_ym = (contest_dt.year, contest_dt.month)
            return start_ym <= contest_ym <= end_ym

        dataset = dataset.filter(_in_range)

    def _add_meta(x: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "subset_for_metrics": x.get("difficulty"),
            "release_version": release_version,
        }
        if metric_ks_list is not None:
            out["metric_ks"] = metric_ks_list
        return out

    dataset = dataset.map(_add_meta, **map_kwargs)

    return datasets.DatasetDict({"test": dataset})


def doc_to_text(doc: dict[str, Any]) -> str:
    return PromptConstants.CODE_GENERATION_PROMPT.format(
        question_content=doc["question_content"],
        formatting_message=doc["formatting_message"],
        starter_code=doc["starter_code"],
    )


def _normalize_release_version_bare(release_version: str) -> str:
    """Normalize release versions to the bare form used by NeMo ("v6", "v5", ...).

    Accepts both "v6" and "release_v6" inputs and always returns "v6".
    """
    rv = (release_version or "").strip()
    if not rv:
        return "v6"
    for prefix in ("release_", "release-"):
        if rv.startswith(prefix):
            rv = rv[len(prefix) :]
            break
    return rv


def _normalize_language(language: str) -> str:
    lang = (language or "").strip().lower()
    if not lang:
        return "python"
    if lang in {"py", "python3"}:
        return "python"
    return lang


def _env_flag(name: str, *, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _distributed_rank() -> int:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except ValueError:
            continue
    return 0


def _should_save_full_generations() -> bool:
    return _env_flag("LMEVAL_LCB_SAVE_FULL_GENERATIONS", default="0")


def _should_merge_full_generations() -> bool:
    return _env_flag("LMEVAL_LCB_MERGE_FULL_GENERATIONS", default="1")


def _full_generations_max_chars() -> int | None:
    raw = os.environ.get("LMEVAL_LCB_FULL_GENERATIONS_MAX_CHARS")
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return max(0, val)


def _truncate_for_log(text: str) -> str:
    limit = _full_generations_max_chars()
    if limit is None:
        return text
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit]


@dataclass
class _JsonlGzipLogger:
    path: Path
    fp: Any
    lines_written: int = 0
    flush_every: int = 32

    def write(self, record: dict[str, Any]) -> None:
        self.fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.lines_written += 1
        if self.flush_every > 0 and (self.lines_written % self.flush_every) == 0:
            self.fp.flush()

    def close(self) -> None:
        try:
            self.fp.flush()
        except Exception:
            pass
        try:
            self.fp.close()
        except Exception:
            pass


def _raw_logger(
    *,
    release_version: str,
    language: str,
    n: int,
) -> _JsonlGzipLogger:
    rank = _distributed_rank()
    pid = os.getpid()
    key = (release_version, language, int(n), int(rank), int(pid))
    existing = _RAW_LOGGERS.get(key)
    if existing is not None:
        return existing

    out_dir = _lcb_artifact_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"livecodebench_full_generations_{release_version}_{language}_n{n}_rank{rank}_pid{pid}.jsonl.gz"
    fp = gzip.open(path, mode="at", encoding="utf-8")
    logger = _JsonlGzipLogger(path=path, fp=fp)
    _RAW_LOGGERS[key] = logger
    atexit.register(logger.close)
    return logger


def _log_full_generations(
    doc: dict[str, Any],
    *,
    raw_completions: list[str],
    extracted_codes: list[str],
    release_version: str,
    language: str,
    expected_n: int,
) -> None:
    if not _should_save_full_generations():
        return

    # Normalize to the task's intended repeat count so the log can be consumed easily.
    raw_list = list(raw_completions)
    code_list = list(extracted_codes)
    if len(raw_list) > expected_n:
        raw_list = raw_list[:expected_n]
        code_list = code_list[:expected_n]
    elif len(raw_list) < expected_n:
        raw_list = raw_list + [""] * (expected_n - len(raw_list))
        code_list = code_list + [""] * (expected_n - len(code_list))

    logger = _raw_logger(release_version=release_version, language=language, n=expected_n)
    task_id = str(doc.get("task_id") or doc.get("question_id") or "")
    subset_for_metrics = doc.get("subset_for_metrics")

    rank = _distributed_rank()
    pid = os.getpid()
    for idx, (raw, code) in enumerate(zip(raw_list, code_list, strict=True)):
        logger.write(
            {
                "task_id": task_id,
                "sample_idx": idx,
                "release_version": release_version,
                "language": language,
                "subset_for_metrics": subset_for_metrics,
                "rank": rank,
                "pid": pid,
                "raw_completion": _truncate_for_log(str(raw)),
                "extracted_code": _truncate_for_log(str(code)),
            }
        )


def _strip_reasoning(text: str) -> str:
    # NeMo's `preprocess_code` removes reasoning traces that are wrapped in tags.
    # Do this conservatively: only strip if we see a *start* tag, and require the end tag.
    for start_tag, end_tag in (
        ("<think>", "</think>"),
        ("<analysis>", "</analysis>"),
    ):
        if start_tag in text:
            if end_tag not in text:
                return ""
            _, _, text = text.partition(end_tag)
    return text


_FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)[ \t]*\n(?P<code>[\s\S]*?)```",
    re.MULTILINE,
)


def _extract_last_fenced_code_block(text: str, *, language: str) -> str | None:
    matches = list(_FENCED_BLOCK_RE.finditer(text))
    if not matches:
        return None

    preferred_langs = {language, ""}
    if language == "python":
        preferred_langs |= {"py", "python3"}

    for match in reversed(matches):
        lang = (match.group("lang") or "").strip().lower()
        if lang in preferred_langs:
            return (match.group("code") or "").strip()

    # Fall back to the last fenced block regardless of language.
    return (matches[-1].group("code") or "").strip()


def preprocess_code(completion: str, *, language: str = "python") -> str:
    """NeMo-aligned generation post-processing for LiveCodeBench.

    - Drops <think>/<analysis> traces if present.
    - Extracts the last fenced code block (```python ... ```) if present.
    - If a fence is opened but not closed, returns "" (invalid).
    """
    completion = completion.replace("\r", "")
    completion = _strip_reasoning(completion)
    if not completion:
        return ""

    language = _normalize_language(language)

    if "```" in completion:
        extracted = _extract_last_fenced_code_block(completion, language=language)
        if extracted is None:
            # Strict: fence opened but not closed (or malformed); treat as invalid.
            return ""
        return extracted

    return completion.strip()


def extract_code(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Filter: preprocess each generation into an executable code string."""
    out: list[list[str]] = []
    for resp, doc in zip(resps, docs, strict=True):
        language = _normalize_language(str(doc.get("language") or "python"))
        release_version = _normalize_release_version_bare(str(doc.get("release_version") or "v6"))

        metric_ks = sorted({k for k in _metric_ks_from_doc(doc) if k > 0})
        expected_n = max(metric_ks) if metric_ks else len(resp)
        if expected_n <= 0:
            expected_n = max(1, len(resp))

        raw_list = list(resp)
        extracted = [preprocess_code(r, language=language) for r in raw_list]
        _log_full_generations(
            doc,
            raw_completions=raw_list,
            extracted_codes=extracted,
            release_version=release_version,
            language=language,
            expected_n=int(expected_n),
        )
        out.append(extracted)
    return out


def _metric_ks_from_doc(doc: dict[str, Any]) -> list[int]:
    metric_ks = doc.get("metric_ks")
    if metric_ks is None:
        return [1]
    if isinstance(metric_ks, (int, float)):
        return [int(metric_ks)]
    if isinstance(metric_ks, str):
        parts = [p.strip() for p in metric_ks.split(",")]
        ks = []
        for part in parts:
            if not part:
                continue
            try:
                ks.append(int(part))
            except ValueError:
                continue
        return ks or [1]
    if isinstance(metric_ks, (list, tuple)):
        out = []
        for k in metric_ks:
            try:
                out.append(int(k))
            except (TypeError, ValueError):
                continue
        return out or [1]
    return [1]


def process_results(doc: dict[str, Any], results: list[list[str]]) -> dict[str, Any]:
    # generate_until -> single request, potentially repeated generations
    metric_ks = sorted({k for k in _metric_ks_from_doc(doc) if k > 0})
    if not metric_ks:
        metric_ks = [1]

    code_list = [c for group in results for c in group]
    expected_n = max(metric_ks) if metric_ks else 1
    if expected_n <= 0:
        expected_n = 1
    if len(code_list) != expected_n:
        task_id = str(doc.get("task_id") or "")
        if len(code_list) > expected_n:
            LOG.warning(
                "livecodebench: task_id=%s produced %d generations, expected %d; truncating extras.",
                task_id,
                len(code_list),
                expected_n,
            )
            code_list = code_list[:expected_n]
        else:
            LOG.warning(
                "livecodebench: task_id=%s produced %d generations, expected %d; padding missing with empty strings.",
                task_id,
                len(code_list),
                expected_n,
            )
            code_list = code_list + [""] * (expected_n - len(code_list))

    n = len(code_list)
    no_answer_count = sum(1 for c in code_list if not str(c).strip())
    payload = {
        "task_id": str(doc["task_id"]),
        "release_version": _normalize_release_version_bare(str(doc.get("release_version") or "v6")),
        "subset_for_metrics": doc.get("subset_for_metrics"),
        "language": _normalize_language(str(doc.get("language") or "python")),
        "code_list": list(code_list),
        "metric_ks": metric_ks,
    }
    out: dict[str, Any] = {
        "no_answer_rate": float(no_answer_count / n) if n else 1.0,
        "no_answer_count": int(no_answer_count),
        "num_generations": int(n),
    }

    for k in metric_ks:
        out[f"pass@{k}"] = payload
        out[f"majority@{k}"] = payload
    return out


def _sandbox_base_url() -> str:
    host = os.environ.get("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1")
    port = os.environ.get("NEMO_SKILLS_SANDBOX_PORT", "6000")
    return f"http://{host}:{port}"


def _http_get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _is_sandbox_healthy(*, timeout_s: float = 2.0) -> bool:
    try:
        data = _http_get_json(f"{_sandbox_base_url()}/health", timeout_s=timeout_s)
        return data.get("status") == "healthy"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False


def _sandbox_execute_shell(
    command: str,
    *,
    timeout_s: float,
    max_output_characters: int = 100_000,
) -> dict[str, Any]:
    return _http_post_json(
        f"{_sandbox_base_url()}/execute",
        payload={
            "generated_code": command,
            "language": "shell",
            "timeout": timeout_s,
            "max_output_characters": max_output_characters,
        },
        timeout_s=timeout_s + 5.0,
    )


def _sandbox_execute_shell_with_retries(
    command: str,
    *,
    timeout_s: float,
    max_output_characters: int = 100_000,
    retries: int = 3,
    initial_backoff_s: float = 1.0,
) -> dict[str, Any]:
    last_err: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return _sandbox_execute_shell(
                command,
                timeout_s=timeout_s,
                max_output_characters=max_output_characters,
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(initial_backoff_s * (2**attempt))
    assert last_err is not None
    raise RuntimeError(f"Sandbox request failed after {retries + 1} attempts: {last_err}") from last_err


def _maybe_hydra_output_dir() -> str | None:
    """Best-effort Hydra output dir discovery.

    lm-eval itself is not Hydra-based, but many wrappers (including this repo)
    use Hydra and store run artifacts under `hydra.runtime.output_dir`.
    """
    try:
        from hydra.core.hydra_config import HydraConfig  # type: ignore
    except Exception:
        return None
    try:
        if HydraConfig.initialized():
            return str(HydraConfig.get().runtime.output_dir)
    except Exception:
        return None
    return None


def _lcb_artifact_dir() -> Path:
    override = os.environ.get("LMEVAL_LCB_ARTIFACT_DIR") or os.environ.get("LMEVAL_LCB_SAVE_DIR")
    if override:
        return Path(override)

    hydra_dir = _maybe_hydra_output_dir()
    if hydra_dir:
        return Path(hydra_dir)

    return Path(os.getcwd())


def _ensure_livecodebench_installed(*, allow_auto_install: bool) -> None:
    if importlib.util.find_spec("livecodebench") is not None:
        return
    if not allow_auto_install:
        raise ModuleNotFoundError(
            "Missing dependency `livecodebench`. Install it in the same environment that runs the sandbox server, "
            "e.g. `pip install git+https://github.com/wasiahmad/livecodebench.git`. "
            "Set `LMEVAL_LCB_AUTO_INSTALL=1` to auto-install (requires network access)."
        )

    pip_spec = os.environ.get("LMEVAL_LCB_PIP_SPEC", "git+https://github.com/wasiahmad/livecodebench.git")
    cmd = f"{shlex.quote(os.environ.get('LMEVAL_LCB_INTERPRETER', sys.executable))} -m pip install {shlex.quote(pip_spec)}"

    if _is_sandbox_healthy():
        out = _sandbox_execute_shell_with_retries(cmd, timeout_s=60 * 20)
        if out.get("process_status") != "completed":
            raise RuntimeError(f"Failed to install livecodebench in sandbox: {out}")
    else:
        raise ModuleNotFoundError(
            "livecodebench is not installed and sandbox is not reachable for auto-install. "
            "Start the sandbox server or install livecodebench manually."
        )


_LCB_EVAL_CACHE: dict[tuple[Any, ...], dict[str, list[bool]]] = {}


def _eval_cache_key(
    items: list[dict[str, Any]],
    *,
    release_version: str,
    language: str,
    max_candidates: int,
    num_proc: int,
    timeout_per_sample: int,
    interpreter: str,
    use_sandbox: bool,
) -> tuple[Any, ...]:
    # Use object identity to prevent accidental reuse across *different* evaluator runs
    # in the same python process, while still allowing reuse across metrics in one run.
    first_payload_id = id(items[0]) if items else 0
    return (
        first_payload_id,
        len(items),
        release_version,
        language,
        max_candidates,
        num_proc,
        timeout_per_sample,
        interpreter,
        use_sandbox,
    )


def _run_lcb_eval(items: list[dict[str, Any]]) -> dict[str, list[bool]]:
    if not items:
        return {}

    allow_auto_install = os.environ.get("LMEVAL_LCB_AUTO_INSTALL", "0") == "1"
    _ensure_livecodebench_installed(allow_auto_install=allow_auto_install)

    use_sandbox = _is_sandbox_healthy()
    if not use_sandbox and os.environ.get("LMEVAL_LCB_ALLOW_LOCAL_EVAL", "0") != "1":
        raise RuntimeError(
            "LiveCodeBench evaluation requires a running sandbox server. "
            "Start it (see lm_eval/tasks/livecodebench/README.md) or set `LMEVAL_LCB_ALLOW_LOCAL_EVAL=1` (unsafe)."
        )

    # Validate release version consistency (NeMo convention).
    release_versions = {_normalize_release_version_bare(str(x.get("release_version", ""))) for x in items}
    if len(release_versions) != 1:
        raise ValueError(f"LiveCodeBench samples must share one release_version, got: {sorted(release_versions)}")
    (release_version,) = release_versions

    # Validate language consistency.
    languages = {_normalize_language(str(x.get("language") or "python")) for x in items}
    if len(languages) != 1:
        raise ValueError(f"LiveCodeBench samples must share one language, got: {sorted(languages)}")
    (language,) = languages

    code_lens = [len(x.get("code_list") or []) for x in items]
    if not code_lens:
        return {}
    min_candidates = min(code_lens)
    max_candidates = max(code_lens)
    if max_candidates <= 0:
        max_candidates = 1

    if min_candidates != max_candidates:
        # LiveCodeBench's official evaluator asserts that all samples in one file have the
        # same number of generations. Normalize to the most common length.
        counter = Counter([n for n in code_lens if n > 0])
        target_candidates = counter.most_common(1)[0][0] if counter else max_candidates
        LOG.warning(
            "livecodebench: inconsistent candidate counts across tasks (min=%d max=%d). Normalizing to %d.",
            min_candidates,
            max_candidates,
            target_candidates,
        )
        for payload in items:
            code_list = list(payload.get("code_list") or [])
            if len(code_list) > target_candidates:
                payload["code_list"] = code_list[:target_candidates]
            elif len(code_list) < target_candidates:
                payload["code_list"] = code_list + [""] * (target_candidates - len(code_list))
        max_candidates = target_candidates

    # Merge per-rank full generation logs (written during filtering) into one file.
    if _should_save_full_generations() and _should_merge_full_generations() and _distributed_rank() == 0:
        out_dir = _lcb_artifact_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = out_dir / f"livecodebench_full_generations_{release_version}_{language}_n{max_candidates}.jsonl.gz"
        manifest = out_dir / f"livecodebench_full_generations_{release_version}_{language}_n{max_candidates}.manifest.txt"
        if not merged.exists():
            parts = sorted(
                out_dir.glob(
                    f"livecodebench_full_generations_{release_version}_{language}_n{max_candidates}_rank*_pid*.jsonl.gz"
                )
            )
            if parts:
                manifest.write_text("\n".join(str(p) for p in parts) + "\n", encoding="utf-8")
                with merged.open("wb") as out_f:
                    for p in parts:
                        with p.open("rb") as in_f:
                            shutil.copyfileobj(in_f, out_f, length=1024 * 1024)
            else:
                LOG.warning(
                    "livecodebench: requested full generations merge, but no per-rank logs found for %s/%s n=%d in %s",
                    release_version,
                    language,
                    max_candidates,
                    out_dir,
                )

    num_proc = int(os.environ.get("LMEVAL_LCB_NUM_PROCESSES", "4"))
    timeout_per_sample = int(os.environ.get("LMEVAL_LCB_TIMEOUT", "6"))
    timeout_buffer = int(os.environ.get("LMEVAL_LCB_TIMEOUT_BUFFER", "60"))
    num_retries = int(os.environ.get("LMEVAL_LCB_NUM_RETRIES", "3"))

    interpreter = os.environ.get("LMEVAL_LCB_INTERPRETER", sys.executable)

    cache_key = _eval_cache_key(
        items,
        release_version=release_version,
        language=language,
        max_candidates=max_candidates,
        num_proc=num_proc,
        timeout_per_sample=timeout_per_sample,
        interpreter=str(interpreter),
        use_sandbox=use_sandbox,
    )
    cached = _LCB_EVAL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # LiveCodeBench evaluator expects python release names like "release_v6".
    lcb_release_version = f"release_{release_version}" if language == "python" else release_version

    tmp_root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or None
    save_files = os.environ.get("LMEVAL_LCB_SAVE_FILES", "0") == "1"
    sandbox_eval_out: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="lcb_", dir=tmp_root) as td:
        td_path = Path(td)
        jsonl_path = td_path / "livecodebench_predictions.jsonl"
        tag = f"{release_version}_{language}_n{max_candidates}_{td_path.name}_pid{os.getpid()}"

        samples: list[dict[str, Any]] = []
        for payload in items:
            task_id = str(payload["task_id"])
            samples.append(
                {
                    "task_id": task_id,
                    # LiveCodeBench evaluator expects this name.
                    "question_id": task_id,
                    "code_list": list(payload.get("code_list") or []),
                    "subset_for_metrics": payload.get("subset_for_metrics"),
                    "release_version": release_version,
                }
            )

        with jsonl_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        if save_files:
            out_dir = _lcb_artifact_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jsonl_path, out_dir / f"livecodebench_predictions_{tag}.jsonl")

        # Ensure the evaluator grades *all* candidates so we can compute pass@k estimators
        # using the full sample count n, matching the AIME-style formulation.
        k_list = [1] if max_candidates == 1 else [1, max_candidates]
        eval_code = (
            "from livecodebench.evaluate import evaluate\n"
            "evaluate(\n"
            f"  custom_output_file={json.dumps(str(jsonl_path))},\n"
            f"  release_version={json.dumps(lcb_release_version)},\n"
            "  test_file=None,\n"
            f"  k_list={json.dumps(k_list)},\n"
            f"  language={json.dumps(language)},\n"
            f"  num_process_evaluate={num_proc},\n"
            f"  timeout={timeout_per_sample},\n"
            ")\n"
        )
        cmd = f"{shlex.quote(str(interpreter))} -c {shlex.quote(eval_code)}"

        if save_files:
            out_dir = _lcb_artifact_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            run_meta = {
                "release_version": release_version,
                "lcb_release_version": lcb_release_version,
                "language": language,
                "max_candidates": max_candidates,
                "num_process_evaluate": num_proc,
                "timeout_per_sample_s": timeout_per_sample,
                "timeout_buffer_s": timeout_buffer,
                "use_sandbox": use_sandbox,
                "sandbox_base_url": _sandbox_base_url() if use_sandbox else None,
                "command": cmd,
            }
            (out_dir / f"livecodebench_eval_meta_{tag}.json").write_text(
                json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if use_sandbox:
            total_candidates = sum(len(s.get("code_list") or []) for s in samples)
            timeout_total = int(math.ceil(timeout_per_sample * (total_candidates / max(1, num_proc)) + timeout_buffer))
            sandbox_eval_out = _sandbox_execute_shell_with_retries(cmd, timeout_s=float(timeout_total), retries=num_retries)
            if save_files and sandbox_eval_out is not None:
                out_dir = _lcb_artifact_dir()
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"livecodebench_eval_sandbox_response_{tag}.json").write_text(
                    json.dumps(sandbox_eval_out, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                (out_dir / f"livecodebench_eval_stdout_{tag}.txt").write_text(
                    str(sandbox_eval_out.get("stdout") or ""),
                    encoding="utf-8",
                )
                (out_dir / f"livecodebench_eval_stderr_{tag}.txt").write_text(
                    str(sandbox_eval_out.get("stderr") or ""),
                    encoding="utf-8",
                )
            if sandbox_eval_out.get("process_status") != "completed":
                raise RuntimeError(f"LiveCodeBench evaluator failed in sandbox: {sandbox_eval_out}")
            rc = sandbox_eval_out.get("return_code")
            if rc != 0:
                raise RuntimeError(f"LiveCodeBench evaluator exited non-zero in sandbox (rc={rc}): {sandbox_eval_out}")
        else:
            # Explicitly unsafe; allow only when opted in.
            import subprocess

            subprocess.run(cmd, shell=True, check=True)  # noqa: S602

        results_path = jsonl_path.with_name(jsonl_path.stem + "_eval_results.json")
        if not results_path.exists():
            if sandbox_eval_out is not None:
                raise RuntimeError(
                    f"LiveCodeBench did not create results at {results_path}. "
                    f"Sandbox stdout/stderr (truncated) is in the sandbox response: {sandbox_eval_out}"
                )
            raise FileNotFoundError(f"Expected LiveCodeBench results at {results_path}, but it was not created.")

        eval_results = json.loads(results_path.read_text(encoding="utf-8"))
        graded = eval_results.get("eval", {}) or {}

        scores_by_task_id: dict[str, list[bool]] = {}
        for payload in items:
            task_id = str(payload["task_id"])
            graded_list = (graded.get(task_id) or {}).get("graded_list") or []
            scores_by_task_id[task_id] = [bool(x) for x in graded_list]

        if save_files:
            out_dir = _lcb_artifact_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results_path, out_dir / f"livecodebench_eval_results_{tag}.json")

            if sandbox_eval_out is not None:
                # Already persisted above; keep for backward compatibility if this
                # function is called without `save_files` set until after eval.
                pass

    _LCB_EVAL_CACHE[cache_key] = scores_by_task_id
    return scores_by_task_id


def _aligned_scores(payload: dict[str, Any], scores_by_task_id: dict[str, list[bool]]) -> list[bool]:
    task_id = str(payload["task_id"])
    code_list = list(payload.get("code_list") or [])
    n = len(code_list)
    scores = list(scores_by_task_id.get(task_id) or [])
    if len(scores) < n:
        scores.extend([False] * (n - len(scores)))
    return [bool(x) for x in scores[:n]]


def _pass_at_k(scores: list[bool], *, k: int) -> float:
    n = len(scores)
    if n == 0 or k <= 0:
        return 0.0
    k_eff = min(k, n)
    correct = int(sum(1 for s in scores if s))
    incorrect = n - correct
    if incorrect < k_eff:
        return 1.0
    return float(1.0 - (math.comb(incorrect, k_eff) / math.comb(n, k_eff)))


def _majority_at_k(code_list: list[str], scores: list[bool], *, k: int) -> float:
    if k <= 0:
        return 0.0
    k_eff = min(k, len(code_list), len(scores))
    valid = [
        (code, float(score))
        for code, score in zip(code_list[:k_eff], scores[:k_eff], strict=False)
        if str(code).strip()
    ]
    if not valid:
        return 0.0
    counter = Counter(valid)
    majority_count = counter.most_common(1)[0][1]
    tied_scores = [score for (_, score), cnt in counter.items() if cnt == majority_count]
    return float(sum(tied_scores) / len(tied_scores))


def _aggregate_pass_at_k(items: list[dict[str, Any]], *, k: int) -> float:
    if not items:
        return float("nan")
    scores_by_task_id = _run_lcb_eval(items)
    per_doc = [_pass_at_k(_aligned_scores(payload, scores_by_task_id), k=k) for payload in items]
    return float(sum(per_doc) / max(1, len(per_doc)))


def _aggregate_majority_at_k(items: list[dict[str, Any]], *, k: int) -> float:
    if not items:
        return float("nan")
    scores_by_task_id = _run_lcb_eval(items)
    per_doc: list[float] = []
    for payload in items:
        code_list = list(payload.get("code_list") or [])
        scores = _aligned_scores(payload, scores_by_task_id)
        per_doc.append(_majority_at_k(code_list, scores, k=k))
    return float(sum(per_doc) / max(1, len(per_doc)))


def aggregate_pass_at_1(items: list[dict[str, Any]]) -> float:
    return _aggregate_pass_at_k(items, k=1)


def aggregate_pass_at_2(items: list[dict[str, Any]]) -> float:
    return _aggregate_pass_at_k(items, k=2)


def aggregate_pass_at_4(items: list[dict[str, Any]]) -> float:
    return _aggregate_pass_at_k(items, k=4)


def aggregate_pass_at_8(items: list[dict[str, Any]]) -> float:
    return _aggregate_pass_at_k(items, k=8)


def aggregate_pass_at_16(items: list[dict[str, Any]]) -> float:
    return _aggregate_pass_at_k(items, k=16)


def aggregate_pass_at_32(items: list[dict[str, Any]]) -> float:
    return _aggregate_pass_at_k(items, k=32)


def aggregate_majority_at_1(items: list[dict[str, Any]]) -> float:
    return _aggregate_majority_at_k(items, k=1)


def aggregate_majority_at_2(items: list[dict[str, Any]]) -> float:
    return _aggregate_majority_at_k(items, k=2)


def aggregate_majority_at_4(items: list[dict[str, Any]]) -> float:
    return _aggregate_majority_at_k(items, k=4)


def aggregate_majority_at_8(items: list[dict[str, Any]]) -> float:
    return _aggregate_majority_at_k(items, k=8)


def aggregate_majority_at_16(items: list[dict[str, Any]]) -> float:
    return _aggregate_majority_at_k(items, k=16)


def aggregate_majority_at_32(items: list[dict[str, Any]]) -> float:
    return _aggregate_majority_at_k(items, k=32)
