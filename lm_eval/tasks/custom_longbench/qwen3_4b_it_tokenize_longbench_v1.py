from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


class _TaskYamlLoader(yaml.SafeLoader):
    pass


def _construct_function(loader: _TaskYamlLoader, node: yaml.Node) -> Any:
    # Treat `!function some.module.fn` as a plain scalar for this report script.
    return loader.construct_scalar(node)  # type: ignore[no-any-return]


_TaskYamlLoader.add_constructor("!function", _construct_function)


def _summarize(values: list[int]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def _render_doc_to_text(template: str, doc: dict[str, Any]) -> str:
    # LongBench doc_to_text templates are simple and only use {{context}} and {{question}}.
    text = template
    text = text.replace("{{context}}", str(doc.get("context", "")))
    text = text.replace("{{question}}", str(doc.get("question", "")))
    return text


def _iter_leaf_task_yamls(task_dir: Path) -> Iterable[Path]:
    for p in sorted(task_dir.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        if p.name.endswith("_cot.yaml"):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to the xlstm-distillation model YAML (to read `model.pretrained_model_name_or_path`).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write the JSONL + Markdown report into.",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not hit the network for tokenizer/model files (requires local cache).",
    )
    args = ap.parse_args()

    model_cfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    model_id = model_cfg["model"]["pretrained_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=bool(args.local_files_only)
    )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(
            "Expected a fast tokenizer with `backend_tokenizer` for efficient long-context token counting."
        )

    task_dir = Path(__file__).resolve().parent
    task_cfgs: list[dict[str, Any]] = []
    for yml in _iter_leaf_task_yamls(task_dir):
        cfg = yaml.load(yml.read_text(encoding="utf-8"), Loader=_TaskYamlLoader)
        if not isinstance(cfg, dict):
            continue
        if "dataset_path" not in cfg or "dataset_name" not in cfg or "doc_to_text" not in cfg:
            continue
        task_cfgs.append(
            {
                "task_file": yml.name,
                "dataset_path": cfg["dataset_path"],
                "dataset_name": cfg["dataset_name"],
                "split": cfg.get("test_split", "test"),
                "doc_to_text": cfg["doc_to_text"],
            }
        )

    # De-duplicate by (dataset_path, dataset_name, split) in case multiple YAMLs point at the same config.
    seen = set()
    deduped: list[dict[str, Any]] = []
    for c in task_cfgs:
        key = (c["dataset_path"], c["dataset_name"], c["split"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # LongBench v1 on HF ships as a dataset script + a `data.zip` blob. Newer
    # `datasets` versions can refuse dataset scripts, so we load JSONL files
    # from `data.zip` directly.
    zip_by_repo: dict[str, Path] = {}
    for c in deduped:
        repo_id = str(c["dataset_path"])
        if repo_id in zip_by_repo:
            continue
        repo_root = Path(snapshot_download(repo_id=repo_id, repo_type="dataset")).resolve()
        zip_path = repo_root / "data.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Expected {zip_path} for dataset repo {repo_id}")
        zip_by_repo[repo_id] = zip_path

    jsonl_path = out_dir / "qwen3_4b_it_longbench_v1_tokens.jsonl"
    by_dataset_prompt: dict[str, list[int]] = {}
    by_dataset_context: dict[str, list[int]] = {}
    overall_prompt: list[int] = []
    overall_context: list[int] = []

    with jsonl_path.open("w", encoding="utf-8") as f:
        for cfg in sorted(deduped, key=lambda x: str(x["dataset_name"])):
            repo_id = str(cfg["dataset_path"])
            zip_path = zip_by_repo[repo_id]
            member = f"data/{cfg['dataset_name']}.jsonl"
            prompt_counts: list[int] = []
            context_counts: list[int] = []

            template = str(cfg["doc_to_text"])
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(member) as fp:
                    for idx, raw in enumerate(fp):
                        line = raw.decode("utf-8").strip()
                        if not line:
                            continue
                        doc = json.loads(line)
                        prompt = _render_doc_to_text(template, doc)
                        context = str(doc.get("context", ""))
                        prompt_tok = int(len(backend.encode(prompt)))
                        context_tok = int(len(backend.encode(context)))

                        prompt_counts.append(prompt_tok)
                        context_counts.append(context_tok)
                        overall_prompt.append(prompt_tok)
                        overall_context.append(context_tok)

                        doc_id = (
                            doc.get("_id")
                            or doc.get("id")
                            or doc.get("qid")
                            or doc.get("question_id")
                            or None
                        )
                        f.write(
                            json.dumps(
                                {
                                    "dataset_name": cfg["dataset_name"],
                                    "task_file": cfg["task_file"],
                                    "id": doc_id,
                                    "idx": int(idx),
                                    "length": doc.get("length"),
                                    "qwen_4b_it_tokens": prompt_tok,
                                    "qwen_4b_it_context_tokens": context_tok,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            by_dataset_prompt[str(cfg["dataset_name"])] = prompt_counts
            by_dataset_context[str(cfg["dataset_name"])] = context_counts

    overall_prompt_summary = _summarize(overall_prompt)
    overall_context_summary = _summarize(overall_context)
    by_dataset_prompt_summary = {
        k: _summarize(v) for k, v in sorted(by_dataset_prompt.items())
    }
    by_dataset_context_summary = {
        k: _summarize(v) for k, v in sorted(by_dataset_context.items())
    }

    md_path = out_dir / "qwen3_4b_it_longbench_v1_token_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-4B-Instruct token lengths on LongBench v1\n\n")
        f.write(f"- Model config: `{args.model_config}`\n")
        f.write(f"- Tokenizer: `{model_id}`\n")
        f.write(
            "- `qwen_4b_it_tokens` counts tokens of the *full harness prompt* (doc_to_text) for each sample.\n"
        )
        f.write(
            "- `qwen_4b_it_context_tokens` counts tokens of the raw `context` field (for reference).\n\n"
        )

        f.write("## Overall (prompt tokens)\n\n")
        f.write("```json\n")
        f.write(json.dumps(overall_prompt_summary, indent=2, sort_keys=True) + "\n")
        f.write("```\n\n")

        f.write("## Overall (context tokens)\n\n")
        f.write("```json\n")
        f.write(json.dumps(overall_context_summary, indent=2, sort_keys=True) + "\n")
        f.write("```\n\n")

        f.write("## By dataset (context tokens)\n\n")
        for dataset_name, summary in by_dataset_context_summary.items():
            f.write(f"### {dataset_name}\n\n")
            f.write("```json\n")
            f.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            f.write("```\n\n")

        f.write("## By dataset (prompt tokens)\n\n")
        for dataset_name, summary in by_dataset_prompt_summary.items():
            f.write(f"### {dataset_name}\n\n")
            f.write("```json\n")
            f.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            f.write("```\n\n")

        f.write("## Raw token counts\n\n")
        f.write(f"- JSONL: `{jsonl_path.name}`\n")

    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
