from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from lm_eval.tasks.custom_longbench2 import utils


DEFAULT_MODEL_CONFIG = Path(
    "/nfs-gpu/xlstm-distillation/work_niklas/"
    "xlstm-distillation-internal/configs/model/qwen3_4b/qwen3_4b_it_baseline_fft.yaml"
)
CUMULATIVE_BUCKET_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("16k", ("0_16k",)),
    ("32k", ("0_16k", "16k_32k")),
    ("64k", ("0_16k", "16k_32k", "32k_64k")),
    ("128k", ("0_16k", "16k_32k", "32k_64k", "64k_128k")),
    ("512k", ("0_16k", "16k_32k", "32k_64k", "64k_128k", "128k_512k")),
)


@dataclass(frozen=True)
class TaskVariant:
    family_prefix: str
    group_alias_prefix: str
    base_include: str
    bucket_tag: str


TASK_VARIANTS: tuple[TaskVariant, ...] = (
    TaskVariant(
        family_prefix="custom_longbench2",
        group_alias_prefix="LongBench v2",
        base_include="_custom_longbench2_base.yaml",
        bucket_tag="custom_longbench2_bucket_tasks",
    ),
    TaskVariant(
        family_prefix="custom_longbench2_cot",
        group_alias_prefix="LongBench v2 CoT",
        base_include="_custom_longbench2_cot_base.yaml",
        bucket_tag="custom_longbench2_cot_bucket_tasks",
    ),
)


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


def _write_leaf_task_yaml(
    output_path: Path,
    *,
    variant: TaskVariant,
    task_name: str,
    task_alias: str,
    sub_domain: str,
    bucket_label: str,
) -> None:
    lines = [
        f"include: {variant.base_include}",
        "tag:",
        f"  - {variant.bucket_tag}",
        f"task: {task_name}",
        f'task_alias: "{task_alias}"',
        "dataset_kwargs:",
        f'  sub_domain: "{sub_domain}"',
        f"  token_length_bucket: {bucket_label}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_group_yaml(
    output_path: Path,
    *,
    group_name: str,
    group_alias: str,
    tasks: list[str],
) -> None:
    lines = [
        f"group: {group_name}",
        f'group_alias: "{group_alias}"',
        "task:",
    ]
    lines.extend(f"  - {task_name}" for task_name in tasks)
    lines.extend(
        [
            "aggregate_metric_list:",
            "  - metric: acc",
            "    weight_by_size: true",
            "metadata:",
            "  version: 1.0",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _generate_bucket_configs(
    *,
    task_dir: Path,
    non_empty_pairs: dict[str, list[str]],
    variant: TaskVariant,
) -> None:
    generated_files = set(task_dir.glob(f"{variant.family_prefix}_*_len_*.yaml"))
    generated_files.update(task_dir.glob(f"{variant.family_prefix}_len_*.yaml"))
    generated_files.update(task_dir.glob(f"{variant.family_prefix}_le_*.yaml"))
    generated_files.add(task_dir / f"{variant.family_prefix}_bucketed.yaml")
    for sub_domain in non_empty_pairs:
        slug = utils.slugify_sub_domain(sub_domain)
        generated_files.add(task_dir / f"{variant.family_prefix}_{slug}.yaml")
    for path in generated_files:
        if path.exists():
            path.unlink()

    by_bucket: dict[str, list[str]] = defaultdict(list)
    all_leaf_tasks: list[str] = []

    for sub_domain in sorted(non_empty_pairs):
        slug = utils.slugify_sub_domain(sub_domain)
        leaf_tasks: list[str] = []
        for bucket_label in non_empty_pairs[sub_domain]:
            task_name = f"{variant.family_prefix}_{slug}_len_{bucket_label}"
            task_alias = (
                f"{variant.group_alias_prefix} {sub_domain} "
                f"({bucket_label.replace('_', '-')} tokens)"
            )
            _write_leaf_task_yaml(
                task_dir / f"{task_name}.yaml",
                variant=variant,
                task_name=task_name,
                task_alias=task_alias,
                sub_domain=sub_domain,
                bucket_label=bucket_label,
            )
            leaf_tasks.append(task_name)
            by_bucket[bucket_label].append(task_name)
            all_leaf_tasks.append(task_name)

        _write_group_yaml(
            task_dir / f"{variant.family_prefix}_{slug}.yaml",
            group_name=f"{variant.family_prefix}_{slug}",
            group_alias=f"{variant.group_alias_prefix} {sub_domain}",
            tasks=leaf_tasks,
        )

    for bucket_label, _, _ in utils.TOKEN_LENGTH_BUCKETS:
        bucket_tasks = by_bucket.get(bucket_label, [])
        if not bucket_tasks:
            continue
        _write_group_yaml(
            task_dir / f"{variant.family_prefix}_len_{bucket_label}.yaml",
            group_name=f"{variant.family_prefix}_len_{bucket_label}",
            group_alias=(
                f"{variant.group_alias_prefix} {bucket_label.replace('_', '-')} tokens"
            ),
            tasks=bucket_tasks,
        )

    for threshold_label, included_buckets in CUMULATIVE_BUCKET_GROUPS:
        threshold_tasks: list[str] = []
        for bucket_label in included_buckets:
            threshold_tasks.extend(by_bucket.get(bucket_label, []))
        if not threshold_tasks:
            continue
        _write_group_yaml(
            task_dir / f"{variant.family_prefix}_le_{threshold_label}.yaml",
            group_name=f"{variant.family_prefix}_le_{threshold_label}",
            group_alias=f"{variant.group_alias_prefix} <= {threshold_label} tokens",
            tasks=threshold_tasks,
        )

    _write_group_yaml(
        task_dir / f"{variant.family_prefix}_bucketed.yaml",
        group_name=f"{variant.family_prefix}_bucketed",
        group_alias=f"{variant.group_alias_prefix} bucketed by Qwen token length",
        tasks=all_leaf_tasks,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to the xlstm-distillation model YAML used to resolve the tokenizer.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write the JSONL, report, and generated YAML configs into.",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not hit the network for tokenizer files (requires local cache).",
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
            "Expected a fast tokenizer with `backend_tokenizer` for efficient token counting."
        )

    ds = load_dataset(utils.DEFAULT_REPO_ID, split=utils.DEFAULT_SPLIT)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / utils.DEFAULT_TOKEN_ANNOTATIONS_PATH.name
    md_path = out_dir / "qwen3_4b_it_longbench_v2_token_report.md"

    overall_context: list[int] = []
    by_sub_domain_context: dict[str, list[int]] = defaultdict(list)
    by_sub_domain_bucket_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for idx, doc in enumerate(ds):
            context = str(doc.get("context", ""))
            context_tok = int(len(backend.encode(context)))
            bucket_label = utils.bucket_for_token_count(context_tok)
            sub_domain = str(doc["sub_domain"])

            overall_context.append(context_tok)
            by_sub_domain_context[sub_domain].append(context_tok)
            if bucket_label is not None:
                by_sub_domain_bucket_counts[sub_domain][bucket_label] += 1

            record = {
                "_id": str(doc["_id"]),
                "idx": int(idx),
                "domain": doc["domain"],
                "sub_domain": sub_domain,
                "difficulty": doc["difficulty"],
                "length": doc["length"],
                utils.TOKEN_COUNT_FIELD: context_tok,
                utils.CONTEXT_TOKEN_COUNT_FIELD: context_tok,
                utils.TOKEN_LENGTH_BUCKET_FIELD: bucket_label,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    overall_context_summary = _summarize(overall_context)
    by_sub_domain_context_summary = {
        sub_domain: _summarize(values)
        for sub_domain, values in sorted(by_sub_domain_context.items())
    }
    non_empty_pairs = {
        sub_domain: [
            bucket_label
            for bucket_label, _, _ in utils.TOKEN_LENGTH_BUCKETS
            if bucket_counts.get(bucket_label, 0) > 0
        ]
        for sub_domain, bucket_counts in sorted(by_sub_domain_bucket_counts.items())
    }

    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("# Qwen3-4B-Instruct context token lengths on LongBench v2\n\n")
        fp.write(f"- Model config: `{args.model_config}`\n")
        fp.write(f"- Tokenizer: `{model_id}`\n")
        fp.write(
            "- Bucketing is prompt-agnostic and uses only the raw `context` field so "
            "different prompt strategies evaluate the exact same documents in each bucket.\n"
        )
        fp.write(
            f"- `{utils.TOKEN_COUNT_FIELD}` and `{utils.CONTEXT_TOKEN_COUNT_FIELD}` "
            "both store the raw `context` token count for compatibility with the "
            "existing task loader.\n"
        )
        fp.write(
            f"- `{utils.TOKEN_LENGTH_BUCKET_FIELD}` uses context-token buckets "
            "`0_16k`, `16k_32k`, `32k_64k`, `64k_128k`, `128k_512k`, `512k_1024k`.\n\n"
        )

        fp.write("## Overall (context tokens)\n\n```json\n")
        fp.write(json.dumps(overall_context_summary, indent=2, sort_keys=True) + "\n")
        fp.write("```\n\n")

        fp.write("## By sub-domain (context tokens)\n\n")
        for sub_domain, summary in by_sub_domain_context_summary.items():
            fp.write(f"### {sub_domain}\n\n```json\n")
            fp.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            fp.write("```\n\n")

        fp.write("## Non-empty context-token buckets by sub-domain\n\n")
        for sub_domain, bucket_labels in non_empty_pairs.items():
            bucket_counts = {
                bucket_label: by_sub_domain_bucket_counts[sub_domain][bucket_label]
                for bucket_label in bucket_labels
            }
            fp.write(f"### {sub_domain}\n\n```json\n")
            fp.write(json.dumps(bucket_counts, indent=2, sort_keys=True) + "\n")
            fp.write("```\n\n")

        fp.write("## Raw token counts\n\n")
        fp.write(f"- JSONL: `{jsonl_path.name}`\n")

    for variant in TASK_VARIANTS:
        _generate_bucket_configs(
            task_dir=out_dir,
            non_empty_pairs=non_empty_pairs,
            variant=variant,
        )

    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {md_path}")
    for variant in TASK_VARIANTS:
        print(f"Generated bucket configs for {variant.family_prefix}:")
        for sub_domain, bucket_labels in non_empty_pairs.items():
            print(
                f"  - {sub_domain}: "
                f"{', '.join(bucket_labels) if bucket_labels else '(none)'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
