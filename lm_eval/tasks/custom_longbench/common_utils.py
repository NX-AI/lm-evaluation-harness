from __future__ import annotations

import json
import zipfile
from functools import cache
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import snapshot_download


@cache
def _repo_root(repo_id: str) -> Path:
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset")).resolve()


@cache
def _load_longbench_split(repo_id: str, dataset_name: str) -> datasets.Dataset:
    repo_root = _repo_root(repo_id)
    zip_path = repo_root / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected {zip_path} for dataset repo {repo_id}")

    member = f"data/{dataset_name}.jsonl"
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fp:
            for raw in fp:
                line = raw.decode("utf-8").strip()
                if line:
                    doc = json.loads(line)
                    if "question" not in doc and "input" in doc:
                        doc["question"] = doc["input"]
                    doc.setdefault("question", "")
                    doc.setdefault("context", "")
                    doc.setdefault("answers", [])
                    doc.setdefault("all_classes", [])
                    records.append(doc)

    return datasets.Dataset.from_list(records)


def load_longbench_dataset(
    *,
    repo_id: str,
    dataset_name: str,
    split: str = "test",
    **_: Any,
) -> dict[str, datasets.Dataset]:
    return {split: _load_longbench_split(repo_id, dataset_name)}
