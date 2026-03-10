from __future__ import annotations

import json
from collections.abc import Callable
from difflib import SequenceMatcher

import datasets
import pytest

from lm_eval.tasks import TaskManager
from lm_eval.tasks.custom_mrcr import task as mrcr_task


def _make_row(n_needles: int, prefix: str, suffix: str) -> dict[str, object]:
    prompt = json.dumps(
        [
            {"role": "user", "content": f"Context for {n_needles} needles."},
            {"role": "assistant", "content": "Earlier assistant turn."},
            {"role": "user", "content": f"Return the final answer for {n_needles}."},
        ]
    )
    return {
        "prompt": prompt,
        "answer": f"{prefix}{suffix}",
        "random_string_to_prepend": prefix,
        "n_needles": n_needles,
        "desired_msg_index": 1,
        "total_messages": 3,
        "n_chars": len(prompt),
        "date_added": "2025-01-01",
    }


def _make_length_row(
    length_bin: str,
    token_count: int,
    n_needles: int = 2,
) -> dict[str, object]:
    row = _make_row(n_needles, f"{length_bin[:4]}HASH00", f"{length_bin} answer")
    row["qwen3_4b_it_token_count"] = token_count
    row["length_bin"] = length_bin
    return row


def _load_from_disk_factory(rows: list[dict[str, object]]) -> Callable[..., datasets.DatasetDict]:
    def _loader(path: str, *args, **kwargs) -> datasets.DatasetDict:
        del args, kwargs
        assert path == mrcr_task.MRCR_LOCAL_DATASET_PATH
        return datasets.DatasetDict({"train": datasets.Dataset.from_list(rows)})

    return _loader


def _make_task(monkeypatch, n_needles: int = 2) -> mrcr_task.MRCRTask:
    rows = [
        _make_row(2, "HASH000001", "gold answer 2"),
        _make_row(4, "HASH000002", "gold answer 4"),
        _make_row(8, "HASH000003", "gold answer 8"),
    ]
    monkeypatch.setattr(
        mrcr_task.datasets,
        "load_from_disk",
        _load_from_disk_factory(rows),
    )
    return mrcr_task.MRCRTask(
        config={
            "task": f"custom_mrcr_{n_needles}_test",
            "dataset_path": "openai/mrcr",
            "test_split": "test",
            "output_type": "generate_until",
            "dataset_kwargs": {"n_needles": n_needles},
            "num_fewshot": 0,
            "generation_kwargs": {
                "max_gen_toks": 4096,
                "do_sample": False,
                "temperature": 0.0,
                "until": [],
            },
            "metric_list": [
                {
                    "metric": "score",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
        }
    )


def test_score_response_exact_match() -> None:
    assert mrcr_task.score_response("HASHgold", "HASHgold", "HASH") == 1.0


def test_score_response_missing_prefix_returns_zero() -> None:
    assert mrcr_task.score_response("gold", "HASHgold", "HASH") == 0.0


def test_score_response_prefix_not_at_start_returns_zero() -> None:
    assert mrcr_task.score_response("xHASHgold", "HASHgold", "HASH") == 0.0


def test_score_response_partial_overlap_returns_fractional_score() -> None:
    expected = SequenceMatcher(None, "hello there", "hello world").ratio()
    actual = mrcr_task.score_response(
        "HASHhello there",
        "HASHhello world",
        "HASH",
    )
    assert 0.0 < actual < 1.0
    assert actual == expected


def test_load_mrcr_dataset_reads_local_preprocessed_dataset(monkeypatch) -> None:
    rows = [_make_length_row("4k_8k", 5000, n_needles=2)]
    monkeypatch.setattr(
        mrcr_task.datasets,
        "load_from_disk",
        _load_from_disk_factory(rows),
    )

    loaded = mrcr_task.load_mrcr_dataset(length_bin="4k_8k")

    docs = loaded["test"]
    assert len(docs) == 1
    assert docs[0]["length_bin"] == "4k_8k"
    assert docs[0]["qwen3_4b_it_token_count"] == 5000


def test_plaintext_prompt_format_preserves_turn_order_and_appends_assistant(
    monkeypatch,
) -> None:
    task = _make_task(monkeypatch, n_needles=2)
    doc = task.test_docs()[0]

    prompt = task.doc_to_text(doc)

    assert prompt == (
        "User: Context for 2 needles.\n\n"
        "Assistant: Earlier assistant turn.\n\n"
        "User: Return the final answer for 2.\n\n"
        "Assistant:"
    )


def test_chat_template_path_preserves_original_messages(monkeypatch) -> None:
    task = _make_task(monkeypatch, n_needles=2)
    doc = task.test_docs()[0]
    calls = []

    def _chat_template(messages, add_generation_prompt):
        calls.append((messages, add_generation_prompt))
        return "CHAT_PROMPT"

    prompt = task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=_chat_template,
    )

    assert prompt == "CHAT_PROMPT"
    assert calls == [
        (
            [
                {"role": "user", "content": "Context for 2 needles."},
                {"role": "assistant", "content": "Earlier assistant turn."},
                {"role": "user", "content": "Return the final answer for 2."},
            ],
            True,
        )
    ]


def test_nonzero_fewshot_is_rejected(monkeypatch) -> None:
    task = _make_task(monkeypatch, n_needles=2)
    doc = task.test_docs()[0]

    with pytest.raises(ValueError, match="zero-shot only"):
        task.fewshot_context(doc, num_fewshot=1)


def test_system_instruction_is_ignored_with_warning(monkeypatch, caplog) -> None:
    task = _make_task(monkeypatch, n_needles=2)
    doc = task.test_docs()[0]

    with caplog.at_level("WARNING"):
        prompt = task.fewshot_context(
            doc,
            num_fewshot=0,
            system_instruction="Do something else",
        )

    assert prompt.endswith("Assistant:")
    assert "ignores supplied system instructions" in caplog.text


def test_process_results_uses_official_score(monkeypatch) -> None:
    task = _make_task(monkeypatch, n_needles=2)
    doc = task.test_docs()[0]

    result = task.process_results(doc, [doc["answer"]])

    assert result == {"score": 1.0}


def test_task_config_is_zero_shot_and_disables_default_stop(monkeypatch) -> None:
    task = _make_task(monkeypatch, n_needles=2)

    assert task.config.num_fewshot == 0
    assert task.config.generation_kwargs["until"] == []
    assert task.config.generation_kwargs["max_gen_toks"] == 4096
    assert callable(task.config.process_results)
    assert task._metric_fn_list["score"] is None


def test_task_manager_loads_group_and_filters_each_subset(monkeypatch) -> None:
    rows = [
        _make_row(2, "HASH000001", "gold answer 2"),
        _make_row(4, "HASH000002", "gold answer 4"),
        _make_row(8, "HASH000003", "gold answer 8"),
    ]
    monkeypatch.setattr(
        mrcr_task.datasets,
        "load_from_disk",
        _load_from_disk_factory(rows),
    )

    task_manager = TaskManager()
    loaded = task_manager.load(["custom_mrcr"])

    assert set(loaded["tasks"]) == {
        "custom_mrcr_2needle",
        "custom_mrcr_4needle",
        "custom_mrcr_8needle",
    }
    assert "custom_mrcr" in loaded["groups"]
    assert loaded["group_map"]["custom_mrcr"] == [
        "custom_mrcr_2needle",
        "custom_mrcr_4needle",
        "custom_mrcr_8needle",
    ]

    group = loaded["groups"]["custom_mrcr"]
    assert group.aggregate_metric_list is not None
    assert group.aggregate_metric_list[0].metric == "score"
    assert group.aggregate_metric_list[0].weight_by_size is True
    assert set(loaded["tasks"]["custom_mrcr_2needle"].test_docs()["n_needles"]) == {2}
    assert set(loaded["tasks"]["custom_mrcr_4needle"].test_docs()["n_needles"]) == {4}
    assert set(loaded["tasks"]["custom_mrcr_8needle"].test_docs()["n_needles"]) == {8}


def test_task_manager_loads_length_group_and_filters_each_bin(monkeypatch) -> None:
    rows = [
        _make_length_row("4k_8k", 5000, n_needles=2),
        _make_length_row("8k_16k", 10000, n_needles=4),
        _make_length_row("16k_32k", 20000, n_needles=8),
        _make_length_row("32k_64k", 40000, n_needles=2),
        _make_length_row("64k_128k", 80000, n_needles=4),
        _make_length_row("128k_256k", 160000, n_needles=8),
        _make_length_row("256k_512k", 300000, n_needles=2),
        _make_length_row("512k_1024k", 600000, n_needles=4),
    ]
    monkeypatch.setattr(
        mrcr_task.datasets,
        "load_from_disk",
        _load_from_disk_factory(rows),
    )

    task_manager = TaskManager()
    loaded = task_manager.load(["custom_mrcr_by_length"])

    expected_tasks = {
        "custom_mrcr_len_4k_8k",
        "custom_mrcr_len_8k_16k",
        "custom_mrcr_len_16k_32k",
        "custom_mrcr_len_32k_64k",
        "custom_mrcr_len_64k_128k",
        "custom_mrcr_len_128k_256k",
        "custom_mrcr_len_256k_512k",
        "custom_mrcr_len_512k_1024k",
    }
    assert set(loaded["tasks"]) == expected_tasks
    assert "custom_mrcr_by_length" in loaded["groups"]

    for task_name, expected_bin in {
        "custom_mrcr_len_4k_8k": "4k_8k",
        "custom_mrcr_len_8k_16k": "8k_16k",
        "custom_mrcr_len_16k_32k": "16k_32k",
        "custom_mrcr_len_32k_64k": "32k_64k",
        "custom_mrcr_len_64k_128k": "64k_128k",
        "custom_mrcr_len_128k_256k": "128k_256k",
        "custom_mrcr_len_256k_512k": "256k_512k",
        "custom_mrcr_len_512k_1024k": "512k_1024k",
    }.items():
        docs = loaded["tasks"][task_name].test_docs()
        assert set(docs["length_bin"]) == {expected_bin}


def test_task_manager_loads_combined_needle_and_length_tasks(monkeypatch) -> None:
    rows = [
        _make_length_row("4k_8k", 5000, n_needles=2),
        _make_length_row("4k_8k", 5001, n_needles=4),
        _make_length_row("8k_16k", 10000, n_needles=2),
        _make_length_row("512k_1024k", 600000, n_needles=8),
    ]
    monkeypatch.setattr(
        mrcr_task.datasets,
        "load_from_disk",
        _load_from_disk_factory(rows),
    )

    task_manager = TaskManager()
    loaded = task_manager.load(
        [
            "custom_mrcr_2needle_len_4k_8k",
            "custom_mrcr_4needle_len_4k_8k",
            "custom_mrcr_8needle_len_512k_1024k",
        ]
    )

    assert set(loaded["tasks"]) == {
        "custom_mrcr_2needle_len_4k_8k",
        "custom_mrcr_4needle_len_4k_8k",
        "custom_mrcr_8needle_len_512k_1024k",
    }
    docs = loaded["tasks"]["custom_mrcr_2needle_len_4k_8k"].test_docs()
    assert set(docs["n_needles"]) == {2}
    assert set(docs["length_bin"]) == {"4k_8k"}

    docs = loaded["tasks"]["custom_mrcr_4needle_len_4k_8k"].test_docs()
    assert set(docs["n_needles"]) == {4}
    assert set(docs["length_bin"]) == {"4k_8k"}

    docs = loaded["tasks"]["custom_mrcr_8needle_len_512k_1024k"].test_docs()
    assert set(docs["n_needles"]) == {8}
    assert set(docs["length_bin"]) == {"512k_1024k"}
