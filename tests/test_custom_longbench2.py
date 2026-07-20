from __future__ import annotations

from collections.abc import Callable

import datasets

from lm_eval.tasks import TaskManager
from lm_eval.tasks.custom_longbench2 import utils as longbench2_utils


def _make_row(answer: str = "B") -> dict[str, object]:
    return {
        "_id": "row-1",
        "context": "A long document about an event with many details.",
        "question": "Which option best matches the document?",
        "choice_A": "Option one",
        "choice_B": "Option two",
        "choice_C": "Option three",
        "choice_D": "Option four",
        "answer": answer,
        "domain": "Single-Document QA",
        "sub_domain": "Academic",
        "difficulty": "hard",
        "length": "medium",
    }


def _load_dataset_factory(
    rows: list[dict[str, object]],
) -> Callable[..., datasets.Dataset]:
    def _loader(path: str, split: str, *args, **kwargs) -> datasets.Dataset:
        del args, kwargs
        assert path == longbench2_utils.DEFAULT_REPO_ID
        assert split == longbench2_utils.DEFAULT_SPLIT
        return datasets.Dataset.from_list(rows)

    return _loader


def test_build_prompt_formats_default_longbench2_prompt() -> None:
    prompt = longbench2_utils.build_prompt(_make_row())

    assert prompt.startswith(
        "Please read the following text and answer the question below."
    )
    assert "<text>\nA long document about an event with many details.\n</text>" in prompt
    assert "What is the correct answer to this question: Which option best matches the document?" in prompt
    assert "(A) Option one" in prompt
    assert "(D) Option four" in prompt
    assert r"On a new line, output the best choice as \boxed{<LETTER>}" in prompt


def test_build_prompt_formats_paper_cot_longbench2_prompt() -> None:
    prompt = longbench2_utils.build_prompt_for_style(
        _make_row(),
        longbench2_utils.COT_PROMPT_STYLE,
    )

    assert prompt.startswith(
        "Please read the following text and answer the question below.\n<text>\n"
    )
    assert "<text>\nA long document about an event with many details.\n</text>" in prompt
    assert "What is the correct answer to this question: Which option best matches the document?" in prompt
    assert "(A) Option one" in prompt
    assert "(D) Option four" in prompt
    assert r"\boxed{<LETTER>}" in prompt
    assert prompt.endswith(
        r"Let's think step by step. On a new line, output the best choice as \boxed{<LETTER>} where <LETTER> is one of A, B, C or D."
    )


def test_extract_answer_label_handles_expected_response_formats() -> None:
    assert longbench2_utils.extract_answer_label(r"\boxed{C}") == "C"
    assert longbench2_utils.extract_answer_label(r"\boxed{\text{B}}") == "B"
    assert (
        longbench2_utils.extract_answer_label("<think>reasoning</think>\n\\boxed{D}")
        == "D"
    )
    assert longbench2_utils.extract_answer_label("The correct answer is (C).") == "C"
    assert longbench2_utils.extract_answer_label("Answer: B") == "B"
    assert longbench2_utils.extract_answer_label("(D)") == "D"
    assert longbench2_utils.extract_answer_label("   a   ") == "A"


def test_bucket_helpers_normalize_expected_labels() -> None:
    assert longbench2_utils.normalize_token_length_bucket("0_16k") == "0_16k"
    assert longbench2_utils.normalize_token_length_bucket("128k_512k") == "128k_512k"
    assert longbench2_utils.normalize_prompt_style("paper_cot") == "cot"
    assert (
        longbench2_utils.get_token_annotations_path("cot")
        == longbench2_utils.DEFAULT_TOKEN_ANNOTATIONS_PATH
    )
    assert longbench2_utils.bucket_for_token_count(8192) == "0_16k"
    assert longbench2_utils.bucket_for_token_count(2000) == "0_16k"
    assert longbench2_utils.bucket_for_token_count(1_500_000) == "512k_1024k"


def test_process_results_scores_accuracy_from_extracted_label() -> None:
    result = longbench2_utils.process_results(
        _make_row(answer="B"),
        ["<think>read carefully</think>\nTherefore the answer is\n\\boxed{B}."],
    )

    assert result == {"acc": 1.0}


def test_doc_to_target_uses_boxed_letter() -> None:
    assert longbench2_utils.doc_to_target(_make_row(answer="C")) == r"\boxed{C}"


def test_load_longbench2_dataset_reads_requested_split(monkeypatch) -> None:
    rows = [_make_row()]
    monkeypatch.setattr(
        longbench2_utils.datasets,
        "load_dataset",
        _load_dataset_factory(rows),
    )

    loaded = longbench2_utils.load_longbench2_dataset()

    docs = loaded[longbench2_utils.DEFAULT_SPLIT]
    assert len(docs) == 1
    assert docs[0]["answer"] == "B"
    assert docs[0]["sub_domain"] == "Academic"


def test_load_longbench2_dataset_ignores_harness_metadata_kwargs(monkeypatch) -> None:
    rows = [_make_row()]
    seen_kwargs: dict[str, object] = {}

    def _loader(path: str, split: str, *args, **kwargs) -> datasets.Dataset:
        del args
        seen_kwargs.update(kwargs)
        assert path == longbench2_utils.DEFAULT_REPO_ID
        assert split == longbench2_utils.DEFAULT_SPLIT
        return datasets.Dataset.from_list(rows)

    monkeypatch.setattr(longbench2_utils.datasets, "load_dataset", _loader)

    loaded = longbench2_utils.load_longbench2_dataset(
        config_source="generated.yaml",
        version="1.0",
        pretrained="/tmp/model",
        batch_size=1,
    )

    assert len(loaded[longbench2_utils.DEFAULT_SPLIT]) == 1
    assert seen_kwargs == {}


def test_load_longbench2_dataset_filters_with_annotations(
    monkeypatch,
    tmp_path,
) -> None:
    rows = [
        _make_row(answer="A"),
        {
            **_make_row(answer="D"),
            "_id": "row-2",
            "sub_domain": "Legal",
            "question": "Which legal option is correct?",
        },
    ]
    monkeypatch.setattr(
        longbench2_utils.datasets,
        "load_dataset",
        _load_dataset_factory(rows),
    )

    annotations_path = tmp_path / "longbench2_tokens.jsonl"
    annotations_path.write_text(
        "\n".join(
            [
                '{"_id":"row-1","qwen_4b_it_tokens":12000,"qwen_4b_it_context_tokens":11000,"qwen_4b_it_length_bucket":"0_16k"}',
                '{"_id":"row-2","qwen_4b_it_tokens":40000,"qwen_4b_it_context_tokens":39000,"qwen_4b_it_length_bucket":"32k_64k"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = longbench2_utils.load_longbench2_dataset(
        sub_domain="Academic",
        token_length_bucket="0_16k",
        annotations_path=str(annotations_path),
    )

    docs = loaded[longbench2_utils.DEFAULT_SPLIT]
    assert len(docs) == 1
    assert docs[0]["_id"] == "row-1"
    assert docs[0][longbench2_utils.TOKEN_COUNT_FIELD] == 12000
    assert docs[0][longbench2_utils.TOKEN_LENGTH_BUCKET_FIELD] == "0_16k"
def test_task_manager_loads_custom_longbench2_group(monkeypatch) -> None:
    rows = [_make_row(answer="C")]
    monkeypatch.setattr(
        longbench2_utils.datasets,
        "load_dataset",
        _load_dataset_factory(rows),
    )

    task_manager = TaskManager()
    loaded = task_manager.load(["custom_longbench2"])

    assert "custom_longbench2" in loaded["groups"]
    assert loaded["group_map"]["custom_longbench2"] == ["custom_longbench2_0shot"]

    task = loaded["tasks"]["custom_longbench2_0shot"]
    docs = task.test_docs()
    assert len(docs) == 1
    assert docs[0]["answer"] == "C"
    assert task.config.num_fewshot == 0
    assert (
        task.config.generation_kwargs["max_gen_toks"]
        == longbench2_utils.DEFAULT_MAX_GEN_TOKS
    )


def test_task_manager_loads_custom_longbench2_cot_group(monkeypatch) -> None:
    rows = [_make_row(answer="A")]
    monkeypatch.setattr(
        longbench2_utils.datasets,
        "load_dataset",
        _load_dataset_factory(rows),
    )

    task_manager = TaskManager()
    loaded = task_manager.load(["custom_longbench2_cot"])

    assert "custom_longbench2_cot" in loaded["groups"]
    assert loaded["group_map"]["custom_longbench2_cot"] == ["custom_longbench2_cot_0shot"]

    task = loaded["tasks"]["custom_longbench2_cot_0shot"]
    docs = task.test_docs()
    assert len(docs) == 1
    assert docs[0]["answer"] == "A"
    assert task.config.num_fewshot == 0
    assert (
        task.config.generation_kwargs["max_gen_toks"]
        == longbench2_utils.DEFAULT_COT_MAX_GEN_TOKS
    )
