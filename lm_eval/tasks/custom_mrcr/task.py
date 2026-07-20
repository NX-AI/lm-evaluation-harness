from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from typing import Any

import datasets

from lm_eval.api.task import ConfigurableTask


eval_logger = logging.getLogger(__name__)

MRCR_DATASET_PATH = "openai/mrcr"
MRCR_LOCAL_DATASET_PATH = (
    "/nfs-gpu/xlstm-distillation/lolcat_data/harness_datasets/mrcr"
)
MRCR_LENGTH_BINS: tuple[tuple[str, int, int], ...] = (
    ("4k_8k", 4096, 8192),
    ("8k_16k", 8192, 16384),
    ("16k_32k", 16384, 32768),
    ("32k_64k", 32768, 65536),
    ("64k_128k", 65536, 131072),
    ("128k_256k", 131072, 262144),
    ("256k_512k", 262144, 524288),
    ("512k_1024k", 524288, 1048576),
)
_ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
}


def load_dataset_from_disk(path: str, split: str = "train") -> datasets.Dataset:
    """Load a Dataset or DatasetDict materialized with save_to_disk."""
    loaded = datasets.load_from_disk(path)
    if isinstance(loaded, datasets.DatasetDict):
        if split not in loaded:
            raise KeyError(f"MRCR dataset at {path!r} does not contain split {split!r}.")
        return loaded[split]
    if isinstance(loaded, datasets.Dataset):
        return loaded
    raise TypeError(f"Unsupported dataset type loaded from {path!r}: {type(loaded)!r}")


def load_mrcr_dataset(
    *,
    n_needles: int | None = None,
    length_bin: str | None = None,
    split: str = "train",
    **_: Any,
) -> datasets.DatasetDict:
    """Load the locally preprocessed MRCR dataset and expose the subset as test."""
    dataset = load_dataset_from_disk(MRCR_LOCAL_DATASET_PATH, split=split)
    if n_needles is not None:
        needle_count = int(n_needles)
        dataset = dataset.filter(lambda row: int(row["n_needles"]) == needle_count)
    if length_bin is not None:
        normalized_length_bin = normalize_length_bin(length_bin)
        dataset = dataset.filter(lambda row: row["length_bin"] == normalized_length_bin)
    return datasets.DatasetDict({"test": dataset})


def parse_prompt_messages(
    prompt: str | list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Parse the serialized MRCR chat prompt into role/content messages."""
    raw_messages = json.loads(prompt) if isinstance(prompt, str) else prompt
    if not isinstance(raw_messages, list):
        raise TypeError("MRCR prompt must decode to a list of messages.")

    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise TypeError("MRCR prompt messages must be dictionaries.")
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        messages.append(
            {
                "role": role,
                "content": "" if content is None else str(content),
            }
        )
    return messages


def format_plaintext_transcript(messages: list[dict[str, str]]) -> str:
    """Render MRCR chat turns as a deterministic plain-text transcript."""
    transcript = [
        f"{_ROLE_LABELS.get(message['role'], message['role'].title())}: {message['content']}"
        for message in messages
    ]
    if not messages or messages[-1]["role"] != "assistant":
        transcript.append("Assistant:")
    return "\n\n".join(transcript)


def normalize_length_bin(length_bin: str) -> str:
    """Normalize user-facing length bin labels to the internal task form."""
    normalized = length_bin.strip().lower()
    if normalized in {label for label, _, _ in MRCR_LENGTH_BINS}:
        return normalized
    raise ValueError(
        f"Unsupported MRCR length bin {length_bin!r}. Expected one of "
        f"{', '.join(label for label, _, _ in MRCR_LENGTH_BINS)}."
    )


def score_response(response: str, answer: str, required_prefix: str) -> float:
    """Score a response using the official MRCR rule from the dataset README."""
    if not response.startswith(required_prefix):
        return 0.0

    stripped_response = response.removeprefix(required_prefix)
    stripped_answer = answer.removeprefix(required_prefix)
    return SequenceMatcher(None, stripped_response, stripped_answer).ratio()


def extract_response_text(results: list[Any]) -> str:
    """Normalize generate_until results to the first generated string."""
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


def process_results(doc, results):
    """Compute the MRCR score for a generated response."""
    response = extract_response_text(results)
    return {
        "score": score_response(
            response=response,
            answer=doc["answer"],
            required_prefix=doc["random_string_to_prepend"],
        )
    }


class MRCRTask(ConfigurableTask):
    """Custom task wrapper for the MRCR benchmark."""

    DATASET_PATH = MRCR_DATASET_PATH

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config: dict[str, Any] | None = None,
    ) -> None:
        clean_config = {k: v for k, v in (config or {}).items() if k != "class"}
        clean_config.setdefault("process_results", process_results)
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=clean_config,
        )

    def download(
        self, dataset_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        del kwargs
        self.dataset = load_mrcr_dataset(**(dataset_kwargs or {}))

    def doc_to_text(self, doc: dict[str, Any], doc_to_text=None) -> str:
        del doc_to_text
        return format_plaintext_transcript(parse_prompt_messages(doc["prompt"]))

    def doc_to_target(self, doc: dict[str, Any], doc_to_target=None) -> str:
        del doc_to_target
        return doc["answer"]

    def fewshot_context(
        self,
        doc,
        num_fewshot,
        rnd=None,
        description=None,
        **kwargs,
    ) -> str:
        del rnd, description

        if num_fewshot != 0:
            raise ValueError("MRCR is zero-shot only and does not support few-shot examples.")

        if kwargs.get("system_instruction"):
            eval_logger.warning(
                "%s ignores supplied system instructions to preserve benchmark prompts.",
                self.config.task,
            )

        if kwargs.get("apply_chat_template"):
            chat_template = kwargs.get("chat_template")
            if chat_template is not None:
                return chat_template(
                    parse_prompt_messages(doc["prompt"]),
                    add_generation_prompt=True,
                )
            eval_logger.warning(
                "%s received apply_chat_template=True without a chat_template; falling back to plain-text prompt formatting.",
                self.config.task,
            )

        return self.doc_to_text(doc)

    def process_results(self, doc, results):
        return process_results(doc, results)
