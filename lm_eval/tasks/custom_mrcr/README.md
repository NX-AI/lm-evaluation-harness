# custom_mrcr

MRCR is integrated as a built-in custom benchmark backed by the locally
preprocessed dataset at
`/nfs-gpu/xlstm-distillation/lolcat_data/harness_datasets/mrcr`.

## Task Names

- `custom_mrcr`
- `custom_mrcr_2needle`
- `custom_mrcr_4needle`
- `custom_mrcr_8needle`
- `custom_mrcr_by_length`
- `custom_mrcr_len_4k_8k`
- `custom_mrcr_len_8k_16k`
- `custom_mrcr_len_16k_32k`
- `custom_mrcr_len_32k_64k`
- `custom_mrcr_len_64k_128k`
- `custom_mrcr_len_128k_256k`
- `custom_mrcr_len_256k_512k`
- `custom_mrcr_len_512k_1024k`
- `custom_mrcr_{2,4,8}needle_len_<bin>` for all 24 needle-by-length combinations

`custom_mrcr` is an aggregate group over the three subset tasks and reports a
size-weighted mean `score`.

`custom_mrcr_by_length` is an aggregate group over the dataset-card token-length
bins.

## Prompting

MRCR is zero-shot only. The task ignores external few-shot examples and ignores
runtime system instructions so the benchmark prompt stays unchanged.

When `--apply_chat_template` is enabled, the task parses the dataset's
serialized chat prompt and renders the original `user` / `assistant` turns with
`add_generation_prompt=True`.

Without `--apply_chat_template`, the same prompt is flattened into a plain-text
transcript ending in `Assistant:` for completion-style models.

## Scoring

Scoring follows the official dataset README:

1. The model response must start with `random_string_to_prepend`.
2. That prefix is stripped from both the model response and the gold answer.
3. `difflib.SequenceMatcher(...).ratio()` is used as the final score.

If the required prefix is missing, the score is `0.0`.

## Length Bins

Length-bin tasks follow the MRCR dataset card boundaries based on `prompt +
answer` token count:

- `4k_8k`: `[4096, 8192]`
- `8k_16k`: `(8192, 16384]`
- `16k_32k`: `(16384, 32768]`
- `32k_64k`: `(32768, 65536]`
- `64k_128k`: `(65536, 131072]`
- `128k_256k`: `(131072, 262144]`
- `256k_512k`: `(262144, 524288]`
- `512k_1024k`: `(524288, 1048576]`

These bins are already materialized in the saved local dataset. The harness does
not tokenize MRCR at evaluation time. Rows that fall below `4096` or above
`1048576` Qwen-tokenized `prompt + answer` tokens are clamped into the nearest
edge bin in the saved dataset.

## Example Usage

```bash
lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
  --tasks custom_mrcr \
  --apply_chat_template
```

```bash
lm_eval \
  --model hf \
  --model_args pretrained=EleutherAI/pythia-410m \
  --tasks custom_mrcr_8needle
```

```bash
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507 \
  --tasks custom_mrcr_by_length \
  --apply_chat_template
```

```bash
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507 \
  --tasks custom_mrcr_2needle_len_4k_8k \
  --apply_chat_template
```
