# Custom LongBench v2

This folder adds a namespaced `custom_longbench2` entry point for LongBench v2.

- Task and tag names are prefixed with `custom_longbench2_*` to avoid collisions with upstream tasks.
- The task loads the raw `zai-org/LongBench-v2` dataset and reconstructs the default 0-shot prompt locally so the original documents and metadata stay accessible for later tokenizer-length bucketing.
- A separate paper-style CoT family is included as `custom_longbench2_cot*`, using a prompt that starts with `Let's think step by step.` and ends with the same boxed-answer instruction.
- Both prompt families instruct the model to place the final multiple-choice answer in `\boxed{<LETTER>}`, and scoring strips any visible `<think>...</think>` block before extracting the prediction.
- Evaluation uses the harness `generate_until` path and scores accuracy by extracting the predicted `A`/`B`/`C`/`D` label from the model response.
- Qwen3-4B token-count annotations and generated bucket configs are included for the `0_16k`, `16k_32k`, `32k_64k`, `64k_128k`, `128k_512k`, and `512k_1024k` context-length ranges, shared across both the default and paper-CoT prompt families.

### Paper

Title: `LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks`

Homepage: `https://longbench2.github.io/`

Dataset: `https://huggingface.co/datasets/zai-org/LongBench-v2`

### Dataset

The default config loads the official raw benchmark from `zai-org/LongBench-v2`, using the `train` split exposed by the dataset card. Prompt text is built in `utils.py` instead of relying on a pre-rendered prompt mirror, which keeps `context`, `question`, `choice_A`-`choice_D`, `domain`, `sub_domain`, `difficulty`, and `length` directly available in the loaded docs.

Precomputed Qwen3-4B token annotations live in `qwen3_4b_it_longbench_v2_tokens.jsonl`, with a summary in `qwen3_4b_it_longbench_v2_token_report.md`. Bucketing is prompt-agnostic: `qwen_4b_it_length_bucket` is derived from the raw `context` token count so `custom_longbench2_*` and `custom_longbench2_cot_*` evaluate the exact same documents in each bucket.

### Usage

- Group: `custom_longbench2`
- Leaf task: `custom_longbench2_0shot`
- CoT group: `custom_longbench2_cot`
- CoT leaf task: `custom_longbench2_cot_0shot`
- Bucketed all-leaf group: `custom_longbench2_bucketed`
- CoT bucketed all-leaf group: `custom_longbench2_cot_bucketed`
- Per-bucket groups: `custom_longbench2_len_0_16k`, `custom_longbench2_len_16k_32k`, `custom_longbench2_len_32k_64k`, `custom_longbench2_len_64k_128k`, `custom_longbench2_len_128k_512k`, `custom_longbench2_len_512k_1024k`
- CoT per-bucket groups: `custom_longbench2_cot_len_0_16k`, `custom_longbench2_cot_len_16k_32k`, `custom_longbench2_cot_len_32k_64k`, `custom_longbench2_cot_len_64k_128k`, `custom_longbench2_cot_len_128k_512k`, `custom_longbench2_cot_len_512k_1024k`
- Cumulative union groups: `custom_longbench2_le_16k`, `custom_longbench2_le_32k`, `custom_longbench2_le_64k`, `custom_longbench2_le_128k`, `custom_longbench2_le_512k`
- CoT cumulative union groups: `custom_longbench2_cot_le_16k`, `custom_longbench2_cot_le_32k`, `custom_longbench2_cot_le_64k`, `custom_longbench2_cot_le_128k`, `custom_longbench2_cot_le_512k`
- Per-sub-domain groups: `custom_longbench2_academic`, `custom_longbench2_code_repo_qa`, `custom_longbench2_governmental`, etc.
- CoT per-sub-domain groups: `custom_longbench2_cot_academic`, `custom_longbench2_cot_code_repo_qa`, `custom_longbench2_cot_governmental`, etc.

Example:

```bash
lm-eval --tasks custom_longbench2 --model hf --model_args pretrained=...
lm-eval --tasks custom_longbench2_bucketed --model hf --model_args pretrained=...
lm-eval --tasks custom_longbench2_academic_len_0_16k --model hf --model_args pretrained=...
lm-eval --tasks custom_longbench2_le_32k --model hf --model_args pretrained=...
lm-eval --tasks custom_longbench2_cot --model hf --model_args pretrained=...
lm-eval --tasks custom_longbench2_cot_le_32k --model hf --model_args pretrained=...
```

To regenerate the token annotations and bucket YAMLs:

```bash
conda run -n torch29 python -m lm_eval.tasks.custom_longbench2.qwen3_4b_it_tokenize_longbench_v2
```

### Citation

```bibtex
@inproceedings{bai2025longbenchv2,
  title = {LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
  author = {Bai, Yushi and Tu, Shangqing and Zhang, Jiajie and Peng, Hao and Wang, Xiaozhi and Lv, Xin and Cao, Shulin and Xu, Jiazheng and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2025},
  pages = {3639--3664},
  url = {https://aclanthology.org/2025.acl-long.183/}
}
```
