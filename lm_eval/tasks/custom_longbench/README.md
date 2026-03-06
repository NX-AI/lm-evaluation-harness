# Custom LongBench

This folder is a namespaced copy of `lm_eval/tasks/longbench` with two changes:

- All task/group/tag names are prefixed with `custom_longbench_*` to avoid collisions with the original harness tasks.
- A generative CoT variant is added (`custom_longbench_cot`) that prompts concise step-by-step reasoning and asks the model to end with `Answer: <text>`, while preserving task-specific constraints on the final answer span. The scorer strips optional `<think>...</think>` / `<analysis>...</analysis>` traces and uses the final `Answer:` span directly when present.

### Paper

Title: `LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding`

Abstract: `In this paper, we introduce LongBench, the first bilingual, multi-task benchmark for long context understanding, enabling a more rigorous evaluation of long context understanding. LongBench comprises 21 datasets across 6 task categories in both English and Chinese, with an average length of 6,711 words (English) and 13,386 characters (Chinese). These tasks cover key long-text application areas including single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, and code completion. All datasets in LongBench are standardized into a unified format, allowing for effortless automatic evaluation of LLMs`

Homepage: `https://github.com/THUDM/LongBench`

### Dataset

These tasks are configured to load from `zai-org/LongBench` (as requested). Because newer `datasets` versions no longer execute dataset scripts such as `LongBench.py`, the task configs load each split from the repo's `data.zip` via a local `custom_dataset` helper instead of calling `datasets.load_dataset(repo_id, name=...)` directly.


### Citation

```
@inproceedings{bai2024longbench,
    title = "{L}ong{B}ench: A Bilingual, Multitask Benchmark for Long Context Understanding",
    author = "Bai, Yushi and Lv, Xin  and Zhang, Jiajie  and Lyu, Hongchang  and
      Tang, Jiankai  and Huang, Zhidian  and Du, Zhengxiao  and Liu, Xiao  and Zeng, Aohan  and Hou, Lei  and Dong, Yuxiao  and Tang, Jie  and Li, Juanzi",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.172",
    doi = "10.18653/v1/2024.acl-long.172",
    pages = "3119--3137",
}
```
> [!NOTE]
> The original implementation suggest not to use `chat_template` for these tasks for instruct models (with add_bos_token=True but model dependent):
> - custom_longbench_fewshot
>    - custom_longbench_trec
>    - custom_longbench_triviaqa
>    - custom_longbench_samsum
>    - custom_longbench_lsht
> - custom_longbench_code
>   - custom_longbench_lcc
>   - custom_longbench_repobench-p


### Groups, Tags, and Tasks

#### Groups

* `custom_longbench_single`: Single-Document QA tasks requiring comprehension of individual documents
* `custom_longbench_multi`: Multi-Document QA tasks requiring information synthesis across multiple documents
* `custom_longbench_summarization`: Summarization tasks for long documents and conversations
* `custom_longbench_fewshot`: Few-shot learning tasks with in-context examples
* `custom_longbench_synthetic`: Synthetic tasks including passage retrieval and counting
* `custom_longbench_code`: Code completion tasks for long code contexts
* `custom_longbench_cot`: CoT-prompted generative variant for all tasks in this folder, with optional reasoning traces stripped before scoring
* `custom_longbench_cot_origprompt`: Original-prompt ablation that keeps the current extraction/scoring path while reusing the original LongBench prompts
* `custom_longbench_cot_en`: English-only CoT group that excludes the Chinese subtasks
* `custom_longbench_cot_en_base`: English-only CoT subgroup for the base LongBench tasks `NarrativeQA`, `Qasper`, `MultiFieldQA`, `HotpotQA`, `2WikiMultihopQA`, `Musique`, `GovReport`, `QMSum`, `MultiNews`, `TREC`, `TriviaQA`, `SAMSum`, `LCC`, and `RepoBench-P`
* `custom_longbench_cot_en_origprompt`: English-only original-prompt ablation group
* `custom_longbench_cot_en_base_origprompt`: English-only base-task original-prompt ablation subgroup

#### Tags

* `custom_longbench_tasks`: All baseline tasks in this folder
* `custom_longbench_tasks_e`: LongBench-E baseline tasks in this folder
* `custom_longbench_cot_tasks`: All CoT leaf tasks in this folder
* `custom_longbench_cot_origprompt_tasks`: All original-prompt ablation leaf tasks in this folder

#### Tasks

Same leaf tasks as the original `longbench` folder, but namespaced as `custom_longbench_*`, with CoT variants `custom_longbench_*_cot`, and with original-prompt ablation variants `custom_longbench_*_cot_origprompt` that keep the current extraction/scoring behavior.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
v2.: fix doc_to_target; add vcsum

v3: properly use all answers for metric calculation; trim whitespace from resps; fix stop sequences not parsing correctly.

v4: fixed special characters in prompts; use greedy decoding by default.

v5: folded `*_think` stripping behavior into `*_cot`; removed separate think task/group files.

v6: added `custom_longbench_cot_en` as an English-only CoT group.

v7: added `custom_longbench_cot_en_base` for the base English LongBench CoT subset.

v8: switched `custom_longbench` dataset loading to a local `data.zip` reader because modern `datasets` rejects LongBench dataset scripts.

v9: added `*_cot_origprompt` ablation tasks and matching group configs so the original LongBench prompts can be evaluated with the current extraction/scoring path.
