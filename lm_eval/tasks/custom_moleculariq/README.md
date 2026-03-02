# MolecularIQ Benchmark Task

MolecularIQ is a comprehensive chemistry benchmark for evaluating language models on molecular understanding tasks. This implementation provides Pass@k metrics.

## Dataset

- **Dataset**: `tschouis/moleculariq_arxiv`
- **Split**: test
- **Columns**: `uid`, `task_type`, `features`, `question`, `target`, `constraints`, `original_smiles`, `complexity_bin`, `multi_task_load`, `metadata`

## Task Types

The benchmark includes 5111 samples across three task types:

1. **Count** (1800 samples): Counting molecular features (e.g. rotatable bonds, sp3 carbons, rings, halogens). Includes single-property and multi-property questions.

2. **Index** (1740 samples): Identifying atom indices for specific features (e.g. longest carbon chain, stereocenters, branch points). Includes single-feature and multi-feature questions.

3. **Generation** (1571 samples): Generating molecules (SMILES) that satisfy given constraints (e.g. reaction feasibility, structural properties).

## Metrics

- **Pass@1**: Accuracy on first attempt
- **Pass@3**: Any correct answer in first 3 attempts
- **avg_accuracy**: Average accuracy across all attempts

## Available Tasks

| Task | Description | Repeats |
|------|-------------|---------|
| `moleculariq_pass_at_k` | System prompt via `description` field + raw question (for chat models) | 3 |
| `moleculariq_inline` | Question with inline prompt baked into the user message (for base models) | 3 |

**Key difference**: `moleculariq_pass_at_k` uses the YAML `description` field as a system prompt, which is automatically applied by lm-eval for chat-templated models. `moleculariq_inline` embeds all instructions directly in the user message, making it suitable for base models without a chat template.

## Usage

### Chat Models (Recommended: `moleculariq_pass_at_k`)

The `description` field in the YAML is automatically used as the system prompt for chat-templated models. No need for `--system_instruction`.

```bash
lm-eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks moleculariq_pass_at_k \
    --batch_size auto \
    --output_path ./results
```

With vLLM:

```bash
lm-eval --model vllm \
    --model_args pretrained=your-model-name \
    --tasks moleculariq_pass_at_k \
    --batch_size auto \
    --output_path ./results
```

### Base Models (Use `moleculariq_inline`)

For models without a chat template, the inline variant bakes instructions into the prompt:

```bash
lm-eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks moleculariq_inline \
    --batch_size auto \
    --output_path ./results
```

### Quick Test (Limited Samples)

```bash
lm-eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
    --tasks moleculariq_pass_at_k \
    --limit 10 \
    --batch_size auto \
    --output_path ./results
```

### Passing Extra Chat Template Args

You can pass additional kwargs (e.g. `reasoning_effort`) to the chat template via `chat_template_args` in `model_args`:

```bash
lm-eval --model hf \
    --model_args '{"pretrained":"openai/gpt-oss-20b","chat_template_args":{"reasoning_effort":"low"}}' \
    --tasks moleculariq_pass_at_k \
    --batch_size auto \
    --output_path ./results
```

## Directory Structure

```
moleculariq/
├── __init__.py                    # Package init
├── moleculariq_pass_at_k.yaml     # Task with system prompt via description (for chat models)
├── moleculariq_inline.yaml        # Task with inline prompt (for base models)
├── task_processor.py              # Processing hooks (uses moleculariq_core)
├── extractors.py                  # Answer extraction functions
└── README.md
```

## Dependencies

- `moleculariq_core`: Core library for molecular reasoning and reward computation
- `rdkit`: Chemistry toolkit (dependency of moleculariq_core)
- `datasets`: For loading the HuggingFace dataset

Install dependencies:
```bash
pip install moleculariq-core rdkit
```

## Answer Format

Models should return answers in JSON format within answer tags:

```
<answer>{"property_name": value}</answer>
```

For count tasks:
```
<answer>{"ring_count": 2}</answer>
```

For index tasks:
```
<answer>{"carbon_indices": [0, 1, 3]}</answer>
```

For constraint generation:
```
<answer>{"smiles": "CCO"}</answer>
```


## Atom Indexing Convention

Atoms are indexed from 0 to N-1, reading the SMILES string left to right, counting only heavy atoms (non-hydrogen). Examples:

- `"CCO"`: C(0), C(1), O(2)
- `"CC(C)O"`: C(0), C(1), C(2), O(3)
- `"CC(=O)N"`: C(0), C(1), O(2), N(3)

## Customization

### Custom Extraction via Filters

The lm-eval `filter_list` mechanism allows users to customize output extraction. Users can create a task variant YAML:

```yaml
# my_model_moleculariq.yaml
include: moleculariq_pass_at_k.yaml
task: my_model_moleculariq

filter_list:
  - name: my_extraction
    filter:
      - function: custom
        filter_fn: !function my_extractors.extract_my_format
```

### Custom Preprocessing via `doc_to_text`

Users can customize prompt preprocessing by overriding `doc_to_text`:

```yaml
# my_model_moleculariq.yaml
include: moleculariq_pass_at_k.yaml
task: my_model_moleculariq

doc_to_text: !function my_utils.custom_doc_to_text
```

### Using Regex Filters

For simple pattern extraction, use the built-in regex filter:

```yaml
filter_list:
  - name: extract_answer
    filter:
      - function: regex
        regex_pattern: r"<my_answer>(.*?)</my_answer>"
        group_select: 0
        fallback: ""
```

## Citation

If you use this benchmark, please cite the MolecularIQ paper:

```bibtex
@article{bartmann2026moleculariq,
  title={MolecularIQ: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs},
  author={Bartmann, Christoph and Schimunek, Johannes and Ielanskyi, Mykyta and Seidl, Philipp and Klambauer, G{\"u}nter and Luukkonen, Sohvi},
  journal={arXiv preprint arXiv:2601.15279},
  year={2026}
}
```
