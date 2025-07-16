# ProcessBench-Style Evaluation for DeltaBench

This script (`deltabench_processbench.py`) adapts the ProcessBench methodology to evaluate DeltaBench data, focusing on **first-error identification accuracy** with voting mechanisms.

## Key Differences from DeltaBench Original

| Aspect | DeltaBench Original | ProcessBench Style |
|--------|--------------------|--------------------|
| **Focus** | All error sections | **First error only** |
| **Temperature** | 1.0 | **0.7** |
| **Voting** | Single prediction | **8-vote majority consensus** |
| **Metrics** | Precision/Recall/F1 across all errors | **Binary accuracy: error detection + correct recognition** |
| **Evaluation** | Section-by-section analysis | **First mistake identification** |

## Usage

```bash
python deltabench_processbench.py \
    --dataset Deltabench_v1 \
    --model gpt-4o-mini \
    --api-key YOUR_API_KEY \
    --n-votes 8 \
    --processes 4
```

## Arguments

- `--dataset`: Dataset name (e.g., `Deltabench_v1`)
- `--model`: OpenAI model name (default: `gpt-4o-mini`)
- `--api-key`: OpenAI API key (required)
- `--n-votes`: Number of votes for consensus (default: 8, as per ProcessBench)
- `--processes`: Number of parallel processes (default: 4)

## Data Format Conversion

The script automatically converts DeltaBench format to ProcessBench format:

- **DeltaBench sections** → **ProcessBench steps** (array of content)
- **reason_error_section_numbers[0]** → **ProcessBench label** (first error, 0-indexed)
- **No errors** → **label = -1** (ProcessBench convention)

## ProcessBench Template

Uses the exact critique template from ProcessBench reproduction guide:

```
The following is a problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Problem]
{problem}

[Solution]
{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1.

Please put your final answer (i.e., the index) in \boxed{}.
```

## Evaluation Metrics

ProcessBench-style metrics focus on binary classification:

- **Error Accuracy**: Accuracy on examples with errors (label ≠ -1)
- **Correct Accuracy**: Accuracy on examples without errors (label = -1)  
- **F1 Score**: F1 score between the two accuracies
- **By-task breakdown**: Results grouped by task_l1

## Output Files

- `results/processbench/{dataset}_{model}_votes{n}.jsonl`: Detailed results
- `results/processbench/{dataset}_{model}_votes{n}.csv`: Summary by task

## Testing

Run the conversion test:
```bash
python test_processbench_conversion.py
```

This verifies that DeltaBench data is properly converted to ProcessBench format.