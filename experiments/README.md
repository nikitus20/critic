# DeltaBench Experiments

This folder contains two self-contained experimental scripts that reproduce key findings about DeltaBench evaluation methodology.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API key:** You'll need a valid OpenAI API key for the original reproduction experiment.

## Experiments

### 1. Original DeltaBench Reproduction

**Purpose:** Reproduces the exact methodology from the original DeltaBench paper.

**Usage:**
```bash
python deltabench_original.py --call_modelname gpt-4o-mini --dataset Deltabench_v1 --api-key YOUR_API_KEY_HERE
```

**What it does:**
- Uses the original DeltaBench prompt and evaluation logic
- Calls OpenAI API to evaluate reasoning steps
- Saves results to `results/original/`
- Outputs detailed metrics matching the paper

**Example output:**
```
Results saved to: results/original/Deltabench_v1_gpt-4o-mini.jsonl
Metrics saved to: results/original/Deltabench_v1_gpt-4o-mini.csv
```

### 2. Trivial Baseline (Evaluation Flaw Demo)

**Purpose:** Demonstrates a critical flaw in DeltaBench's evaluation methodology.

**Usage:**
```bash
python deltabench_trivial.py --dataset data/Deltabench_v1.jsonl
```

**What it does:**
- Always predicts sections 3-36 as error sections (no reasoning)
- Achieves ~47% F1 score by gaming the filtering logic
- **No API calls needed** - pure prediction strategy
- Saves results to `results/trivial/`

**Key insight:** This trivial approach outperforms sophisticated reasoning critics, exposing that the evaluation rewards prediction strategies rather than actual reasoning quality.

## Results

Both experiments save results in their respective folders:
- `results/original/` - Original DeltaBench reproduction results
- `results/trivial/` - Trivial baseline results

Each produces:
- `.jsonl` file with detailed per-example results
- `.csv` file with summary metrics
- Console output with performance breakdown

## Understanding the Results

**Original reproduction** shows how the intended DeltaBench methodology performs.

**Trivial baseline** reveals that a critic with zero reasoning ability can achieve high performance by exploiting evaluation logic flaws.

The comparison demonstrates important limitations in the DeltaBench evaluation framework that should be considered when interpreting results.

## Notes

- Both experiments are completely self-contained
- No dependencies on the main framework (`src/` folder)
- Dataset (`data/Deltabench_v1.jsonl`) is included
- Scripts are designed for one-time execution and analysis