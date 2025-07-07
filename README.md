# DeltaBench: Simple Reasoning Critic Framework

A minimal framework for evaluating reasoning errors in step-by-step solutions using the DeltaBench dataset.

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Place dataset:**
   Put `Deltabench_v1.jsonl` in the root directory.

## Usage

### Basic Example

```python
from src import DeltaBenchDataset, LLMCritic, DeltaBenchEvaluator, plot_results

# Load dataset
dataset = DeltaBenchDataset()
dataset.load_jsonl('Deltabench_v1.jsonl')

# Initialize critic
critic = LLMCritic(model_name="gpt-4o-mini")

# Create evaluator
evaluator = DeltaBenchEvaluator(dataset, critic)

# Run evaluation
results = evaluator.evaluate_dataset(num_examples=10)

# Calculate metrics
metrics = evaluator.calculate_metrics(results)
print(f"F1 Score: {metrics['f1_mean']:.3f}")

# Plot results
plot_results(results)
```

### Quick Test

```bash
python test_simplified.py  # Test without API key
python example.py          # Run with API key
```

## Project Structure

```
deltabench/
├── src/
│   ├── __init__.py         # Main exports
│   ├── config.py           # Configuration
│   ├── data_loader.py      # Dataset loading
│   ├── critic.py           # LLM critic
│   ├── evaluator.py        # Evaluation engine
│   ├── visualizer.py       # Basic plots
│   └── utils.py            # Data structures
├── requirements.txt        # Dependencies
├── example.py             # Usage example
└── README.md              # This file
```

## Core Components

- **DeltaBenchDataset**: Load JSONL/CSV datasets
- **LLMCritic**: Evaluate reasoning with OpenAI models
- **DeltaBenchEvaluator**: Run evaluations and calculate metrics
- **plot_results**: Basic visualization

## What Was Simplified

Removed from original codebase:
- Verbose logging and print statements
- Complex configuration management
- Dataset exploration utilities
- Directory creation logic
- Extensive error handling
- Multiple visualization styles
- Jupyter notebook dependencies

Kept only the essential functionality for research experiments.