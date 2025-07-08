# DeltaBench: Reasoning Critic Framework

A comprehensive research framework for evaluating LLM-based critics that detect errors in mathematical reasoning steps. Built for the DeltaBench dataset with support for multiple prompting strategies and detailed analysis tools.

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
   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Dataset:**
   The dataset files are in the `data/` directory (1,236 examples).

## Usage

### Basic Example

```python
from src import DeltaBenchDataset, LLMCritic, DeltaBenchEvaluator, plot_results

# Load dataset
dataset = DeltaBenchDataset()
dataset.load_jsonl('data/Deltabench_v1.jsonl')

# Initialize critic with different prompts
critic = LLMCritic(model_name="gpt-4o-mini", prompt_type="deltabench")  # or "pedcot"

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

### Analysis Example

```python
from src import display_example, display_critic_comparison, summarize_results

# Display examples
error_examples = dataset.get_examples_with_errors(limit=5)
display_example(error_examples[0], dataset)

# Compare critics
deltabench_critic = LLMCritic(prompt_type="deltabench")
pedcot_critic = LLMCritic(prompt_type="pedcot")

# Evaluate and compare
results = {"deltabench": deltabench_result, "pedcot": pedcot_result}
display_critic_comparison(example, results, dataset)
```

### Quick Test

```bash
python example.py          # Run basic evaluation on 5 examples
python critic_evaluation_40.py  # Comprehensive evaluation on 40 examples
jupyter notebook analysis_notebook.ipynb  # Full analysis and insights
```

## Project Structure

```
deltabench/
├── src/
│   ├── __init__.py         # Main exports
│   ├── config.py           # Configuration & prompts
│   ├── data_loader.py      # Dataset loading & parsing
│   ├── critic.py           # LLM critic with prompt types
│   ├── evaluator.py        # Evaluation engine
│   ├── analysis_utils.py   # Analysis & display tools
│   ├── visualizer.py       # Basic plots
│   └── utils.py            # Data structures
├── data/                   # Dataset files
│   ├── Deltabench_v1.jsonl
│   └── Deltabench_v1.csv
├── analysis_notebook.ipynb # Analysis & experiments
├── critic_evaluation_40.py # Comprehensive evaluation script
├── requirements.txt        # Dependencies
├── example.py             # Basic usage example
└── README.md              # This file
```

## Core Components

### Data Management (`src/data_loader.py`)
- **DeltaBenchDataset**: Load datasets (JSONL/CSV), parse sections, filter by task type
- **get_statistics()**: Dataset statistics and error distribution analysis
- **parse_sections()**: Extract individual reasoning sections from text
- **get_examples_with_errors()**: Filter examples containing reasoning errors

### Critic Implementation (`src/critic.py`)
- **LLMCritic**: Evaluate reasoning with multiple prompt strategies (OpenAI GPT)
- **prompt_type**: Support for "deltabench" and "pedcot" prompts
- **parse_output()**: Extract predictions and calculate precision/recall/F1 metrics
- **Token tracking**: Monitor API usage and costs

### Evaluation & Analysis (`src/evaluator.py`, `src/analysis_utils.py`)
- **DeltaBenchEvaluator**: Run evaluations with detailed results and progress tracking
- **display_example()**: Show examples with error highlighting and metadata
- **display_critic_comparison()**: Compare different critics side-by-side
- **summarize_results()**: Generate evaluation summaries with performance breakdowns
- **generate_dataset_insights()**: Create comprehensive markdown reports

## Key Features

- **Clean & Modular Architecture**: Each component is focused and well-structured
- **Multiple Prompting Strategies**: Easy comparison of different critic approaches
- **Rich Analysis Tools**: Comprehensive data exploration and error pattern analysis
- **Research Ready**: Detailed results storage, visualization, and reporting
- **Extensible Design**: Simple to add new prompts, models, or analysis methods
- **Performance Tracking**: Token usage monitoring and cost estimation
- **Comprehensive Evaluation**: Multi-metric assessment with statistical analysis

## Research Workflow

1. **Explore Data**: Use `analysis_notebook.ipynb` to understand the dataset structure and error patterns
2. **Quick Test**: Run `python example.py` to test basic functionality (5 examples)
3. **Comprehensive Evaluation**: Use `critic_evaluation_40.py` for detailed analysis (40 examples)
4. **Compare Critics**: Test DeltaBench vs PedCOT prompts on sample data
5. **Analyze Results**: Identify patterns in critic errors and performance bottlenecks
6. **Generate Reports**: Use analysis tools to create comprehensive evaluation summaries
7. **Iterate**: Refine prompts and strategies based on insights

## Dataset Details

The DeltaBench dataset contains:
- **1,236 mathematical reasoning examples** from various domains
- **Multi-step solutions** broken into numbered sections
- **Ground truth annotations** for error and unuseful sections
- **Task categories**: Number Theory, Algebra, Geometry, and more
- **Difficulty levels**: From basic to advanced mathematical reasoning