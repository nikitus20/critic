# DeltaBench: Reasoning Critic Framework

A comprehensive research framework for evaluating LLM-based critics that detect errors in mathematical reasoning steps. This implementation compares two core approaches: **DeltaBench Direct** (single-stage prompting) and **PedCOT** (pedagogical chain-of-thought with two-stage evaluation).

## 🎯 Project Overview

This framework implements and compares two approaches for detecting reasoning errors:

### **DirectCritic (DeltaBench Approach)**
- **Single-stage evaluation**: Analyzes complete reasoning chain in one prompt
- **Holistic analysis**: Sees all sections simultaneously for context
- **Efficient**: ~500-1000 tokens per evaluation
- **Output**: Direct error section identification

### **PedCoTCritic (Pedagogical Approach)** 
- **Two-stage process**: Regenerate ideal reasoning, then compare with actual
- **Section-by-section**: Individual analysis with sequential context
- **Pedagogical principles**: Based on Bloom's taxonomy (Remember-Understand-Apply)
- **Domain-aware**: Specialized handling for math, programming, science, general domains
- **Detailed**: ~2000-3000 tokens per evaluation

## 📁 Project Structure

```
deltabench/
├── src/                           # Core framework
│   ├── critics/                   # Critic implementations
│   │   ├── direct_critic.py       # DeltaBench direct approach
│   │   ├── pedcot_critic.py       # PedCOT pedagogical approach
│   │   ├── base_critic.py         # Abstract base class
│   │   ├── critic_factory.py      # Factory for creating critics
│   │   └── pedcot/                # PedCOT-specific components
│   │       ├── pedagogical_principles.py  # Bloom's taxonomy implementation
│   │       ├── tip_processor.py          # Two-stage interaction process
│   │       └── error_mapping.py          # Error taxonomy mapping
│   ├── prompts/                   # Prompt templates
│   ├── config.py                  # Configuration and prompts
│   ├── evaluator.py              # Evaluation engine
│   ├── data_loader.py            # Dataset handling
│   └── analysis_utils.py         # Analysis and visualization tools
├── data/                         # DeltaBench dataset (1,236 examples)
├── results/                      # Evaluation outputs and metrics
├── docs/                         # Documentation and analysis reports
├── example.py                    # Quick usage example
├── critic_evaluation_40.py       # Comprehensive evaluation script
└── analysis_notebook.ipynb       # Full analysis and insights
```

## 🚀 Quick Start

### 1. Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-api-key-here"
# Or create a .env file with: OPENAI_API_KEY=your-api-key-here
```

### 2. Basic Usage

```python
from src import DeltaBenchDataset, DeltaBenchEvaluator, create_critic

# Load dataset
dataset = DeltaBenchDataset()
dataset.load_jsonl('data/Deltabench_v1.jsonl')

# Compare both approaches
direct_critic = create_critic('direct', model='gpt-4o-mini')
pedcot_critic = create_critic('pedcot', model='gpt-4o-mini')

# Run evaluations
evaluator = DeltaBenchEvaluator(dataset, direct_critic)
direct_results = evaluator.evaluate_dataset(num_examples=10)

evaluator = DeltaBenchEvaluator(dataset, pedcot_critic)
pedcot_results = evaluator.evaluate_dataset(num_examples=10)

# Compare metrics
print(f"Direct F1: {evaluator.calculate_metrics(direct_results)['f1_mean']:.3f}")
print(f"PedCOT F1: {evaluator.calculate_metrics(pedcot_results)['f1_mean']:.3f}")
```

### 3. Command Line Usage

```bash
# Quick test with DirectCritic
python example.py --critic direct --examples 5

# Test with PedCOT
python example.py --critic pedcot --examples 5

# Comprehensive evaluation
python critic_evaluation_40.py --critic direct
python critic_evaluation_40.py --critic pedcot

# Full analysis
jupyter notebook analysis_notebook.ipynb
```

## 🔬 Core Components

### **DirectCritic Implementation**
- Uses original DeltaBench prompt format
- Single LLM call per example
- Evaluates entire reasoning chain holistically
- Simple section-by-section error identification

### **PedCOTCritic Implementation**
- **Stage 1**: Regenerate expected reasoning using pedagogical principles
- **Stage 2**: Compare actual vs expected across Remember-Understand-Apply dimensions
- Domain detection (math/programming/science/general)
- Error mapping from pedagogical principles to DeltaBench taxonomy

### **Evaluation Pipeline**
- Unified interface supporting both critics
- Precision/Recall/F1 metrics calculation
- Task-specific performance analysis
- Token usage and cost tracking

## 📊 Dataset Details

The DeltaBench dataset contains:
- **1,236 mathematical reasoning examples** from various domains
- **Multi-step solutions** broken into numbered sections
- **Ground truth annotations** for error and unuseful sections
- **Task categories**: Math (45.5%), Code (30.2%), Physics/Chemistry/Biology (12.5%), General (11.9%)
- **Error complexity**: 60.2% of examples have multiple errors

## 📈 Results and Analysis

### **Performance Comparison**
Results are automatically saved to `results/` directory:
- Raw evaluation data in JSONL format
- Metrics summaries in CSV format
- Task-specific performance breakdowns

### **Key Findings** (from preliminary evaluation)
- Both approaches show task-specific performance variations
- Math problems consistently show lower performance than other domains
- PedCOT provides more detailed error analysis but at higher computational cost
- DirectCritic is more efficient for large-scale evaluation

## 🛠️ Advanced Usage

### **Custom Critic Configuration**
```python
from src import CriticFactory

# Custom PedCOT configuration
pedcot_config = {
    'principle_weighting': {'remember': 0.3, 'understand': 0.4, 'apply': 0.3},
    'error_mapping_strategy': 'weighted_consensus',
    'confidence_threshold': 0.5
}

critic = CriticFactory.create_critic('pedcot', 'gpt-4o-mini', pedcot_config)
```

### **Analysis Tools**
```python
from src import display_example, display_critic_comparison, summarize_results

# Display specific examples with error highlighting
display_example(dataset.data[0], dataset)

# Compare critics side-by-side
display_critic_comparison(example, {'direct': direct_result, 'pedcot': pedcot_result}, dataset)

# Generate comprehensive analysis
summarize_results(results, output_file='analysis_report.md')
```

## 📖 Documentation

- **[Development Notes](docs/development_notes.md)**: Current project status and implementation details
- **[Dataset Analysis](docs/dataset_analysis.md)**: Comprehensive dataset statistics and insights
- **[PedCOT Implementation Plan](docs/pedcot_implementation_plan.md)**: Detailed implementation roadmap
- **[Original Evaluation](docs/original_evaluation.md)**: Reference implementation from the original paper

## 🔍 Research Focus

This framework is designed for researchers comparing:
- **Direct vs Pedagogical prompting** for error detection
- **Single-stage vs Two-stage evaluation** approaches  
- **Domain-specific performance** across mathematical reasoning tasks
- **Computational efficiency vs Analysis depth** trade-offs

## 🚨 Known Limitations

- PedCOT approach requires 2-3x more API calls than DirectCritic
- Domain detection relies on keyword matching (could be improved with classification models)
- Error mapping between pedagogical principles and DeltaBench taxonomy is heuristic-based
- Current implementation focuses on mathematical reasoning (extensible to other domains)

## 📝 Citation

If you use this framework in your research, please cite:
- Original DeltaBench paper for the dataset and baseline approach
- PedCOT paper for the pedagogical methodology
- This implementation for the comparative framework

## 🤝 Contributing

The framework is designed to be extensible:
- Add new critic types by inheriting from `BaseCritic`
- Extend domain coverage in `pedagogical_principles.py`
- Add new error types in `error_mapping.py`
- Contribute analysis tools in `analysis_utils.py`

---

*Framework developed for comparing direct and pedagogical approaches to mathematical reasoning error detection.*