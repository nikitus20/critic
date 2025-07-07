#!/usr/bin/env python3
"""
Simple example of using DeltaBench.

Before running: export OPENAI_API_KEY="your-key-here"
"""

import sys
sys.path.append('src')

from src import DeltaBenchDataset, LLMCritic, DeltaBenchEvaluator, plot_results

def main():
    # Load dataset
    dataset = DeltaBenchDataset()
    dataset.load_jsonl('Deltabench_v1.jsonl')
    
    # Initialize critic
    critic = LLMCritic(model_name="gpt-4o-mini")
    
    # Create evaluator
    evaluator = DeltaBenchEvaluator(dataset, critic)
    
    # Run evaluation on first 5 examples
    print("Running evaluation on 5 examples...")
    results = evaluator.evaluate_dataset(num_examples=5)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    
    # Show results
    print("\n📊 Results:")
    print(f"F1 Score: {metrics['f1_mean']:.3f}")
    print(f"Precision: {metrics['precision_mean']:.3f}")
    print(f"Recall: {metrics['recall_mean']:.3f}")
    print(f"Avg Tokens: {metrics['avg_tokens']:.0f}")
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()