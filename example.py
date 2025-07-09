#!/usr/bin/env python3
"""
Simple example of using DeltaBench with different critics.

Before running: export OPENAI_API_KEY="your-key-here"

Usage:
    python example.py                    # Uses DirectCritic (default)
    python example.py --critic pedcot    # Uses PedCOT critic
    python example.py --critic direct    # Uses DirectCritic explicitly
"""

import sys
import argparse
sys.path.append('src')

from src import DeltaBenchDataset, LLMCritic, DeltaBenchEvaluator, plot_results, CriticFactory

def main():
    parser = argparse.ArgumentParser(description='Run DeltaBench evaluation example')
    parser.add_argument('--critic', default='direct', choices=['direct', 'pedcot'], 
                       help='Type of critic to use (default: direct)')
    parser.add_argument('--examples', type=int, default=5, 
                       help='Number of examples to evaluate (default: 5)')
    parser.add_argument('--model', default='gpt-4o-mini', 
                       help='Model to use (default: gpt-4o-mini)')
    
    args = parser.parse_args()
    
    print(f"🔬 DeltaBench Example with {args.critic.upper()} critic")
    print(f"📊 Model: {args.model}, Examples: {args.examples}")
    print("=" * 50)
    
    # Load dataset
    print("📁 Loading dataset...")
    dataset = DeltaBenchDataset()
    dataset.load_jsonl('data/Deltabench_v1.jsonl')
    print(f"✅ Dataset loaded: {len(dataset.data)} examples")
    
    # Initialize critic using factory
    print(f"🤖 Creating {args.critic} critic...")
    if args.critic == 'direct':
        # Use LLMCritic for backward compatibility
        critic = LLMCritic(model_name=args.model)
    else:
        # Use factory for PedCOT
        critic = CriticFactory.create_critic(args.critic, args.model)
    
    print(f"✅ {critic.__class__.__name__} created")
    
    # Create evaluator
    evaluator = DeltaBenchEvaluator(dataset, critic)
    
    # Run evaluation
    print(f"🔄 Running evaluation on {args.examples} examples...")
    results = evaluator.evaluate_dataset(num_examples=args.examples)
    
    if len(results) == 0:
        print("❌ No results obtained!")
        return
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    
    # Show results
    print("\n📊 Results:")
    print(f"   F1 Score (macro): {metrics.get('f1_macro', 0):.3f}")
    print(f"   Precision (macro): {metrics.get('precision_macro', 0):.3f}")
    print(f"   Recall (macro): {metrics.get('recall_macro', 0):.3f}")
    print(f"   Examples processed: {len(results)}")
    print(f"   Average tokens: {metrics.get('avg_tokens', 0):.0f}")
    
    # Show critic-specific info
    if args.critic == 'pedcot':
        print("\n🎓 PedCOT-specific info:")
        pedcot_examples = results[results['prompt_type'] == 'pedcot'] if 'prompt_type' in results.columns else results
        if len(pedcot_examples) > 0:
            print(f"   Two-stage evaluations: {len(pedcot_examples)}")
            print("   Uses pedagogical principles (Remember-Understand-Apply)")
    
    # Plot results
    try:
        plot_results(results)
        print("📈 Results visualization saved")
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

if __name__ == "__main__":
    main()