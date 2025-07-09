#!/usr/bin/env python3
"""
Evaluate DeltaBench critic on 40 examples and provide comprehensive analysis.
"""

import sys
sys.path.append('src')
import os
from dotenv import load_dotenv
import pandas as pd
import random
import numpy as np
from collections import Counter

# Load environment
load_dotenv('.env')

from src import DeltaBenchDataset, LLMCritic, DeltaBenchEvaluator

def main():
    print('=== DELTABENCH CRITIC EVALUATION (40 Examples) ===\n')
    
    # Load dataset
    dataset = DeltaBenchDataset()
    data = dataset.load_jsonl('data/Deltabench_v1.jsonl')
    print(f'Dataset loaded: {len(data)} examples')
    
    # Create reproducible sample
    random.seed(42)
    sample_indices = random.sample(range(len(data)), 40)
    
    # Initialize critic
    critic = LLMCritic(model_name='gpt-4o-mini', prompt_type='deltabench')
    evaluator = DeltaBenchEvaluator(dataset, critic)
    print('Critic initialized\n')
    
    # Run evaluation
    print('🚀 Running evaluation...')
    results = []
    total_tokens = 0
    
    for i, idx in enumerate(sample_indices):
        if (i + 1) % 10 == 0:
            print(f'  Completed {i + 1}/40 examples')
        
        try:
            example = dataset.data[idx]
            result = evaluator.evaluate_example(example)
            
            if result and result.get('result'):
                critic_result = result['result']
                
                results.append({
                    'example_idx': idx,
                    'task_l1': example.get('task_l1', 'unknown'),
                    'true_sections': example.get('reason_error_section_numbers', []),
                    'predicted_sections': critic_result.predicted_error_sections,
                    'precision': critic_result.precision,
                    'recall': critic_result.recall,
                    'f1_score': critic_result.f1_score,
                    'true_errors': len(example.get('reason_error_section_numbers', [])),
                    'predicted_errors': len(critic_result.predicted_error_sections),
                    'tokens_used': result.get('token_info', {}).get('total_tokens', 0)
                })
                
                total_tokens += result.get('token_info', {}).get('total_tokens', 0)
                
        except Exception as e:
            print(f'  Error on example {i+1}: {str(e)[:50]}...')
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # ANALYSIS
    print(f'\n=== COMPREHENSIVE ANALYSIS ===')
    print(f'Successfully evaluated: {len(results_df)}/40 examples\n')
    
    # Basic Performance Metrics
    print('📊 OVERALL PERFORMANCE:')
    print(f'  Mean F1 Score:    {results_df["f1_score"].mean():.3f} ± {results_df["f1_score"].std():.3f}')
    print(f'  Mean Precision:   {results_df["precision"].mean():.3f} ± {results_df["precision"].std():.3f}')
    print(f'  Mean Recall:      {results_df["recall"].mean():.3f} ± {results_df["recall"].std():.3f}')
    
    # Performance Distribution
    print(f'\n📈 PERFORMANCE DISTRIBUTION:')
    perfect = (results_df['f1_score'] == 1.0).sum()
    good = (results_df['f1_score'] >= 0.75).sum()
    moderate = ((results_df['f1_score'] >= 0.25) & (results_df['f1_score'] < 0.75)).sum()
    poor = (results_df['f1_score'] < 0.25).sum()
    failed = (results_df['f1_score'] == 0.0).sum()
    
    print(f'  Perfect predictions (F1=1.0):     {perfect}/40 ({perfect/40*100:.1f}%)')
    print(f'  Good predictions (F1≥0.75):       {good}/40 ({good/40*100:.1f}%)')
    print(f'  Moderate predictions (0.25≤F1<0.75): {moderate}/40 ({moderate/40*100:.1f}%)')
    print(f'  Poor predictions (F1<0.25):       {poor}/40 ({poor/40*100:.1f}%)')
    print(f'  Failed predictions (F1=0.0):      {failed}/40 ({failed/40*100:.1f}%)')
    
    # Error Detection Analysis
    print(f'\n🎯 ERROR DETECTION ANALYSIS:')
    total_true_errors = results_df['true_errors'].sum()
    total_predicted_errors = results_df['predicted_errors'].sum()
    print(f'  Total true errors:      {total_true_errors}')
    print(f'  Total predicted errors: {total_predicted_errors}')
    print(f'  Over/Under prediction:  {total_predicted_errors - total_true_errors:+d} ({(total_predicted_errors/total_true_errors - 1)*100:+.1f}%)')
    
    # Calculate aggregate precision/recall
    # True positives across all examples
    total_tp = sum(len(set(row['true_sections']) & set(row['predicted_sections'])) 
                   for _, row in results_df.iterrows())
    total_fp = total_predicted_errors - total_tp
    total_fn = total_true_errors - total_tp
    
    aggregate_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    aggregate_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    print(f'  Aggregate Precision:    {aggregate_precision:.3f}')
    print(f'  Aggregate Recall:       {aggregate_recall:.3f}')
    
    # Task type breakdown
    print(f'\n🏷️  PERFORMANCE BY TASK TYPE:')
    for task in sorted(results_df['task_l1'].unique()):
        task_data = results_df[results_df['task_l1'] == task]
        print(f'  {task}: n={len(task_data)}, F1={task_data["f1_score"].mean():.3f}±{task_data["f1_score"].std():.3f}')
    
    # Error complexity analysis
    print(f'\n📝 PERFORMANCE BY ERROR COMPLEXITY:')
    results_df['error_category'] = results_df['true_errors'].apply(
        lambda x: 'No errors' if x == 0 else ('Single error' if x == 1 else 'Multiple errors')
    )
    
    for category in ['No errors', 'Single error', 'Multiple errors']:
        cat_data = results_df[results_df['error_category'] == category]
        if len(cat_data) > 0:
            print(f'  {category}: n={len(cat_data)}, F1={cat_data["f1_score"].mean():.3f}±{cat_data["f1_score"].std():.3f}')
    
    # Resource usage
    print(f'\n💰 RESOURCE USAGE:')
    print(f'  Total tokens used:    {total_tokens:,}')
    print(f'  Average per example:  {total_tokens/len(results_df):.0f} tokens')
    print(f'  Token efficiency:     {results_df["f1_score"].mean()*1000/results_df["tokens_used"].mean():.2f} F1 per 1k tokens')
    
    # Cost estimation
    cost_per_1k_tokens = 0.00015  # gpt-4o-mini cost
    estimated_cost = total_tokens * cost_per_1k_tokens / 1000
    print(f'  Estimated cost:       ${estimated_cost:.3f} USD')
    
    # Error pattern analysis
    print(f'\n🔍 ERROR PATTERN ANALYSIS:')
    
    # Most commonly predicted error sections
    all_predicted = []
    for sections in results_df['predicted_sections']:
        all_predicted.extend(sections)
    
    if all_predicted:
        predicted_counter = Counter(all_predicted)
        print(f'  Most predicted sections: {dict(predicted_counter.most_common(5))}')
    
    # Most commonly missed sections  
    all_missed = []
    for _, row in results_df.iterrows():
        true_set = set(row['true_sections'])
        pred_set = set(row['predicted_sections'])
        missed = true_set - pred_set
        all_missed.extend(missed)
    
    if all_missed:
        missed_counter = Counter(all_missed)
        print(f'  Most missed sections: {dict(missed_counter.most_common(5))}')
    
    # Save results
    results_df.to_csv('results/critic_evaluation_40_results.csv', index=False)
    print(f'\n📁 Results saved to: results/critic_evaluation_40_results.csv')
    
    # Key insights
    print(f'\n💡 KEY INSIGHTS:')
    print(f'  • Critic tends to {"over" if total_predicted_errors > total_true_errors else "under"}-predict errors')
    print(f'  • {failed/40*100:.1f}% complete failures suggest room for prompt improvement')
    print(f'  • Best performance on {results_df.groupby("task_l1")["f1_score"].mean().idxmax()} tasks')
    print(f'  • Token usage is {"high" if results_df["tokens_used"].mean() > 5000 else "moderate"} at {results_df["tokens_used"].mean():.0f} tokens/example')

if __name__ == '__main__':
    main()