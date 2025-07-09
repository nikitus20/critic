#!/usr/bin/env python3
"""
Comprehensive evaluation of DeltaBench critic on 100 samples.
Includes detailed metrics, error type analysis, runtime, and cost tracking.
"""

import sys
sys.path.append('src')
import os
import time
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from dotenv import load_dotenv

# Load environment
load_dotenv('.env')

from src import DeltaBenchDataset, LLMCritic, DeltaBenchEvaluator

def categorize_error_complexity(true_errors):
    """Categorize examples by error complexity."""
    if len(true_errors) == 0:
        return "No Errors"
    elif len(true_errors) == 1:
        return "Single Error"
    elif len(true_errors) <= 3:
        return "Multiple Errors"
    else:
        return "Complex Errors"

def categorize_error_position(true_errors, total_sections):
    """Categorize errors by position in reasoning."""
    if not true_errors:
        return "No Errors"
    
    early_errors = [e for e in true_errors if e <= 3]
    late_errors = [e for e in true_errors if e > max(1, total_sections - 3)]
    
    if early_errors and late_errors:
        return "Mixed Position"
    elif early_errors:
        return "Early Errors"
    elif late_errors:
        return "Late Errors"
    else:
        return "Middle Errors"

def main():
    print('=== DELTABENCH CRITIC EVALUATION (100 SAMPLES) ===\n')
    
    # Load dataset
    print('📁 Loading dataset...')
    dataset = DeltaBenchDataset()
    data = dataset.load_jsonl('data/Deltabench_v1.jsonl')
    print(f'   Dataset loaded: {len(data)} examples')
    
    # Get dataset statistics
    stats = dataset.get_statistics()
    print(f'   Examples with errors: {stats["examples_with_errors"]} ({stats["error_rate"]:.1%})')
    
    # Create reproducible sample of 100 examples
    random.seed(42)
    sample_indices = random.sample(range(len(data)), 100)
    print(f'   Selected 100 random examples (seed=42)\n')
    
    # Initialize critic and evaluator
    print('🤖 Initializing critic...')
    critic = LLMCritic(model_name='gpt-4o-mini', prompt_type='deltabench')
    evaluator = DeltaBenchEvaluator(dataset, critic)
    print('   Critic initialized successfully\n')
    
    # Track metrics
    results = []
    total_tokens = 0
    start_time = time.time()
    failed_evaluations = 0
    
    # Error categorization
    complexity_metrics = defaultdict(list)
    position_metrics = defaultdict(list)
    task_metrics = defaultdict(list)
    
    print('🚀 Running evaluation...')
    print('   Progress: ', end='', flush=True)
    
    for i, idx in enumerate(sample_indices):
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f'{i + 1}', end=' ', flush=True)
        elif (i + 1) % 5 == 0:
            print('.', end='', flush=True)
        
        try:
            example = dataset.data[idx]
            result = evaluator.evaluate_example(example)
            
            if result and result.get('result'):
                critic_result = result['result']
                
                # Basic metrics
                true_errors = example.get('reason_error_section_numbers', [])
                unuseful_errors = example.get('reason_unuseful_section_numbers', [])
                all_true_errors = list(set(true_errors + unuseful_errors))
                
                # Get total sections for position analysis
                sections_content = example.get('sections_content', '') or example.get('section_content', '')
                total_sections = len(dataset.parse_sections(sections_content)) if sections_content else 10
                
                # Categorize errors
                complexity_cat = categorize_error_complexity(all_true_errors)
                position_cat = categorize_error_position(all_true_errors, total_sections)
                task_type = example.get('task_l1', 'Unknown')
                
                # Store result
                result_dict = {
                    'example_idx': idx,
                    'example_id': example.get('id', ''),
                    'task_type': task_type,
                    'complexity_category': complexity_cat,
                    'position_category': position_cat,
                    'true_sections': all_true_errors,
                    'predicted_sections': critic_result.predicted_error_sections,
                    'precision': critic_result.precision,
                    'recall': critic_result.recall,
                    'f1_score': critic_result.f1_score,
                    'true_errors': len(all_true_errors),
                    'predicted_errors': len(critic_result.predicted_error_sections),
                    'tokens_used': result.get('token_info', {}).get('total_tokens', 0),
                    'total_sections': total_sections
                }
                results.append(result_dict)
                
                # Track by category
                complexity_metrics[complexity_cat].append(critic_result.f1_score)
                position_metrics[position_cat].append(critic_result.f1_score)
                task_metrics[task_type].append(critic_result.f1_score)
                
                total_tokens += result.get('token_info', {}).get('total_tokens', 0)
                
            else:
                failed_evaluations += 1
                
        except Exception as e:
            print(f'\n   Error on example {i+1}: {str(e)[:50]}...')
            failed_evaluations += 1
            continue
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f'\n\n✅ Evaluation completed!')
    print(f'   Successfully evaluated: {len(results)}/100 examples')
    print(f'   Failed evaluations: {failed_evaluations}')
    print(f'   Total runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)')
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("❌ No results to analyze")
        return
    
    # === COMPREHENSIVE ANALYSIS ===
    print(f'\n{"="*60}')
    print('📊 COMPREHENSIVE ANALYSIS REPORT')
    print(f'{"="*60}\n')
    
    # Overall Performance
    print('🎯 OVERALL PERFORMANCE:')
    print(f'   Mean F1 Score:    {results_df["f1_score"].mean():.3f} ± {results_df["f1_score"].std():.3f}')
    print(f'   Mean Precision:   {results_df["precision"].mean():.3f} ± {results_df["precision"].std():.3f}')
    print(f'   Mean Recall:      {results_df["recall"].mean():.3f} ± {results_df["recall"].std():.3f}')
    print(f'   Median F1 Score:  {results_df["f1_score"].median():.3f}')
    
    # Performance Distribution
    print(f'\n📈 PERFORMANCE DISTRIBUTION:')
    perfect = (results_df['f1_score'] == 1.0).sum()
    excellent = ((results_df['f1_score'] >= 0.8) & (results_df['f1_score'] < 1.0)).sum()
    good = ((results_df['f1_score'] >= 0.6) & (results_df['f1_score'] < 0.8)).sum()
    moderate = ((results_df['f1_score'] >= 0.3) & (results_df['f1_score'] < 0.6)).sum()
    poor = ((results_df['f1_score'] > 0.0) & (results_df['f1_score'] < 0.3)).sum()
    failed = (results_df['f1_score'] == 0.0).sum()
    
    total_evaluated = len(results_df)
    print(f'   Perfect (F1=1.0):       {perfect:2d} ({perfect/total_evaluated*100:4.1f}%)')
    print(f'   Excellent (0.8≤F1<1.0): {excellent:2d} ({excellent/total_evaluated*100:4.1f}%)')
    print(f'   Good (0.6≤F1<0.8):      {good:2d} ({good/total_evaluated*100:4.1f}%)')
    print(f'   Moderate (0.3≤F1<0.6):  {moderate:2d} ({moderate/total_evaluated*100:4.1f}%)')
    print(f'   Poor (0<F1<0.3):        {poor:2d} ({poor/total_evaluated*100:4.1f}%)')
    print(f'   Failed (F1=0.0):        {failed:2d} ({failed/total_evaluated*100:4.1f}%)')
    
    # Error Detection Analysis
    print(f'\n🎯 ERROR DETECTION ANALYSIS:')
    total_true_errors = results_df['true_errors'].sum()
    total_predicted_errors = results_df['predicted_errors'].sum()
    
    # Calculate aggregate TP, FP, FN
    total_tp = 0
    for _, row in results_df.iterrows():
        true_set = set(row['true_sections'])
        pred_set = set(row['predicted_sections'])
        total_tp += len(true_set & pred_set)
    
    total_fp = total_predicted_errors - total_tp
    total_fn = total_true_errors - total_tp
    
    aggregate_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    aggregate_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    aggregate_f1 = 2 * aggregate_precision * aggregate_recall / (aggregate_precision + aggregate_recall) if (aggregate_precision + aggregate_recall) > 0 else 0
    
    print(f'   Total true errors:       {total_true_errors}')
    print(f'   Total predicted errors:  {total_predicted_errors}')
    print(f'   Prediction bias:         {total_predicted_errors - total_true_errors:+d} ({(total_predicted_errors/total_true_errors - 1)*100:+.1f}% vs truth)')
    print(f'   Aggregate Precision:     {aggregate_precision:.3f}')
    print(f'   Aggregate Recall:        {aggregate_recall:.3f}')
    print(f'   Aggregate F1:            {aggregate_f1:.3f}')
    
    # === ERROR TYPE ANALYSIS ===
    print(f'\n🏷️  PERFORMANCE BY ERROR COMPLEXITY:')
    for complexity in ['No Errors', 'Single Error', 'Multiple Errors', 'Complex Errors']:
        subset = results_df[results_df['complexity_category'] == complexity]
        if len(subset) > 0:
            mean_f1 = subset['f1_score'].mean()
            std_f1 = subset['f1_score'].std()
            print(f'   {complexity:15s}: n={len(subset):2d}, F1={mean_f1:.3f}±{std_f1:.3f}')
    
    print(f'\n📍 PERFORMANCE BY ERROR POSITION:')
    for position in ['No Errors', 'Early Errors', 'Middle Errors', 'Late Errors', 'Mixed Position']:
        subset = results_df[results_df['position_category'] == position]
        if len(subset) > 0:
            mean_f1 = subset['f1_score'].mean()
            std_f1 = subset['f1_score'].std()
            print(f'   {position:15s}: n={len(subset):2d}, F1={mean_f1:.3f}±{std_f1:.3f}')
    
    print(f'\n📚 PERFORMANCE BY TASK TYPE:')
    task_performance = results_df.groupby('task_type').agg({
        'f1_score': ['count', 'mean', 'std']
    }).round(3)
    
    for task in sorted(results_df['task_type'].unique()):
        subset = results_df[results_df['task_type'] == task]
        mean_f1 = subset['f1_score'].mean()
        std_f1 = subset['f1_score'].std()
        print(f'   {task:15s}: n={len(subset):2d}, F1={mean_f1:.3f}±{std_f1:.3f}')
    
    # === RUNTIME & COST ANALYSIS ===
    print(f'\n💰 RUNTIME & COST ANALYSIS:')
    avg_time_per_example = runtime / len(results_df)
    avg_tokens_per_example = total_tokens / len(results_df)
    
    # Cost estimation (GPT-4o-mini pricing)
    cost_per_1k_tokens = 0.00015  # USD per 1K tokens
    total_cost = total_tokens * cost_per_1k_tokens / 1000
    cost_per_example = total_cost / len(results_df)
    
    print(f'   Total runtime:           {runtime:.1f} seconds ({runtime/60:.1f} minutes)')
    print(f'   Average per example:     {avg_time_per_example:.1f} seconds')
    print(f'   Total tokens used:       {total_tokens:,}')
    print(f'   Average tokens/example:  {avg_tokens_per_example:.0f}')
    print(f'   Total estimated cost:    ${total_cost:.3f} USD')
    print(f'   Cost per example:        ${cost_per_example:.4f} USD')
    print(f'   Token efficiency:        {results_df["f1_score"].mean()*1000/avg_tokens_per_example:.2f} F1 per 1k tokens')
    
    # === ERROR PATTERN ANALYSIS ===
    print(f'\n🔍 ERROR PATTERN ANALYSIS:')
    
    # Most commonly predicted sections
    all_predicted = []
    for sections in results_df['predicted_sections']:
        all_predicted.extend(sections)
    
    if all_predicted:
        predicted_counter = Counter(all_predicted)
        print(f'   Most predicted sections: {dict(predicted_counter.most_common(5))}')
    
    # Most commonly missed sections
    all_missed = []
    for _, row in results_df.iterrows():
        true_set = set(row['true_sections'])
        pred_set = set(row['predicted_sections'])
        missed = true_set - pred_set
        all_missed.extend(missed)
    
    if all_missed:
        missed_counter = Counter(all_missed)
        print(f'   Most missed sections:    {dict(missed_counter.most_common(5))}')
    
    # === SAVE RESULTS ===
    output_file = 'results/evaluation_100_samples_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f'\n📁 Results saved to: {output_file}')
    
    # === KEY INSIGHTS ===
    print(f'\n💡 KEY INSIGHTS:')
    best_task = results_df.groupby('task_type')['f1_score'].mean().idxmax()
    worst_task = results_df.groupby('task_type')['f1_score'].mean().idxmin()
    
    print(f'   • Overall F1 score of {results_df["f1_score"].mean():.3f} indicates {"good" if results_df["f1_score"].mean() > 0.6 else "moderate" if results_df["f1_score"].mean() > 0.3 else "poor"} performance')
    print(f'   • Critic {"over" if total_predicted_errors > total_true_errors else "under"}-predicts errors by {abs(total_predicted_errors - total_true_errors)} sections')
    print(f'   • {failed/total_evaluated*100:.1f}% complete failures suggest {"significant" if failed/total_evaluated > 0.2 else "some"} room for improvement')
    print(f'   • Best performance on {best_task} tasks (F1={results_df[results_df["task_type"]==best_task]["f1_score"].mean():.3f})')
    print(f'   • Weakest performance on {worst_task} tasks (F1={results_df[results_df["task_type"]==worst_task]["f1_score"].mean():.3f})')
    print(f'   • Average cost of ${cost_per_example:.4f} per example is {"reasonable" if cost_per_example < 0.01 else "high"} for research')
    print(f'   • Token efficiency of {results_df["f1_score"].mean()*1000/avg_tokens_per_example:.2f} F1/1k tokens')
    
    print(f'\n{"="*60}')
    print('✅ EVALUATION COMPLETE')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()