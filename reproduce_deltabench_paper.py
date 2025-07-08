#!/usr/bin/env python3
"""
Reproduce DeltaBench paper results using GPT-4o-mini.
Implements the exact methodology from the original paper with parallel processing.
"""

import sys
sys.path.append('src')
import os
import json
import time
import random
import argparse
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment
load_dotenv('.env')

from src import DeltaBenchDataset, LLMCritic, DeltaBenchEvaluator

def process_single_example(args_data):
    """Process a single example - designed for multiprocessing."""
    example, model_name, prompt_type = args_data
    
    try:
        # Initialize critic for this process
        critic = LLMCritic(model_name=model_name, prompt_type=prompt_type)
        
        # Extract data
        question = example['question']
        model_output = example.get('sections_content', '') or example.get('section_content', '')
        
        if not question or not model_output:
            return None
            
        # Get ground truth (combine error and unuseful sections)
        error_sections = example.get('reason_error_section_numbers', [])
        unuseful_sections = example.get('reason_unuseful_section_numbers', [])
        all_error_sections = list(set(error_sections + unuseful_sections))
        
        # Get critic evaluation
        critic_output, token_info = critic.evaluate_reasoning(question, model_output)
        
        if critic_output is None:
            return {
                'example': example,
                'parsing_success': 0,
                'error': 'Failed to get critic response'
            }
        
        # Parse critic output
        result = critic.parse_output(critic_output, all_error_sections)
        
        if result is None:
            return {
                'example': example,
                'parsing_success': 0,
                'error': 'Failed to parse critic output'
            }
        
        # Prepare result in original paper format
        return {
            'id': example.get('id', ''),
            'question': question,
            'sections_content': model_output,
            'task_l1': example.get('task_l1', ''),
            'task_l2': example.get('task_l2', ''),
            'reason_error_section_numbers': error_sections,
            'reason_unuseful_section_numbers': unuseful_sections,
            'all_error_sections': all_error_sections,
            
            # Critic results
            'critic': critic_output,
            'token_info': token_info,
            'predicted_sections': result.predicted_error_sections,
            'true_sections': all_error_sections,
            
            # Metrics  
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'tp_step': len(set(result.predicted_error_sections) & set(all_error_sections)),
            'fp_step': len(set(result.predicted_error_sections) - set(all_error_sections)),
            'fn_step': len(set(all_error_sections) - set(result.predicted_error_sections)),
            'judge': 1 if result.predicted_error_sections else 0,
            'parsing_success': 1,
            
            # Additional info
            'explanations': result.explanations,
            'raw_output': result.raw_output
        }
        
    except Exception as e:
        return {
            'example': example,
            'parsing_success': 0,
            'error': str(e)
        }

def load_existing_results(output_file):
    """Load existing results to support resume functionality."""
    completed = {}
    if os.path.exists(output_file):
        print(f"Found existing results file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('parsing_success', 0) == 1:
                        question = data.get('question', '')
                        if question:
                            completed[question] = data
                except:
                    continue
        print(f"Loaded {len(completed)} completed examples")
    return completed

def save_result(result, output_file):
    """Save a single result to JSONL file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

def calculate_paper_metrics(results_file):
    """Calculate metrics exactly as in the original paper."""
    print(f"\n📊 Calculating paper metrics from {results_file}...")
    
    # Load results
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('parsing_success', 0) == 1:
                    results.append(data)
            except:
                continue
    
    if not results:
        print("No valid results found!")
        return
    
    df = pd.DataFrame(results)
    
    # Overall metrics (macro and micro)
    def calculate_accuracies_v2(group):
        precision_macro = group['precision'].mean()
        recall_macro = group['recall'].mean()
        f1_score_macro = group['f1_score'].mean()
        
        sum_tp = group['tp_step'].sum()
        sum_fp = group['fp_step'].sum()
        sum_fn = group['fn_step'].sum()
        
        precision_micro = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
        recall_micro = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
        
        return pd.Series({
            'recall_macro': recall_macro,
            'precision_macro': precision_macro,
            'f1_score_macro': f1_score_macro,
            'recall_micro': recall_micro,
            'precision_micro': precision_micro,
            'f1_micro': f1_micro,
        })
    
    # Calculate overall metrics
    overall_metrics = calculate_accuracies_v2(df)
    
    # Calculate by task type
    task_metrics = df.groupby('task_l1').apply(calculate_accuracies_v2).reset_index()
    
    # Create final results table
    overall_row = pd.DataFrame({
        'task_l1': ['Overall'],
        'recall_macro': [overall_metrics['recall_macro']],
        'precision_macro': [overall_metrics['precision_macro']],
        'f1_score_macro': [overall_metrics['f1_score_macro']],
        'recall_micro': [overall_metrics['recall_micro']],
        'precision_micro': [overall_metrics['precision_micro']],
        'f1_micro': [overall_metrics['f1_micro']],
        'count': [len(df)]
    })
    
    final_metrics = pd.concat([overall_row, task_metrics], ignore_index=True)
    
    # Save to CSV
    csv_file = results_file.replace('.jsonl', '_metrics.csv')
    final_metrics.to_csv(csv_file, index=False)
    
    print(f"📈 Results Summary:")
    print(f"   Total examples: {len(df)}")
    print(f"   Overall F1 (macro): {overall_metrics['f1_score_macro']:.3f}")
    print(f"   Overall F1 (micro): {overall_metrics['f1_micro']:.3f}")
    print(f"   Overall Precision (macro): {overall_metrics['precision_macro']:.3f}")
    print(f"   Overall Recall (macro): {overall_metrics['recall_macro']:.3f}")
    print(f"\n📋 Task breakdown:")
    for _, row in task_metrics.iterrows():
        print(f"   {row['task_l1']:20s}: F1={row['f1_score_macro']:.3f} (n={df[df['task_l1']==row['task_l1']].shape[0]})")
    
    print(f"\n💾 Detailed metrics saved to: {csv_file}")
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description='Reproduce DeltaBench paper results')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model name to use')
    parser.add_argument('--prompt_type', default='deltabench', help='Prompt type (deltabench/pedcot)')
    parser.add_argument('--dataset', default='Deltabench_v1', help='Dataset name')
    parser.add_argument('--processes', type=int, default=min(10, cpu_count()), help='Number of processes')
    parser.add_argument('--test_size', type=int, default=None, help='Test on subset (for debugging)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print('=' * 80)
    print('🚀 REPRODUCING DELTABENCH PAPER RESULTS')
    print('=' * 80)
    print(f"Model: {args.model}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Processes: {args.processes}")
    if args.test_size:
        print(f"Test size: {args.test_size} examples")
    print()
    
    start_time = time.time()
    
    # Load dataset
    print("📁 Loading dataset...")
    dataset = DeltaBenchDataset()
    data = dataset.load_jsonl(f'data/{args.dataset}.jsonl')
    
    if not data:
        print("❌ Failed to load dataset!")
        return
    
    print(f"   Dataset loaded: {len(data)} examples")
    
    # Prepare output file
    output_file = f'results/{args.dataset}_{args.model}_{args.prompt_type}.jsonl'
    os.makedirs('results', exist_ok=True)
    
    # Load existing results for resume capability
    completed = load_existing_results(output_file)
    
    # Filter out completed examples
    remaining_data = []
    for example in data:
        question = example.get('question', '')
        if question not in completed:
            remaining_data.append(example)
        else:
            # Re-save completed results to maintain order
            save_result(completed[question], output_file)
    
    print(f"   Remaining to process: {len(remaining_data)} examples")
    
    # For testing, limit dataset size
    if args.test_size and args.test_size < len(remaining_data):
        random.seed(args.seed)
        remaining_data = random.sample(remaining_data, args.test_size)
        print(f"   Using test subset: {len(remaining_data)} examples")
    
    if not remaining_data:
        print("✅ All examples already processed!")
    else:
        # Prepare arguments for multiprocessing
        process_args = [(example, args.model, args.prompt_type) for example in remaining_data]
        
        print(f"🔄 Processing {len(remaining_data)} examples with {args.processes} processes...")
        
        # Process with multiprocessing
        successful = 0
        failed = 0
        
        with Pool(processes=args.processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_example, process_args),
                total=len(process_args),
                desc="Processing examples"
            ))
            
            # Save results and count successes
            for result in results:
                if result:
                    save_result(result, output_file)
                    if result.get('parsing_success', 0) == 1:
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
        
        print(f"\n✅ Processing complete!")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {successful / (successful + failed) * 100:.1f}%")
    
    # Calculate and display metrics
    metrics = calculate_paper_metrics(output_file)
    
    # Runtime and cost estimation
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n💰 Runtime & Cost Analysis:")
    print(f"   Total runtime: {runtime/60:.1f} minutes")
    
    # Estimate token usage and cost
    total_examples = len(data) - len(completed) + len(completed)
    avg_tokens_estimate = 5500  # Based on previous tests
    total_tokens_estimate = total_examples * avg_tokens_estimate
    cost_estimate = total_tokens_estimate * 0.00015 / 1000  # GPT-4o-mini pricing
    
    print(f"   Estimated total tokens: {total_tokens_estimate:,}")
    print(f"   Estimated total cost: ${cost_estimate:.2f}")
    
    print(f"\n🎉 Experiment complete! Results saved to:")
    print(f"   Raw data: {output_file}")
    print(f"   Metrics: {output_file.replace('.jsonl', '_metrics.csv')}")

if __name__ == '__main__':
    main()