#!/usr/bin/env python3
"""
ProcessBench-Style Evaluation Script for DeltaBench Data

This script adapts the ProcessBench methodology to evaluate DeltaBench data,
focusing on first-error identification accuracy with voting mechanisms.

Key differences from DeltaBench original approach:
1. Uses ProcessBench critique template and prompt format
2. Focuses only on first error identification (not all errors)  
3. Uses temperature 0.7 with voting mechanism (8 votes)
4. Evaluates binary classification: error detection vs correct recognition
5. Simplified metrics: accuracy on errors + accuracy on correct + F1 score

Based on ProcessBench reproduction guide and DeltaBench data structure.
"""

import argparse
import json
import os
import re
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ProcessBench critique template (adapted from reproduction guide)
PROCESSBENCH_CRITIQUE_TEMPLATE = """The following is a problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}.
"""

# Global OpenAI client configuration
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"


def convert_deltabench_to_processbench(deltabench_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert DeltaBench data format to ProcessBench format.
    
    Args:
        deltabench_item: DeltaBench data item
        
    Returns:
        ProcessBench format item with 'problem', 'steps', and 'label'
    """
    # Extract problem and steps from DeltaBench format
    problem = deltabench_item['question']
    
    # Convert sections to steps array
    steps = []
    for section in deltabench_item['sections']:
        steps.append(section['content'])
    
    # Get first error index (ProcessBench focuses on first error only)
    error_sections = deltabench_item.get('reason_error_section_numbers', [])
    
    if not error_sections:
        # No errors - label as -1 (ProcessBench convention)
        label = -1
    else:
        # Get first error (convert to 0-indexed)
        first_error_section = min(error_sections)
        # DeltaBench sections are 1-indexed, ProcessBench steps are 0-indexed
        label = first_error_section - 1
    
    return {
        'id': deltabench_item['id'],
        'problem': problem,
        'steps': steps,
        'label': label,
        'task_l1': deltabench_item.get('task_l1', 'unknown'),
        'origin': deltabench_item.get('origin', 'unknown')
    }


def prepare_processbench_input(template: str, input_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Prepare input for ProcessBench evaluation using the critique template.
    
    Args:
        template: ProcessBench critique template
        input_data: Dictionary with 'problem' and 'steps'
        
    Returns:
        List of message dictionaries for OpenAI API
    """
    problem = input_data['problem']
    steps = input_data['steps']
    
    # Tag each step with paragraph indices (ProcessBench format)
    tagged_response = ''
    for idx, step in enumerate(steps):
        tagged_response += f'<paragraph_{idx}>\n{step}\n</paragraph_{idx}>\n\n'
    tagged_response = tagged_response.strip()
    
    # Format the prompt
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages


def extract_processbench_answer(response_text: str) -> Optional[int]:
    """
    Extract the answer from ProcessBench model response.
    
    Args:
        response_text: Model's response string
        
    Returns:
        Integer index or None if not found
    """
    # ProcessBench uses \boxed{} format for answers
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, response_text)
    if matches:
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None
    return None


def call_openai_api(client: OpenAI, messages: List[Dict[str, str]], 
                   model: str = "gpt-4o-mini", temperature: float = 0.7, 
                   max_tokens: int = 4096) -> Tuple[str, Dict[str, Any]]:
    """
    Call OpenAI API with ProcessBench parameters.
    
    Args:
        client: OpenAI client instance
        messages: List of message dictionaries
        model: Model name
        temperature: Sampling temperature (0.7 for ProcessBench voting)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (response text, token usage info)
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42  # For reproducibility
            )
            
            output = response.choices[0].message.content
            token_info = {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
            if output and output.strip():
                return output, token_info
                
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return "", {}


def process_with_processbench_voting(client: OpenAI, template: str, 
                                   input_data: Dict[str, Any], 
                                   model: str = "gpt-4o-mini", 
                                   n_votes: int = 8) -> Dict[str, Any]:
    """
    Process single example with ProcessBench voting mechanism.
    
    Args:
        client: OpenAI client
        template: ProcessBench critique template
        input_data: Input example
        model: Model name
        n_votes: Number of votes (ProcessBench uses 8)
        
    Returns:
        Dictionary with prediction and voting details
    """
    messages = prepare_processbench_input(template, input_data)
    
    # Generate multiple responses with temperature 0.7 for diversity
    votes = []
    for _ in range(n_votes):
        try:
            response_text, _ = call_openai_api(
                client, messages, model, temperature=0.7
            )
            prediction = extract_processbench_answer(response_text)
            if prediction is not None:
                votes.append(prediction)
        except Exception as e:
            print(f"Error in voting round: {e}")
            continue
    
    # Determine final prediction by majority vote
    if votes:
        vote_counts = Counter(votes)
        final_prediction = vote_counts.most_common(1)[0][0]
    else:
        final_prediction = None
    
    return {
        'prediction': final_prediction,
        'votes': votes,
        'vote_distribution': dict(Counter(votes)) if votes else {},
        'n_valid_votes': len(votes)
    }


def calculate_processbench_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate ProcessBench-style metrics.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with ProcessBench accuracy metrics
    """
    # Split into error and correct examples
    error_examples = [r for r in results if r['label'] != -1]
    correct_examples = [r for r in results if r['label'] == -1]
    
    # Calculate accuracies
    if error_examples:
        error_matches = sum(1 for r in error_examples if r['prediction'] == r['label'])
        error_accuracy = error_matches / len(error_examples) * 100
    else:
        error_accuracy = 0.0
    
    if correct_examples:
        correct_matches = sum(1 for r in correct_examples if r['prediction'] == r['label'])
        correct_accuracy = correct_matches / len(correct_examples) * 100
    else:
        correct_accuracy = 0.0
    
    # Calculate F1 score (ProcessBench uses F1 of the two accuracies)
    if error_accuracy + correct_accuracy > 0:
        f1_score = 2 * error_accuracy * correct_accuracy / (error_accuracy + correct_accuracy)
    else:
        f1_score = 0.0
    
    return {
        'error_accuracy': error_accuracy,
        'correct_accuracy': correct_accuracy,
        'f1_score': f1_score,
        'error_count': len(error_examples),
        'correct_count': len(correct_examples),
        'total_count': len(results)
    }


def process_single_example(args_tuple: Tuple[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Process a single example with ProcessBench methodology.
    
    Args:
        args_tuple: Tuple of (example, args)
        
    Returns:
        Result dictionary
    """
    example, args = args_tuple
    
    try:
        # Convert DeltaBench format to ProcessBench format
        pb_example = convert_deltabench_to_processbench(example)
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"]
        )
        
        # Process with voting mechanism
        result = process_with_processbench_voting(
            client, PROCESSBENCH_CRITIQUE_TEMPLATE, pb_example, 
            args.model, args.n_votes
        )
        
        # Add metadata
        result.update({
            'id': pb_example['id'],
            'label': pb_example['label'],
            'task_l1': pb_example['task_l1'],
            'origin': pb_example['origin'],
            'match': result['prediction'] == pb_example['label'] if result['prediction'] is not None else False,
            'success': 1 if result['prediction'] is not None else 0
        })
        
        return result
        
    except Exception as e:
        print(f"Error processing example {example.get('id', 'unknown')}: {e}")
        print(traceback.format_exc())
        return {
            'id': example.get('id', 'unknown'),
            'label': convert_deltabench_to_processbench(example)['label'],
            'prediction': None,
            'match': False,
            'success': 0,
            'error': str(e)
        }


def load_deltabench_data(filepath: str) -> List[Dict[str, Any]]:
    """Load DeltaBench data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_results(results: List[Dict[str, Any]], filepath: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def print_processbench_results(metrics: Dict[str, float], task_metrics: pd.DataFrame):
    """Print ProcessBench-style results."""
    print("\n" + "="*80)
    print("PROCESSBENCH-STYLE EVALUATION RESULTS")
    print("="*80)
    print(f"Overall Results:")
    print(f"  Error Accuracy: {metrics['error_accuracy']:.1f}%")
    print(f"  Correct Accuracy: {metrics['correct_accuracy']:.1f}%")
    print(f"  F1 Score: {metrics['f1_score']:.1f}")
    print(f"  Total Examples: {metrics['total_count']}")
    print(f"  Error Examples: {metrics['error_count']}")
    print(f"  Correct Examples: {metrics['correct_count']}")
    print("\nBy Task:")
    print(task_metrics.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='ProcessBench-style evaluation for DeltaBench data'
    )
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., Deltabench_v1)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model name')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--n-votes', type=int, default=8, help='Number of votes for consensus')
    parser.add_argument('--processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--subsample', type=float, default=1.0, help='Fraction of data to use (0.1 = 10%, 1.0 = 100%)')
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["OPENAI_API_KEY"] = args.api_key
    
    print(f"ProcessBench-style evaluation starting...")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Votes: {args.n_votes}")
    print(f"Processes: {args.processes}")
    print(f"Subsample: {args.subsample * 100:.1f}%")
    
    # Load data
    data_path = f"data/{args.dataset}.jsonl"
    
    # Add subsample info to results filename
    subsample_str = f"_sub{int(args.subsample * 100)}" if args.subsample < 1.0 else ""
    results_path = f"results/processbench/{args.dataset}_{args.model}_votes{args.n_votes}{subsample_str}.jsonl"
    
    print(f"Loading data from {data_path}...")
    data = load_deltabench_data(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Apply subsampling if requested
    if args.subsample < 1.0:
        import random
        random.seed(42)  # For reproducibility
        sample_size = int(len(data) * args.subsample)
        data = random.sample(data, sample_size)
        print(f"Subsampled to {len(data)} examples ({args.subsample * 100:.1f}%)")
    
    # Process examples
    start_time = time.time()
    
    args_list = [(example, args) for example in data]
    
    with Pool(processes=args.processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_example, args_list), 
            total=len(data),
            desc="Processing examples"
        ))
    
    # Calculate metrics
    metrics = calculate_processbench_metrics(results)
    
    # Calculate by-task metrics
    df = pd.DataFrame(results)
    task_metrics = df.groupby('task_l1').apply(
        lambda group: pd.Series(calculate_processbench_metrics(group.to_dict('records')))
    ).reset_index()
    
    # Save results
    save_results(results, results_path)
    
    # Print results
    print_processbench_results(metrics, task_metrics)
    
    # Save CSV summary
    csv_path = results_path.replace('.jsonl', '.csv')
    task_metrics.to_csv(csv_path, index=False)
    
    end_time = time.time()
    print(f"Evaluation completed in {(end_time - start_time) / 60:.1f} minutes")
    print(f"Results saved to {results_path}")
    print(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    main()