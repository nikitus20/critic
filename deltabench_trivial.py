#!/usr/bin/env python3
"""
DeltaBench Simple Baseline Experiment
Simulates a critic that always predicts sections 3-36 as error sections.
"""

import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


def parse_output_simple_baseline(all_error_section_indexs, max_section=50):
    """
    Simulate the simple baseline approach: always predict sections 3-36 as errors.
    
    Args:
        all_error_section_indexs (list): List of all actual error section indices
        max_section (int): Maximum section number to consider
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Simple baseline: always predict sections 3-36 as error sections
    predicted_error_sections = list(range(3, 37))  # 3 to 36 inclusive
    
    # Filter based on max label (same as original code)
    if all_error_section_indexs:
        max_label_error_section = max(all_error_section_indexs)
        predicted_error_sections = [x for x in predicted_error_sections if x <= max_label_error_section]
    
    # Calculate true positives, false positives, and false negatives
    true_positives = len(set(predicted_error_sections) & set(all_error_section_indexs))
    false_positives = len(set(predicted_error_sections) - set(all_error_section_indexs))
    false_negatives = len(set(all_error_section_indexs) - set(predicted_error_sections))
    
    # Calculate precision, recall and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "error_sections_nums": predicted_error_sections,
        "parsing_success": 1,
        "judge": 1 if predicted_error_sections else 0,  # 1 if we predict any errors, 0 otherwise
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp_step": true_positives,
        "fp_step": false_positives,
        "fn_step": false_negatives,
    }


def process_data_simple_baseline(data):
    """
    Process a single data point with the simple baseline approach.
    
    Args:
        data (dict): Single data point from the dataset
        
    Returns:
        dict: Processed data point with evaluation metrics
    """
    # Extract ground truth error sections
    idea_error_section_numbers = data.get('reason_unuseful_section_numbers', [])
    error_section_numbers = data.get('reason_error_section_numbers', [])
    
    all_section_indexs = idea_error_section_numbers + error_section_numbers
    all_section_indexs = list(set(all_section_indexs))
    
    # Apply simple baseline approach
    info = parse_output_simple_baseline(all_section_indexs)
    
    # Update the data point with results
    result_data = data.copy()
    result_data.update(info)
    
    return result_data


def calculate_accuracies_v2(group):
    """
    Calculate macro and micro averaged metrics for a group.
    
    Args:
        group (pd.DataFrame): Group of data points
        
    Returns:
        pd.Series: Calculated metrics
    """
    total_questions = len(group)
    
    # Macro averages (average of individual scores)
    precision_macro = group['precision'].mean()
    recall_macro = group['recall'].mean()
    f1_score_macro = group['f1_score'].mean()
    
    # Micro averages (aggregate then calculate)
    sum_tp = group['tp_step'].sum()
    sum_fp = group['fp_step'].sum()
    sum_fn = group['fn_step'].sum()
    
    precision_micro = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall_micro = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
    
    return pd.Series({
        'count': total_questions,
        'recall_macro': recall_macro,
        'precision_macro': precision_macro,
        'f1_score_macro': f1_score_macro,
        'recall_micro': recall_micro, 
        'precision_micro': precision_micro,  
        'f1_micro': f1_micro,
    })


def load_and_process_data(input_file):
    """
    Load data from JSONL file and process with simple baseline.
    
    Args:
        input_file (str): Path to input JSONL file
        
    Returns:
        list: Processed data points
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'question' not in data:
                    print(f"Warning: Line {line_num} missing 'question' field, skipping")
                    continue
                
                processed_item = process_data_simple_baseline(data)
                processed_data.append(processed_item)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return processed_data


def print_results_table(df, dataset_name):
    """
    Print results in a formatted table.
    
    Args:
        df (pd.DataFrame): Results dataframe
        dataset_name (str): Name of the dataset
    """
    print(f"\n{'='*80}")
    print(f"DELTABENCH SIMPLE BASELINE RESULTS - {dataset_name.upper()}")
    print(f"Approach: Always predict sections 3-36 as error sections")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Task':<15} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'':->15} {'':->8} {'':->12} {'':->12} {'':->12}")
    print(f"{'MACRO AVERAGES':<47}")
    print(f"{'':->47}")
    
    # Print results for each task
    for _, row in df.iterrows():
        task = row['task_l1'] if 'task_l1' in row else 'Overall'
        count = int(row['count']) if pd.notna(row['count']) else 0
        precision = f"{row['precision_macro']:.4f}" if pd.notna(row['precision_macro']) else "0.0000"
        recall = f"{row['recall_macro']:.4f}" if pd.notna(row['recall_macro']) else "0.0000"
        f1 = f"{row['f1_score_macro']:.4f}" if pd.notna(row['f1_score_macro']) else "0.0000"
        
        print(f"{task:<15} {count:<8} {precision:<12} {recall:<12} {f1:<12}")
    
    print(f"\n{'MICRO AVERAGES':<47}")
    print(f"{'':->47}")
    
    # Print micro averages
    for _, row in df.iterrows():
        task = row['task_l1'] if 'task_l1' in row else 'Overall'
        count = int(row['count']) if pd.notna(row['count']) else 0
        precision = f"{row['precision_micro']:.4f}" if pd.notna(row['precision_micro']) else "0.0000"
        recall = f"{row['recall_micro']:.4f}" if pd.notna(row['recall_micro']) else "0.0000"
        f1 = f"{row['f1_micro']:.4f}" if pd.notna(row['f1_micro']) else "0.0000"
        
        print(f"{task:<15} {count:<8} {precision:<12} {recall:<12} {f1:<12}")


def get_metrics_and_save(processed_data, output_file, dataset_name):
    """
    Calculate metrics and save results.
    
    Args:
        processed_data (list): Processed data points
        output_file (str): Output file path
        dataset_name (str): Name of the dataset
    """
    # Convert to DataFrame
    df = pd.json_normalize(processed_data)
    
    # Calculate overall metrics
    overall_metrics = calculate_accuracies_v2(df)
    overall_row = pd.DataFrame({
        'task_l1': ['Overall'],
        'count': [overall_metrics['count']],
        'recall_macro': [overall_metrics['recall_macro']],
        'precision_macro': [overall_metrics['precision_macro']],
        'f1_score_macro': [overall_metrics['f1_score_macro']],
        'recall_micro': [overall_metrics['recall_micro']],
        'precision_micro': [overall_metrics['precision_micro']],
        'f1_micro': [overall_metrics['f1_micro']],
    })
    
    # Calculate metrics by task if task_l1 exists
    final_df = overall_row.copy()
    if 'task_l1' in df.columns and df['task_l1'].notna().any():
        task_metrics = df.groupby('task_l1').apply(calculate_accuracies_v2).reset_index()
        final_df = pd.concat([overall_row, task_metrics], ignore_index=True)
    
    # Save results
    if output_file:
        final_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Also save detailed results
        detailed_file = output_file.replace('.csv', '_detailed.jsonl')
        with open(detailed_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Detailed results saved to: {detailed_file}")
    
    # Print results table
    print_results_table(final_df, dataset_name)
    
    return final_df


def main():
    parser = argparse.ArgumentParser(description='DeltaBench Simple Baseline Experiment')
    parser.add_argument('--dataset', required=True, 
                        help='Path to the dataset JSONL file (e.g., data/your_dataset.jsonl)')
    parser.add_argument('--output', required=False, default=None,
                        help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Extract dataset name for display
    dataset_name = args.dataset.split('/')[-1].replace('.jsonl', '')
    
    print(f"Loading dataset: {args.dataset}")
    print(f"Simple baseline approach: Always predict sections 3-36 as error sections")
    
    try:
        # Load and process data
        processed_data = load_and_process_data(args.dataset)
        
        if not processed_data:
            print("Error: No valid data points found in the input file.")
            return
        
        print(f"Processed {len(processed_data)} data points")
        
        # Generate output filename if not provided
        output_file = args.output
        if output_file is None:
            output_file = f"simple_baseline_{dataset_name}_results.csv"
        
        # Calculate and display metrics
        results_df = get_metrics_and_save(processed_data, output_file, dataset_name)
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{args.dataset}' not found.")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
