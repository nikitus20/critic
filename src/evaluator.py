"""Evaluation engine for DeltaBench."""

import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

from .data_loader import DeltaBenchDataset
from .critic import LLMCritic


class DeltaBenchEvaluator:
    """Simple evaluation engine for DeltaBench."""
    
    def __init__(self, dataset: DeltaBenchDataset, critic: LLMCritic):
        self.dataset = dataset
        self.critic = critic
        
    def evaluate_example(self, example: Dict) -> Optional[Dict]:
        """Evaluate a single example."""
        question = example.get('question', '')
        model_output = example.get('sections_content', '') or example.get('section_content', '')
        
        if not question or not model_output:
            return None
        
        # Get ground truth - combine error and unuseful sections (matching original paper)
        error_sections = example.get('reason_error_section_numbers', [])
        unuseful_sections = example.get('reason_unuseful_section_numbers', [])
        all_errors = list(set(error_sections + unuseful_sections))  # Remove duplicates
        
        # Get critic evaluation
        critic_output, token_info = self.critic.evaluate_reasoning(question, model_output)
        
        if critic_output is None:
            return None
        
        # Parse results
        result = self.critic.parse_output(critic_output, all_errors)
        
        if result is None:
            return None
        
        return {
            'example': example,
            'result': result,
            'token_info': token_info,
            'ground_truth': all_errors
        }
    
    def evaluate_dataset(self, num_examples: Optional[int] = None, 
                        store_raw: bool = False) -> pd.DataFrame:
        """Evaluate multiple examples."""
        if self.dataset.data is None:
            return pd.DataFrame()
        
        examples = self.dataset.data[:num_examples] if num_examples else self.dataset.data
        results = []
        self.raw_results = [] if store_raw else None
        
        for i, example in enumerate(tqdm(examples, desc="Evaluating")):
            eval_result = self.evaluate_example(example)
            
            if eval_result and eval_result['result']:
                result_dict = {
                    'example_idx': i,
                    'example_id': example.get('id', ''),
                    'task_l1': example.get('task_l1', ''),
                    'task_l2': example.get('task_l2', ''),
                    'precision': eval_result['result'].precision,
                    'recall': eval_result['result'].recall,
                    'f1_score': eval_result['result'].f1_score,
                    'predicted_errors': len(eval_result['result'].predicted_error_sections),
                    'true_errors': len(eval_result['ground_truth']),
                    'predicted_sections': eval_result['result'].predicted_error_sections,
                    'true_sections': eval_result['ground_truth'],
                    'tokens_used': eval_result['token_info']['total_tokens'] if eval_result['token_info'] else 0,
                    'prompt_type': eval_result['token_info'].get('prompt_type', 'unknown') if eval_result['token_info'] else 'unknown'
                }
                results.append(result_dict)
                
                # Store raw results if requested
                if store_raw:
                    self.raw_results.append(eval_result)
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate aggregate metrics matching original paper methodology."""
        if len(results_df) == 0:
            return {}
        
        # Macro metrics (mean of per-example metrics)
        precision_macro = results_df['precision'].mean()
        recall_macro = results_df['recall'].mean()
        f1_macro = results_df['f1_score'].mean()
        
        # Micro metrics (aggregated across all examples)
        sum_tp = sum(len(set(row['predicted_sections']) & set(row['true_sections'])) 
                    for _, row in results_df.iterrows())
        sum_fp = sum(len(set(row['predicted_sections']) - set(row['true_sections'])) 
                    for _, row in results_df.iterrows())
        sum_fn = sum(len(set(row['true_sections']) - set(row['predicted_sections'])) 
                    for _, row in results_df.iterrows())
        
        precision_micro = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
        recall_micro = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
        
        return {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'total_examples': len(results_df),
            'avg_tokens': results_df['tokens_used'].mean()
        }
    
    def calculate_metrics_by_task(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics grouped by task type matching original paper."""
        if len(results_df) == 0:
            return pd.DataFrame()
        
        def calculate_task_metrics(group):
            # Macro metrics
            precision_macro = group['precision'].mean()
            recall_macro = group['recall'].mean()
            f1_macro = group['f1_score'].mean()
            
            # Micro metrics  
            sum_tp = sum(len(set(row['predicted_sections']) & set(row['true_sections'])) 
                        for _, row in group.iterrows())
            sum_fp = sum(len(set(row['predicted_sections']) - set(row['true_sections'])) 
                        for _, row in group.iterrows())
            sum_fn = sum(len(set(row['true_sections']) - set(row['predicted_sections'])) 
                        for _, row in group.iterrows())
            
            precision_micro = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
            recall_micro = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
            f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
            
            return pd.Series({
                'recall_macro': recall_macro,
                'precision_macro': precision_macro,
                'f1_macro': f1_macro,
                'recall_micro': recall_micro,
                'precision_micro': precision_micro,
                'f1_micro': f1_micro,
                'count': len(group)
            })
        
        # Calculate by task type
        task_metrics = results_df.groupby('task_l1').apply(calculate_task_metrics).reset_index()
        
        # Add overall metrics
        overall_metrics = self.calculate_metrics(results_df)
        overall_row = pd.DataFrame({
            'task_l1': ['Overall'],
            'recall_macro': [overall_metrics['recall_macro']],
            'precision_macro': [overall_metrics['precision_macro']],
            'f1_macro': [overall_metrics['f1_macro']],
            'recall_micro': [overall_metrics['recall_micro']],
            'precision_micro': [overall_metrics['precision_micro']],
            'f1_micro': [overall_metrics['f1_micro']],
            'count': [len(results_df)]
        })
        
        return pd.concat([overall_row, task_metrics], ignore_index=True)