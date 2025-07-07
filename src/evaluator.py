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
        
        # Get ground truth
        error_sections = example.get('reason_error_section_numbers', [])
        unuseful_sections = example.get('reason_unuseful_section_numbers', [])
        all_errors = list(set(error_sections + unuseful_sections))
        
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
    
    def evaluate_dataset(self, num_examples: Optional[int] = None) -> pd.DataFrame:
        """Evaluate multiple examples."""
        if self.dataset.data is None:
            return pd.DataFrame()
        
        examples = self.dataset.data[:num_examples] if num_examples else self.dataset.data
        results = []
        
        for i, example in enumerate(tqdm(examples, desc="Evaluating")):
            eval_result = self.evaluate_example(example)
            
            if eval_result and eval_result['result']:
                results.append({
                    'example_idx': i,
                    'precision': eval_result['result'].precision,
                    'recall': eval_result['result'].recall,
                    'f1_score': eval_result['result'].f1_score,
                    'predicted_errors': len(eval_result['result'].predicted_error_sections),
                    'true_errors': len(eval_result['ground_truth']),
                    'tokens_used': eval_result['token_info']['total_tokens'] if eval_result['token_info'] else 0
                })
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate aggregate metrics."""
        if len(results_df) == 0:
            return {}
        
        return {
            'precision_mean': results_df['precision'].mean(),
            'recall_mean': results_df['recall'].mean(),
            'f1_mean': results_df['f1_score'].mean(),
            'total_examples': len(results_df),
            'avg_tokens': results_df['tokens_used'].mean()
        }