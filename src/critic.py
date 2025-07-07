"""LLM critic for evaluating reasoning steps."""

import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

from . import config
from .utils import CriticResult


class LLMCritic:
    """LLM critic for evaluating reasoning steps."""
    
    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self.model_name = model_name
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
    def evaluate_reasoning(self, question: str, model_output: str) -> Tuple[str, Dict]:
        """Evaluate reasoning using LLM critic."""
        prompt = config.CRITIC_PROMPT.format(
            question=question,
            model_output=model_output
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P
            )
            
            output = response.choices[0].message.content
            token_info = {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
            return output, token_info
            
        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            return None, None
    
    def parse_output(self, critic_output: str, true_error_sections: List[int]) -> CriticResult:
        """Parse critic output into structured result."""
        try:
            # Extract conclusion
            result = critic_output.split("Error Section Number:")[0].split("Conclusion:")[-1].strip()
            has_errors = "yes" in result.lower()
            
            predicted_errors = []
            explanations = []
            
            if has_errors:
                # Parse error sections
                sections = critic_output.split("Error Section Number:")[1:]
                for section in sections:
                    # Extract error number
                    number_match = re.search(r'\d+', section.split("Explanation:")[0])
                    if number_match:
                        error_num = int(number_match.group())
                        predicted_errors.append(error_num)
                    
                    # Extract explanation
                    if "Explanation:" in section:
                        explanation = section.split("Explanation:")[-1].strip()
                        explanations.append(explanation)
            
            # Filter predictions to valid range
            if true_error_sections:
                max_section = max(true_error_sections)
                predicted_errors = [x for x in predicted_errors if x <= max_section]
            
            # Calculate metrics
            tp = len(set(predicted_errors) & set(true_error_sections))
            fp = len(set(predicted_errors) - set(true_error_sections))
            fn = len(set(true_error_sections) - set(predicted_errors))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return CriticResult(
                predicted_error_sections=predicted_errors,
                explanations=explanations,
                precision=precision,
                recall=recall,
                f1_score=f1,
                raw_output=critic_output
            )
            
        except Exception as e:
            print(f"Error parsing output: {e}")
            return None