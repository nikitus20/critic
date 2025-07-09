"""Direct critic for evaluating reasoning steps (original DeltaBench approach)."""

import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import httpx

from .. import config
from ..utils import CriticResult
from .base_critic import BaseCritic


class DirectCritic(BaseCritic):
    """Direct critic for evaluating reasoning steps using single-stage prompting."""
    
    def __init__(self, model: str = config.DEFAULT_MODEL, prompt_type: str = "deltabench", config_dict: Optional[Dict] = None, model_name: Optional[str] = None):
        # Support both 'model' and 'model_name' for backward compatibility
        if model_name is not None:
            model = model_name
        
        super().__init__(model, config_dict)
        self.prompt_type = prompt_type
        
        # Create OpenAI client with custom httpx client to avoid proxy issues
        http_client = httpx.Client()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY, http_client=http_client)
        
        # Validate prompt type
        if prompt_type not in config.PROMPTS:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(config.PROMPTS.keys())}")
    
    def evaluate_reasoning(self, question: str, model_output: str) -> Tuple[str, Dict]:
        """Evaluate reasoning using LLM critic with retry mechanism."""
        # Get the appropriate prompt template
        prompt_template = config.PROMPTS[self.prompt_type]
        prompt = prompt_template.format(
            question=question,
            model_output=model_output
        )
        
        # Retry mechanism matching original paper
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P
                )
                
                output = response.choices[0].message.content
                token_info = {
                    "total_tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_type": self.prompt_type
                }
                
                # Check if output is valid (not None or empty)
                if output and output.strip():
                    return output, token_info
                    
            except Exception as e:
                print(f"Error calling {self.model} (attempt {retry_count + 1}): {e}")
                
            retry_count += 1
            
        # All retries failed
        print(f"Failed to get response after {max_retries} attempts")
        return None, None
    
    def parse_output(self, critic_output: str, true_error_sections: List[int], max_valid_section: Optional[int] = None) -> CriticResult:
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
            
            # Filter predictions to valid range based on actual section count or ground truth range
            if max_valid_section is not None:
                # Use actual section count if provided
                predicted_errors = [x for x in predicted_errors if 1 <= x <= max_valid_section]
            elif true_error_sections:
                # Fallback: use ground truth range (but this may not capture all valid sections)
                max_section = max(true_error_sections + [1])  # At least section 1 should exist
                predicted_errors = [x for x in predicted_errors if 1 <= x <= max_section]
            else:
                # No filtering if we don't know the valid range
                predicted_errors = [x for x in predicted_errors if x >= 1]
            
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


# Backward compatibility alias
LLMCritic = DirectCritic