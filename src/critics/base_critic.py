"""Base critic interface for DeltaBench evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from ..utils import CriticResult


class BaseCritic(ABC):
    """Abstract base class for all critics."""
    
    def __init__(self, model: str, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}
        
    @abstractmethod
    def evaluate_reasoning(self, question: str, model_output: str) -> Tuple[str, Dict]:
        """
        Evaluate reasoning using the critic.
        
        Args:
            question: The original question
            model_output: The model's reasoning output to evaluate
            
        Returns:
            Tuple of (critic_output, token_info)
        """
        pass
    
    @abstractmethod
    def parse_output(self, critic_output: str, true_error_sections: List[int], max_valid_section: Optional[int] = None) -> CriticResult:
        """
        Parse critic output into structured result.
        
        Args:
            critic_output: Raw output from the critic
            true_error_sections: Ground truth error sections
            max_valid_section: Maximum valid section number (total section count)
            
        Returns:
            CriticResult object with predictions and metrics
        """
        pass
    
    def analyze_section(self, question: str, section: str, context: List[str]) -> Dict:
        """
        Analyze a single section of reasoning.
        
        Args:
            question: The original question
            section: The section content to analyze
            context: Previous sections for context
            
        Returns:
            Dictionary with analysis results
        """
        # Default implementation uses full evaluate_reasoning
        full_output = "\n".join(context + [section])
        critic_output, token_info = self.evaluate_reasoning(question, full_output)
        return {
            "raw_output": critic_output,
            "token_info": token_info
        }
    
    def analyze_full_cot(self, question: str, sections: List[str]) -> Dict:
        """
        Analyze complete reasoning chain.
        
        Args:
            question: The original question
            sections: List of all sections in order
            
        Returns:
            Dictionary with complete analysis results
        """
        # Default implementation uses full evaluate_reasoning
        full_output = "\n".join(sections)
        critic_output, token_info = self.evaluate_reasoning(question, full_output)
        return {
            "raw_output": critic_output,
            "token_info": token_info
        }
    
    def get_predictions(self, question: str, model_output: str, true_error_sections: List[int], max_valid_section: Optional[int] = None) -> CriticResult:
        """
        Get predictions from the critic.
        
        Args:
            question: The original question
            model_output: The model's reasoning output
            true_error_sections: Ground truth error sections
            max_valid_section: Maximum valid section number (total section count)
            
        Returns:
            CriticResult with predictions and metrics
        """
        critic_output, _ = self.evaluate_reasoning(question, model_output)
        if critic_output is None:
            return None
        return self.parse_output(critic_output, true_error_sections, max_valid_section)