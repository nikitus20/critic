"""
DeltaBench: Simple framework for evaluating reasoning critics.
"""

from .data_loader import DeltaBenchDataset
from .critic import LLMCritic
from .evaluator import DeltaBenchEvaluator
from .visualizer import plot_results
from .utils import CriticResult
from .analysis_utils import display_example, display_critic_comparison, summarize_results

__version__ = "0.2.0"
__all__ = [
    "DeltaBenchDataset",
    "LLMCritic", 
    "DeltaBenchEvaluator",
    "plot_results",
    "CriticResult",
    "display_example",
    "display_critic_comparison", 
    "summarize_results"
]