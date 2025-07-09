"""
DeltaBench: Simple framework for evaluating reasoning critics.
"""

from .data_loader import DeltaBenchDataset
from .critic import LLMCritic
from .evaluator import DeltaBenchEvaluator
from .visualizer import plot_results
from .utils import CriticResult
from .analysis_utils import display_example, display_critic_comparison, summarize_results
from .critics import DirectCritic, PedCoTCritic, CriticFactory, create_critic

__version__ = "0.2.0"
__all__ = [
    "DeltaBenchDataset",
    "LLMCritic", 
    "DirectCritic",
    "PedCoTCritic",
    "CriticFactory",
    "create_critic",
    "DeltaBenchEvaluator",
    "plot_results",
    "CriticResult",
    "display_example",
    "display_critic_comparison", 
    "summarize_results"
]