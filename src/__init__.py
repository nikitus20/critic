"""
DeltaBench: Simple framework for evaluating reasoning critics.
"""

from .data_loader import DeltaBenchDataset
from .critic import LLMCritic
from .evaluator import DeltaBenchEvaluator
from .visualizer import plot_results
from .utils import CriticResult

__version__ = "0.1.0"
__all__ = [
    "DeltaBenchDataset",
    "LLMCritic", 
    "DeltaBenchEvaluator",
    "plot_results",
    "CriticResult"
]