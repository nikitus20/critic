"""
Common utilities and data structures for DeltaBench.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class ReasoningStep:
    """Represents a single reasoning step"""
    step_number: int
    content: str
    is_error: bool = False
    error_type: Optional[str] = None


@dataclass
class ReasoningTrace:
    """Represents a complete reasoning trace"""
    question: str
    steps: List[ReasoningStep]
    final_answer: Optional[str] = None
    error_section_numbers: List[int] = None
    unuseful_section_numbers: List[int] = None


@dataclass
class CriticResult:
    """Results from critic evaluation"""
    predicted_error_sections: List[int]
    explanations: List[str]
    precision: float
    recall: float
    f1_score: float
    raw_output: str