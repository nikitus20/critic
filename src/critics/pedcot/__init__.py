"""PedCOT (Pedagogical Chain-of-Thought) critic implementation."""

from .pedagogical_principles import PedagogicalPrinciples
from .tip_processor import TIPProcessor
from .error_mapping import PedCoTErrorMapper

__all__ = ['PedagogicalPrinciples', 'TIPProcessor', 'PedCoTErrorMapper']