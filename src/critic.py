"""LLM critic for evaluating reasoning steps."""

# Import the new DirectCritic for backward compatibility
from .critics.direct_critic import DirectCritic, LLMCritic

__all__ = ['LLMCritic', 'DirectCritic']