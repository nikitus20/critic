"""Factory pattern for creating different critic types."""

from typing import Dict, Optional, Type

from .base_critic import BaseCritic
from .direct_critic import DirectCritic
from .pedcot_critic import PedCoTCritic


class CriticFactory:
    """Factory for creating different types of critics."""
    
    # Registry of available critic types
    CRITIC_TYPES = {
        'direct': DirectCritic,
        'deltabench': DirectCritic,  # Alias for backward compatibility
        'llm': DirectCritic,         # Alias for backward compatibility
        'pedcot': PedCoTCritic,
        'pedagogical': PedCoTCritic  # Alias
    }
    
    @classmethod
    def create_critic(cls, critic_type: str, model: str, config: Optional[Dict] = None) -> BaseCritic:
        """
        Create a critic of the specified type.
        
        Args:
            critic_type: Type of critic to create ('direct', 'pedcot', etc.)
            model: Model name to use for the critic
            config: Optional configuration dictionary
            
        Returns:
            Configured critic instance
            
        Raises:
            ValueError: If critic_type is not recognized
        """
        critic_type = critic_type.lower()
        
        if critic_type not in cls.CRITIC_TYPES:
            available_types = list(cls.CRITIC_TYPES.keys())
            raise ValueError(f"Unknown critic type: {critic_type}. Available types: {available_types}")
        
        critic_class = cls.CRITIC_TYPES[critic_type]
        
        # Create and configure the critic
        if critic_type in ['direct', 'deltabench', 'llm']:
            # DirectCritic uses prompt_type parameter
            prompt_type = config.get('prompt_type', 'deltabench') if config else 'deltabench'
            return critic_class(model=model, prompt_type=prompt_type, config_dict=config)
        else:
            # PedCoTCritic uses config_dict parameter
            return critic_class(model=model, config_dict=config)
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available critic types."""
        return list(cls.CRITIC_TYPES.keys())
    
    @classmethod
    def register_critic_type(cls, name: str, critic_class: Type[BaseCritic]) -> None:
        """
        Register a new critic type.
        
        Args:
            name: Name to register the critic under
            critic_class: Critic class that inherits from BaseCritic
        """
        if not issubclass(critic_class, BaseCritic):
            raise ValueError(f"Critic class must inherit from BaseCritic")
        
        cls.CRITIC_TYPES[name.lower()] = critic_class
    
    @classmethod
    def get_critic_info(cls, critic_type: str) -> Dict:
        """
        Get information about a specific critic type.
        
        Args:
            critic_type: Type of critic to get info for
            
        Returns:
            Dictionary with critic information
        """
        critic_type = critic_type.lower()
        
        if critic_type not in cls.CRITIC_TYPES:
            raise ValueError(f"Unknown critic type: {critic_type}")
        
        critic_class = cls.CRITIC_TYPES[critic_type]
        
        info = {
            'name': critic_type,
            'class': critic_class.__name__,
            'module': critic_class.__module__,
            'description': critic_class.__doc__ or 'No description available'
        }
        
        # Add specific information based on critic type
        if critic_type in ['direct', 'deltabench', 'llm']:
            info['type'] = 'Single-stage direct prompting critic'
            info['supports_prompt_types'] = ['deltabench', 'pedcot']
        elif critic_type in ['pedcot', 'pedagogical']:
            info['type'] = 'Two-stage pedagogical critic with Bloom\'s taxonomy'
            info['features'] = ['Pedagogical principles', 'Two-stage interaction', 'Domain awareness']
        
        return info


# Convenience function for backward compatibility
def create_critic(critic_type: str, model: str, config: Optional[Dict] = None) -> BaseCritic:
    """
    Convenience function to create a critic.
    
    Args:
        critic_type: Type of critic to create
        model: Model name to use
        config: Optional configuration
        
    Returns:
        Configured critic instance
    """
    return CriticFactory.create_critic(critic_type, model, config)