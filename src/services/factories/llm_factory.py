"""
Factory for creating LLM service instances.
This allows easy switching between providers without changing client code.
"""
from src.services.llm_service import LLMService
from src.config.settings import settings
from src.utils.logger import logger


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str = None) -> LLMService:
        """
        Create an LLM service instance based on provider type.
        
        Args:
            provider_type: 'groq' or 'local'. Defaults to settings.LLM_PROVIDER
            
        Returns:
            LLMService instance
        """
        if provider_type is None:
            provider_type = settings.LLM_PROVIDER.lower()
        
        # For now, we just return the singleton instance
        # In a more complex setup, you might want different instances
        # This is a simple factory - can be extended later
        logger.info(f"Creating LLM provider: {provider_type}")
        
        service = LLMService()
        if provider_type != service.provider:
            # If different provider requested, we'd need to handle that
            # For now, just log it
            logger.warning(f"Requested {provider_type} but service is configured for {service.provider}")
        
        return service
    
    @staticmethod
    def get_available_providers():
        """Get list of available LLM providers."""
        providers = ["groq"]
        
        # Check if local model support is available
        try:
            from langchain_community.llms import CTransformers
            providers.append("local")
        except ImportError:
            pass
        
        return providers



