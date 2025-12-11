"""
Tests for configuration management.
"""
import pytest
from src.config.settings import Settings


class TestSettings:
    """Test cases for Settings class."""
    
    def test_settings_initialization(self):
        """Test that settings can be initialized."""
        settings = Settings()
        assert settings.MODEL_NAME is not None
        assert settings.MAX_TOKENS > 0
        assert 0 <= settings.TEMPERATURE <= 2
    
    def test_settings_validation(self):
        """Test settings validation."""
        settings = Settings()
        errors = settings.validate()
        
        # Should have at least one error if PEXELS_API_KEY is not set
        assert isinstance(errors, list)
    
    def test_settings_defaults(self):
        """Test that default values are set correctly."""
        settings = Settings()
        assert settings.DEFAULT_WORD_LIMIT == 800
        assert settings.MIN_WORD_COUNT == 200
        assert settings.MAX_WORD_COUNT == 2000






