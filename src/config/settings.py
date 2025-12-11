"""
Configuration settings management.
Handles environment variables, config files, and default values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


class Settings:
    """Application settings and configuration."""
    
    # API Keys
    PEXELS_API_KEY: str = os.getenv("PEXELS_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # LLM Provider Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "local"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "TheBloke/Llama-2-7B-Chat-GGML")
    MODEL_FILE: str = os.getenv("MODEL_FILE", "llama-2-7b-chat.ggmlv3.q8_0.bin")
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "llama")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("TOP_K", "50"))
    
    # Article Generation Settings
    DEFAULT_WORD_LIMIT: int = int(os.getenv("DEFAULT_WORD_LIMIT", "800"))
    MIN_WORD_COUNT: int = int(os.getenv("MIN_WORD_COUNT", "200"))
    MAX_WORD_COUNT: int = int(os.getenv("MAX_WORD_COUNT", "2000"))
    
    # Image Settings
    IMAGES_PER_PAGE: int = int(os.getenv("IMAGES_PER_PAGE", "1"))
    IMAGE_WIDTH_INCHES: float = float(os.getenv("IMAGE_WIDTH_INCHES", "4.0"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = str(LOGS_DIR / "app.log")
    
    # Streamlit Settings
    PAGE_TITLE: str = "Article Forge - NLP Engineering"
    PAGE_LAYOUT: str = "wide"
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration settings."""
        errors = []
        
        if not cls.PEXELS_API_KEY:
            errors.append("PEXELS_API_KEY is not set")
        
        if cls.MAX_TOKENS < 1:
            errors.append("MAX_TOKENS must be greater than 0")
        
        if not (0 <= cls.TEMPERATURE <= 2):
            errors.append("TEMPERATURE must be between 0 and 2")
        
        return errors


# Global settings instance
settings = Settings()
