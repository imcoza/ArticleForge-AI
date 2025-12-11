"""
Initialize database and create tables.
Run this once to set up the database schema.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.base import Base, engine
from src.models.article import Article
from src.utils.logger import logger


def init_database():
    """Create all database tables."""
    logger.info("Initializing database...")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    logger.info("Database initialized successfully!")
    logger.info("Tables created: articles")


if __name__ == "__main__":
    init_database()



