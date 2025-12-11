"""
Repository for article data access.
Implements repository pattern for cleaner separation of concerns.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.models.article import Article


class ArticleRepository:
    """Repository for article operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, topic: str, content: str, word_count: int, metadata: dict = None) -> Article:
        """Create a new article record."""
        article = Article(
            topic=topic,
            content=content,
            word_count=word_count,
            metadata=metadata or {}
        )
        self.db.add(article)
        self.db.commit()
        self.db.refresh(article)
        return article
    
    def get_by_id(self, article_id: int) -> Optional[Article]:
        """Get article by ID."""
        return self.db.query(Article).filter(Article.id == article_id).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Article]:
        """Get all articles with pagination."""
        return self.db.query(Article).order_by(desc(Article.created_at)).offset(offset).limit(limit).all()
    
    def get_by_topic(self, topic: str, limit: int = 10) -> List[Article]:
        """Get articles by topic (fuzzy search)."""
        return self.db.query(Article).filter(
            Article.topic.ilike(f"%{topic}%")
        ).order_by(desc(Article.created_at)).limit(limit).all()
    
    def count(self) -> int:
        """Get total article count."""
        return self.db.query(Article).count()
    
    def delete(self, article_id: int) -> bool:
        """Delete an article."""
        article = self.get_by_id(article_id)
        if article:
            self.db.delete(article)
            self.db.commit()
            return True
        return False



