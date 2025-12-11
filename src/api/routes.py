"""
API routes for article generation endpoints.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

# Optional database imports
try:
    from sqlalchemy.orm import Session
    from src.models.base import get_db
    from src.repositories.article_repository import ArticleRepository
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    Session = None
    get_db = None
    ArticleRepository = None

from src.services.llm_service import llm_service
from src.services.image_service import image_service
from src.services.document_service import DocumentService
from src.utils.text_processing import TextProcessor
from src.utils.logger import logger
from src.config.settings import settings

router = APIRouter(tags=["articles"])


class ArticleRequest(BaseModel):
    """Request model for article generation."""
    topic: str = Field(..., min_length=3, max_length=500, description="Article topic")
    word_limit: Optional[int] = Field(None, ge=200, le=2000, description="Target word count")
    image_query: Optional[str] = Field(None, max_length=100, description="Image search keywords")
    
    @validator('topic')
    def topic_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        return v.strip()


class ArticleResponse(BaseModel):
    """Response model for generated article."""
    success: bool
    article_id: Optional[str] = None
    topic: str
    content: str
    word_count: int
    keywords: List[tuple] = []
    metadata: dict = {}


class ArticleListResponse(BaseModel):
    """Response for article list."""
    articles: List[dict]
    total: int


@router.post("/articles/generate", response_model=ArticleResponse)
async def generate_article(
    request: ArticleRequest,
    db: Session = Depends(get_db) if DB_AVAILABLE else None
):
    """
    Generate an article on the given topic.
    
    This endpoint handles the full pipeline:
    - LLM-based article generation
    - Text processing and analysis
    - Optional image retrieval
    - Database persistence
    """
    try:
        word_limit = request.word_limit or settings.DEFAULT_WORD_LIMIT
        
        # Generate article with validation
        article_result = llm_service.generate_article(request.topic, word_limit)
        
        if not article_result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=article_result.get('error', 'Failed to generate article')
            )
        
        generated_text = article_result['text']
        actual_word_count = article_result.get('word_count', len(generated_text.split()))
        
        # Process text
        processor = TextProcessor()
        cleaned_text = processor.clean_text(generated_text)
        keywords = processor.extract_keywords(cleaned_text, top_n=10)
        
        # Validate with proper word count range
        min_acceptable = max(settings.MIN_WORD_COUNT, int(word_limit * 0.8))
        validation = processor.validate_text_quality(cleaned_text, min_words=min_acceptable)
        validation['word_count'] = actual_word_count
        
        # Check for warnings
        warnings = []
        if 'warning' in article_result:
            warnings.append(article_result['warning'])
        
        max_acceptable = word_limit + 100
        if actual_word_count < min_acceptable:
            warnings.append(f"Article shorter than expected: {actual_word_count} words (target: {word_limit})")
        elif actual_word_count > max_acceptable:
            warnings.append(f"Article longer than expected: {actual_word_count} words (target: {word_limit})")
        
        # Get image if requested (non-blocking, optional)
        image_url = None
        if request.image_query:
            image_result = image_service.get_image_url(request.image_query)
            if image_result.get('success'):
                image_url = image_result['url']
        
        # Save to database if available
        article_id = None
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'word_limit': word_limit,
            'image_url': image_url,
            'readability': validation.get('readability', {}),
            'keywords': [{"word": k[0], "count": k[1]} for k in keywords]
        }
        
        if DB_AVAILABLE and db:
            try:
                repo = ArticleRepository(db)
                article = repo.create(
                    topic=request.topic,
                    content=generated_text,
                    word_count=validation['word_count'],
                    metadata=metadata
                )
                article_id = str(article.id)
            except Exception as e:
                logger.warning(f"Failed to save to database: {e}")
        
        # Add warnings to metadata if any
        if warnings:
            metadata['warnings'] = warnings
            metadata['completion_status'] = 'partial' if actual_word_count < min_acceptable else 'complete'
        else:
            metadata['completion_status'] = 'complete'
        
        return ArticleResponse(
            success=True,
            article_id=article_id,
            topic=request.topic,
            content=generated_text,
            word_count=validation['word_count'],
            keywords=keywords,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating article: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/articles/{article_id}")
async def get_article(article_id: str, db: Session = Depends(get_db) if DB_AVAILABLE else None):
    """Get a specific article by ID."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        repo = ArticleRepository(db)
        article = repo.get_by_id(int(article_id))
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article.to_dict()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid article ID")


@router.get("/articles", response_model=ArticleListResponse)
async def list_articles(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db) if DB_AVAILABLE else None
):
    """List all articles with pagination."""
    if not DB_AVAILABLE:
        return ArticleListResponse(articles=[], total=0)
    
    repo = ArticleRepository(db)
    articles = repo.get_all(limit=min(limit, 100), offset=offset)
    total = repo.count()
    
    return ArticleListResponse(
        articles=[a.to_dict() for a in articles],
        total=total
    )


@router.post("/articles/{article_id}/download")
async def download_article(article_id: str):
    """Download article as Word document."""
    # TODO: implement when database is added
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/keywords")
async def extract_keywords(text: str, top_n: int = 10):
    """Extract keywords from provided text."""
    if not text or len(text) < 10:
        raise HTTPException(status_code=400, detail="Text too short")
    
    try:
        processor = TextProcessor()
        keywords = processor.extract_keywords(text, top_n=min(top_n, 50))
        return {"keywords": keywords, "count": len(keywords)}
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check with service status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "llm": "unknown",  # TODO: add actual health checks
            "image": "unknown"
        }
    }
    
    # Quick check if services are configured
    if settings.GROQ_API_KEY:
        health_status["services"]["llm"] = "configured"
    if settings.PEXELS_API_KEY:
        health_status["services"]["image"] = "configured"
    
    return health_status

