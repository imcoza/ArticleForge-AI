"""
Tests for text processing utilities.
"""
import pytest
from src.utils.text_processing import TextProcessor


class TestTextProcessor:
    """Test cases for TextProcessor class."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        processor = TextProcessor()
        
        text = "Hello   World!!!  http://example.com test@email.com"
        cleaned = processor.clean_text(text)
        
        assert "http://example.com" not in cleaned
        assert "test@email.com" not in cleaned
        assert "Hello" in cleaned.lower()
        assert "World" in cleaned.lower()
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        processor = TextProcessor()
        
        text = "Python is a great programming language. Python is used for data science and machine learning."
        keywords = processor.extract_keywords(text, top_n=5)
        
        assert len(keywords) <= 5
        assert any("python" in word.lower() for word, _ in keywords)
    
    def test_calculate_readability_metrics(self):
        """Test readability metrics calculation."""
        processor = TextProcessor()
        
        text = "This is a simple sentence. It has multiple sentences. Each sentence is easy to read."
        metrics = processor.calculate_readability_metrics(text)
        
        assert 'flesch_reading_ease' in metrics
        assert 'word_count' in metrics
        assert 'sentence_count' in metrics
        assert metrics['word_count'] > 0
        assert metrics['sentence_count'] > 0
    
    def test_validate_text_quality(self):
        """Test text quality validation."""
        processor = TextProcessor()
        
        # Short text should have issues
        short_text = "This is short."
        validation = processor.validate_text_quality(short_text, min_words=200)
        assert not validation['is_valid']
        assert len(validation['issues']) > 0
        
        # Longer text should be valid
        long_text = " ".join(["This is a sentence."] * 50)
        validation = processor.validate_text_quality(long_text, min_words=200)
        assert validation['is_valid'] or len(validation['issues']) == 0
    
    def test_summarize_text(self):
        """Test text summarization."""
        processor = TextProcessor()
        
        text = ". ".join(["This is sentence number " + str(i) + "." for i in range(10)])
        summary = processor.summarize_text(text, max_sentences=3)
        
        assert len(summary) <= len(text)
        assert summary.count('.') <= 3






