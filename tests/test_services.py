"""
Tests for service layer.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.services.image_service import ImageService
from src.services.document_service import DocumentService


class TestImageService:
    """Test cases for ImageService."""
    
    @patch('src.services.image_service.requests.get')
    def test_get_image_url_success(self, mock_get):
        """Test successful image URL retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'photos': [{
                'src': {'original': 'https://example.com/image.jpg'},
                'photographer': 'Test Photographer',
                'alt': 'Test image',
                'width': 1920,
                'height': 1080
            }]
        }
        mock_get.return_value = mock_response
        
        service = ImageService(api_key="test_key")
        result = service.get_image_url("test query")
        
        assert result['success'] is True
        assert result['url'] == 'https://example.com/image.jpg'
    
    @patch('src.services.image_service.requests.get')
    def test_get_image_url_no_results(self, mock_get):
        """Test image URL retrieval with no results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'photos': []}
        mock_get.return_value = mock_response
        
        service = ImageService(api_key="test_key")
        result = service.get_image_url("test query")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_get_image_url_no_api_key(self):
        """Test image URL retrieval without API key."""
        service = ImageService(api_key="")
        result = service.get_image_url("test query")
        
        assert result['success'] is False
        assert 'API key not configured' in result['error']


class TestDocumentService:
    """Test cases for DocumentService."""
    
    def test_create_word_document(self):
        """Test Word document creation."""
        doc = DocumentService.create_word_document(
            title="Test Article",
            content="This is test content.",
            metadata={'date': '2024-01-01', 'word_count': 4}
        )
        
        assert doc is not None
        assert len(doc.paragraphs) > 0
    
    def test_save_document_to_bytes(self):
        """Test document saving to bytes."""
        doc = DocumentService.create_word_document(
            title="Test",
            content="Test content"
        )
        
        doc_bytes = DocumentService.save_document_to_bytes(doc)
        
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 0






