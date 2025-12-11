"""
Image service for fetching images from Pexels API.
Handles image retrieval and caching.
"""
import requests
from typing import Optional, Dict, Any
from src.config.settings import settings
from src.utils.logger import logger


class ImageService:
    """Service for managing image operations."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.PEXELS_API_KEY
        self.base_url = "https://api.pexels.com/v1/search"
    
    def get_image_url(self, query: str, per_page: int = 1) -> Dict[str, Any]:
        """
        Fetch image URL from Pexels API.
        
        Args:
            query: Search query for images
            per_page: Number of images to fetch
            
        Returns:
            Dictionary with image URL and metadata
        """
        if not self.api_key:
            logger.warning("Pexels API key not configured")
            return {
                'success': False,
                'url': None,
                'error': 'API key not configured'
            }
        
        try:
            # Pexels API authentication format: Authorization: YOUR_API_KEY
            # Note: Pexels uses the API key directly, NOT "Bearer YOUR_API_KEY"
            # Reference: https://www.pexels.com/api/documentation/
            headers = {
                'Authorization': self.api_key if self.api_key else None
            }
            if not headers['Authorization']:
                return {
                    'success': False,
                    'url': None,
                    'error': 'API key not configured'
                }
            params = {
                'query': query,
                'per_page': per_page,
            }
            
            logger.info(f"Fetching image for query: {query}")
            response = requests.get(self.base_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                photos = data.get('photos', [])
                
                if photos:
                    photo = photos[0]
                    image_url = photo['src']['original']
                    photographer = photo.get('photographer', 'Unknown')
                    
                    logger.info(f"Image fetched successfully: {image_url}")
                    
                    return {
                        'success': True,
                        'url': image_url,
                        'photographer': photographer,
                        'alt': photo.get('alt', query),
                        'width': photo.get('width'),
                        'height': photo.get('height')
                    }
                else:
                    logger.warning(f"No photos found for query: {query}")
                    return {
                        'success': False,
                        'url': None,
                        'error': 'No photos found for the given query'
                    }
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'url': None,
                    'error': error_msg
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'url': None,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'url': None,
                'error': error_msg
            }
    
    def download_image(self, url: str) -> Optional[bytes]:
        """
        Download image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            Image bytes or None if download fails
        """
        try:
            logger.info(f"Downloading image from: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}", exc_info=True)
            return None


# Global service instance
image_service = ImageService()

