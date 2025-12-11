"""
LLM Service for article generation.
Handles model loading, inference, and prompt management.
Supports both Groq API and local CTransformers models.
"""
from typing import Dict, Optional, Any

# Groq API support
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Handle LangChain version differences (for local models)
try:
    # New LangChain structure (v0.1.0+) - modular imports
    from langchain_community.llms import CTransformers
    from langchain.chains import LLMChain
    try:
        from langchain.prompts import PromptTemplate
    except ImportError:
        from langchain import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Old LangChain structure (v0.0.x) - legacy imports
    try:
        from langchain.llms import CTransformers
        from langchain.chains import LLMChain
        from langchain import PromptTemplate
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        CTransformers = None
        LLMChain = None
        PromptTemplate = None

from src.config.settings import settings
from src.utils.logger import logger


class LLMService:
    """Service for managing LLM operations."""
    
    def __init__(self):
        self.llm_chain: Optional[LLMChain] = None
        self._model_loaded = False
        self.groq_client: Optional[Groq] = None
        self.provider = settings.LLM_PROVIDER.lower()
    
    def _initialize_groq(self) -> bool:
        """Initialize Groq API client."""
        if not GROQ_AVAILABLE:
            logger.error("Groq package not installed. Install with: pip install groq")
            return False
        
        if not settings.GROQ_API_KEY:
            logger.error("GROQ_API_KEY not set in environment variables")
            return False
        
        try:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("Groq API client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}", exc_info=True)
            return False
    
    def load_model(self) -> bool:
        """
        Load the LLM model based on provider setting.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.provider == "groq":
            return self._initialize_groq()
        
        # Local model loading (CTransformers)
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not installed. Install with: pip install langchain langchain-community ctransformers")
            return False
        
        try:
            logger.info(f"Loading local model: {settings.MODEL_NAME}")
            
            llm = CTransformers(
                model=settings.MODEL_NAME,
                model_file=settings.MODEL_FILE,
                model_type=settings.MODEL_TYPE,
                max_new_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                top_k=settings.TOP_K,
                context_length=4096,
                threads=4,
                gpu_layers=0  # Set to > 0 if GPU available
            )
            
            prompt_template = self._get_prompt_template()
            self.llm_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate.from_template(prompt_template)
            )
            
            self._model_loaded = True
            logger.info("Local model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}", exc_info=True)
            self._model_loaded = False
            return False
    
    def generate_article(self, topic: str, word_limit: int = None, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate an article on the given topic with validation.
        
        Args:
            topic: Article topic
            word_limit: Target word count (defaults to settings)
            max_retries: Maximum retry attempts if article is incomplete
            
        Returns:
            Dictionary with generated text and metadata
        """
        if word_limit is None:
            word_limit = settings.DEFAULT_WORD_LIMIT
        
        # Ensure word_limit is within acceptable range
        word_limit = max(settings.MIN_WORD_COUNT, min(word_limit, settings.MAX_WORD_COUNT))
        
        # Use Groq API if configured
        if self.provider == "groq":
            if not self.groq_client:
                if not self._initialize_groq():
                    return {
                        'success': False,
                        'text': '',
                        'error': 'Failed to initialize Groq API client'
                    }
            
            return self._generate_with_groq(topic, word_limit, max_retries)
        
        # Use local model (CTransformers)
        if not self._model_loaded:
            if not self.load_model():
                return {
                    'success': False,
                    'text': '',
                    'error': 'Failed to load model'
                }
        
        try:
            logger.info(f"Generating article for topic: {topic} using local model")
            
            # Update prompt with word limit
            prompt_template = self._get_prompt_template(word_limit)
            self.llm_chain.prompt = PromptTemplate.from_template(prompt_template)
            
            result = self.llm_chain.run(user_input=topic)
            
            # Extract text from result
            if isinstance(result, dict):
                text = result.get('text', str(result))
            else:
                text = str(result)
            
            # Validate local model output
            word_count = len(text.split())
            min_words = max(settings.MIN_WORD_COUNT, int(word_limit * 0.8))
            max_words = word_limit + 100
            
            validation = self._validate_article_completeness(text, word_count, min_words, max_words)
            
            if not validation['valid'] and word_count < min_words:
                # If too short, try increasing tokens and retry once
                logger.warning(f"Article too short ({word_count} words), retrying with more tokens...")
                # This would require regenerating, but for now we'll return with warning
                return {
                    'success': True,
                    'text': text,
                    'topic': topic,
                    'word_limit': word_limit,
                    'word_count': word_count,
                    'warning': validation['reason']
                }
            
            logger.info(f"Article generated successfully. Words: {word_count}, Target: {word_limit}")
            
            return {
                'success': True,
                'text': text,
                'topic': topic,
                'word_limit': word_limit,
                'word_count': word_count
            }
            
        except Exception as e:
            logger.error(f"Error generating article: {str(e)}", exc_info=True)
            return {
                'success': False,
                'text': '',
                'error': str(e)
            }
    
    def _generate_with_groq(self, topic: str, word_limit: int, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate article using Groq API with validation and retry logic.
        
        Args:
            topic: Article topic
            word_limit: Target word count
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with generated text and metadata
        """
        min_words = max(settings.MIN_WORD_COUNT, int(word_limit * 0.8))
        max_words = word_limit + 100  # Allow some tolerance
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating article for topic: {topic} using Groq API (attempt {attempt + 1}/{max_retries})")
                
                # Build prompt
                prompt = self._get_prompt_template(word_limit).format(user_input=topic)
                
                # Calculate max tokens for the response
                # Estimate: 1 token ≈ 0.75 words for English text
                # Prompt is now concise (~50 tokens), so we focus on response tokens
                # Add buffer: word_limit * 1.5 tokens for the article content
                # Then add extra buffer for safety and model variations
                estimated_response_tokens = int(word_limit * 1.5)  # Base estimate
                prompt_tokens_estimate = 100  # Concise prompt uses ~50-100 tokens
                safety_buffer = 500  # Extra buffer for model variations
                
                max_tokens = min(
                    estimated_response_tokens + prompt_tokens_estimate + safety_buffer,
                    32768  # Groq model limit
                )
                
                logger.debug(f"Token calculation: word_limit={word_limit}, max_tokens={max_tokens}, "
                           f"estimated_response={estimated_response_tokens}")
                
                # Create completion
                try:
                    completion = self.groq_client.chat.completions.create(
                        model=settings.GROQ_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=settings.TEMPERATURE,
                        max_completion_tokens=max_tokens,
                        top_p=settings.TOP_P,
                        reasoning_effort="medium",
                        stream=False,
                        stop=None
                    )
                except Exception as api_error:
                    # Log API error details
                    error_msg = str(api_error)
                    logger.error(f"Groq API call failed (attempt {attempt + 1}): {error_msg}")
                    
                    # Check for specific error types
                    if 'rate limit' in error_msg.lower() or '429' in error_msg:
                        if attempt < max_retries - 1:
                            import time
                            wait_time = (attempt + 1) * 3  # Longer wait for rate limits
                            logger.info(f"Rate limited, waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                    
                    # Re-raise to be caught by outer exception handler
                    raise
                
                # Extract text and finish reason from response
                text = completion.choices[0].message.content
                finish_reason = getattr(completion.choices[0], 'finish_reason', 'unknown')
                
                # Handle "length" finish_reason - response was truncated due to token limit
                if finish_reason == 'length':
                    logger.warning(
                        f"Response truncated due to token limit (attempt {attempt + 1}/{max_retries}). "
                        f"Current max_tokens: {max_tokens}, Word limit: {word_limit}"
                    )
                    
                    # If we have some content but it was truncated, increase token limit and retry
                    if attempt < max_retries - 1:
                        # Significantly increase token limit for next attempt
                        # Use a more aggressive multiplier to ensure completion
                        new_estimated = int(word_limit * 2.0)  # 2 tokens per word
                        new_max_tokens = min(new_estimated + 1000, 32768)  # Add buffer, cap at model limit
                        if new_max_tokens > max_tokens:
                            max_tokens = new_max_tokens
                            logger.info(f"Increasing max_tokens from {max_tokens - (new_max_tokens - max_tokens)} to {max_tokens} for retry...")
                            import time
                            time.sleep(1)  # Brief delay before retry
                            continue
                    
                    # If we have truncated content, use it with a warning
                    if text and len(text.strip()) > 100:
                        logger.warning(f"Using truncated response ({len(text)} chars). Consider reducing word_limit.")
                        # Continue with truncated content - it's better than nothing
                    else:
                        # No usable content even after truncation
                        # This happens when the prompt itself is too long, leaving no room for response
                        return {
                            'success': False,
                            'text': '',
                            'error': f'Response truncated due to token limit after {max_retries} attempts. '
                                   f'Try reducing word_limit (current: {word_limit}) or increase MAX_TOKENS in .env file. '
                                   f'The prompt may be consuming too many tokens, leaving insufficient room for the article.'
                        }
                
                # Debug: Log response details for empty responses
                if not text:
                    logger.warning(
                        f"Empty response from Groq API (attempt {attempt + 1}/{max_retries}). "
                        f"Finish reason: {finish_reason}, Model: {settings.GROQ_MODEL}"
                    )
                    
                    # Add a small delay before retry to avoid rate limiting
                    if attempt < max_retries - 1:
                        import time
                        wait_time = (attempt + 1) * 1  # Progressive delay: 1s, 2s, 3s
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    
                    return {
                        'success': False,
                        'text': '',
                        'error': f'Empty response from Groq API after {max_retries} attempts. Finish reason: {finish_reason}'
                    }
                
                # Validate the generated article
                word_count = len(text.split())
                is_complete = self._validate_article_completeness(text, word_count, min_words, max_words)
                
                if is_complete['valid']:
                    logger.info(f"Article generated successfully via Groq. Words: {word_count}, Target: {word_limit}")
                    return {
                        'success': True,
                        'text': text,
                        'topic': topic,
                        'word_limit': word_limit,
                        'word_count': word_count
                    }
                else:
                    # Validation failed - check if it's a critical issue or just word count
                    is_critical = 'incomplete' in is_complete['reason'].lower() or 'cut off' in is_complete['reason'].lower()
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Article validation failed: {is_complete['reason']}. Retrying... (attempt {attempt + 1})")
                        continue
                    else:
                        # Last attempt - be more lenient
                        # If it's just word count issue but article seems complete, accept it
                        if not is_critical and word_count >= min_words * 0.7:  # At least 70% of minimum
                            logger.warning(f"Article generated but slightly out of range: {is_complete['reason']}")
                            return {
                                'success': True,
                                'text': text,
                                'topic': topic,
                                'word_limit': word_limit,
                                'word_count': word_count,
                                'warning': f"Article generated but {is_complete['reason']}"
                            }
                        else:
                            # Critical issue or too short - return with warning but still success
                            logger.warning(f"Article generated but validation failed: {is_complete['reason']}")
                            return {
                                'success': True,
                                'text': text,
                                'topic': topic,
                                'word_limit': word_limit,
                                'word_count': word_count,
                                'warning': is_complete['reason']
                            }
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error generating article with Groq (attempt {attempt + 1}): {error_msg}", exc_info=True)
                
                # Check if it's a rate limit or temporary error
                if 'rate limit' in error_msg.lower() or '429' in error_msg:
                    if attempt < max_retries - 1:
                        import time
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        logger.info(f"Rate limited, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                
                # On last attempt, return the error
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'text': '',
                        'error': f"Groq API error after {max_retries} attempts: {error_msg}"
                    }
        
        # Should not reach here, but just in case
        return {
            'success': False,
            'text': '',
            'error': 'Failed to generate article after all retry attempts'
        }
    
    def _validate_article_completeness(self, text: str, word_count: int, min_words: int, max_words: int) -> Dict[str, Any]:
        """
        Validate that the article is complete and within word count range.
        Uses lenient validation - only flags critical issues.
        
        Args:
            text: Generated article text
            word_count: Actual word count
            min_words: Minimum acceptable word count
            max_words: Maximum acceptable word count
            
        Returns:
            Dictionary with validation result
        """
        # Lenient word count check - allow 70% of minimum as acceptable
        lenient_min = int(min_words * 0.7)
        
        if word_count < lenient_min:
            return {
                'valid': False,
                'reason': f'Article too short: {word_count} words (minimum: {min_words}, got: {word_count})'
            }
        
        # Allow up to 150% of target as maximum (more lenient)
        lenient_max = int(max_words * 1.5)
        if word_count > lenient_max:
            return {
                'valid': False,
                'reason': f'Article too long: {word_count} words (maximum: {max_words}, got: {word_count})'
            }
        
        # Check for incomplete sentences (ends without punctuation) - only if very short
        text_stripped = text.strip()
        if text_stripped and text_stripped[-1] not in '.!?' and word_count < lenient_min:
            return {
                'valid': False,
                'reason': 'Article appears incomplete - missing ending punctuation and too short'
            }
        
        # Only check for incomplete patterns if article is significantly short
        if word_count < lenient_min:
            incomplete_patterns = [
                '...',
                'to be continued',
                'more to come'
            ]
            text_lower = text.lower()
            for pattern in incomplete_patterns:
                if pattern in text_lower[-100:]:  # Check last 100 chars
                    return {
                        'valid': False,
                        'reason': f'Article may be incomplete - contains "{pattern}" near the end'
                    }
        
        # If word count is reasonable (even if not perfect), consider it valid
        if word_count >= lenient_min and word_count <= lenient_max:
            return {
                'valid': True,
                'reason': 'Article is complete and within acceptable word count range'
            }
        
        # Default to valid if we got here (lenient approach)
        return {
            'valid': True,
            'reason': 'Article generated successfully'
        }
    
    def _get_prompt_template(self, word_limit: int = None) -> str:
        """
        Get the prompt template for article generation.
        
        Args:
            word_limit: Target word count
            
        Returns:
            Prompt template string
        """
        if word_limit is None:
            word_limit = settings.DEFAULT_WORD_LIMIT
        
        min_words = max(settings.MIN_WORD_COUNT, int(word_limit * 0.8))  # At least 80% of target
        
        # Concise prompt to save tokens - essential instructions only
        return f"""Write a complete, professional article on: {{user_input}}

Requirements:
- Target: {word_limit} words (range: {min_words}-{word_limit + 50})
- Structure: Introduction, body paragraphs, conclusion
- Complete the article - do not cut off mid-sentence
- Use clear, engaging language with proper grammar
- Do NOT include author name, "by [name]", date, or publication metadata
- Start directly with the article content

Article:"""


# Global service instance
llm_service = LLMService()

