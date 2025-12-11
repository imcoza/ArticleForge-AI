"""
Text preprocessing and postprocessing utilities.
Demonstrates NLP engineering best practices for text manipulation.
"""
import re
import string
from typing import List, Dict, Optional, Any
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from src.utils.logger import logger

# Try to import spaCy, but handle import errors gracefully
try:
    import spacy
    SPACY_AVAILABLE = True
except (ImportError, Exception) as e:
    spacy = None
    SPACY_AVAILABLE = False
    logger.warning(f"spaCy not available: {e}. NER features will be limited.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Download punkt_tab (required for newer NLTK versions)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass  # punkt_tab may not be available in all NLTK versions

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize NLP components
nlp = None
if SPACY_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_sm")
    except (OSError, Exception):
        nlp = None  # Will use NLTK fallback

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def _tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of tokenized words
    """
    tokens = word_tokenize(text.lower())
    # Filter out punctuation and short words
    tokens = [
        token for token in tokens
        if token not in string.punctuation and len(token) > 2
    ]
    return tokens


class TextProcessor:
    """Text preprocessing and analysis utilities."""
    
    @staticmethod
    def clean_text(text: str, remove_stopwords: bool = False) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation for readability
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if remove_stopwords:
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in stop_words]
            text = ' '.join(tokens)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10, use_tfidf: bool = True) -> List[tuple]:
        """
        Extract top keywords from text using TF-IDF algorithm.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            use_tfidf: Whether to use true TF-IDF (requires corpus) or frequency fallback
            
        Returns:
            List of (keyword, score) tuples where score is TF-IDF or frequency
        """
        if use_tfidf:
            try:
                from src.services.ml.tfidf_processor import TFIDFProcessor
                processor = TFIDFProcessor()
                # Use single-document mode (IDF = 1, so effectively TF)
                keywords = processor.extract_keywords_tfidf(text, top_n, use_corpus_idf=False)
                if keywords:
                    return keywords
            except Exception as e:
                logger.warning(f"TF-IDF extraction failed: {e}. Using frequency fallback.")
        
        # Fallback to frequency-based (original implementation)
        cleaned_text = TextProcessor.clean_text(text, remove_stopwords=True)
        tokens = word_tokenize(cleaned_text)
        
        # Filter out punctuation and short words
        tokens = [token for token in tokens 
                 if token not in string.punctuation and len(token) > 2]
        
        # Count frequencies
        word_freq = Counter(tokens)
        
        # Return top N keywords with frequency as score
        return word_freq.most_common(top_n)
    
    @staticmethod
    def calculate_readability_metrics(text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with readability metrics
        """
        return {
            'flesch_reading_ease': flesch_reading_ease(text),
            'flesch_kincaid_grade': flesch_kincaid_grade(text),
            'automated_readability_index': automated_readability_index(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_sentence_length': len(text.split()) / max(len(sent_tokenize(text)), 1)
        }
    
    @staticmethod
    def extract_entities(text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with text, label, and start/end positions
        """
        entities = []
        use_spacy = False
        
        # Check if spaCy is available and loaded
        if SPACY_AVAILABLE and nlp is not None:
            try:
                doc = nlp(text)
                use_spacy = True
                # Filter out common false positives
                common_words = {'this', 'that', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 
                              'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                              'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                              'would', 'could', 'should', 'may', 'might', 'must', 'can'}
                
                # Expanded common words list for spaCy filtering
                expanded_common_words = {
                    'this', 'that', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'one',
                    'over', 'under', 'above', 'below', 'through', 'during', 'before', 'after',
                    'advances', 'understanding', 'protecting', 'addressing', 'equally',
                    'species', 'forests', 'international', 'climate', 'change', 'panel',
                    'intergovernmental', 'earth', 'yat', 'equally', 'addressing', 'protecting'
                }
                
                for ent in doc.ents:
                    ent_text_lower = ent.text.lower().strip()
                    
                    # Filter out single-word entities that are common words
                    if len(ent.text.split()) == 1 and ent_text_lower in expanded_common_words:
                        continue
                    # Filter out very short entities (likely false positives)
                    if len(ent.text.strip()) < 3:
                        continue
                    # Filter out entities that are just punctuation or numbers
                    if ent.text.strip().isdigit() or not any(c.isalpha() for c in ent.text):
                        continue
                    # Filter out common false positive patterns
                    if ent_text_lower in expanded_common_words:
                        continue
                    # Filter out entities that start with common words
                    if any(ent.text.lower().startswith(word + ' ') for word in ['the ', 'a ', 'an ', 'this ', 'that ']):
                        if len(ent.text.split()) <= 2:  # Short phrases are likely not entities
                            continue
                    # Filter out certain label types that are often wrong (like GPE for common words)
                    if ent.label_ == 'GPE' and len(ent.text.split()) == 1:
                        # Only accept single-word GPE if it's a known place or long enough
                        if len(ent.text) < 5 or ent_text_lower in expanded_common_words:
                            continue
                    
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            except Exception as e:
                logger.warning(f"Error in spaCy NER: {e}. Using fallback.")
                use_spacy = False
        
        # Fallback: improved pattern matching (only if spaCy failed or no entities found)
        if not use_spacy or not entities:
            # Comprehensive common words to exclude (expanded list)
            common_words = {
                # Articles and determiners
                'the', 'a', 'an', 'this', 'that', 'these', 'those',
                # Prepositions
                'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into', 'onto',
                # Conjunctions
                'and', 'or', 'but', 'so', 'yet', 'nor',
                # Pronouns
                'it', 'its', 'he', 'she', 'they', 'them', 'we', 'us', 'you', 'i', 'me',
                # Common verbs
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
                # Common nouns (context-dependent, but these are too generic)
                'one', 'two', 'three', 'first', 'second', 'third', 'time', 'way', 'year', 'work',
                'day', 'man', 'thing', 'world', 'life', 'hand', 'part', 'child', 'eye', 'woman',
                'place', 'case', 'point', 'government', 'company', 'number', 'group', 'problem',
                'fact', 'system', 'data', 'model', 'algorithm', 'machine', 'learning', 'technology',
                # Common adjectives/adverbs
                'new', 'old', 'good', 'great', 'small', 'large', 'long', 'short', 'high', 'low',
                'more', 'most', 'less', 'least', 'many', 'much', 'some', 'any', 'all', 'every',
                # Sentence starters (often capitalized but not entities)
                'over', 'under', 'above', 'below', 'through', 'during', 'before', 'after',
                'advances', 'understanding', 'protecting', 'addressing', 'equally', 'species',
                'forests', 'international', 'climate', 'change', 'panel', 'intergovernmental'
            }
            
            # More restrictive patterns - only match likely entities
            patterns = {
                'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',  # Full names (First Last or First Middle Last)
                'ORG': r'\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+)* )?(?:Inc|Corp|LLC|Ltd|Company|Corporation|University|Institute|Organization|Foundation|Society|Association)\b',
                # GPE: Only match known place patterns or multi-word capitalized phrases that aren't common
                'GPE': r'\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+)+)\b',  # Multi-word place names only
            }
            
            # Known place names and organizations (whitelist approach for single words)
            known_places = {
                'arctic', 'antarctic', 'africa', 'asia', 'europe', 'america', 'australia',
                'london', 'paris', 'tokyo', 'beijing', 'moscow', 'berlin', 'rome', 'madrid',
                'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
                'united states', 'united kingdom', 'south africa', 'south korea', 'north korea',
                'united nations', 'european union', 'nato', 'who', 'unicef'
            }
            
            found_entities = set()  # Avoid duplicates
            
            for label, pattern in patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_text = match.group().strip()
                    entity_lower = entity_text.lower()
                    
                    # Skip if it's a common word
                    if entity_lower in common_words:
                        continue
                    # Skip if it's too short
                    if len(entity_text) < 3:
                        continue
                    # Skip if it's just numbers
                    if entity_text.isdigit():
                        continue
                    # Skip if it's a single word that's not a known place
                    if label == 'GPE' and len(entity_text.split()) == 1 and entity_lower not in known_places:
                        continue
                    # Skip if it starts with common sentence starters
                    if entity_text.lower().startswith(('the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ')):
                        continue
                    # Skip if already found
                    if entity_text in found_entities:
                        continue
                    # Skip if it's a common phrase pattern
                    if re.match(r'^(The|A|An|This|That|These|Those|One|Over|Under|Above|Below)\s+[A-Z]', entity_text):
                        if len(entity_text.split()) <= 2:  # Short phrases like "The One" are likely not entities
                            continue
                    
                    found_entities.add(entity_text)
                    entities.append({
                        'text': entity_text,
                        'label': label,
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Remove duplicates and sort by position
        seen = set()
        unique_entities = []
        for ent in entities:
            key = (ent['text'].lower(), ent['start'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        # Sort by start position
        unique_entities.sort(key=lambda x: x['start'])
        
        return unique_entities[:20]  # Limit to top 20 entities
    
    @staticmethod
    def summarize_text(text: str, max_sentences: int = 3, method: str = "textrank") -> str:
        """
        Advanced extractive summarization using TextRank or simple method.
        
        Args:
            text: Input text
            max_sentences: Maximum number of sentences in summary
            method: Summarization method ("textrank" or "simple")
            
        Returns:
            Summarized text
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        if method == "textrank":
            try:
                from src.services.ml.advanced_nlp import TextRankSummarizer
                return TextRankSummarizer.summarize_textrank(text, max_sentences)
            except Exception as e:
                logger.warning(f"TextRank summarization failed: {e}. Using simple method.")
                method = "simple"
        
        # Simple approach: return first and last sentences
        if method == "simple":
            summary_sentences = sentences[:max_sentences//2] + sentences[-max_sentences//2:]
            return ' '.join(summary_sentences)
        
        # Default fallback
        return ' '.join(sentences[:max_sentences])
    
    @staticmethod
    def validate_text_quality(text: str, min_words: int = 200) -> Dict[str, Any]:
        """
        Validate text quality and provide feedback.
        
        Args:
            text: Input text
            min_words: Minimum word count requirement
            
        Returns:
            Dictionary with validation results
        """
        word_count = len(text.split())
        sentence_count = len(sent_tokenize(text))
        
        issues = []
        warnings = []  # Warnings removed as requested - no warnings will be generated
        
        if word_count < min_words:
            issues.append(f"Text too short: {word_count} words (minimum: {min_words})")
        
        readability = TextProcessor.calculate_readability_metrics(text)
        
        return {
            'is_valid': len(issues) == 0,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'issues': issues,
            'warnings': warnings,  # Always empty now
            'readability': readability
        }

