"""
Strategy pattern for different text processing approaches.
Allows swapping processing algorithms without changing client code.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.utils.text_processing import TextProcessor


class TextProcessingStrategy(ABC):
    """Base strategy for text processing."""
    
    @abstractmethod
    def extract_keywords(self, text: str, top_n: int = 10) -> List[tuple]:
        """Extract keywords from text."""
        pass
    
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """Perform full text analysis."""
        pass


class TFIDFStrategy(TextProcessingStrategy):
    """TF-IDF based keyword extraction strategy."""
    
    def __init__(self):
        self.processor = TextProcessor()
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[tuple]:
        """Extract keywords using TF-IDF approach."""
        cleaned = self.processor.clean_text(text, remove_stopwords=True)
        return self.processor.extract_keywords(cleaned, top_n=top_n)
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Full analysis using TF-IDF methods."""
        cleaned = self.processor.clean_text(text)
        keywords = self.extract_keywords(text)
        readability = self.processor.calculate_readability_metrics(text)
        validation = self.processor.validate_text_quality(text)
        
        return {
            "keywords": keywords,
            "readability": readability,
            "validation": validation,
            "method": "tfidf"
        }


class SimpleFrequencyStrategy(TextProcessingStrategy):
    """Simple word frequency based strategy (faster, less accurate)."""
    
    def __init__(self):
        self.processor = TextProcessor()
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[tuple]:
        """Extract keywords using simple frequency counting."""
        from collections import Counter
        from nltk.tokenize import word_tokenize
        
        cleaned = self.processor.clean_text(text, remove_stopwords=True)
        tokens = word_tokenize(cleaned)
        # Filter short words
        tokens = [t for t in tokens if len(t) > 3]
        freq = Counter(tokens)
        return freq.most_common(top_n)
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Full analysis using simple frequency method."""
        keywords = self.extract_keywords(text)
        readability = self.processor.calculate_readability_metrics(text)
        
        return {
            "keywords": keywords,
            "readability": readability,
            "method": "frequency"
        }


class TextProcessingContext:
    """Context class that uses a strategy."""
    
    def __init__(self, strategy: TextProcessingStrategy = None):
        self.strategy = strategy or TFIDFStrategy()
    
    def set_strategy(self, strategy: TextProcessingStrategy):
        """Change the processing strategy at runtime."""
        self.strategy = strategy
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text using the current strategy."""
        return self.strategy.analyze(text)



