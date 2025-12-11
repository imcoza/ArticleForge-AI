"""
True TF-IDF Implementation for Keyword Extraction
Demonstrates proper TF-IDF algorithm with corpus-based IDF calculation.
"""
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np

# Optional sklearn import
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None

from src.utils.text_processing import TextProcessor
from src.utils.logger import logger


class TFIDFProcessor:
    """
    True TF-IDF implementation for keyword extraction.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure
    used to evaluate how important a word is to a document in a collection of documents.
    
    Formula: TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
    - TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
    - IDF(t, D) = log(Total documents / Number of documents containing term t)
    """
    
    def __init__(self, corpus: Optional[List[str]] = None):
        """
        Initialize TF-IDF processor.
        
        Args:
            corpus: Optional list of documents to build IDF from.
                   If None, uses the document itself (single-document TF-IDF)
        """
        self.corpus = corpus or []
        self.idf_scores: Dict[str, float] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        
        if corpus:
            self._build_idf_from_corpus()
    
    def _build_idf_from_corpus(self):
        """Build IDF scores from corpus using scikit-learn or manual calculation."""
        if SKLEARN_AVAILABLE and TfidfVectorizer:
            try:
                self.vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2),  # Unigrams and bigrams
                    min_df=1,  # Minimum document frequency
                    max_df=0.95  # Maximum document frequency (remove very common words)
                )
                
                # Fit on corpus to build vocabulary and IDF
                self.vectorizer.fit(self.corpus)
                
                # Extract IDF scores
                feature_names = self.vectorizer.get_feature_names_out()
                idf_values = self.vectorizer.idf_
                
                self.idf_scores = dict(zip(feature_names, idf_values))
                logger.info(f"Built IDF from corpus of {len(self.corpus)} documents using scikit-learn")
                return
                
            except Exception as e:
                logger.warning(f"Error building IDF with scikit-learn: {e}. Using manual calculation.")
        else:
            logger.info("scikit-learn not available. Using manual IDF calculation.")
        
        # Fallback to manual calculation
        self._build_idf_manual()
    
    def _build_idf_manual(self):
        """Manual IDF calculation from corpus."""
        if not self.corpus:
            return
        
        # Tokenize all documents
        doc_tokens = []
        for doc in self.corpus:
            cleaned = TextProcessor.clean_text(doc, remove_stopwords=True)
            # Use helper function for tokenization
            from nltk.tokenize import word_tokenize
            import string
            tokens = word_tokenize(cleaned.lower())
            tokens = [t for t in tokens if t not in string.punctuation and len(t) > 2]
            doc_tokens.append(set(tokens))
        
        # Calculate document frequency for each term
        doc_freq = defaultdict(int)
        total_docs = len(self.corpus)
        
        for tokens in doc_tokens:
            for token in tokens:
                doc_freq[token] += 1
        
        # Calculate IDF: log(total_docs / doc_freq)
        for term, df in doc_freq.items():
            if df > 0:
                self.idf_scores[term] = math.log(total_docs / df)
            else:
                self.idf_scores[term] = 0.0
        
        logger.info(f"Built IDF manually from {total_docs} documents")
    
    def calculate_tf(self, text: str) -> Dict[str, float]:
        """
        Calculate Term Frequency (TF) for a document.
        
        Args:
            text: Input document text
            
        Returns:
            Dictionary mapping terms to their TF scores
        """
        cleaned = TextProcessor.clean_text(text, remove_stopwords=True)
        # Use helper function for tokenization
        from nltk.tokenize import word_tokenize
        import string
        tokens = word_tokenize(cleaned.lower())
        tokens = [t for t in tokens if t not in string.punctuation and len(t) > 2]
        
        if not tokens:
            return {}
        
        # Count term frequencies
        term_counts = Counter(tokens)
        total_terms = len(tokens)
        
        # Calculate TF: count / total_terms
        tf_scores = {
            term: count / total_terms
            for term, count in term_counts.items()
        }
        
        return tf_scores
    
    def calculate_tfidf(self, text: str, use_corpus_idf: bool = True) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for a document.
        
        Args:
            text: Input document text
            use_corpus_idf: Whether to use corpus-based IDF or single-document IDF
            
        Returns:
            Dictionary mapping terms to their TF-IDF scores
        """
        # Calculate TF
        tf_scores = self.calculate_tf(text)
        
        if not tf_scores:
            return {}
        
        # Calculate or retrieve IDF
        if use_corpus_idf and self.corpus:
            # Use pre-computed IDF from corpus
            idf_scores = self.idf_scores
        else:
            # Single-document IDF (IDF = 1 for all terms)
            # This is a simplified case - true IDF requires a corpus
            idf_scores = {term: 1.0 for term in tf_scores.keys()}
            logger.debug("Using single-document mode (IDF = 1.0 for all terms)")
        
        # Calculate TF-IDF: TF × IDF
        tfidf_scores = {}
        for term, tf in tf_scores.items():
            idf = idf_scores.get(term, 0.0)
            tfidf_scores[term] = tf * idf
        
        return tfidf_scores
    
    def extract_keywords_tfidf(
        self,
        text: str,
        top_n: int = 10,
        use_corpus_idf: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Extract top keywords using TF-IDF scores.
        
        Args:
            text: Input document text
            top_n: Number of top keywords to return
            use_corpus_idf: Whether to use corpus-based IDF
            
        Returns:
            List of (keyword, tfidf_score) tuples, sorted by score descending
        """
        tfidf_scores = self.calculate_tfidf(text, use_corpus_idf=use_corpus_idf)
        
        if not tfidf_scores:
            return []
        
        # Sort by TF-IDF score (descending)
        sorted_keywords = sorted(
            tfidf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_keywords[:top_n]
    
    def extract_keywords_with_scikit(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords using scikit-learn's TfidfVectorizer.
        More efficient and includes advanced features like n-grams.
        
        Args:
            text: Input document text
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, tfidf_score) tuples
        """
        if not SKLEARN_AVAILABLE or not self.vectorizer:
            # Fallback to manual calculation
            return self.extract_keywords_tfidf(text, top_n)
        
        try:
            # Transform document to TF-IDF vector
            tfidf_matrix = self.vectorizer.transform([text])
            
            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Create (feature, score) pairs and sort
            feature_scores = list(zip(feature_names, scores))
            feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
            
            # Filter out zero scores and return top N
            non_zero = [(feat, score) for feat, score in feature_scores if score > 0]
            return non_zero[:top_n]
            
        except Exception as e:
            logger.warning(f"Error with scikit-learn TF-IDF: {e}. Using manual calculation.")
            return self.extract_keywords_tfidf(text, top_n)
    
    def update_corpus(self, new_documents: List[str]):
        """
        Update corpus and rebuild IDF scores.
        
        Args:
            new_documents: List of new documents to add to corpus
        """
        if new_documents:
            self.corpus.extend(new_documents)
            self._build_idf_from_corpus()
            logger.info(f"Updated corpus: {len(self.corpus)} total documents")



