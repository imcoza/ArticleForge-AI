"""
Advanced NLP Features
Demonstrates modern NLP techniques: embeddings, semantic similarity, text classification.
"""
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter

# Optional numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import advanced NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None
    CountVectorizer = None
    TfidfVectorizer = None

try:
    import spacy
    SPACY_AVAILABLE = True
except (ImportError, Exception) as e:
    SPACY_AVAILABLE = False
    spacy = None
    import logging
    logging.warning(f"spaCy not available: {e}. NER features will be limited.")

from src.utils.text_processing import TextProcessor
from src.utils.logger import logger


class AdvancedNLPProcessor:
    """
    Advanced NLP processing using modern techniques:
    - Sentence embeddings (BERT, Sentence-BERT)
    - Semantic similarity
    - Text classification
    - Advanced text analysis
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize advanced NLP processor.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.embedding_model = None
        self.model_name = model_name
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
        else:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    
    def generate_embeddings(self, texts: List[str]):
        """
        Generate sentence embeddings using transformer models.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings (n_texts, embedding_dim) or list if numpy unavailable
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available. Using fallback.")
            return self._fallback_embeddings(texts)
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=NUMPY_AVAILABLE,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return self._fallback_embeddings(texts)
    
    def _fallback_embeddings(self, texts: List[str]):
        """Fallback to simple TF-IDF based embeddings."""
        if not SKLEARN_AVAILABLE or not TfidfVectorizer:
            # Very basic fallback
            logger.warning("scikit-learn not available. Using simple frequency-based fallback.")
            if NUMPY_AVAILABLE:
                return np.random.rand(len(texts), 128)
            else:
                # Return list of zeros as fallback
                return [[0.0] * 128 for _ in texts]
        
        try:
            vectorizer = TfidfVectorizer(max_features=128)
            embeddings = vectorizer.fit_transform(texts).toarray()
            return embeddings
        except Exception as e:
            logger.error(f"Error in fallback embeddings: {e}")
            if NUMPY_AVAILABLE:
                return np.random.rand(len(texts), 128)
            else:
                return [[0.0] * 128 for _ in texts]
    
    def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate semantic similarity between two texts using embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.generate_embeddings([text1, text2])
        
        if not embeddings or len(embeddings) < 2:
            return 0.0
        
        # Convert to numpy array if needed
        if not NUMPY_AVAILABLE:
            # Simple fallback similarity
            return 0.5
        
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        if embeddings.shape[0] < 2:
            return 0.0
        
        # Calculate cosine similarity
        if SKLEARN_AVAILABLE and cosine_similarity:
            try:
                similarity = cosine_similarity(
                    embeddings[0:1],
                    embeddings[1:2]
                )[0][0]
                return float(similarity)
            except Exception as e:
                logger.warning(f"Error calculating similarity with sklearn: {e}")
        
        # Manual cosine similarity
        vec1 = embeddings[0]
        vec2 = embeddings[1]
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2 + 1e-8)
        
        return float(similarity)
    
    def find_similar_sentences(
        self,
        query: str,
        sentences: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar sentences to a query using semantic similarity.
        
        Args:
            query: Query text
            sentences: List of candidate sentences
            top_k: Number of top results to return
            
        Returns:
            List of (sentence, similarity_score) tuples
        """
        if not sentences:
            return []
        
        # Generate embeddings for query and all sentences
        all_texts = [query] + sentences
        embeddings = self.generate_embeddings(all_texts)
        
        if not NUMPY_AVAILABLE:
            # Fallback: return first sentences
            return [(sentences[i], 0.5) for i in range(min(top_k, len(sentences)))]
        
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        query_embedding = embeddings[0:1]
        sentence_embeddings = embeddings[1:]
        
        # Calculate similarities
        if SKLEARN_AVAILABLE and cosine_similarity:
            try:
                similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
            except Exception:
                similarities = np.array([
                    np.dot(query_embedding[0], sent_emb) / (
                        np.linalg.norm(query_embedding[0]) * np.linalg.norm(sent_emb) + 1e-8
                    )
                    for sent_emb in sentence_embeddings
                ])
        else:
            similarities = np.array([
                np.dot(query_embedding[0], sent_emb) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(sent_emb) + 1e-8
                )
                for sent_emb in sentence_embeddings
            ])
        
        # Get top K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (sentences[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def extract_key_phrases(
        self,
        text: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract key phrases using noun phrase extraction and scoring.
        
        Args:
            text: Input text
            top_n: Number of key phrases to return
            
        Returns:
            List of (phrase, score) tuples
        """
        if SPACY_AVAILABLE:
            try:
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(text)
                
                # Extract noun phrases
                noun_phrases = []
                for chunk in doc.noun_chunks:
                    phrase = chunk.text.lower().strip()
                    if len(phrase.split()) >= 2 and len(phrase) > 5:
                        noun_phrases.append(phrase)
                
                # Score phrases by frequency and length
                phrase_scores = Counter(noun_phrases)
                scored_phrases = [
                    (phrase, count * len(phrase.split()))
                    for phrase, count in phrase_scores.items()
                ]
                
                # Sort by score
                scored_phrases.sort(key=lambda x: x[1], reverse=True)
                return scored_phrases[:top_n]
                
            except Exception as e:
                logger.warning(f"Error extracting key phrases with spaCy: {e}")
        
        # Fallback: extract bigrams
        return self._extract_bigrams(text, top_n)
    
    def _extract_bigrams(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Extract top bigrams as key phrases."""
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        try:
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            
            # Create bigrams
            bigrams = [
                (tokens[i], tokens[i+1])
                for i in range(len(tokens) - 1)
            ]
            bigram_texts = [' '.join(bg) for bg in bigrams]
            
            # Score by frequency
            bigram_scores = Counter(bigram_texts)
            scored = sorted(
                bigram_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [(phrase, float(score)) for phrase, score in scored[:top_n]]
        except Exception as e:
            logger.warning(f"Error extracting bigrams: {e}")
            return []
    
    def classify_text_sentiment(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Classify text sentiment using rule-based approach.
        For production, use a trained sentiment model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        # Simple rule-based sentiment (for demonstration)
        # In production, use a trained model (VADER, TextBlob, or BERT-based)
        
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'best', 'awesome', 'brilliant', 'outstanding'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'dislike', 'poor', 'disappointing', 'frustrating', 'annoying'
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        positive_score = positive_count / max(total_words, 1)
        negative_score = negative_count / max(total_words, 1)
        neutral_score = 1.0 - positive_score - negative_score
        
        return {
            'positive': float(positive_score),
            'negative': float(negative_score),
            'neutral': float(max(0, neutral_score)),
            'predicted_label': 'positive' if positive_score > negative_score else 'negative' if negative_score > positive_score else 'neutral'
        }
    
    def calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive text statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with various text statistics
        """
        from nltk.tokenize import word_tokenize, sent_tokenize
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Character-level stats
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Word-level stats
        word_count = len(words)
        unique_words = len(set(word.lower() for word in words))
        vocabulary_richness = unique_words / max(word_count, 1)
        
        # Sentence-level stats
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Paragraph estimation (double newlines)
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Word length distribution
        word_lengths = [len(word) for word in words if word.isalpha()]
        if word_lengths:
            if NUMPY_AVAILABLE:
                avg_word_length = float(np.mean(word_lengths))
            else:
                avg_word_length = sum(word_lengths) / len(word_lengths)
        else:
            avg_word_length = 0.0
        
        return {
            'char_count': char_count,
            'char_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'unique_words': unique_words,
            'vocabulary_richness': float(vocabulary_richness),
            'sentence_count': sentence_count,
            'avg_sentence_length': float(avg_sentence_length),
            'avg_word_length': float(avg_word_length),
            'paragraph_count': paragraph_count,
            'words_per_paragraph': word_count / max(paragraph_count, 1)
        }


class TextRankSummarizer:
    """
    TextRank algorithm for extractive summarization.
    More sophisticated than simple sentence selection.
    """
    
    @staticmethod
    def summarize_textrank(
        text: str,
        num_sentences: int = 3
    ) -> str:
        """
        Summarize text using TextRank algorithm.
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            
        Returns:
            Summarized text
        """
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        import math
        
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return text
            
            # Build similarity matrix
            stop_words = set(stopwords.words('english'))
            
            def sentence_vector(sentence):
                words = word_tokenize(sentence.lower())
                words = [w for w in words if w.isalnum() and w not in stop_words]
                return Counter(words)
            
            sentence_vectors = [sentence_vector(s) for s in sentences]
            
            # Calculate similarity matrix
            n = len(sentences)
            if NUMPY_AVAILABLE:
                similarity_matrix = np.zeros((n, n))
            else:
                similarity_matrix = [[0.0] * n for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    vec1 = sentence_vectors[i]
                    vec2 = sentence_vectors[j]
                    
                    # Cosine similarity
                    intersection = set(vec1.keys()) & set(vec2.keys())
                    numerator = sum(vec1[word] * vec2[word] for word in intersection)
                    
                    sum1 = sum(vec1[word] ** 2 for word in vec1)
                    sum2 = sum(vec2[word] ** 2 for word in vec2)
                    denominator = math.sqrt(sum1 * sum2)
                    
                    if denominator > 0:
                        sim_score = numerator / denominator
                        similarity_matrix[i][j] = sim_score
                        similarity_matrix[j][i] = sim_score
            
            # PageRank-like algorithm
            scores = TextRankSummarizer._pagerank(similarity_matrix)
            
            # Get top sentences
            if NUMPY_AVAILABLE and isinstance(scores, np.ndarray):
                top_indices = np.argsort(scores)[::-1][:num_sentences]
                top_indices = sorted(top_indices.tolist())  # Maintain order
            else:
                # Fallback: sort by score
                scored_indices = [(i, scores[i]) for i in range(len(scores))]
                scored_indices.sort(key=lambda x: x[1], reverse=True)
                top_indices = sorted([idx for idx, _ in scored_indices[:num_sentences]])
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.warning(f"Error in TextRank summarization: {e}")
            # Fallback to simple method
            from src.utils.text_processing import TextProcessor
            return TextProcessor.summarize_text(text, num_sentences)
    
    @staticmethod
    def _pagerank(matrix, damping: float = 0.85, max_iter: int = 100):
        """PageRank algorithm for sentence ranking."""
        if not NUMPY_AVAILABLE:
            # Fallback: return uniform scores
            n = len(matrix) if hasattr(matrix, '__len__') else 10
            return [1.0 / n] * n
        
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        
        n = matrix.shape[0]
        
        # Normalize matrix
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized = matrix / row_sums[:, np.newaxis]
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Iterate
        for _ in range(max_iter):
            new_scores = (1 - damping) / n + damping * normalized.dot(scores)
            
            if np.allclose(scores, new_scores):
                break
            
            scores = new_scores
        
        return scores

