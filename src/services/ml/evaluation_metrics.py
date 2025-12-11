"""
ML Evaluation Metrics
Demonstrates model evaluation using standard NLP metrics.
"""
from typing import List, Dict, Optional, Any
import math
from collections import Counter

# Optional imports for advanced metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    NLTK_METRICS_AVAILABLE = True
except ImportError:
    NLTK_METRICS_AVAILABLE = False
    sentence_bleu = None
    SmoothingFunction = None
    meteor_score = None

from src.utils.logger import logger


class EvaluationMetrics:
    """
    Calculate evaluation metrics for text generation and NLP tasks.
    Includes BLEU, ROUGE, METEOR, and custom metrics.
    """
    
    @staticmethod
    def calculate_bleu(reference: str, candidate: str, n_gram: int = 4) -> Dict[str, float]:
        """
        Calculate BLEU score for text generation.
        
        Args:
            reference: Reference text (ground truth)
            candidate: Generated text (prediction)
            n_gram: Maximum n-gram order (default: 4 for BLEU-4)
            
        Returns:
            Dictionary with BLEU scores for different n-grams
        """
        if not NLTK_METRICS_AVAILABLE:
            logger.warning("NLTK metrics not available. Install with: pip install nltk")
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        try:
            from nltk.tokenize import word_tokenize
            
            # Tokenize
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            # Calculate BLEU for different n-grams
            smoothing = SmoothingFunction().method1
            scores = {}
            
            for n in range(1, min(n_gram + 1, 5)):
                try:
                    score = sentence_bleu(
                        [ref_tokens],
                        cand_tokens,
                        weights=tuple([1.0/n] * n + [0.0] * (4-n)),
                        smoothing_function=smoothing
                    )
                    scores[f'bleu_{n}'] = float(score)
                except Exception as e:
                    logger.warning(f"Error calculating BLEU-{n}: {e}")
                    scores[f'bleu_{n}'] = 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in BLEU calculation: {e}")
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    @staticmethod
    def calculate_rouge_l(reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE-L (Longest Common Subsequence) score.
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        try:
            from nltk.tokenize import word_tokenize
            
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            # Calculate LCS
            def lcs_length(seq1, seq2):
                m, n = len(seq1), len(seq2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if seq1[i-1] == seq2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]
            
            lcs = lcs_length(ref_tokens, cand_tokens)
            
            if len(ref_tokens) == 0 or len(cand_tokens) == 0:
                return {'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}
            
            precision = lcs / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
            recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'rouge_l_precision': float(precision),
                'rouge_l_recall': float(recall),
                'rouge_l_f1': float(f1)
            }
            
        except Exception as e:
            logger.error(f"Error in ROUGE-L calculation: {e}")
            return {'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}
    
    @staticmethod
    def calculate_meteor(reference: str, candidate: str) -> float:
        """
        Calculate METEOR score.
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            METEOR score
        """
        if not NLTK_METRICS_AVAILABLE or meteor_score is None:
            return 0.0
        
        try:
            from nltk.tokenize import word_tokenize
            
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            score = meteor_score([ref_tokens], cand_tokens)
            return float(score)
            
        except Exception as e:
            logger.warning(f"Error calculating METEOR: {e}")
            return 0.0
    
    @staticmethod
    def calculate_perplexity(text: str, model_vocab: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate perplexity (simplified version).
        In production, use a trained language model.
        
        Args:
            text: Input text
            model_vocab: Optional vocabulary with word probabilities
            
        Returns:
            Perplexity score (lower is better)
        """
        try:
            from nltk.tokenize import word_tokenize
            from collections import Counter
            
            tokens = word_tokenize(text.lower())
            if len(tokens) == 0:
                return float('inf')
            
            # Simple unigram model
            word_counts = Counter(tokens)
            total_words = len(tokens)
            
            # Calculate log probability
            log_prob = 0.0
            for word, count in word_counts.items():
                prob = count / total_words
                log_prob += count * math.log(prob) if prob > 0 else 0
            
            # Perplexity = exp(-log_prob / N)
            avg_log_prob = log_prob / total_words
            perplexity = math.exp(-avg_log_prob) if avg_log_prob < 0 else float('inf')
            
            return float(perplexity)
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_text_quality_metrics(text: str) -> Dict[str, float]:
        """
        Calculate overall text quality metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with quality metrics
        """
        from nltk.tokenize import word_tokenize, sent_tokenize
        from collections import Counter
        
        try:
            words = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
            
            # Vocabulary diversity
            unique_words = len(set(words))
            total_words = len(words)
            vocab_diversity = unique_words / total_words if total_words > 0 else 0.0
            
            # Average word length
            avg_word_length = sum(len(w) for w in words) / total_words if total_words > 0 else 0.0
            
            # Sentence length variation
            sentence_lengths = [len(word_tokenize(s)) for s in sentences]
            if sentence_lengths:
                avg_sent_len = sum(sentence_lengths) / len(sentence_lengths)
                sent_len_variance = sum((x - avg_sent_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            else:
                avg_sent_len = 0.0
                sent_len_variance = 0.0
            
            return {
                'vocabulary_diversity': float(vocab_diversity),
                'avg_word_length': float(avg_word_length),
                'avg_sentence_length': float(avg_sent_len),
                'sentence_length_variance': float(sent_len_variance),
                'unique_words': int(unique_words),
                'total_words': int(total_words)
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {
                'vocabulary_diversity': 0.0,
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0,
                'sentence_length_variance': 0.0,
                'unique_words': 0,
                'total_words': 0
            }
    
    @staticmethod
    def calculate_all_metrics(
        reference: Optional[str] = None,
        candidate: str = "",
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available evaluation metrics.
        
        Args:
            reference: Reference text (for BLEU, ROUGE, METEOR)
            candidate: Generated text (for comparison)
            text: Single text for quality metrics
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Comparison metrics (if reference provided)
        if reference and candidate:
            results.update(EvaluationMetrics.calculate_bleu(reference, candidate))
            results.update(EvaluationMetrics.calculate_rouge_l(reference, candidate))
            results['meteor'] = EvaluationMetrics.calculate_meteor(reference, candidate)
        
        # Quality metrics (if text provided)
        if text:
            results.update(EvaluationMetrics.calculate_text_quality_metrics(text))
            results['perplexity'] = EvaluationMetrics.calculate_perplexity(text)
        
        return results



