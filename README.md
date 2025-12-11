# ArticleForge

A production-ready NLP/ML system demonstrating advanced algorithms, evaluation metrics, and feature engineering. Implements corpus-based TF-IDF, TextRank summarization, Sentence-BERT embeddings, semantic similarity computation, and comprehensive evaluation using BLEU, ROUGE-L, METEOR, and perplexity metrics. Emphasizes ML/NLP engineering practices with proper algorithm understanding, vector space operations, and quantitative evaluation methodologies.

## Overview

Article generation system using LLMs with focus on ML/NLP capabilities. Core implementation includes advanced NLP algorithms, ML evaluation metrics, feature extraction pipelines, and production-grade text processing workflows. Demonstrates corpus-based TF-IDF with mathematical precision, custom TextRank for extractive summarization, semantic similarity via Sentence-BERT embeddings, and comprehensive model evaluation using standard metrics (BLEU, ROUGE-L, METEOR, perplexity).

## Demo Recording

<div align="center">

[![ArticleForge AI Demo Video](./Article_forge_recording_thumbnail.jpg)](https://github.com/imcoza/ArticleForge-AI/raw/main/Article_forge_recording.mp4)

**Click the image above to watch the demo video**

</div>

> **Note:** Click the thumbnail to watch the demo. The video demonstrates article generation workflow, NLP analysis with Plotly visualizations, TF-IDF keyword extraction, sentiment analysis, TextRank summarization, Sentence-BERT embeddings, semantic similarity, NER, and comprehensive evaluation metrics (BLEU, ROUGE-L, METEOR, perplexity).

## Machine Learning Evaluation Metrics

**24 metrics** computed in real-time: **9 Text Generation** (BLEU-1/2/3/4, ROUGE-L P/R/F1, METEOR, Perplexity), **7 Text Quality** (Vocabulary Diversity, Word/Sentence Length, Readability), **8 NLP Processing** (TF-IDF, Semantic Similarity, Sentiment scores, Key Phrases, TextRank).

### Text Generation Metrics

- **BLEU** (BLEU-1/2/3/4): N-gram precision with geometric mean, Method 1 smoothing, range 0.0-1.0 (typical 0.2-0.5). NLTK sentence_bleu with custom smoothing.
- **ROUGE-L**: Dynamic programming LCS computation, O(mn) complexity. Precision, Recall, F1 components (0.0-1.0). Custom LCS algorithm.
- **METEOR**: Synonym matching via WordNet, stemming, word order consideration. Range 0.0-1.0. NLTK meteor_score with WordNet integration.
- **Perplexity**: exp(-average_log_probability), unigram language model. Lower values indicate better fit (typically 10-1000).

### Text Quality Metrics

- **Vocabulary Diversity**: unique_words/total_words (0.3-0.6 typical for articles)
- **Average Word/Sentence Length**: Mean characters per word, words per sentence
- **Sentence Length Variance**: Var(X) = E[(X - μ)²] measures style consistency
- **Readability**: Automated Readability Index (0-14+, textstat library)

### NLP Processing Metrics

- **TF-IDF Score**: Term frequency-inverse document frequency (corpus-based IDF)
- **Semantic Similarity**: Cosine similarity on Sentence-BERT embeddings (0-1)
- **Sentiment Scores**: Positive/Negative/Neutral probabilities (0-1, rule-based lexicon, ~70-75% accuracy)
- **Key Phrase Score**: Frequency × phrase length
- **TextRank Score**: Sentence importance via PageRank algorithm (0-1)

## ML Models and Technical Specifications

### Model Inventory

- **Sentence Embedding**: `all-MiniLM-L6-v2` (DistilBERT-based, 384 dimensions, ~90MB, mean pooling)
- **Primary LLM**: `openai/gpt-oss-120b` (120B parameters, Groq API, LPU inference, 4096-16384 token context, temp=0.7, top_p=0.9, top_k=50)
- **Fallback LLM**: `TheBloke/Llama-2-7B-Chat-GGML` (7B parameters, Q8_0 quantization, 4096 tokens, CTransformers)
- **spaCy Model**: `en_core_web_sm` (dependency parsing, NER, noun phrase extraction)

### Algorithm Specifications

**TF-IDF Vectorization**
- Implementation: scikit-learn TfidfVectorizer + manual calculation
- Formula: TF(t,d) = count(t,d)/|d|, IDF(t,D) = log(|D|/|{d ∈ D : t ∈ d}|), TF-IDF = TF × IDF
- N-grams: Unigrams, bigrams. min_df=1, max_df=0.95, max_features=10,000
- Matrix: Sparse (scipy.sparse). Performance: ~5,000 words/second

**TextRank Algorithm**
- PageRank-inspired graph ranking, damping factor α=0.85
- Convergence: max 100 iterations, ε<1e-6
- Similarity: Cosine similarity on word frequency vectors
- Complexity: O(n²) similarity matrix, O(n²×k) PageRank
- Performance: <1s for 100 sentences, ROUGE-L F1 ~0.35-0.45

**Semantic Similarity**
- Method: Cosine similarity on Sentence-BERT embeddings
- Formula: similarity(A,B) = (A·B)/(||A|| × ||B||)
- Model: all-MiniLM-L6-v2 (384 dimensions)
- Complexity: O(n) pairwise, O(n²) all-pairs
- Performance: ~100-200 sentences/second

**Sentiment Classification**
- Rule-based lexicon matching, expandable dictionaries
- Neutral: 1 - (positive + negative), argmax prediction
- Accuracy: ~70-75% baseline, ~50,000 words/second

### Hyperparameters

- **LLM**: Temperature=0.7, top_p=0.9, top_k=50, max_tokens=1000 (dynamic up to 16384), retry=3, token_multiplier=6x/8x
- **TextRank**: Damping=0.85, convergence=1e-6, max_iterations=100
- **TF-IDF**: min_df=1, max_df=0.95, ngram_range=(1,2), max_features=10,000
- **Embeddings**: Batch processing, L2 normalization, 384 dimensions

## Machine Learning Architecture

Production-grade NLP pipelines: Sentence-BERT embeddings for semantic representation, corpus-based TF-IDF (scikit-learn) for keyword extraction, custom TextRank for extractive summarization, comprehensive evaluation metrics.

### Core NLP Algorithms

**TF-IDF Vectorization**: Dual-mode (TfidfVectorizer + manual), N-gram support, document frequency filtering, sparse matrix representation. Performance: ~5,000 words/second, linear corpus scaling.

**TextRank**: Graph-based extractive summarization using PageRank. Sentence similarity matrix via cosine similarity on word frequency vectors, weighted undirected graph (nodes=sentences, edges=similarity). Performance: <1s for 100 sentences.

**Semantic Similarity**: Cosine similarity on Sentence-BERT embeddings. Use cases: key phrase extraction via semantic clustering, top-k sentence retrieval, document deduplication. Performance: ~100-200 sentences/second.

### Feature Engineering

**Text Preprocessing**: Unicode normalization, URL/email removal (regex), special character filtering, whitespace normalization, optional stopword removal, NLTK tokenization, WordNet lemmatization. Performance: ~10,000 words/second.

**Key Phrase Extraction**: Method 1—spaCy dependency parsing (min 2 words, length >5 chars, frequency × phrase length scoring). Method 2—bigram extraction with frequency scoring (fallback).

### Performance Benchmarks

**Text Processing** (Intel i7, 16GB RAM):
- Article generation (1000 words): 3-5s (Groq API)
- TF-IDF extraction: <100ms (1000-word document)
- Sentence embeddings: ~50ms (20 sentences)
- TextRank summarization: ~200ms (50 sentences)
- Full NLP pipeline: ~1-2s end-to-end

**Model Accuracy**:
- Sentence similarity (STS Benchmark): Spearman correlation >0.80
- Keyword extraction: Top-10 precision ~0.70-0.80 (domain-dependent)
- TextRank summarization: ROUGE-L F1 ~0.35-0.45 (news articles)

### ML Pipeline Architecture

```
Input Text → Preprocessing → Feature Extraction → ML Processing → Evaluation → Output
                ↓                    ↓                    ↓              ↓
         Tokenization,        TF-IDF / Embeddings   Keyword Ext.   Quality Metrics
         Cleaning,            / N-grams             Summarization  Similarity Scores
         Normalization                              Sentiment
```

**Implementation**: scikit-learn TfidfVectorizer with sparse matrix optimization, batch processing for embeddings, NumPy-optimized cosine similarity, custom PageRank with convergence detection (ε<1e-6), sparse matrix storage, lazy loading, graceful fallbacks.

**Feature Representations**: TF-IDF vectors (sparse scipy.sparse), sentence embeddings (dense 384-dim numpy arrays), N-gram features (unigrams/bigrams), dynamic vocabulary building.

## Installation and Setup

### Prerequisites

Python 3.8+, pip, Groq API key, PostgreSQL (optional), 8GB RAM minimum for local inference.

### Installation

```bash
git clone https://github.com/imcoza/ArticleForge-AI
cd ArticleForge-AI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

NLTK data auto-downloads on first execution. Manual: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`

### Configuration

Create `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
PEXELS_API_KEY=your_pexels_api_key_here
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-120b
MAX_TOKENS=1000
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
DEFAULT_WORD_LIMIT=800
MIN_WORD_COUNT=200
MAX_WORD_COUNT=2000
LOG_LEVEL=INFO
```

**Parameters**: `GROQ_API_KEY` (required), `PEXELS_API_KEY` (optional), `LLM_PROVIDER` (groq/local), `GROQ_MODEL`, `MAX_TOKENS`, `TEMPERATURE`, `DEFAULT_WORD_LIMIT`, `LOG_LEVEL`.

### Database Setup

```bash
python scripts/init_db.py
```

For production: configure PostgreSQL connection string in environment variables.

## Running the Application

```bash
streamlit run main.py
```

Interface available at `http://localhost:8501` with article generation, text analysis, and document export.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

Test suite includes unit tests for ML/NLP components (TF-IDF, TextRank, evaluation metrics), text processing utilities, and integration tests for end-to-end NLP workflows.

## Deployment

Docker deployment: `docker build -t articleforge-ai .`. Docker Compose available for multi-container deployments with database support.

## Development Guidelines

PEP 8 style guidelines with comprehensive type hints. ML/NLP components include unit tests for algorithm correctness, evaluation metric accuracy, and feature extraction pipelines. Test coverage maintains minimum 80% threshold with integration tests for end-to-end NLP workflows.

## Project Structure

```
src/
├── services/
│   └── ml/           # ML/NLP services
│       ├── tfidf_processor.py      # TF-IDF with corpus-based IDF
│       ├── advanced_nlp.py         # Sentence embeddings, semantic similarity, TextRank
│       └── evaluation_metrics.py   # BLEU, ROUGE-L, METEOR, perplexity
├── utils/
│   └── text_processing.py          # Preprocessing, tokenization, NER, feature extraction
├── app/
│   └── streamlit_app.py             # ML/NLP visualization with Plotly charts
├── services/
│   ├── llm_service.py               # LLM integration
│   └── document_service.py         # Document export
├── api/              # REST API endpoints
└── config/           # Configuration management
tests/                 # ML/NLP component tests
```

## Limitations and Future Enhancements

**Current Limitations**: Rule-based sentiment analysis (~70-75% accuracy), extractive-only TextRank, single-document TF-IDF by default, evaluation metrics not used for model selection, no fine-tuning, English-only.

**Potential Enhancements**: BERT/RoBERTa fine-tuning for sentiment, abstractive summarization (T5/BART), topic modeling (LDA/BERTopic), embedding visualization (t-SNE/UMAP), model comparison framework with A/B testing, hyperparameter tuning for TextRank, MLflow integration, multi-language support.
