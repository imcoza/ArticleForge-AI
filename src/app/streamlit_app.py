"""
Streamlit application interface.
Main UI for the article generation application.
"""
import streamlit as st
import re
from datetime import datetime
from src.config.settings import settings
from src.services.llm_service import llm_service
from src.services.image_service import image_service
from src.services.document_service import DocumentService
from src.utils.text_processing import TextProcessor
from src.utils.logger import logger
from src.services.ml.tfidf_processor import TFIDFProcessor
from src.services.ml.advanced_nlp import AdvancedNLPProcessor
from src.services.ml.evaluation_metrics import EvaluationMetrics

# Optional visualization imports
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    pd = None
    px = None
    go = None


def create_app():
    """Create and configure the Streamlit application."""
    st.set_page_config(
        page_title=settings.PAGE_TITLE,
        layout=settings.PAGE_LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        word_limit = st.slider(
            "Target Word Count",
            min_value=settings.MIN_WORD_COUNT,
            max_value=settings.MAX_WORD_COUNT,
            value=settings.DEFAULT_WORD_LIMIT,
            step=50
        )
        
        st.markdown("---")
        st.markdown("**NLP Analysis Options:**")
        show_analysis = st.checkbox("Show NLP Analysis", value=True)
        show_keywords = st.checkbox("Show Keywords (TF-IDF)", value=True)
        show_advanced = st.checkbox("Show Advanced NLP", value=False)
        use_corpus_tfidf = st.checkbox("Use Corpus-based TF-IDF", value=False, help="Requires multiple documents for IDF calculation")
        
        st.markdown("---")
        st.markdown("**Technical Details:**")
        st.caption(f"LLM Provider: {settings.LLM_PROVIDER.upper()}")
        st.caption(f"Model: {settings.GROQ_MODEL if settings.LLM_PROVIDER == 'groq' else 'Local'}")
        st.caption(f"Max Tokens: {settings.MAX_TOKENS}")
        st.caption(f"Temperature: {settings.TEMPERATURE}")
    
    # Main content
    st.title("Article Forge")
    st.markdown("""
    **Advanced NLP and ML System** demonstrating machine learning engineering:
    - **True TF-IDF**: Corpus-based IDF calculation, vector space representation, scikit-learn integration
    - **Advanced NLP**: Sentence embeddings (Sentence-BERT), semantic similarity, key phrase extraction
    - **TextRank Summarization**: Graph-based extractive summarization algorithm
    - **Text Classification**: Sentiment analysis, text statistics, vocabulary analysis
    - **ML Pipeline**: Feature extraction, text preprocessing, model-ready data preparation
    - **LLM Integration**: Groq API with retry logic and token management
    - **Named Entity Recognition**: spaCy-based entity extraction with fallback
    """)
    
    # Input section
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        user_input = st.text_input(
            "📌 Article Topic",
            placeholder="Enter the topic or idea for your article...",
            help="The main subject or theme of the article you want to generate"
        )
    
    with col_input2:
        image_query = st.text_input(
            "🖼️ Image Search Query",
            placeholder="Enter keywords for reference image...",
            help="Keywords to search for a relevant image on Pexels"
        )
    
    # Generate button
    if st.button("🚀 Generate Article", type="primary", use_container_width=True):
        if not user_input:
            st.error("⚠️ Please enter an article topic!")
            return
        
        if not image_query:
            st.warning("⚠️ Image query is optional, but recommended for better results.")
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate article
            status_text.text("Generating article content...")
            progress_bar.progress(20)
            
            article_result = llm_service.generate_article(user_input, word_limit)
            
            if not article_result['success']:
                error_msg = article_result.get('error', 'Unknown error')
                st.error(f"❌ Failed to generate article: {error_msg}")
                
                # Provide helpful suggestions
                if 'rate limit' in error_msg.lower() or '429' in error_msg:
                    st.info("💡 **Tip:** You've hit the rate limit. Please wait a moment and try again.")
                elif 'api key' in error_msg.lower() or 'authentication' in error_msg.lower():
                    st.info("💡 **Tip:** Check your GROQ_API_KEY in the .env file.")
                elif 'timeout' in error_msg.lower():
                    st.info("💡 **Tip:** The request timed out. Try reducing the word count or try again.")
                else:
                    st.info("💡 **Tip:** Try again with a different topic or reduce the word count.")
                
                return
            
            generated_text = article_result['text']
            actual_word_count = article_result.get('word_count', len(generated_text.split()))
            
            # Enforce strict word limit - truncate if necessary
            if actual_word_count > word_limit:
                words = generated_text.split()
                truncated_words = words[:word_limit]
                generated_text = ' '.join(truncated_words)
                # Ensure it ends properly at a sentence boundary
                if generated_text and generated_text[-1] not in '.!?':
                    last_period = generated_text.rfind('.')
                    last_exclamation = generated_text.rfind('!')
                    last_question = generated_text.rfind('?')
                    last_sentence_end = max(last_period, last_exclamation, last_question)
                    if last_sentence_end > len(generated_text) * 0.8:  # Only if we keep at least 80%
                        generated_text = generated_text[:last_sentence_end + 1]
                    else:
                        generated_text = generated_text.rstrip() + '.'
                actual_word_count = len(generated_text.split())
                st.info(f"ℹ️ Article was truncated to {actual_word_count} words (limit: {word_limit})")
            
            # Check for warnings
            if 'warning' in article_result:
                st.warning(f"⚠️ {article_result['warning']}")
            
            # Validate word count range (strict limit - no exceeding)
            min_acceptable = max(settings.MIN_WORD_COUNT, int(word_limit * 0.8))
            max_acceptable = word_limit  # Strict limit
            
            if actual_word_count < min_acceptable:
                st.warning(f"⚠️ Article is shorter than expected: {actual_word_count} words (target: {word_limit}, minimum: {min_acceptable})")
            elif actual_word_count > max_acceptable:
                st.warning(f"⚠️ Article exceeded limit: {actual_word_count} words (target: {word_limit})")
            
            progress_bar.progress(50)
            
            # Step 2: Fetch image
            status_text.text("Fetching reference image...")
            image_result = image_service.get_image_url(image_query) if image_query else {'success': False}
            progress_bar.progress(70)
            
            # Step 3: Process and analyze text
            status_text.text("Processing text with NLP pipeline...")
            text_processor = TextProcessor()
            
            # Clean and validate text
            cleaned_text = text_processor.clean_text(generated_text)
            validation = text_processor.validate_text_quality(cleaned_text, min_words=min_acceptable)
            
            # Update validation with actual word count from LLM
            validation['word_count'] = actual_word_count
            
            progress_bar.progress(90)
            
            # Display results
            status_text.text("Complete")
            progress_bar.progress(100)
            
            st.success(f"Article generated successfully ({actual_word_count} words)")
            
            # Main content display - Improved 2-column layout
            col_article, col_sidebar = st.columns([2, 1])
            
            with col_article:
                st.subheader("Generated Article")
                
                # Technical metrics
                word_count = validation['word_count']
                readability = validation.get('readability', {})
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Words", word_count, f"{word_count - word_limit}")
                with metric_col2:
                    st.metric("Sentences", readability.get('sentence_count', 0))
                with metric_col3:
                    avg_sent_len = readability.get('avg_sentence_length', 0)
                    st.metric("Avg Sentence", f"{avg_sent_len:.1f}")
                
                # Display article
                st.markdown("---")
                # Highlight topic prominently
                st.markdown(f"""
                <div style="background-color: #e6f7ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0050b3; margin-bottom: 20px;">
                    <h3 style="color: #0050b3; margin: 0; font-size: 1.3em;">Topic: {user_input}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
                
                # Clean generated text to remove any "by" or date metadata
                cleaned_article = generated_text
                # Remove "By [name]" at the start (with optional date)
                cleaned_article = re.sub(r'^By\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+on\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})?\s*\n?', '', cleaned_article, flags=re.MULTILINE)
                # Remove "Published on", "Posted on", "Written on" lines
                cleaned_article = re.sub(r'^(Published|Posted|Written)\s+on\s+.*?\n', '', cleaned_article, flags=re.MULTILINE | re.IGNORECASE)
                # Remove standalone date patterns at the start (MM/DD/YYYY or DD/MM/YYYY)
                cleaned_article = re.sub(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\n?', '', cleaned_article, flags=re.MULTILINE)
                # Remove "by [name]" anywhere in first few lines (case insensitive)
                lines = cleaned_article.split('\n')
                cleaned_lines = []
                for i, line in enumerate(lines):
                    # Skip lines that are just "by [name]" in first 3 lines
                    if i < 3 and re.match(r'^\s*by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$', line, re.IGNORECASE):
                        continue
                    cleaned_lines.append(line)
                cleaned_article = '\n'.join(cleaned_lines).strip()
                
                st.markdown(cleaned_article)
                
                # Technical validation info
                if validation['issues']:
                    st.markdown("---")
                    st.markdown("**Validation Issues:**")
                    for issue in validation['issues']:
                        st.write(f"- {issue}")
            
            with col_sidebar:
                # NLP Analysis Section - ML/NLP Focus
                if show_analysis:
                    st.subheader("NLP Analysis")
                    
                    # Quick Metrics Overview Dashboard (always visible when article is generated)
                    if VISUALIZATION_AVAILABLE and pd is not None and px is not None and go is not None:
                        try:
                            # Calculate quick metrics for overview
                            from nltk.tokenize import word_tokenize, sent_tokenize
                            words = word_tokenize(generated_text.lower())
                            sentences = sent_tokenize(generated_text)
                            
                            # Quick metrics (excluding readability to avoid overlap)
                            quick_metrics = {
                                'Total Words': len(words),
                                'Unique Words': len(set(words)),
                                'Total Sentences': len(sentences),
                                'Avg Words/Sentence': len(words) / max(len(sentences), 1),
                                'Vocabulary Diversity': len(set(words)) / max(len(words), 1)
                            }
                            
                            # Metric definitions for tooltips
                            metric_definitions = {
                                'Total Words': 'Total number of words in the article',
                                'Unique Words': 'Number of distinct words used',
                                'Total Sentences': 'Total number of sentences',
                                'Avg Words/Sentence': 'Average words per sentence',
                                'Vocabulary Diversity': 'Ratio of unique to total words (0-1, higher = more diverse)'
                            }
                            
                            # Create a quick overview chart
                            df_quick = pd.DataFrame(list(quick_metrics.items()), columns=['Metric', 'Value'])
                            
                            # Add definitions column for hover text
                            df_quick['Definition'] = df_quick['Metric'].map(metric_definitions)
                            
                            # Normalize values for better visualization (except counts)
                            normalized_values = []
                            for metric, value in quick_metrics.items():
                                if 'Diversity' in metric:
                                    normalized_values.append(value)  # Already 0-1
                                elif 'Avg' in metric:
                                    normalized_values.append(min(value / 30, 1.0))  # Normalize to 0-1
                                else:
                                    # For counts, use log scale normalization
                                    normalized_values.append(min(value / 1000, 1.0))
                            
                            df_quick_norm = pd.DataFrame({
                                'Metric': list(quick_metrics.keys()),
                                'Normalized Value': normalized_values,
                                'Definition': [metric_definitions[m] for m in quick_metrics.keys()]
                            })
                            
                            # Create and display the chart with custom hover text
                            fig_quick = px.bar(
                                df_quick_norm,
                                x='Metric',
                                y='Normalized Value',
                                title='📊 Quick Metrics Overview',
                                color='Normalized Value',
                                color_continuous_scale='Plasma',
                                labels={'Normalized Value': 'Normalized Score (0-1)', 'Metric': 'Metric'},
                                hover_data={'Definition': True, 'Normalized Value': ':.3f'}
                            )
                            
                            # Update hover template to show definitions
                            fig_quick.update_traces(
                                hovertemplate='<b>%{x}</b><br>' +
                                            'Value: %{y:.3f}<br>' +
                                            '%{customdata[0]}<extra></extra>',
                                customdata=df_quick_norm[['Definition']].values
                            )
                            
                            fig_quick.update_layout(
                                height=250, 
                                showlegend=False, 
                                xaxis_tickangle=-45,
                                margin=dict(b=100, l=100, r=50, t=50),  # Increased left margin to 100 for y-axis label
                                xaxis=dict(tickfont=dict(size=10), automargin=True),
                                yaxis=dict(
                                    title=dict(text='Normalized Score<br>(0-1)', font=dict(size=10)),
                                    tickfont=dict(size=9),
                                    automargin=True,  # Auto-adjust to prevent overlap
                                    title_standoff=20  # Increased space between title and axis
                                )
                            )
                            st.plotly_chart(fig_quick, use_container_width=True)
                            
                            # Display actual values in columns with definitions
                            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                            with col_q1:
                                st.metric(
                                    "Words", 
                                    f"{quick_metrics['Total Words']:,}",
                                    help="Total number of words in the article"
                                )
                            with col_q2:
                                st.metric(
                                    "Unique", 
                                    f"{quick_metrics['Unique Words']:,}",
                                    help="Number of distinct words used"
                                )
                            with col_q3:
                                st.metric(
                                    "Sentences", 
                                    f"{quick_metrics['Total Sentences']:,}",
                                    help="Total number of sentences"
                                )
                            with col_q4:
                                st.metric(
                                    "Diversity", 
                                    f"{quick_metrics['Vocabulary Diversity']:.3f}",
                                    help="Ratio of unique to total words (higher = more diverse vocabulary)"
                                )
                            
                            st.markdown("---")
                        except Exception as e:
                            logger.error(f"Error creating quick metrics overview: {e}", exc_info=True)
                            st.error(f"Error creating charts: {str(e)}")
                            st.info("Please ensure pandas and plotly are installed: pip install pandas plotly")
                    elif not VISUALIZATION_AVAILABLE:
                        st.warning("⚠️ Visualization libraries not available. Charts will not be displayed.")
                        st.info("Install with: pip install pandas plotly")
                    
                    # Keywords with True TF-IDF scores
                    if show_keywords:
                        st.markdown("**Keywords (TF-IDF Scores):**")
                        try:
                            # Use true TF-IDF processor
                            tfidf_processor = TFIDFProcessor()
                            keywords = tfidf_processor.extract_keywords_tfidf(
                                generated_text,
                                top_n=10,
                                use_corpus_idf=use_corpus_tfidf
                            )
                            
                            if keywords:
                                if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                                    try:
                                        # Display as bar chart
                                        df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'TF-IDF Score'])
                                        df_keywords = df_keywords.sort_values('TF-IDF Score', ascending=True)
                                        
                                        fig = px.bar(
                                            df_keywords,
                                            x='TF-IDF Score',
                                            y='Keyword',
                                            orientation='h',
                                            title='Top Keywords (TF-IDF Scores)',
                                            labels={'TF-IDF Score': 'TF-IDF Score', 'Keyword': 'Keyword'},
                                            color='TF-IDF Score',
                                            color_continuous_scale='Blues'
                                        )
                                        fig.update_layout(height=400, showlegend=False)
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Also show as text for reference
                                        with st.expander("View Keywords List"):
                                            for word, tfidf_score in keywords:
                                                st.text(f"{word}: {tfidf_score:.4f}")
                                    except Exception as chart_error:
                                        logger.warning(f"Error creating chart: {chart_error}")
                                        # Fallback to text display
                                        for word, tfidf_score in keywords:
                                            st.text(f"{word}: {tfidf_score:.4f}")
                                else:
                                    # Fallback to text display
                                    for word, tfidf_score in keywords:
                                        st.text(f"{word}: {tfidf_score:.4f}")
                            else:
                                # Fallback to basic extraction
                                keywords = text_processor.extract_keywords(cleaned_text, top_n=10, use_tfidf=False)
                                if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                                    try:
                                        df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
                                        df_keywords = df_keywords.sort_values('Frequency', ascending=True)
                                        
                                        fig = px.bar(
                                            df_keywords,
                                            x='Frequency',
                                            y='Keyword',
                                            orientation='h',
                                            title='Top Keywords (Frequency)',
                                            labels={'Frequency': 'Frequency', 'Keyword': 'Keyword'},
                                            color='Frequency',
                                            color_continuous_scale='Greens'
                                        )
                                        fig.update_layout(height=400, showlegend=False)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as chart_error:
                                        logger.warning(f"Error creating chart: {chart_error}")
                                        for word, count in keywords:
                                            st.text(f"{word}: {count} (freq)")
                                else:
                                    for word, count in keywords:
                                        st.text(f"{word}: {count} (freq)")
                        except Exception as e:
                            logger.warning(f"TF-IDF extraction error: {e}")
                            keywords = text_processor.extract_keywords(cleaned_text, top_n=10, use_tfidf=False)
                            if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                                try:
                                    df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
                                    df_keywords = df_keywords.sort_values('Frequency', ascending=True)
                                    
                                    fig = px.bar(
                                        df_keywords,
                                        x='Frequency',
                                        y='Keyword',
                                        orientation='h',
                                        title='Top Keywords (Frequency)',
                                        color='Frequency',
                                        color_continuous_scale='Greens'
                                    )
                                    fig.update_layout(height=400, showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as chart_error:
                                    logger.warning(f"Error creating chart: {chart_error}")
                                    for word, count in keywords:
                                        st.text(f"{word}: {count} (freq)")
                            else:
                                for word, count in keywords:
                                    st.text(f"{word}: {count} (freq)")
                        st.markdown("---")
                    
                    # Advanced NLP Features
                    if show_advanced:
                        try:
                            advanced_nlp = AdvancedNLPProcessor()
                            
                            # Key Phrases
                            st.markdown("**Key Phrases:**")
                            key_phrases = advanced_nlp.extract_key_phrases(generated_text, top_n=10)
                            
                            # Key Phrases visualization
                            if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                                try:
                                    if key_phrases:
                                        df_phrases = pd.DataFrame(key_phrases, columns=['Phrase', 'Score'])
                                        df_phrases = df_phrases.sort_values('Score', ascending=True)
                                        
                                        fig_phrases = px.bar(
                                            df_phrases,
                                            x='Score',
                                            y='Phrase',
                                            orientation='h',
                                            title='Key Phrases (Extracted Phrases with Scores)',
                                            color='Score',
                                            color_continuous_scale='Oranges',
                                            labels={'Score': 'Importance Score', 'Phrase': 'Key Phrase'}
                                        )
                                        fig_phrases.update_layout(height=400, showlegend=False)
                                        st.plotly_chart(fig_phrases, use_container_width=True)
                                        
                                        # Also show as text in expander
                                        with st.expander("View Key Phrases List"):
                                            for phrase, score in key_phrases:
                                                st.text(f"{phrase}: {score:.2f}")
                                    else:
                                        st.info("No key phrases extracted")
                                except Exception as chart_error:
                                    logger.warning(f"Error creating key phrases chart: {chart_error}")
                                    # Fallback to text display
                                    for phrase, score in key_phrases:
                                        st.text(f"{phrase}: {score:.2f}")
                            else:
                                # Fallback to text display
                                for phrase, score in key_phrases:
                                    st.text(f"{phrase}: {score:.2f}")
                            st.markdown("---")
                            
                            # Sentiment Analysis
                            st.markdown("**Sentiment Analysis:**")
                            sentiment = advanced_nlp.classify_text_sentiment(generated_text)
                            
                            # Sentiment visualization
                            if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                                try:
                                    # Pie chart for sentiment distribution
                                    sentiment_data = {
                                        'Sentiment': ['Positive', 'Negative', 'Neutral'],
                                        'Score': [
                                            sentiment['positive'],
                                            sentiment['negative'],
                                            sentiment['neutral']
                                        ]
                                    }
                                    df_sentiment = pd.DataFrame(sentiment_data)
                                    
                                    fig_sentiment = px.pie(
                                        df_sentiment,
                                        values='Score',
                                        names='Sentiment',
                                        title='Sentiment Distribution',
                                        color='Sentiment',
                                        color_discrete_map={
                                            'Positive': '#2ecc71',
                                            'Negative': '#e74c3c',
                                            'Neutral': '#95a5a6'
                                        }
                                    )
                                    fig_sentiment.update_layout(height=300, showlegend=True)
                                    st.plotly_chart(fig_sentiment, use_container_width=True)
                                    
                                    # Bar chart for sentiment scores
                                    fig_sentiment_bar = px.bar(
                                        df_sentiment,
                                        x='Sentiment',
                                        y='Score',
                                        title='Sentiment Scores',
                                        color='Sentiment',
                                        color_discrete_map={
                                            'Positive': '#2ecc71',
                                            'Negative': '#e74c3c',
                                            'Neutral': '#95a5a6'
                                        },
                                        range_y=[0, 1]
                                    )
                                    fig_sentiment_bar.update_layout(height=250, showlegend=False)
                                    st.plotly_chart(fig_sentiment_bar, use_container_width=True)
                                except Exception as chart_error:
                                    logger.warning(f"Error creating sentiment chart: {chart_error}")
                            
                            # Text display
                            st.text(f"Positive: {sentiment['positive']:.3f}")
                            st.text(f"Negative: {sentiment['negative']:.3f}")
                            st.text(f"Neutral: {sentiment['neutral']:.3f}")
                            st.text(f"Predicted: {sentiment['predicted_label']}")
                            st.markdown("---")
                            
                            # Advanced Text Statistics
                            st.markdown("**Advanced Statistics:**")
                            stats = advanced_nlp.calculate_text_statistics(generated_text)
                            
                            # Word length distribution visualization
                            if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                                try:
                                    from nltk.tokenize import word_tokenize
                                    words = word_tokenize(generated_text.lower())
                                    word_lengths = [len(w) for w in words if w.isalpha() and len(w) > 0]
                                    
                                    if word_lengths:
                                        df_word_lengths = pd.DataFrame({'Word Length': word_lengths})
                                        
                                        fig_word_hist = px.histogram(
                                            df_word_lengths,
                                            x='Word Length',
                                            nbins=20,
                                            title='Word Length Distribution',
                                            labels={'Word Length': 'Characters per Word', 'count': 'Frequency'},
                                            color_discrete_sequence=['#3498db']
                                        )
                                        fig_word_hist.update_layout(height=300, showlegend=False)
                                        st.plotly_chart(fig_word_hist, use_container_width=True)
                                except Exception as chart_error:
                                    logger.warning(f"Error creating word length chart: {chart_error}")
                            
                            # Sentence length distribution visualization
                            if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                                try:
                                    from nltk.tokenize import sent_tokenize, word_tokenize
                                    sentences = sent_tokenize(generated_text)
                                    sentence_lengths = [len(word_tokenize(s)) for s in sentences if len(word_tokenize(s)) > 0]
                                    
                                    if sentence_lengths:
                                        df_sent_lengths = pd.DataFrame({'Sentence Length': sentence_lengths})
                                        
                                        fig_sent_hist = px.histogram(
                                            df_sent_lengths,
                                            x='Sentence Length',
                                            nbins=15,
                                            title='Sentence Length Distribution (Words per Sentence)',
                                            labels={'Sentence Length': 'Words per Sentence', 'count': 'Frequency'},
                                            color_discrete_sequence=['#9b59b6']
                                        )
                                        fig_sent_hist.update_layout(height=300, showlegend=False)
                                        st.plotly_chart(fig_sent_hist, use_container_width=True)
                                except Exception as chart_error:
                                    logger.warning(f"Error creating sentence length chart: {chart_error}")
                            
                            # Text display
                            st.text(f"Vocabulary Richness: {stats['vocabulary_richness']:.3f}")
                            st.text(f"Avg Word Length: {stats['avg_word_length']:.2f} chars")
                            st.text(f"Paragraphs: {stats['paragraph_count']}")
                            st.text(f"Words/Paragraph: {stats['words_per_paragraph']:.1f}")
                            st.markdown("---")
                            
                        except Exception as e:
                            logger.warning(f"Advanced NLP features error: {e}")
                            st.caption("Advanced NLP features unavailable")
                            st.markdown("---")
                    
                    # Text Statistics
                    st.markdown("**Text Statistics:**")
                    
                    # Text Statistics visualization
                    if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                        try:
                            stats_data = {
                                'Metric': ['Total Characters', 'Total Words', 'Total Sentences'],
                                'Count': [
                                    len(generated_text),
                                    word_count,
                                    readability.get('sentence_count', 0)
                                ]
                            }
                            df_stats = pd.DataFrame(stats_data)
                            
                            fig_stats = px.bar(
                                df_stats,
                                x='Metric',
                                y='Count',
                                title='Basic Text Statistics',
                                color='Count',
                                color_continuous_scale='Purples',
                                labels={'Count': 'Count', 'Metric': 'Metric'}
                            )
                            fig_stats.update_layout(
                                height=300, 
                                showlegend=False, 
                                xaxis_tickangle=-45,
                                margin=dict(b=100, l=50, r=50, t=50),
                                xaxis=dict(tickfont=dict(size=10), automargin=True)
                            )
                            st.plotly_chart(fig_stats, use_container_width=True)
                            
                            # Additional metrics in columns
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("Characters", f"{len(generated_text):,}")
                            with col_stat2:
                                st.metric("Words", f"{word_count:,}")
                            with col_stat3:
                                st.metric("Sentences", f"{readability.get('sentence_count', 0):,}")
                            
                            if readability.get('sentence_count', 0) > 0:
                                st.metric("Words per Sentence", f"{readability.get('avg_sentence_length', 0):.2f}")
                        except Exception as chart_error:
                            logger.warning(f"Error creating text statistics chart: {chart_error}")
                            # Fallback to text display
                            st.text(f"Total Characters: {len(generated_text)}")
                            st.text(f"Total Words: {word_count}")
                            st.text(f"Total Sentences: {readability.get('sentence_count', 0)}")
                            if readability.get('sentence_count', 0) > 0:
                                st.text(f"Words per Sentence: {readability.get('avg_sentence_length', 0):.2f}")
                    else:
                        # Fallback to text display
                        st.text(f"Total Characters: {len(generated_text)}")
                        st.text(f"Total Words: {word_count}")
                        st.text(f"Total Sentences: {readability.get('sentence_count', 0)}")
                        if readability.get('sentence_count', 0) > 0:
                            st.text(f"Words per Sentence: {readability.get('avg_sentence_length', 0):.2f}")
                    
                    # Summary with TextRank
                    st.markdown("---")
                    st.markdown("**Extractive Summary (TextRank):**")
                    summary = text_processor.summarize_text(generated_text, max_sentences=3, method="textrank")
                    
                    # Summary visualization with comparison
                    if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
                        try:
                            from nltk.tokenize import sent_tokenize, word_tokenize
                            original_sentences = sent_tokenize(generated_text)
                            summary_sentences = sent_tokenize(summary)
                            
                            comparison_data = {
                                'Document': ['Original', 'Summary'],
                                'Sentences': [len(original_sentences), len(summary_sentences)],
                                'Words': [
                                    len(word_tokenize(generated_text)),
                                    len(word_tokenize(summary))
                                ]
                            }
                            
                            # Create comparison chart
                            fig_comparison = go.Figure()
                            
                            fig_comparison.add_trace(go.Bar(
                                name='Sentences',
                                x=comparison_data['Document'],
                                y=comparison_data['Sentences'],
                                marker_color='#3498db'
                            ))
                            
                            fig_comparison.add_trace(go.Bar(
                                name='Words',
                                x=comparison_data['Document'],
                                y=comparison_data['Words'],
                                marker_color='#2ecc71'
                            ))
                            
                            fig_comparison.update_layout(
                                title='Summary Comparison (Original vs Summary)',
                                xaxis_title='Document Type',
                                yaxis_title='Count',
                                barmode='group',
                                height=300,
                                showlegend=True
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            # Compression ratio
                            compression_ratio = len(summary_sentences) / len(original_sentences) if len(original_sentences) > 0 else 0
                            st.metric("Compression Ratio", f"{compression_ratio:.2%}", 
                                     help="Percentage of original sentences in summary")
                        except Exception as chart_error:
                            logger.warning(f"Error creating summary comparison chart: {chart_error}")
                    
                    st.write(summary)
                    
                    # Evaluation Metrics
                    st.markdown("---")
                    st.markdown("**Model Evaluation Metrics:**")
                    try:
                        # Calculate all evaluation metrics
                        eval_metrics = EvaluationMetrics.calculate_text_quality_metrics(generated_text)
                        
                        # Calculate perplexity
                        try:
                            perplexity = EvaluationMetrics.calculate_perplexity(generated_text)
                            if perplexity != float('inf') and perplexity < 10000:
                                eval_metrics['perplexity'] = perplexity
                        except Exception:
                            pass
                        
                        # Display key metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Vocabulary Diversity", f"{eval_metrics['vocabulary_diversity']:.3f}", 
                                     help="Ratio of unique words to total words")
                        with col2:
                            st.metric("Avg Word Length", f"{eval_metrics['avg_word_length']:.2f}",
                                     help="Average characters per word")
                        with col3:
                            st.metric("Avg Sentence Length", f"{eval_metrics['avg_sentence_length']:.1f}",
                                     help="Average words per sentence")
                        with col4:
                            st.metric("Unique Words", f"{eval_metrics['unique_words']}",
                                     help="Number of distinct words")
                        
                        # Comprehensive metrics visualization
                        if VISUALIZATION_AVAILABLE and pd is not None and px is not None and go is not None:
                            try:
                                # Create metrics dataframe
                                metrics_data = {
                                    'Metric': [
                                        'Vocabulary Diversity',
                                        'Avg Word Length (normalized)',
                                        'Sentence Variation (normalized)',
                                        'Perplexity (normalized)' if 'perplexity' in eval_metrics else None
                                    ],
                                    'Score': [
                                        eval_metrics['vocabulary_diversity'],
                                        min(eval_metrics['avg_word_length'] / 10, 1.0),
                                        min(eval_metrics['sentence_length_variance'] / 100, 1.0),
                                        min(eval_metrics.get('perplexity', 100) / 1000, 1.0) if 'perplexity' in eval_metrics else None
                                    ]
                                }
                                
                                # Remove None values
                                metrics_data = {k: [v for v in vals if v is not None] for k, vals in metrics_data.items()}
                                
                                if metrics_data['Metric']:
                                    df_quality = pd.DataFrame(metrics_data)
                                    fig = px.bar(
                                        df_quality,
                                        x='Metric',
                                        y='Score',
                                        title='Text Quality Metrics (Normalized)',
                                        color='Score',
                                        color_continuous_scale='Viridis',
                                        range_y=[0, 1],
                                        labels={'Score': 'Normalized Score (0-1)', 'Metric': 'Metric'}
                                    )
                                    fig.update_layout(
                                        height=350, 
                                        showlegend=False, 
                                        xaxis_tickangle=-45,
                                        margin=dict(b=100, l=50, r=50, t=50),
                                        xaxis=dict(tickfont=dict(size=10))
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Radar/Spider chart for comprehensive metrics overview
                                try:
                                    # Normalize metrics for radar chart (0-1 scale)
                                    radar_metrics = {
                                        'Vocabulary Diversity': eval_metrics['vocabulary_diversity'],
                                        'Word Length (norm)': min(eval_metrics['avg_word_length'] / 10, 1.0),
                                        'Sentence Length (norm)': min(eval_metrics['avg_sentence_length'] / 50, 1.0),
                                        'Sentence Variation (norm)': min(eval_metrics['sentence_length_variance'] / 100, 1.0)
                                    }
                                    
                                    if 'perplexity' in eval_metrics and eval_metrics['perplexity'] != float('inf'):
                                        # Invert perplexity for radar (lower is better, so we invert)
                                        radar_metrics['Perplexity (norm)'] = min(1.0 / (eval_metrics['perplexity'] / 100 + 1), 1.0)
                                    
                                    categories = list(radar_metrics.keys())
                                    values = list(radar_metrics.values())
                                    
                                    # Create radar chart
                                    fig_radar = go.Figure()
                                    
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=values + [values[0]],  # Close the loop
                                        theta=categories + [categories[0]],
                                        fill='toself',
                                        name='Text Quality',
                                        line_color='#3498db'
                                    ))
                                    
                                    fig_radar.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1]
                                            )),
                                        showlegend=True,
                                        title='Comprehensive Metrics Overview (Radar Chart)',
                                        height=400
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)
                                except Exception as radar_error:
                                    logger.warning(f"Error creating radar chart: {radar_error}")
                                
                                # Additional metrics visualization
                                st.markdown("**Detailed Metrics:**")
                                detailed_metrics = {
                                    'Total Words': eval_metrics['total_words'],
                                    'Unique Words': eval_metrics['unique_words'],
                                    'Avg Word Length': eval_metrics['avg_word_length'],
                                    'Avg Sentence Length': eval_metrics['avg_sentence_length'],
                                    'Sentence Variance': eval_metrics['sentence_length_variance']
                                }
                                
                                df_detailed = pd.DataFrame(list(detailed_metrics.items()), columns=['Metric', 'Value'])
                                
                                fig2 = px.bar(
                                    df_detailed,
                                    x='Metric',
                                    y='Value',
                                    title='Detailed Text Statistics',
                                    color='Value',
                                    color_continuous_scale='Blues',
                                    labels={'Value': 'Count/Score', 'Metric': 'Metric'}
                                )
                                fig2.update_layout(
                                    height=300, 
                                    showlegend=False, 
                                    xaxis_tickangle=-45,
                                    margin=dict(b=120, l=50, r=50, t=50),
                                    xaxis=dict(tickfont=dict(size=10), automargin=True)
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # Perplexity gauge chart if available
                                if 'perplexity' in eval_metrics and eval_metrics['perplexity'] != float('inf'):
                                    try:
                                        perplexity_value = eval_metrics['perplexity']
                                        # Normalize perplexity for gauge (assuming good range is 0-500)
                                        normalized_perplexity = min(perplexity_value / 500, 1.0)
                                        
                                        fig_gauge = go.Figure(go.Indicator(
                                            mode="gauge+number+delta",
                                            value=perplexity_value,
                                            domain={'x': [0, 1], 'y': [0, 1]},
                                            title={'text': "Perplexity (Lower is Better)"},
                                            delta={'reference': 200},
                                            gauge={
                                                'axis': {'range': [None, 500]},
                                                'bar': {'color': "darkblue"},
                                                'steps': [
                                                    {'range': [0, 100], 'color': "lightgreen"},
                                                    {'range': [100, 300], 'color': "yellow"},
                                                    {'range': [300, 500], 'color': "red"}
                                                ],
                                                'threshold': {
                                                    'line': {'color': "red", 'width': 4},
                                                    'thickness': 0.75,
                                                    'value': 400
                                                }
                                            }
                                        ))
                                        fig_gauge.update_layout(height=300)
                                        st.plotly_chart(fig_gauge, use_container_width=True)
                                    except Exception as gauge_error:
                                        logger.warning(f"Error creating perplexity gauge: {gauge_error}")
                                        st.metric("Perplexity", f"{eval_metrics['perplexity']:.2f}", 
                                                 help="Lower is better - measures model uncertainty")
                                elif 'perplexity' in eval_metrics:
                                    st.metric("Perplexity", f"{eval_metrics['perplexity']:.2f}", 
                                             help="Lower is better - measures model uncertainty")
                                
                            except Exception as chart_error:
                                logger.warning(f"Error creating evaluation charts: {chart_error}")
                                # Fallback to text display
                                st.text(f"Vocabulary Diversity: {eval_metrics['vocabulary_diversity']:.3f}")
                                st.text(f"Avg Word Length: {eval_metrics['avg_word_length']:.2f}")
                                st.text(f"Avg Sentence Length: {eval_metrics['avg_sentence_length']:.1f}")
                                st.text(f"Sentence Variation: {eval_metrics['sentence_length_variance']:.2f}")
                                if 'perplexity' in eval_metrics:
                                    st.text(f"Perplexity: {eval_metrics['perplexity']:.2f}")
                        else:
                            # Text fallback
                            st.text(f"Vocabulary Diversity: {eval_metrics['vocabulary_diversity']:.3f}")
                            st.text(f"Avg Word Length: {eval_metrics['avg_word_length']:.2f}")
                            st.text(f"Avg Sentence Length: {eval_metrics['avg_sentence_length']:.1f}")
                            st.text(f"Sentence Variation: {eval_metrics['sentence_length_variance']:.2f}")
                            if 'perplexity' in eval_metrics:
                                st.text(f"Perplexity: {eval_metrics['perplexity']:.2f}")
                            
                    except Exception as e:
                        logger.warning(f"Error calculating evaluation metrics: {e}")
                        st.caption("Evaluation metrics unavailable")
                        st.error(f"Error: {str(e)}")
                
                # Reference Image Section
                st.subheader("Reference Image")
                if image_result.get('success'):
                    st.image(image_result['url'], use_container_width=True)
                    if image_result.get('photographer'):
                        st.caption(f"Photo by {image_result['photographer']} on Pexels")
                else:
                    if image_query:
                        st.caption("Image API: Not configured")
                    else:
                        st.caption("No image query provided")
            
            # Download section
            st.markdown("---")
            st.subheader("💾 Download Document")
            
            try:
                doc = DocumentService.create_word_document(
                    title=user_input,
                    content=generated_text,
                    image_url=image_result.get('url') if image_result.get('success') else None,
                    metadata={
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'word_count': validation['word_count'],
                        'topic': user_input
                    }
                )
                
                doc_bytes = DocumentService.save_document_to_bytes(doc)
                
                st.download_button(
                    label="📥 Download Word Document",
                    data=doc_bytes,
                    file_name=f"article_{user_input.replace(' ', '_')[:30]}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating document: {str(e)}")
                logger.error(f"Document creation error: {str(e)}", exc_info=True)
        
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}", exc_info=True)
        finally:
            progress_bar.empty()
            status_text.empty()
    
    # Footer
    st.markdown("---")
    st.caption("Article Forge - Production NLP System | LLM Integration | Text Processing Pipeline | NLP Metrics")


if __name__ == "__main__":
    create_app()




