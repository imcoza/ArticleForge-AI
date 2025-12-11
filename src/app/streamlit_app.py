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
            
            # Check for warnings
            if 'warning' in article_result:
                st.warning(f"⚠️ {article_result['warning']}")
            
            # Validate word count range
            min_acceptable = max(settings.MIN_WORD_COUNT, int(word_limit * 0.8))
            max_acceptable = word_limit + 100
            
            if actual_word_count < min_acceptable:
                st.warning(f"⚠️ Article is shorter than expected: {actual_word_count} words (target: {word_limit}, minimum: {min_acceptable})")
            elif actual_word_count > max_acceptable:
                st.warning(f"⚠️ Article is longer than expected: {actual_word_count} words (target: {word_limit}, maximum: {max_acceptable})")
            
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
                            key_phrases = advanced_nlp.extract_key_phrases(generated_text, top_n=5)
                            for phrase, score in key_phrases:
                                st.text(f"{phrase}: {score:.2f}")
                            st.markdown("---")
                            
                            # Sentiment Analysis
                            st.markdown("**Sentiment Analysis:**")
                            sentiment = advanced_nlp.classify_text_sentiment(generated_text)
                            st.text(f"Positive: {sentiment['positive']:.3f}")
                            st.text(f"Negative: {sentiment['negative']:.3f}")
                            st.text(f"Neutral: {sentiment['neutral']:.3f}")
                            st.text(f"Predicted: {sentiment['predicted_label']}")
                            st.markdown("---")
                            
                            # Advanced Text Statistics
                            st.markdown("**Advanced Statistics:**")
                            stats = advanced_nlp.calculate_text_statistics(generated_text)
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
                    st.text(f"Total Characters: {len(generated_text)}")
                    st.text(f"Total Words: {word_count}")
                    st.text(f"Total Sentences: {readability.get('sentence_count', 0)}")
                    if readability.get('sentence_count', 0) > 0:
                        st.text(f"Words per Sentence: {readability.get('avg_sentence_length', 0):.2f}")
                    
                    # Summary with TextRank
                    st.markdown("---")
                    st.markdown("**Extractive Summary (TextRank):**")
                    summary = text_processor.summarize_text(generated_text, max_sentences=3, method="textrank")
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
                        if VISUALIZATION_AVAILABLE and pd is not None and px is not None:
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
                                    fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Additional metrics visualization - Radar/Spider chart style
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
                                fig2.update_layout(height=300, showlegend=False, xaxis_tickangle=-45)
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # Perplexity display if available
                                if 'perplexity' in eval_metrics and eval_metrics['perplexity'] != float('inf'):
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




