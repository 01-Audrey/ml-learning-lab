"""
TextAI Studio - Unified NLP Web Application
===========================================

Week 9 - Day 57
Author: Audrey
Date: December 23, 2024

Features:
- Sentiment Analysis
- Text Summarization  
- Fake News Detection
- Beautiful, professional UI
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import textai_studio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textai_studio_api.textai_studio import TextAIStudio
import time
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="TextAI Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .tool-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'studio' not in st.session_state:
    st.session_state.studio = None
    st.session_state.models_loaded = False

@st.cache_resource
def load_studio():
    """Load TextAI Studio with caching"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_paths = {
        'sentiment': os.path.join(base_dir, 'models', 'bert_sentiment_model'),
        'summarizer': os.path.join(base_dir, 't5_summarization_results', 'final_model'),
        'fake_news': os.path.join(base_dir, 'fake_news_detector_results', 'final_model')
    }
    
    return TextAIStudio(model_paths, device='cpu')

# Header
st.markdown('<h1 class="main-header">🤖 TextAI Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional NLP Tools for Text Analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=TextAI+Studio", width=300)
    
    st.markdown("### 🎯 Available Tools")
    st.markdown("""
    - 😊 **Sentiment Analysis**
    - 📝 **Text Summarization**
    - 🚨 **Fake News Detection**
    - 🔗 **Multi-Tool Pipeline**
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    st.metric("Total Parameters", "280M")
    st.metric("Models", "3")
    st.metric("Avg Accuracy", "94.6%")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    Built with ❤️ by Audrey
    
    **Week 8 Project**  
    ML Learning Journey
    
    [GitHub](https://github.com/01-Audrey/ml-learning-lab)
    """)

# Load models
if not st.session_state.models_loaded:
    with st.spinner("🔄 Loading AI models... This may take a moment..."):
        try:
            st.session_state.studio = load_studio()
            st.session_state.models_loaded = True
            st.success("✅ All models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.stop()

studio = st.session_state.studio

# Main content - Tool selection
st.markdown("## 🛠️ Select Your Tool")

tool = st.selectbox(
    "Choose an NLP tool:",
    ["Sentiment Analysis 😊", "Text Summarization 📝", "Fake News Detection 🚨", "Multi-Tool Pipeline 🔗"],
    label_visibility="collapsed"
)

st.markdown("---")

# Tool 1: Sentiment Analysis
if "Sentiment Analysis" in tool:
    st.markdown("### 😊 Sentiment Analysis")
    st.markdown("Analyze the emotional tone of your text (Positive/Negative)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sentiment_text = st.text_area(
            "Enter text to analyze:",
            height=200,
            placeholder="Type or paste your text here...\n\nExample: 'This product is absolutely amazing! I love it!'"
        )
        
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### 💡 Tips")
        st.info("""
        **Best for:**
        - Product reviews
        - Customer feedback
        - Social media posts
        - Survey responses
        
        **Model:** BERT-base  
        **Accuracy:** 92.5%
        """)
    
    if analyze_btn and sentiment_text:
        with st.spinner("Analyzing sentiment..."):
            result = studio.analyze_sentiment(sentiment_text)
            
            if result['success']:
                res = result['result']
                
                # Display result with color
                if res['sentiment'] == 'Positive':
                    st.markdown(f'<div class="success-box"><h3>✅ Result: POSITIVE</h3><p>Confidence: {res["confidence"]:.1f}%</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="danger-box"><h3>⚠️ Result: NEGATIVE</h3><p>Confidence: {res["confidence"]:.1f}%</p></div>', unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res['confidence'],
                    title={'text': "Confidence Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen" if res['sentiment'] == 'Positive' else "darkred"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightgreen" if res['sentiment'] == 'Positive' else "lightcoral"}
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Score breakdown
                st.markdown("#### 📊 Score Breakdown")
                col1, col2 = st.columns(2)
                col1.metric("Negative Score", f"{res['scores']['negative']:.1f}%")
                col2.metric("Positive Score", f"{res['scores']['positive']:.1f}%")
                
                st.caption(f"⏱️ Analysis time: {result['metadata']['latency_ms']:.1f}ms")

# Tool 2: Text Summarization
elif "Text Summarization" in tool:
    st.markdown("### 📝 Text Summarization")
    st.markdown("Generate concise summaries of long articles or documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        summary_text = st.text_area(
            "Enter text to summarize:",
            height=250,
            placeholder="Paste a long article or document here...\n\nMinimum ~100 words recommended"
        )
        
        length = st.select_slider(
            "Summary length:",
            options=['short', 'medium', 'long'],
            value='medium'
        )
        
        summarize_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### 💡 Tips")
        st.info("""
        **Best for:**
        - News articles
        - Research papers
        - Long documents
        - Reports
        
        **Length Guide:**
        - Short: ~20-40 words
        - Medium: ~40-75 words
        - Long: ~75-110 words
        
        **Model:** T5-small  
        **ROUGE-1:** 31.66%
        """)
    
    if summarize_btn and summary_text:
        if len(summary_text.split()) < 50:
            st.warning("⚠️ Text is quite short. Summaries work best with 100+ words.")
        
        with st.spinner(f"Generating {length} summary..."):
            result = studio.summarize(summary_text, length=length)
            
            if result['success']:
                res = result['result']
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("#### ✅ Summary Generated")
                st.markdown(f"**{res['summary']}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Words", res['original_words'])
                col2.metric("Summary Words", res['summary_words'])
                col3.metric("Compression", f"{res['compression_ratio']*100:.0f}%")
                
                st.caption(f"⏱️ Generation time: {result['metadata']['latency_ms']:.0f}ms")

# Tool 3: Fake News Detection
elif "Fake News Detection" in tool:
    st.markdown("### 🚨 Fake News Detection")
    st.markdown("Detect potentially fake or misleading news articles")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fake_news_text = st.text_area(
            "Enter article or news text:",
            height=200,
            placeholder="Paste news article text here..."
        )
        
        detect_btn = st.button("🔍 Check Credibility", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### 💡 Tips")
        st.info("""
        **Red Flags:**
        - ALL CAPS headlines
        - Excessive punctuation!!!
        - Vague sources
        - Emotional language
        - "SHARE before DELETED"
        
        **Model:** BERT-base  
        **Accuracy:** 98.2%  
        **Precision:** 97.8%
        """)
    
    if detect_btn and fake_news_text:
        with st.spinner("Analyzing credibility..."):
            result = studio.detect_fake_news(fake_news_text)
            
            if result['success']:
                res = result['result']
                
                # Display result
                if res['prediction'] == 'REAL':
                    st.markdown(f'<div class="success-box"><h3>✅ Appears to be REAL NEWS</h3><p>Confidence: {res["confidence"]:.1f}%</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="danger-box"><h3>🚨 WARNING: Likely FAKE NEWS</h3><p>Confidence: {res["confidence"]:.1f}%</p></div>', unsafe_allow_html=True)
                
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res['confidence'],
                    title={'text': "Confidence Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen" if res['prediction'] == 'REAL' else "darkred"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 85], 'color': "yellow"},
                            {'range': [85, 100], 'color': "lightgreen" if res['prediction'] == 'REAL' else "lightcoral"}
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Credibility scores
                st.markdown("#### 📊 Credibility Scores")
                col1, col2 = st.columns(2)
                col1.metric("Real News Score", f"{res['scores']['real']:.1f}%", delta="Credible" if res['prediction']=='REAL' else None)
                col2.metric("Fake News Score", f"{res['scores']['fake']:.1f}%", delta="Suspicious" if res['prediction']=='FAKE' else None)
                
                # Recommendations
                if res['prediction'] == 'FAKE':
                    st.warning("""
                    **⚠️ Recommendation:** 
                    - Verify with credible sources
                    - Check author credentials
                    - Look for citations
                    - Cross-reference facts
                    """)
                
                st.caption(f"⏱️ Analysis time: {result['metadata']['latency_ms']:.1f}ms")

# Tool 4: Multi-Tool Pipeline
else:  # Pipeline
    st.markdown("### 🔗 Multi-Tool Pipeline")
    st.markdown("Run multiple analyses in sequence")
    
    pipeline_text = st.text_area(
        "Enter text for comprehensive analysis:",
        height=200,
        placeholder="Paste text to analyze with multiple tools..."
    )
    
    st.markdown("#### Select Analysis Pipeline")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        check_fake = st.checkbox("🚨 Check Fake News", value=True)
    with col2:
        do_summary = st.checkbox("📝 Summarize", value=True)
    with col3:
        do_sentiment = st.checkbox("😊 Sentiment", value=True)
    
    run_pipeline_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
    
    if run_pipeline_btn and pipeline_text:
        tasks = []
        if check_fake:
            tasks.append('fake_news')
        if do_summary:
            tasks.append('summarize')
        if do_sentiment:
            tasks.append('sentiment')
        
        if not tasks:
            st.warning("⚠️ Please select at least one analysis tool")
        else:
            with st.spinner(f"Running {len(tasks)} analyses..."):
                result = studio.pipeline(pipeline_text, tasks=tasks)
                
                if result['success']:
                    res = result['result']
                    
                    # Display each result
                    if 'fake_news' in res:
                        fn = res['fake_news']
                        if fn['prediction'] == 'REAL':
                            st.success(f"✅ Credibility Check: **REAL** ({fn['confidence']:.1f}%)")
                        else:
                            st.error(f"🚨 Credibility Check: **FAKE** ({fn['confidence']:.1f}%)")
                            if res.get('pipeline_stopped'):
                                st.warning("⚠️ Pipeline stopped - content flagged as fake news")
                    
                    if 'summary' in res:
                        st.info(f"📝 Summary: {res['summary']['summary']}")
                    
                    if 'sentiment' in res:
                        sent = res['sentiment']
                        if sent['sentiment'] == 'Positive':
                            st.success(f"😊 Sentiment: **{sent['sentiment']}** ({sent['confidence']:.1f}%)")
                        else:
                            st.warning(f"😔 Sentiment: **{sent['sentiment']}** ({sent['confidence']:.1f}%)")
                    
                    st.caption(f"⏱️ Total pipeline time: {result['metadata']['latency_ms']:.0f}ms")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>TextAI Studio</strong> | Built with Streamlit & Transformers</p>
    <p>Week 8-9 Project | ML Learning Journey | 2024</p>
</div>
""", unsafe_allow_html=True)