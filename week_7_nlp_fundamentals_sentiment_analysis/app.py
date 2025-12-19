
"""
TextAI Studio - LSTM Sentiment Analyzer
Built with Streamlit
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set page config
st.set_page_config(
    page_title="TextAI Studio - Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .negative-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .neutral-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .word-importance {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 3px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL LOADING
# ============================================

class LSTMSentimentClassifier(nn.Module):
    """LSTM-based sentiment classifier."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, dropout, pad_idx, bidirectional=False):
        super(LSTMSentimentClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]

        dropped = self.dropout(hidden)
        output = self.fc(dropped)
        predictions = torch.sigmoid(output)

        return predictions

@st.cache_resource
def load_model():
    """Load trained model and preprocessing objects."""

    # Load preprocessing objects
    with open('data/preprocessing_objects.pkl', 'rb') as f:
        preprocessing_objects = pickle.load(f)

    word2idx = preprocessing_objects['word2idx']
    idx2word = preprocessing_objects['idx2word']
    vocab_size = preprocessing_objects['vocab_size']
    max_len = preprocessing_objects['max_len']

    # Load Day 45 experiments to get best model config
    with open('models/experiments/day45_all_experiments.pkl', 'rb') as f:
        experiments = pickle.load(f)

    # Find best model
    best_exp_name = max(experiments.keys(), 
                        key=lambda x: experiments[x]['best_valid_acc'] if experiments[x]['status'] == 'COMPLETE' else 0)
    best_config = experiments[best_exp_name]['config']

    # Create model
    device = torch.device('cpu')
    pad_idx = word2idx['<PAD>']

    model = LSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=best_config['embedding_dim'],
        hidden_dim=best_config['hidden_dim'],
        output_dim=1,
        n_layers=best_config['n_layers'],
        dropout=best_config['dropout'],
        pad_idx=pad_idx,
        bidirectional=best_config.get('bidirectional', False)
    ).to(device)

    # Load weights
    weight_paths = [
        'models/best_model_extended_v2.pt',
        'models/best_model_extended.pt',
        f'models/experiments/{best_exp_name}.pt',
    ]

    for path in weight_paths:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            break

    model.eval()

    return model, word2idx, idx2word, vocab_size, max_len, device

# ============================================
# HELPER FUNCTIONS
# ============================================

def clean_text(text):
    """Clean text for prediction."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text, model, word2idx, max_len, device):
    """Predict sentiment for given text."""

    # Clean text
    cleaned = clean_text(text)
    words = cleaned.split()

    if len(words) == 0:
        return None, None, []

    # Convert to sequence
    sequence = []
    for word in words:
        idx = word2idx.get(word, word2idx.get('<UNK>', 0))
        sequence.append(idx)

    # Pad
    if len(sequence) > max_len:
        words = words[:max_len]
        sequence = sequence[:max_len]
    else:
        sequence = sequence + [0] * (max_len - len(sequence))

    # Convert to tensor
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)

    # Predict
    with torch.no_grad():
        prob = model(sequence_tensor).squeeze().item()

    # Get word importance (simple occlusion) - only for first 30 words
    importance_scores = []
    words_to_analyze = min(len(words), 30)  # Limit to avoid slowdown

    for i in range(words_to_analyze):
        # Create text without this word
        modified_words = words[:i] + words[i+1:]
        if len(modified_words) == 0:
            importance_scores.append(0)
            continue

        modified_text = ' '.join(modified_words)
        modified_cleaned = clean_text(modified_text)
        modified_word_list = modified_cleaned.split()

        modified_sequence = []
        for word in modified_word_list:
            idx = word2idx.get(word, word2idx.get('<UNK>', 0))
            modified_sequence.append(idx)

        if len(modified_sequence) > max_len:
            modified_sequence = modified_sequence[:max_len]
        else:
            modified_sequence = modified_sequence + [0] * (max_len - len(modified_sequence))

        modified_tensor = torch.tensor([modified_sequence], dtype=torch.long).to(device)

        with torch.no_grad():
            modified_prob = model(modified_tensor).squeeze().item()

        importance = abs(prob - modified_prob)
        importance_scores.append(importance)

    # Normalize importance scores
    if len(importance_scores) > 0 and max(importance_scores) > 0:
        importance_scores = [s / max(importance_scores) for s in importance_scores]

    return prob, words[:words_to_analyze], importance_scores

def get_sentiment_label(prob):
    """Get sentiment label from probability."""
    if prob >= 0.7:
        return "POSITIVE 😊", "positive"
    elif prob >= 0.55:
        return "Slightly Positive 🙂", "slightly-positive"
    elif prob >= 0.45:
        return "Neutral 😐", "neutral"
    elif prob >= 0.3:
        return "Slightly Negative 🙁", "slightly-negative"
    else:
        return "NEGATIVE 😞", "negative"

def get_confidence_level(prob):
    """Get confidence level."""
    confidence = abs(prob - 0.5) * 2
    if confidence > 0.8:
        return "Very High", "🟢"
    elif confidence > 0.6:
        return "High", "🟢"
    elif confidence > 0.4:
        return "Moderate", "🟡"
    else:
        return "Low", "🟠"

# ============================================
# LOAD MODEL
# ============================================

try:
    model, word2idx, idx2word, vocab_size, max_len, device = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Error loading model: {str(e)}")

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application."""

    # Header
    st.markdown('<div class="main-header">🎭 TextAI Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">LSTM-Powered Sentiment Analysis | Built with PyTorch & Streamlit</div>', unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("⚠️ Model could not be loaded. Please check that model files exist.")
        st.write("Make sure you're running from the week_7_nlp_fundamentals_sentiment_analysis directory")
        return

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode:",
        ["🔍 Single Text Analysis", "📊 Batch Processing", "ℹ️ About"]
    )

    st.sidebar.markdown("---")

    # Model info
    with st.sidebar.expander("📈 Model Information"):
        st.write(f"**Architecture:** Bidirectional LSTM")
        st.write(f"**Parameters:** 2.2M+")
        st.write(f"**Test Accuracy:** 80.38%")
        st.write(f"**Training Data:** IMDB 50K Reviews")
        st.write(f"**Vocabulary:** {vocab_size:,} words")

    with st.sidebar.expander("🎯 Performance Metrics"):
        st.write(f"**Precision:** 80.44%")
        st.write(f"**Recall:** 80.36%")
        st.write(f"**F1 Score:** 80.40%")
        st.write(f"**Inference:** <50ms avg")

    # Main content based on mode
    if mode == "🔍 Single Text Analysis":
        single_text_analysis()
    elif mode == "📊 Batch Processing":
        batch_processing()
    else:
        about_page()

def single_text_analysis():
    """Single text analysis interface."""

    st.header("🔍 Single Text Analysis")

    # Input methods
    input_method = st.radio("Input Method:", ["✍️ Type Text", "📄 Example Reviews"])

    if input_method == "✍️ Type Text":
        text_input = st.text_area(
            "Enter your review:",
            height=150,
            placeholder="Type or paste your movie review here...",
            help="Enter a movie review to analyze its sentiment"
        )
    else:
        # Example reviews
        examples = {
            "Positive Example 1": "This movie was absolutely fantastic! The acting was superb and the story kept me engaged from start to finish. Highly recommended!",
            "Positive Example 2": "Amazing cinematography and brilliant direction. One of the best films I've seen this year!",
            "Negative Example 1": "Terrible waste of time. The plot was boring and predictable, and the acting was awful throughout.",
            "Negative Example 2": "Disappointing film with weak characters and a confusing storyline. Do not recommend.",
            "Mixed Example": "Great cinematography and decent acting, but the story was quite boring and predictable overall.",
        }

        selected_example = st.selectbox("Choose an example:", list(examples.keys()))
        text_input = st.text_area(
            "Review text:",
            value=examples[selected_example],
            height=150
        )

    # Analyze button
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if not text_input or len(text_input.strip()) < 10:
            st.warning("⚠️ Please enter at least 10 characters of text.")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Predict
                prob, words, importance = predict_sentiment(text_input, model, word2idx, max_len, device)

                if prob is None:
                    st.error("❌ Could not analyze text. Please check your input.")
                else:
                    # Display results
                    display_results(text_input, prob, words, importance)

def display_results(text, prob, words, importance):
    """Display prediction results."""

    # Get sentiment and confidence
    sentiment_label, sentiment_class = get_sentiment_label(prob)
    confidence_label, confidence_icon = get_confidence_level(prob)
    confidence = abs(prob - 0.5) * 2

    # Results header
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Main prediction display
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if sentiment_class == "positive":
            st.markdown(f'<div class="positive-box"><h2 style="margin:0; color:#2E7D32;">{sentiment_label}</h2></div>', unsafe_allow_html=True)
        elif sentiment_class == "negative":
            st.markdown(f'<div class="negative-box"><h2 style="margin:0; color:#C62828;">{sentiment_label}</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="neutral-box"><h2 style="margin:0; color:#F57C00;">{sentiment_label}</h2></div>', unsafe_allow_html=True)

    with col2:
        st.metric("Probability", f"{prob:.1%}", help="Model's confidence in prediction")

    with col3:
        st.metric("Confidence", f"{confidence_icon} {confidence_label}", help="How certain the model is")

    # Detailed metrics
    st.markdown("### 📈 Detailed Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Positive Score", f"{prob:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Negative Score", f"{(1-prob):.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Words Analyzed", len(words))
        st.markdown('</div>', unsafe_allow_html=True)

    # Word importance visualization
    if len(words) > 0 and len(importance) > 0:
        st.markdown("### 🔍 Word Importance Analysis")
        st.write("Words highlighted by importance (darker = more influential)")

        # Display words with importance coloring
        word_html = ""
        for word, imp in zip(words, importance):
            # Color based on importance and sentiment
            if prob >= 0.5:  # Positive
                bg_color = f"rgba(76, 175, 80, {imp * 0.7})"
                text_color = "#1B5E20" if imp > 0.5 else "#333"
            else:  # Negative
                bg_color = f"rgba(244, 67, 54, {imp * 0.7})"
                text_color = "#B71C1C" if imp > 0.5 else "#333"

            word_html += f'<span class="word-importance" style="background-color: {bg_color}; color: {text_color};">{word}</span> '

        st.markdown(word_html, unsafe_allow_html=True)

        # Top important words
        st.markdown("#### 🏆 Top 5 Most Important Words")
        word_imp_pairs = list(zip(words, importance))
        word_imp_pairs.sort(key=lambda x: x[1], reverse=True)

        top_words = word_imp_pairs[:5]

        for i, (word, imp) in enumerate(top_words, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(imp, text=f"{i}. **{word}**")
            with col2:
                st.write(f"{imp:.2%}")

    # Explanation
    st.markdown("### 💡 Explanation")

    if confidence > 0.7:
        explanation = f"The model is **very confident** that this review is **{sentiment_label.split()[0].lower()}**. "
    elif confidence > 0.4:
        explanation = f"The model is **moderately confident** that this review is **{sentiment_label.split()[0].lower()}**. "
    else:
        explanation = f"The model has **low confidence** in this prediction. The review may contain mixed or unclear sentiment. "

    if len(top_words) > 0:
        key_words = ', '.join([f"'{w}'" for w, _ in top_words[:3]])
        explanation += f"The prediction is primarily influenced by words like {key_words}."

    st.info(explanation)

def batch_processing():
    """Batch processing interface."""

    st.header("📊 Batch Processing")

    st.write("Analyze multiple reviews at once.")

    st.write("Enter multiple reviews, one per line:")

    batch_text = st.text_area(
        "Enter reviews (one per line):",
        height=200,
        placeholder="Review 1\nReview 2\nReview 3\n..."
    )

    if st.button("🚀 Analyze All Reviews", type="primary"):
        if batch_text.strip():
            reviews = [line.strip() for line in batch_text.split('\n') if line.strip() and len(line.strip()) > 10]
            if len(reviews) > 0:
                analyze_batch(reviews)
            else:
                st.warning("⚠️ No valid reviews found. Each review must be at least 10 characters.")
        else:
            st.warning("⚠️ Please enter some reviews to analyze.")

def analyze_batch(reviews):
    """Analyze batch of reviews."""

    st.markdown("---")
    st.subheader("📊 Batch Analysis Results")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    for i, review in enumerate(reviews):
        status_text.text(f"Analyzing review {i+1}/{len(reviews)}...")
        progress_bar.progress((i + 1) / len(reviews))

        prob, words, importance = predict_sentiment(review, model, word2idx, max_len, device)

        if prob is None:
            results.append({
                'review': review[:100] + '...' if len(review) > 100 else review,
                'sentiment': 'Error',
                'probability': 0,
                'confidence': 0
            })
        else:
            sentiment_label, _ = get_sentiment_label(prob)
            confidence = abs(prob - 0.5) * 2

            results.append({
                'review': review[:100] + '...' if len(review) > 100 else review,
                'sentiment': sentiment_label,
                'probability': prob,
                'confidence': confidence
            })

    status_text.text("✅ Analysis complete!")
    progress_bar.empty()

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Summary statistics
    st.markdown("### 📈 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    successful = results_df[results_df['sentiment'] != 'Error']

    if len(successful) > 0:
        positive_count = len(successful[successful['probability'] >= 0.5])
        negative_count = len(successful[successful['probability'] < 0.5])
        avg_confidence = successful['confidence'].mean()

        with col1:
            st.metric("Total Reviews", len(reviews))
        with col2:
            st.metric("Positive", positive_count, f"{positive_count/len(successful)*100:.1f}%")
        with col3:
            st.metric("Negative", negative_count, f"{negative_count/len(successful)*100:.1f}%")
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    # Display results table
    st.markdown("### 📋 Detailed Results")

    display_df = results_df.copy()
    display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}" if x > 0 else "N/A")
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}" if x > 0 else "N/A")

    st.dataframe(display_df, use_container_width=True)

def about_page():
    """About page."""

    st.header("ℹ️ About TextAI Studio")

    st.markdown("""
    ### 🎭 What is TextAI Studio?

    TextAI Studio is an **LSTM-powered sentiment analysis application** built with PyTorch and Streamlit.

    ### 🧠 Model Details

    - **Architecture:** Bidirectional LSTM (2 layers, 128 hidden units)
    - **Training Data:** IMDB 50,000 movie reviews
    - **Performance:** 80.38% test accuracy

    ### 👤 Developer

    **Audrey** - CS Student @ LPU Laguna

    🔗 [GitHub](https://github.com/01-Audrey)
    """)

if __name__ == "__main__":
    main()
