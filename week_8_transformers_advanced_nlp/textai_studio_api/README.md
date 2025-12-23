
# TextAI Studio - Unified NLP API

**Version:** 1.0.0  
**Author:** Audrey  
**Date:** December 22, 2024  

A production-ready unified API integrating three powerful NLP models for sentiment analysis, text summarization, and fake news detection.

---

## 🚀 Features

- **Sentiment Analysis** - BERT-based emotion detection (Positive/Negative)
- **Text Summarization** - T5-based multi-length summaries (Short/Medium/Long)
- **Fake News Detection** - BERT-based credibility analysis with 98.2% accuracy
- **Multi-Model Pipelines** - Chain multiple models together
- **Unified Interface** - Consistent API across all models
- **Production Ready** - Error handling, performance tracking, standardized responses

---

## 📦 Installation
```bash
pip install torch transformers
```

---

## 🎯 Quick Start
```python
from textai_studio import TextAIStudio

# Initialize with model paths
model_paths = {
    'sentiment': 'path/to/sentiment_model.pt',
    'summarizer': 'path/to/t5_model/',
    'fake_news': 'path/to/fake_news_model/'
}

studio = TextAIStudio(model_paths, device='cuda')  # or 'cpu'

# Sentiment Analysis
result = studio.analyze_sentiment("This product is amazing!")
print(result['result']['sentiment'])  # "Positive"

# Text Summarization
result = studio.summarize("Long article text...", length='medium')
print(result['result']['summary'])

# Fake News Detection
result = studio.detect_fake_news("Article text...")
print(result['result']['prediction'])  # "REAL" or "FAKE"

# Pipeline (multiple tasks)
result = studio.pipeline(
    text="Article text...",
    tasks=['fake_news', 'summarize', 'sentiment']
)
```

---

## 📚 API Reference

### TextAIStudio

Main class for accessing all NLP tools.

#### `__init__(model_paths, device='cpu')`

Initialize TextAI Studio.

**Parameters:**
- `model_paths` (dict): Paths to model files
- `device` (str): 'cpu' or 'cuda'

---

### analyze_sentiment(text)

Analyze sentiment of text.

**Parameters:**
- `text` (str): Input text

**Returns:**
```python
{
    'success': True,
    'result': {
        'sentiment': 'Positive' or 'Negative',
        'confidence': 95.3,  # 0-100
        'scores': {
            'negative': 4.7,
            'positive': 95.3
        }
    },
    'metadata': {
        'model': 'sentiment_analyzer',
        'latency_ms': 45.2,
        'timestamp': '2024-12-22 10:30:00'
    }
}
```

---

### summarize(text, length='medium')

Generate text summary with adjustable length.

**Parameters:**
- `text` (str): Input text to summarize
- `length` (str): 'short', 'medium', or 'long'

**Returns:**
```python
{
    'success': True,
    'result': {
        'summary': 'Concise summary text...',
        'length_type': 'medium',
        'original_words': 250,
        'summary_words': 50,
        'compression_ratio': 0.20
    },
    'metadata': {...}
}
```

**Length Options:**
- `short`: 20-50 tokens (~15-40 words)
- `medium`: 40-100 tokens (~30-75 words)
- `long`: 60-150 tokens (~45-110 words)

---

### detect_fake_news(text)

Detect if text is fake news.

**Parameters:**
- `text` (str): Article or news text

**Returns:**
```python
{
    'success': True,
    'result': {
        'prediction': 'REAL' or 'FAKE',
        'confidence': 98.2,
        'scores': {
            'real': 98.2,
            'fake': 1.8
        }
    },
    'metadata': {...}
}
```

---

### pipeline(text, tasks)

Run multiple tasks in sequence.

**Parameters:**
- `text` (str): Input text
- `tasks` (list): List of tasks e.g., ['fake_news', 'summarize', 'sentiment']

**Returns:**
```python
{
    'success': True,
    'result': {
        'fake_news': {...},
        'summary': {...},
        'sentiment': {...}
    },
    'metadata': {...}
}
```

**Available Tasks:**
- `'fake_news'` - Check credibility (stops pipeline if fake detected)
- `'summarize'` - Generate summary (uses summary for subsequent tasks)
- `'sentiment'` - Analyze sentiment

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Sentiment Analysis | 92.5% | 91.8% | 93.2% | 92.5% |
| Text Summarization | ROUGE-1: 31.66% | ROUGE-2: 11.73% | ROUGE-L: 22.66% | - |
| Fake News Detection | 98.2% | 97.8% | 98.9% | 98.3% |

---

## ⚡ Performance Benchmarks

**Hardware:** CPU - Intel i7 / GPU - Tesla T4

| Operation | CPU Latency | GPU Latency |
|-----------|-------------|-------------|
| Sentiment Analysis | ~100ms | ~20ms |
| Fake News Detection | ~110ms | ~22ms |
| Text Summarization | ~800ms | ~150ms |
| Full Pipeline | ~1000ms | ~200ms |

**Throughput:**
- Single model: 8-10 requests/sec (CPU), 40-50 requests/sec (GPU)
- Pipeline: 1-2 requests/sec (CPU), 5-8 requests/sec (GPU)

---

## 🔧 Advanced Usage

### Error Handling
```python
result = studio.analyze_sentiment("Text...")

if result['success']:
    sentiment = result['result']['sentiment']
else:
    print(f"Error: {result['error']}")
```

### Batch Processing
```python
texts = ["Text 1", "Text 2", "Text 3"]
results = [studio.analyze_sentiment(text) for text in texts]
```

### Custom Pipeline Logic
```python
result = studio.pipeline(text, tasks=['fake_news'])

if result['result']['fake_news']['prediction'] == 'REAL':
    # Only summarize if real
    summary_result = studio.summarize(text)
```

---

## 🏗️ Architecture
```
TextAI Studio
├── Sentiment Analyzer (BERT-base, 110M params)
├── Text Summarizer (T5-small fine-tuned, 60M params)
└── Fake News Detector (BERT-base fine-tuned, 110M params)

Total: ~280M parameters, ~1.2GB memory
```

---

## 📝 Use Cases

- **Content Moderation** - Detect fake news + sentiment analysis
- **News Aggregation** - Summarize articles + credibility check
- **Social Media Analysis** - Sentiment tracking + misinformation detection
- **Research Tools** - Automated content analysis pipelines

---

## ⚠️ Limitations

- **Sentiment:** Binary only (Positive/Negative), no neutral or nuanced emotions
- **Summarization:** Best for news/articles, may struggle with technical content
- **Fake News:** Trained on specific patterns, may not detect novel tactics
- **Language:** English only
- **Context:** Models cannot verify factual accuracy, only detect patterns

---

## 🤝 Integration with Week 9

This API is designed for seamless integration into the Week 9 TextAI Studio Streamlit application:
```python
# In Streamlit app
from textai_studio import TextAIStudio

@st.cache_resource
def load_studio():
    return TextAIStudio(model_paths, device='cuda')

studio = load_studio()

# Use in UI
if st.button("Analyze"):
    result = studio.analyze_sentiment(user_input)
    st.success(result['result']['sentiment'])
```

---

## 📄 License

MIT License - For educational and portfolio purposes

---

## 👤 Author

**Audrey**  
ML Learning Journey - Week 8, Day 56  
https://github.com/01-Audrey/ml-learning-lab

---

## 🎯 Week 8 Achievement

Part of comprehensive NLP toolkit built during Week 8:
- Day 51: Sentiment Analysis with BERT
- Day 54: Text Summarization with T5
- Day 55: Fake News Detection with Explainability
- **Day 56: Unified API Integration** ← You are here!

Ready for Week 9: TextAI Studio Web Application! 🚀
