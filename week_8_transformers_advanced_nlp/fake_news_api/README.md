
# Fake News Detector API

BERT-based fake news detection with 98.2% accuracy and explainability features.

## Features
- Binary classification (Real/Fake)
- Confidence scoring (0-100%)
- Attention-based word importance
- Batch processing
- GPU acceleration

## Installation
```bash
pip install transformers torch
```

## Quick Start
```python
from fake_news_detector import FakeNewsDetector

# Initialize
detector = FakeNewsDetector("path/to/model")

# Simple prediction
result = detector.predict("Article text here...")
print(f"{result['prediction']}: {result['confidence']:.1f}%")

# With explanation
result = detector.predict(
    "Article text here...",
    include_explanation=True,
    top_words=5
)
print("Top influential words:")
for word in result['explanation']['top_words']:
    print(f"  - {word['token']}: {word['attention']:.4f}")

# Batch processing
texts = ["Article 1...", "Article 2...", "Article 3..."]
results = detector.batch_predict(texts)
```

## Performance Metrics
- Accuracy: 98.2%
- Precision: 97.8%
- Recall: 98.9%
- F1-Score: 98.3%

## Model Details
- Base model: BERT-base-uncased
- Training samples: 10,000
- Fine-tuning epochs: 3
- Max sequence length: 512 tokens

## Use Cases
- Content moderation
- Fact-checking assistance
- Media literacy tools
- Research analysis

## Limitations
- Trained on specific fake news patterns
- May not detect novel manipulation tactics
- Cannot verify factual accuracy
- Satire may be misclassified
- Should supplement human judgment

## Citation
```
Audrey (2024). Fake News Detector.
ML Learning Journey - Day 55.
```
