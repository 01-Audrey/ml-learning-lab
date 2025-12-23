
"""
Fake News Detection API
=======================

Production-ready BERT-based fake news detector with explainability.

Author: Audrey
Date: December 21, 2024
Model: BERT-base fine-tuned on fake news dataset
Accuracy: 98.2%

Usage:
    from fake_news_detector import FakeNewsDetector

    detector = FakeNewsDetector("path/to/model")
    result = detector.predict("Article text here...")
    print(result['prediction'], result['confidence'])
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FakeNewsDetector:
    """
    Fake news detection with confidence scoring and explainability.

    Features:
    - 98.2% accuracy on test set
    - Confidence scoring (0-100%)
    - Attention-based word importance
    - Batch processing support
    - GPU acceleration
    """

    def __init__(self, model_path, device=None):
        """
        Initialize detector.

        Args:
            model_path: Path to trained model
            device: 'cuda', 'cpu', or None (auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, text, include_explanation=False, top_words=5):
        """
        Predict if text is fake news.

        Args:
            text: Article text
            include_explanation: Return influential words
            top_words: Number of top words

        Returns:
            {
                'prediction': 'REAL' or 'FAKE',
                'confidence': float (0-100),
                'prob_real': float (0-100),
                'prob_fake': float (0-100),
                'explanation': {...} (optional)
            }
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            if include_explanation:
                outputs = self.model(**inputs, output_attentions=True)
            else:
                outputs = self.model(**inputs)

            logits = outputs.logits

        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

        result = {
            'prediction': 'FAKE' if predicted_class.item() == 1 else 'REAL',
            'confidence': confidence.item() * 100,
            'prob_real': probs[0][0].item() * 100,
            'prob_fake': probs[0][1].item() * 100,
            'predicted_class': predicted_class.item()
        }

        if include_explanation:
            attentions = outputs.attentions
            attention_scores = []
            for layer_attention in attentions:
                cls_attention = layer_attention[0, :, 0, :].mean(dim=0)
                attention_scores.append(cls_attention)

            avg_attention = torch.stack(attention_scores).mean(dim=0)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            attention_weights = avg_attention.cpu().numpy()

            token_attention = []
            for token, weight in zip(tokens, attention_weights):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    token_attention.append({
                        'token': token.replace('##', ''),
                        'attention': float(weight)
                    })

            token_attention.sort(key=lambda x: x['attention'], reverse=True)
            result['explanation'] = {'top_words': token_attention[:top_words]}

        return result

    def batch_predict(self, texts, include_explanation=False):
        """Process multiple texts."""
        return [self.predict(text, include_explanation) for text in texts]
