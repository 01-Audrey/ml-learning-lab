"""
TextAI Studio - Unified NLP API
================================

Production-ready unified API for multiple NLP tasks.

Author: Audrey
Date: December 22, 2024
Version: 1.0.0

Components:
- Sentiment Analysis (BERT-base)
- Text Summarization (T5-small fine-tuned)
- Fake News Detection (BERT-base fine-tuned)

Usage:
    from textai_studio import TextAIStudio
    
    # Initialize
    studio = TextAIStudio(model_paths, device='cuda')
    
    # Sentiment analysis
    result = studio.analyze_sentiment("Great product!")
    
    # Text summarization
    result = studio.summarize("Long article...", length='medium')
    
    # Fake news detection
    result = studio.detect_fake_news("Article text...")
    
    # Pipeline (multiple tasks)
    result = studio.pipeline(text, tasks=['fake_news', 'summarize', 'sentiment'])
"""

import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration
)


class TextAIStudio:
    """
    Unified API for NLP tasks.
    
    Integrates three models:
    1. Sentiment Analysis - BERT-base
    2. Text Summarization - T5-small (fine-tuned)
    3. Fake News Detection - BERT-base (fine-tuned)
    """
    
    def __init__(self, model_paths, device='cpu'):
        """
        Initialize TextAI Studio.
        
        Args:
            model_paths: Dictionary with paths to models
                {
                    'sentiment': 'path/to/sentiment_model/',
                    'summarizer': 'path/to/t5_model/',
                    'fake_news': 'path/to/fake_news_model/'
                }
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.models = {}
        self.tokenizers = {}
        
        self._load_models(model_paths)
    
    def _load_models(self, model_paths):
        """Load all models into memory."""
        
        # Load Sentiment Analyzer
        if 'sentiment' in model_paths:
            self.tokenizers['sentiment'] = AutoTokenizer.from_pretrained(model_paths['sentiment'])
            self.models['sentiment'] = AutoModelForSequenceClassification.from_pretrained(model_paths['sentiment'])
            self.models['sentiment'] = self.models['sentiment'].to(self.device)
            self.models['sentiment'].eval()
        
        # Load Text Summarizer
        if 'summarizer' in model_paths:
            self.tokenizers['summarizer'] = AutoTokenizer.from_pretrained(model_paths['summarizer'])
            self.models['summarizer'] = T5ForConditionalGeneration.from_pretrained(model_paths['summarizer'])
            self.models['summarizer'] = self.models['summarizer'].to(self.device)
            self.models['summarizer'].eval()
        
        # Load Fake News Detector
        if 'fake_news' in model_paths:
            self.tokenizers['fake_news'] = AutoTokenizer.from_pretrained(model_paths['fake_news'])
            self.models['fake_news'] = AutoModelForSequenceClassification.from_pretrained(model_paths['fake_news'])
            self.models['fake_news'] = self.models['fake_news'].to(self.device)
            self.models['fake_news'].eval()
    
    def _format_response(self, success, result=None, error=None, metadata=None):
        """Format standardized response."""
        return {
            'success': success,
            'result': result,
            'error': error,
            'metadata': metadata or {}
        }
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
        
        Returns:
            {
                'success': True,
                'result': {
                    'sentiment': 'Positive' or 'Negative',
                    'confidence': float (0-100),
                    'scores': {'negative': float, 'positive': float}
                },
                'metadata': {'model': str, 'latency_ms': float, 'timestamp': str}
            }
        """
        start_time = time.time()
        
        try:
            inputs = self.tokenizers['sentiment'](
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['sentiment'](**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item() * 100
            
            sentiment = "Positive" if predicted_class == 1 else "Negative"
            latency = (time.time() - start_time) * 1000
            
            return self._format_response(
                success=True,
                result={
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'scores': {
                        'negative': probs[0][0].item() * 100,
                        'positive': probs[0][1].item() * 100
                    }
                },
                metadata={
                    'model': 'sentiment_analyzer',
                    'latency_ms': latency,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            )
        except Exception as e:
            return self._format_response(success=False, error=str(e))
    
    def summarize(self, text, length='medium'):
        """
        Summarize text with adjustable length.
        
        Args:
            text: Input text
            length: 'short', 'medium', or 'long'
        
        Returns:
            {
                'success': True,
                'result': {
                    'summary': str,
                    'length_type': str,
                    'original_words': int,
                    'summary_words': int,
                    'compression_ratio': float
                },
                'metadata': {...}
            }
        """
        start_time = time.time()
        
        try:
            length_configs = {
                'short': {'max_length': 50, 'min_length': 20},
                'medium': {'max_length': 100, 'min_length': 40},
                'long': {'max_length': 150, 'min_length': 60}
            }
            config = length_configs.get(length, length_configs['medium'])
            
            input_text = "summarize: " + text
            inputs = self.tokenizers['summarizer'](
                input_text, return_tensors="pt", max_length=512, truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.models['summarizer'].generate(
                    inputs['input_ids'],
                    max_length=config['max_length'],
                    min_length=config['min_length'],
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            summary = self.tokenizers['summarizer'].decode(summary_ids[0], skip_special_tokens=True)
            
            original_words = len(text.split())
            summary_words = len(summary.split())
            compression_ratio = summary_words / original_words if original_words > 0 else 0
            latency = (time.time() - start_time) * 1000
            
            return self._format_response(
                success=True,
                result={
                    'summary': summary,
                    'length_type': length,
                    'original_words': original_words,
                    'summary_words': summary_words,
                    'compression_ratio': compression_ratio
                },
                metadata={
                    'model': 'text_summarizer',
                    'latency_ms': latency,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            )
        except Exception as e:
            return self._format_response(success=False, error=str(e))
    
    def detect_fake_news(self, text):
        """
        Detect if text is fake news.
        
        Args:
            text: Input text
        
        Returns:
            {
                'success': True,
                'result': {
                    'prediction': 'REAL' or 'FAKE',
                    'confidence': float (0-100),
                    'scores': {'real': float, 'fake': float}
                },
                'metadata': {...}
            }
        """
        start_time = time.time()
        
        try:
            inputs = self.tokenizers['fake_news'](
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['fake_news'](**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item() * 100
            
            prediction = "FAKE" if predicted_class == 1 else "REAL"
            latency = (time.time() - start_time) * 1000
            
            return self._format_response(
                success=True,
                result={
                    'prediction': prediction,
                    'confidence': confidence,
                    'scores': {
                        'real': probs[0][0].item() * 100,
                        'fake': probs[0][1].item() * 100
                    }
                },
                metadata={
                    'model': 'fake_news_detector',
                    'latency_ms': latency,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            )
        except Exception as e:
            return self._format_response(success=False, error=str(e))
    
    def pipeline(self, text, tasks):
        """
        Run multiple tasks in sequence.
        
        Args:
            text: Input text
            tasks: List of task names ['fake_news', 'summarize', 'sentiment']
        
        Returns:
            Combined results from all tasks
        """
        start_time = time.time()
        
        try:
            results = {}
            current_text = text
            
            for task in tasks:
                if task == 'fake_news':
                    result = self.detect_fake_news(current_text)
                    results['fake_news'] = result['result']
                    if result['result']['prediction'] == 'FAKE':
                        results['pipeline_stopped'] = True
                        results['stop_reason'] = 'Fake news detected'
                        break
                
                elif task == 'summarize':
                    result = self.summarize(current_text)
                    results['summary'] = result['result']
                    current_text = result['result']['summary']
                
                elif task == 'sentiment':
                    result = self.analyze_sentiment(current_text)
                    results['sentiment'] = result['result']
            
            latency = (time.time() - start_time) * 1000
            
            return self._format_response(
                success=True,
                result=results,
                metadata={
                    'model': 'pipeline',
                    'tasks': tasks,
                    'latency_ms': latency,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            )
        except Exception as e:
            return self._format_response(success=False, error=str(e))


# Quick test function
def test_textai_studio():
    """Quick test of TextAI Studio."""
    import os
    
    # Example paths (update these!)
    model_paths = {
        'sentiment': 'models/bert_sentiment_model',
        'summarizer': 't5_summarization_results/final_model',
        'fake_news': 'fake_news_detector_results/final_model'
    }
    
    studio = TextAIStudio(model_paths, device='cpu')
    
    # Test
    result = studio.analyze_sentiment("This is great!")
    print("Sentiment:", result['result']['sentiment'])
    
    return studio


if __name__ == "__main__":
    studio = test_textai_studio()
    print("✅ TextAI Studio ready!")