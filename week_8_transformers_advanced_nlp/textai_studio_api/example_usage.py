
"""
TextAI Studio - Example Usage
==============================

Demonstrates various ways to use the TextAI Studio API.
"""

from textai_studio import TextAIStudio
import os

# Model paths (update these to your actual paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'sentiment': os.path.join(BASE_DIR, '..', 'models', 'bert_sentiment_model.pt'),
    'summarizer': os.path.join(BASE_DIR, '..', 't5_summarization_results', 'final_model'),
    'fake_news': os.path.join(BASE_DIR, '..', 'fake_news_detector_results', 'final_model')
}

# Initialize
print("Initializing TextAI Studio...")
studio = TextAIStudio(MODEL_PATHS, device='cpu')  # Change to 'cuda' if available
print("✅ Studio initialized!\n")

# Example 1: Sentiment Analysis
print("="*80)
print("EXAMPLE 1: Sentiment Analysis")
print("="*80)

text1 = "I absolutely love this product! It exceeded all my expectations!"
result1 = studio.analyze_sentiment(text1)

if result1['success']:
    print(f"Text: {text1}")
    print(f"Sentiment: {result1['result']['sentiment']}")
    print(f"Confidence: {result1['result']['confidence']:.1f}%")
    print(f"Latency: {result1['metadata']['latency_ms']:.1f}ms\n")

# Example 2: Text Summarization
print("="*80)
print("EXAMPLE 2: Text Summarization")
print("="*80)

text2 = """
Artificial intelligence continues to transform industries worldwide. Machine learning 
algorithms now power everything from recommendation systems to autonomous vehicles. 
Companies invest billions in AI research, driving innovation at unprecedented rates. 
Deep learning has achieved remarkable breakthroughs in computer vision, natural language 
processing, and robotics. Researchers explore new applications daily across healthcare, 
finance, education, and entertainment sectors.
"""

for length in ['short', 'medium', 'long']:
    result2 = studio.summarize(text2, length=length)
    if result2['success']:
        print(f"\n{length.upper()} Summary:")
        print(f"  {result2['result']['summary']}")
        print(f"  Words: {result2['result']['summary_words']} "
              f"(compression: {result2['result']['compression_ratio']:.1%})")

# Example 3: Fake News Detection
print("\n" + "="*80)
print("EXAMPLE 3: Fake News Detection")
print("="*80)

texts = [
    "Scientists at MIT published groundbreaking research in Nature journal on renewable energy.",
    "SHOCKING discovery doctors don't want you to know! Share before DELETED!"
]

for text in texts:
    result3 = studio.detect_fake_news(text)
    if result3['success']:
        print(f"\nText: {text[:70]}...")
        print(f"Prediction: {result3['result']['prediction']} "
              f"({result3['result']['confidence']:.1f}% confidence)")

# Example 4: Pipeline
print("\n" + "="*80)
print("EXAMPLE 4: Multi-Model Pipeline")
print("="*80)

text4 = """
Researchers announced significant progress in clean energy technology. The breakthrough 
could reduce carbon emissions by 40% according to peer-reviewed studies. Industry experts 
praised the development as a major step toward sustainability goals.
"""

result4 = studio.pipeline(text4, tasks=['fake_news', 'summarize', 'sentiment'])

if result4['success']:
    print(f"\nOriginal text: {text4[:100]}...\n")

    if 'fake_news' in result4['result']:
        print(f"Credibility: {result4['result']['fake_news']['prediction']}")

    if 'summary' in result4['result']:
        print(f"Summary: {result4['result']['summary']['summary']}")

    if 'sentiment' in result4['result']:
        print(f"Sentiment: {result4['result']['sentiment']['sentiment']}")

    print(f"\nTotal Pipeline Time: {result4['metadata']['latency_ms']:.0f}ms")

print("\n" + "="*80)
print("✅ All examples completed!")
print("="*80)
