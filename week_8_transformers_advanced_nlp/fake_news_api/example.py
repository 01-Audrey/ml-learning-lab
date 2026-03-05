
"""
Example usage of FakeNewsDetector API
"""

from fake_news_detector import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector("../fake_news_detector_results/final_model")

# Example articles
articles = [
    "The Federal Reserve announced interest rate changes affecting mortgage rates nationwide.",
    "SHOCKING: Scientists discover miracle cure that big pharma doesn't want you to know!",
    "New research published in Nature shows promising developments in renewable energy."
]

print("\nFake News Detection Results:")
print("=" * 80)

for i, article in enumerate(articles, 1):
    print(f"\nArticle {i}:")
    print(f"Text: {article[:60]}...")

    result = detector.predict(article, include_explanation=True)

    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"\nTop words:")
    for word in result['explanation']['top_words'][:3]:
        print(f"  • {word['token']}")
    print("-" * 80)
