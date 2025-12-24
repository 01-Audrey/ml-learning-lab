"""
Fix Sentiment Model - Save in Hugging Face Format
==================================================

Issue: Model saved as .pt weights only
Solution: Re-save with full model structure
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

print("🔧 Fixing Sentiment Model Format...")

# Paths
weights_path = "models/bert_sentiment_model.pt"
output_path = "models/bert_sentiment_model/"

# Load base BERT architecture
print("\n1. Loading BERT-base architecture...")
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Load trained weights
print("2. Loading trained weights...")
state_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(state_dict)

# Save in HuggingFace format
print("3. Saving in HuggingFace format...")
os.makedirs(output_path, exist_ok=True)

model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained(output_path)

print(f"\n✅ Model saved to: {output_path}")
print("✅ Now compatible with TextAI Studio API!")