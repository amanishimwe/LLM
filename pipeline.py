"""
Minimal Hugging Face sentiment-analysis.
Loads a DistilBERT model fine-tuned on SST-2 and scores example sentences
as POSITIVE or NEGATIVE with confidence scores.
"""

from transformers import pipeline

# sentiment-analysis: task name; model= pins a specific checkpoint so results
# are reproducible across runs (default model can change with library updates).
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# Pipeline accepts a single string or a list of strings; returns one dict per input.
response = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
print(response)
