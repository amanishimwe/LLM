# NLP sentiment pipeline

Small example that runs **sentiment analysis** with [Hugging Face Transformers](https://huggingface.co/docs/transformers): each input sentence is labeled `POSITIVE` or `NEGATIVE` with a confidence score.

## What it does

`pipeline.py` builds a `pipeline("sentiment-analysis", ...)` using `distilbert-base-uncased-finetuned-sst-2-english`, a DistilBERT model fine-tuned on the [SST-2](https://nlp.stanford.edu/sentiment/) binary sentiment task. It classifies two hard-coded example strings and prints the model output.

## Requirements

- Python 3.8+ recommended
- [PyTorch](https://pytorch.org/) (or another backend supported by Transformers for this task)

Install dependencies:

```bash
pip install transformers torch
```

## Run

From the repository root:

```bash
python pipeline.py
```

You should see a list of dictionaries, for example each item shaped like `{"label": "POSITIVE", "score": 0.99...}` (exact scores depend on the model and inputs).

## Customizing

- Change the strings passed to `classifier([...])` to try your own sentences.
- To use another model, change the `model=` argument to any compatible sentiment checkpoint on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending).

## Notes

- The first run downloads model weights (several hundred MB); later runs use the local cache.
- Pinning `model=` in code keeps behavior stable when the library’s default model for `sentiment-analysis` changes.
