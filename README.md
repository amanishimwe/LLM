# NLP Transformers pipelines

Examples that run several [Hugging Face Transformers](https://huggingface.co/docs/transformers) pipelines from one script: sentiment, zero-shot classification, text generation, summarization, and English→French translation. Each section prints model output for fixed demo strings.

## What it does

`pipeline.py` runs, in order:

1. **Sentiment** — `distilbert-base-uncased-finetuned-sst-2-english` (DistilBERT on [SST-2](https://nlp.stanford.edu/sentiment/)-style binary labels).
2. **Zero-shot classification** — `facebook/bart-large-mnli` with candidate labels you supply in code.
3. **Text generation** — `distilgpt2` with a short continuation budget.
4. **Summarization** — `sshleifer/distilbart-cnn-12-6` via `AutoModelForSeq2SeqLM.generate` (the built-in `pipeline("summarization", ...)` task is not available in some recent Transformers versions).
5. **Translation** — `Helsinki-NLP/opus-mt-en-fr` the same way (English to French).

Models are pinned in code so results do not drift when library defaults change.

## Requirements

- Python 3.8+ recommended
- [PyTorch](https://pytorch.org/) (or another backend supported by Transformers for these tasks)

Install dependencies:

```bash
pip install transformers torch
```

## Run

From the repository root:

```bash
python pipeline.py
```

The first run downloads each model you have not cached yet (can be several hundred MB to over 1 GB in total); later runs use the local Hugging Face cache.

## Customizing

- Edit the strings and `candidate_labels` in `main()` to try your own inputs.
- Swap any `model=` argument for another compatible checkpoint on the [Hugging Face Hub](https://huggingface.co/models).

## Notes

- Text generation passes `pad_token_id=50256` (GPT-2 family EOS) so DistilGPT-2 does not warn on open-ended generation.
- Summarization uses beam search with modest `max_length` / `min_length` so short inputs still get a usable summary.
