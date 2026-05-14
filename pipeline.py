"""
Hugging Face Transformers pipeline examples: sentiment, zero-shot labels,
text generation, summarization, and English→French translation.

Checkpoints are pinned where supported so behavior stays stable across
Transformers releases (same rationale as the sentiment model name in code).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def extractive_question_answering(
    question: str,
    context: str,
    *,
    model_name: str = "distilbert/distilbert-base-cased-distilled-squad",
) -> dict[str, int | str | float]:
    """Span QA for plain text. ``document-question-answering`` expects LayoutLM-style models, not SQuAD BERT."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    enc = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0]
    token_type_ids = enc.get("token_type_ids")
    with torch.no_grad():
        out = model(**enc)
    start_logits = out.start_logits[0]
    end_logits = out.end_logits[0]
    if token_type_ids is not None and token_type_ids[0].sum() > 0:
        ctx = token_type_ids[0] == 1
    else:
        ctx = torch.tensor(
            [sid == 1 for sid in enc.sequence_ids(0)],
            device=start_logits.device,
        )
    mask = ~ctx
    start_logits = start_logits.masked_fill(mask, torch.finfo(start_logits.dtype).min)
    end_logits = end_logits.masked_fill(mask, torch.finfo(end_logits.dtype).min)
    start_ix = int(start_logits.argmax())
    end_slice = end_logits[start_ix:]
    end_rel = int(end_slice.argmax())
    end_ix = start_ix + end_rel
    p_start = torch.softmax(start_logits, dim=-1)
    p_end = torch.softmax(end_slice, dim=-1)
    score = float(p_start[start_ix] * p_end[end_rel])
    char_start = int(offsets[start_ix][0])
    char_end = int(offsets[end_ix][1])
    return {
        "score": score,
        "start": char_start,
        "end": char_end,
        "answer": context[char_start:char_end],
    }


def main() -> None:
    # Sentiment (binary SST-2 style).
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    sentiment_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    print("sentiment-analysis:", sentiment(sentiment_inputs))

    # Zero-shot: choose candidate labels; the model scores each without task-specific training.
    zero_shot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
    print(
        "zero-shot-classification:",
        zero_shot(
            "This is a course about the Transformers library",
            candidate_labels=["education", "politics", "business"],
        ),
    )

    text_gen = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

    print(text_gen("In this course, we will teach you how to",
    max_new_tokens=40,
    num_return_sequences=2,
    ))
    # Fill mask
    unmasker = pipeline("fill-mask")
    print(unmasker("This course will teach you all about <mask> models.", top_k=2))
    
    #name entity recognition
    ner = pipeline("ner", aggregation_strategy="simple")
    print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

    # Extractive QA on plain text (not document/layout QA).
    print(
        "question-answering",
        extractive_question_answering(
            question="Where do I work?",
            context="My name is Sylvain and I work at Hugging Face in Brooklyn",
        ),
    )
    
    

if __name__ == "__main__":
    main()
