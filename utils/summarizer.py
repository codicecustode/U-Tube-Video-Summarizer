from transformers import pipeline
import torch
from functools import lru_cache


@lru_cache()
def get_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def summarize_text(text: str, max_length: int = 150, min_length: int = 30) -> str:
    split_text = []
    i = 0
    max_truncate_length = 800  # characters, not tokens

    # Split the text into smaller parts
    while len(text[i:]) > 1000:
        end_idx = text[i:i+max_truncate_length].rfind('.')
        if end_idx == -1:
            end_idx = max_truncate_length
        split_text.append(text[i:i+end_idx+1].strip())
        i += end_idx + 1

    # Append remaining text
    if i < len(text):
        split_text.append(text[i:].strip())

    summarized_text = []

    print(f"Total chunks: {len(split_text)}")
    """for idx, chunk in enumerate(split_text):
        print(f"Summarizing chunk {idx+1}/{len(split_text)}")
        if len(chunk.split()) < 50:  # If chunk is too small, skip summarizing
            summarized_text.append(chunk)
        else:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summarized_text.append(summary[0]['summary_text'])
"""
    #return ' '.join(summarized_text)
    summarizer = get_summarizer()
    return summarizer(split_text[0], max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
