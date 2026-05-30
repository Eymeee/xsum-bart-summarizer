"""Inference helpers for the fine-tuned XSum BART model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


DEFAULT_MODEL_DIR = Path("models/bart_xsum_finetuned")
MAX_INPUT_TOKENS = 512
DEFAULT_NUM_BEAMS = 4
DEFAULT_LENGTH_PENALTY = 2.0
DEFAULT_MAX_LENGTH = 64
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model_and_tokenizer(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> tuple[BartTokenizer, BartForConditionalGeneration, torch.device]:
    model_path = Path(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def input_token_count(text: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> int:
    tokenizer, _, _ = load_model_and_tokenizer(model_dir)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncation_warning(text: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> str:
    if not text.strip():
        return ""

    token_count = input_token_count(text, model_dir)
    if token_count <= MAX_INPUT_TOKENS:
        return ""

    return (
        f"Input is about {token_count} BART tokens. The model will summarize only "
        f"the first {MAX_INPUT_TOKENS} tokens because that is the training limit."
    )


def summarize(
    text: str,
    num_beams: int = DEFAULT_NUM_BEAMS,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
    max_length: int = DEFAULT_MAX_LENGTH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> str:
    cleaned_text = text.strip()
    if not cleaned_text:
        return "Paste an article to summarize."
    if len(cleaned_text.split()) < 20:
        return "Please provide a longer news article for summarization."

    tokenizer, model, device = load_model_and_tokenizer(model_dir)
    inputs = tokenizer(
        cleaned_text,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "num_beams": int(num_beams),
        "length_penalty": float(length_penalty),
        "max_length": int(max_length),
        "no_repeat_ngram_size": DEFAULT_NO_REPEAT_NGRAM_SIZE,
        "early_stopping": True,
    }
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def summarize_with_warning(
    text: str,
    num_beams: int,
    length_penalty: float,
    max_length: int,
) -> tuple[str, str]:
    return (
        summarize(
            text,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
        ),
        truncation_warning(text),
    )
