---
license: apache-2.0
base_model: facebook/bart-base
datasets:
- EdinburghNLP/xsum
language:
- en
pipeline_tag: summarization
tags:
- summarization
- bart
- xsum
- pytorch
- transformers
metrics:
- rouge
- bertscore
---

# XSum BART Summarizer

This model is `facebook/bart-base` fine-tuned on the XSum dataset for abstractive
single-sentence news summarization. It is intended to summarize BBC-style news
articles into concise summaries.

## Model Details

- Base model: `facebook/bart-base`
- Dataset: `EdinburghNLP/xsum`
- Task: abstractive summarization
- Language: English
- Fine-tuning run: 1 epoch on the full XSum train split
- Max source length: 512 BART tokens
- Max target/generation length used for evaluation: 64 BART tokens

## Evaluation

The checkpoint was evaluated on all 11,334 XSum test examples.

| Metric | Value |
| --- | ---: |
| ROUGE-1 | 0.3938 |
| ROUGE-2 | 0.1696 |
| ROUGE-L | 0.3197 |
| ROUGE-Lsum | 0.3196 |
| BERTScore precision mean | 0.9136 |
| BERTScore recall mean | 0.9000 |
| BERTScore F1 mean | 0.9066 |

ROUGE was computed with stemming enabled. BERTScore was computed with
`roberta-base`.

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

repo_id = "Eymeee/xsum-bart-summarizer"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

article = """
The government announced a new transport plan after months of consultation with
local councils and passenger groups. Ministers said the proposal would improve
bus and rail services, reduce delays, and give local authorities more control
over routes and fares.
"""

inputs = tokenizer(
    article,
    return_tensors="pt",
    max_length=512,
    truncation=True,
)
output_ids = model.generate(
    **inputs,
    num_beams=4,
    length_penalty=2.0,
    max_length=64,
    no_repeat_ngram_size=3,
    early_stopping=True,
)
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(summary)
```

## Limitations

- Inputs longer than 512 BART tokens are truncated.
- The current checkpoint was fine-tuned for 1 epoch; stronger quality would
  likely require additional epochs and checkpoint selection.
- Generated summaries can contain factual errors, entity mix-ups, or
  hallucinated details.
- The model is tuned on XSum/BBC-style news and may generalize poorly to other
  domains.
- Generated summaries should not be treated as verified facts.

## Training and Evaluation Context

This model is part of an end-to-end portfolio project covering dataset
exploration, preprocessing, fine-tuning, evaluation, and a local Gradio demo.
See the GitHub repository for the full code and evaluation report.
