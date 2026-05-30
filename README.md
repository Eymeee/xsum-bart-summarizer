# XSum BART Summarizer

Fine-tuned `facebook/bart-base` on the XSum dataset for abstractive news
summarization. The project covers the full NLP workflow: EDA, preprocessing,
fine-tuning, evaluation, and a local Gradio demo.

Model Hub: <https://huggingface.co/Eymeee/xsum-bart-summarizer>

## Results

The current checkpoint is a 1-epoch fine-tune on the full XSum train split and
was evaluated on all 11,334 XSum test examples.

| Metric | Value |
| --- | ---: |
| ROUGE-1 | 0.3938 |
| ROUGE-2 | 0.1696 |
| ROUGE-L | 0.3197 |
| ROUGE-Lsum | 0.3196 |
| BERTScore F1 mean | 0.9066 |

See [evaluation_report.md](evaluation_report.md) for details, qualitative
examples, and limitations.

## Setup

```bash
uv sync
```

The project uses Python 3.12 and dependencies declared in `pyproject.toml`.

## Pipeline Commands

Run exploratory data analysis:

```bash
uv run python -m src.eda
```

Preprocess XSum for BART:

```bash
uv run python -m src.preprocess
```

Fine-tune the model:

```bash
uv run python -m src.train --num-train-epochs 1 --eval-steps 5000 --save-steps 5000
```

Evaluate the fine-tuned checkpoint:

```bash
uv run python -m src.evaluate_model
```

Upload the trained checkpoint and model card to Hugging Face Hub:

```bash
uv run python scripts/upload_model.py
```

Run the local Gradio app:

```bash
uv run python app.py
```

Then open <http://127.0.0.1:7860>.

## Inference

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

repo_id = "Eymeee/xsum-bart-summarizer"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

text = "The government announced a new transport plan after months of consultation with local councils and passenger groups."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
output = model.generate(**inputs, num_beams=4, max_length=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Repository Layout

- `src/eda.py`: dataset exploration and plots
- `src/preprocess.py`: tokenization and verification report
- `src/train.py`: BART fine-tuning with `Seq2SeqTrainer`
- `src/evaluate_model.py`: full test-set ROUGE and BERTScore evaluation
- `src/inference.py`: reusable inference helpers
- `app.py`: local Gradio summarization app
- `scripts/upload_model.py`: Hugging Face Hub model publishing helper
- `deploy/model/README.md`: Hugging Face model card
- `context/AGENTS.md`: project phase checklist

## Generated Artifacts

The following directories are intentionally excluded from Git:

- `data/`
- `models/`
- `outputs/`
- `wandb/`
- `runs/`

The fine-tuned checkpoint is published on Hugging Face Hub instead of being
stored in the GitHub repository.

## Limitations

- Inputs longer than 512 BART tokens are truncated.
- The current model is a 1-epoch checkpoint.
- Generated summaries can contain factual errors, entity mix-ups, or
  hallucinated details.
- The model is tuned for XSum/BBC-style news and may not generalize well to
  other domains.
