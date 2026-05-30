# AGENTS.md — Abstractive Text Summarization

## 📌 Project Brief

This project fine-tunes `facebook/bart-base` on the **XSum dataset** to perform **abstractive text summarization** of BBC news articles. Unlike extractive methods, the model generates novel, concise single-sentence summaries in its own words. The final deliverable is a full **Gradio web app** with the fine-tuned model published on **HuggingFace Hub** and the codebase on **GitHub**.

**Primary goal:** Demonstrate an end-to-end NLP fine-tuning pipeline, from raw data to a live interactive demo, as a portfolio project.

---

## ✅ Implementation Checklist

### Phase 1 — Environment Setup
- [x] Install `uv` if not already available (`pip install uv` or via the [official installer](https://docs.astral.sh/uv/getting-started/installation/))
- [x] Initialize the project with `uv init` and create a virtual environment with `uv venv`
- [x] Activate the environment (`.venv/bin/activate` on Unix / `.venv\Scripts\activate` on Windows)
- [x] Install core dependencies with `uv add`:
  - `torch`, `transformers`, `datasets`, `evaluate`
  - `accelerate`, `sentencepiece`, `rouge_score`
  - `bert_score`, `wandb`, `gradio`
- [x] Verify dependencies are locked in `uv.lock` and synced via `uv sync`
- [x] Log in to HuggingFace Hub (`huggingface-cli login`)
- [x] Log in to Weights & Biases (`wandb login`)
- [x] Initialize project repository with `.gitignore`, `pyproject.toml`, and folder structure

### Phase 2 — Data Exploration (EDA)
- [x] Load XSum dataset using `datasets.load_dataset("EdinburghNLP/xsum")`
- [x] Inspect dataset splits: train / validation / test sizes
- [x] Plot distribution of article lengths (token count)
- [x] Plot distribution of summary lengths (token count)
- [x] Compute and report the abstractiveness ratio (n-gram novelty of summaries vs. articles)
- [x] Display a few samples to understand data format (`document`, `summary`, `id`)

### Phase 3 — Preprocessing
- [x] Load `BartTokenizer` from `facebook/bart-base`
- [x] Write a `preprocess_function` that tokenizes both `document` (input) and `summary` (label)
- [x] Set `max_input_length = 512` and `max_target_length = 64`
- [x] Apply `dataset.map(preprocess_function, batched=True)` on all splits
- [x] Set up `DataCollatorForSeq2Seq` with dynamic padding
- [x] Verify tokenized samples and decoded outputs look correct

### Phase 4 — Fine-Tuning
- [x] Load `BartForConditionalGeneration` from `facebook/bart-base`
- [x] Define `compute_metrics` function using ROUGE-1, ROUGE-2, ROUGE-L via `evaluate.load("rouge")`
- [x] Configure `Seq2SeqTrainingArguments`:
  - `output_dir`, `num_train_epochs` (3)
  - `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` for RTX 5050 GPU
  - `learning_rate` (5e-5), `warmup_ratio`, `weight_decay`
  - `predict_with_generate=True`
  - `report_to="wandb"` for full training
- [x] Instantiate `Seq2SeqTrainer` with model, args, datasets, tokenizer, collator, and compute_metrics
- [x] Run full training with `trainer.train()` for 1 epoch on the full train split
- [x] Save full-training checkpoint with `trainer.save_model()` to `models/bart_xsum_finetuned/`

### Phase 5 — Evaluation
- [ ] Run `trainer.evaluate()` on the test set
- [ ] Report **ROUGE-1**, **ROUGE-2**, **ROUGE-L** scores
- [ ] Compute **BERTScore** (Precision, Recall, F1) using `bert_score`
- [ ] Compare results against published XSum baselines
- [ ] Manually inspect at least 10 generated summaries (good cases + failure cases)
- [ ] Document findings in an `evaluation_report.md`

### Phase 6 — Gradio App
- [ ] Write an `inference.py` module:
  - Load tokenizer and fine-tuned model
  - Define `summarize(text)` function using `model.generate()`
  - Tune generation params: `num_beams=4`, `length_penalty=2.0`, `max_length=128`
- [ ] Build `app.py` with Gradio:
  - `gr.Interface` with a text input and summary output
  - Add 3–5 pre-loaded XSum examples
  - Add a short description and project title in the UI
- [ ] Test the app locally with `python app.py`

### Phase 7 — Publishing
- [ ] Push fine-tuned model and tokenizer to HuggingFace Hub (`model.push_to_hub(...)`)
- [ ] Write a `README.md` Model Card including:
  - Model description and base model
  - Dataset and training details
  - ROUGE / BERTScore results
  - How to run inference
  - Limitations and future work
- [ ] Deploy Gradio app to HuggingFace Spaces
- [ ] Push full project to GitHub with a clean `README.md`

---

## ⏭️ Next Step

**Continue with Phase 5 — Evaluation.**

Run test-set evaluation for the fine-tuned `facebook/bart-base` checkpoint saved at `models/bart_xsum_finetuned/`. The completed Phase 4 run used the full train split for 1 epoch and wrote training metrics to `outputs/bart_xsum/training_summary.json`.

The tokenized dataset and tokenizer are generated locally under `data/processed/xsum_bart_base/`.
