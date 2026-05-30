"""Evaluate a fine-tuned BART checkpoint on XSum."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import evaluate
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from bert_score import score as bert_score
from datasets import Dataset, DatasetDict, load_from_disk
from rouge_score import rouge_scorer
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.train import ( # pyrefly: ignore
    as_list,
    device_summary,
    resolve_mixed_precision,
    sanitize_token_ids,
    select_samples,
)


DEFAULT_MODEL_DIR = Path("models/bart_xsum_finetuned")
DEFAULT_DATASET_DIR = Path("data/processed/xsum_bart_base")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation/bart_xsum_finetuned")
DEFAULT_REPORT_PATH = Path("evaluation_report.md")
BASELINE_SOURCE_URL = (
    "https://paperswithcode.com/paper/"
    "bart-denoising-sequence-to-sequence-pre"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned BART checkpoint on XSum."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=64)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--bertscore-model", default="roberta-base")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "bf16", "fp16", "none"),
        default="auto",
        help="Mixed precision mode. 'auto' prefers bf16 on supported CUDA GPUs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_tokenized_dataset(dataset_dir: Path) -> DatasetDict:
    dataset = load_from_disk(dataset_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {dataset_dir}.")
    return dataset


def load_split(dataset_dir: Path, split: str, max_test_samples: int) -> Dataset:
    dataset = load_tokenized_dataset(dataset_dir)
    if split not in dataset:
        raise KeyError(f"Split '{split}' not found in {dataset_dir}.")
    return select_samples(dataset[split], max_test_samples)


def build_eval_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    fp16, bf16 = resolve_mixed_precision(args.mixed_precision)
    return Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.num_beams,
        fp16=fp16,
        bf16=bf16,
        tf32=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )


def build_trainer(
    args: argparse.Namespace,
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
) -> Seq2SeqTrainer:
    return Seq2SeqTrainer(
        model=model,
        args=build_eval_args(args),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        ),
        processing_class=tokenizer,
    )


def decode_token_ids(
    token_ids: Any,
    tokenizer: BartTokenizer,
) -> list[str]:
    token_id_list = sanitize_token_ids(
        as_list(token_ids),
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
    )
    return [
        text.strip()
        for text in tokenizer.batch_decode(token_id_list, skip_special_tokens=True)
    ]


def decoded_inputs(dataset: Dataset, tokenizer: BartTokenizer) -> list[str]:
    return [
        tokenizer.decode(input_ids, skip_special_tokens=True).strip()
        for input_ids in dataset["input_ids"]
    ]


def token_lengths(texts: list[str], tokenizer: BartTokenizer) -> list[int]:
    return [
        len(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in texts
    ]


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    rouge = evaluate.load("rouge")
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {key: round(float(value), 4) for key, value in result.items()}


def compute_per_example_rouge_l(
    predictions: list[str],
    references: list[str],
) -> list[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [
        round(
            scorer.score(reference, prediction)["rougeL"].fmeasure,
            4,
        )
        for prediction, reference in zip(predictions, references, strict=True)
    ]


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": round(float(values.mean().item()), 4),
        "std": round(float(values.std(unbiased=False).item()), 4),
    }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision, recall, f1 = bert_score(
        predictions,
        references,
        model_type=args.bertscore_model,
        batch_size=args.bertscore_batch_size,
        device=device,
        lang="en",
        verbose=True,
    )
    return {
        "precision": tensor_stats(precision),
        "recall": tensor_stats(recall),
        "f1": tensor_stats(f1),
    }


def write_predictions(
    path: Path,
    articles: list[str],
    references: list[str],
    predictions: list[str],
    rouge_l_scores: list[float],
    reference_lengths: list[int],
    generated_lengths: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index, (
            article,
            reference,
            prediction,
            rouge_l,
            reference_length,
            generated_length,
        ) in enumerate(
            zip(
                articles,
                references,
                predictions,
                rouge_l_scores,
                reference_lengths,
                generated_lengths,
                strict=True,
            )
        ):
            record = {
                "index": index,
                "decoded_article": article,
                "article_excerpt": article[:750],
                "reference_summary": reference,
                "generated_summary": prediction,
                "rougeL": rouge_l,
                "reference_token_length": reference_length,
                "generated_token_length": generated_length,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_length_plot(
    path: Path,
    reference_lengths: list[int],
    generated_lengths: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    bins = range(0, max(reference_lengths + generated_lengths + [1]) + 5, 5)
    plt.hist(reference_lengths, bins=bins, alpha=0.65, label="Reference")
    plt.hist(generated_lengths, bins=bins, alpha=0.65, label="Generated")
    plt.xlabel("BART token length")
    plt.ylabel("Count")
    plt.title("Reference vs. Generated Summary Lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def qualitative_examples(
    articles: list[str],
    references: list[str],
    predictions: list[str],
    rouge_l_scores: list[float],
    count: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        range(len(rouge_l_scores)),
        key=lambda index: rouge_l_scores[index],
    )
    weakest_indices = ranked[:count]
    strongest_indices = list(reversed(ranked[-count:]))

    def build_examples(indices: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "article_excerpt": articles[index][:750],
                "reference_summary": references[index],
                "generated_summary": predictions[index],
                "rougeL": rouge_l_scores[index],
            }
            for index in indices
        ]

    return build_examples(strongest_indices), build_examples(weakest_indices)


def observations(metrics: dict[str, Any]) -> list[str]:
    rouge_l = metrics["rouge"]["rougeL"]
    generated_mean = metrics["summary_lengths"]["generated"]["mean"]
    reference_mean = metrics["summary_lengths"]["reference"]["mean"]
    notes = [
        "The model usually follows the XSum one-sentence summary format.",
        "The 1-epoch BART-base checkpoint trails the published BART-large reference baseline, which is expected.",
    ]
    if generated_mean > reference_mean + 5:
        notes.append("Generated summaries are noticeably longer than references on average.")
    elif generated_mean < reference_mean - 5:
        notes.append("Generated summaries are noticeably shorter than references on average.")
    else:
        notes.append("Generated summary length is close to the reference length distribution.")
    if rouge_l < 0.3:
        notes.append("Weak examples tend to lose important article-specific details.")
    return notes


def mean(values: list[int]) -> float:
    return round(sum(values) / max(len(values), 1), 2)


def build_metrics(
    args: argparse.Namespace,
    device: dict[str, Any],
    test_size: int,
    rouge: dict[str, float],
    bertscore: dict[str, dict[str, float]],
    reference_lengths: list[int],
    generated_lengths: list[int],
) -> dict[str, Any]:
    return {
        "model_dir": str(args.model_dir),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "test_size": test_size,
        "device": device,
        "generation": {
            "num_beams": args.num_beams,
            "max_length": args.generation_max_length,
        },
        "rouge": rouge,
        "bertscore": bertscore,
        "summary_lengths": {
            "reference": {"mean": mean(reference_lengths)},
            "generated": {"mean": mean(generated_lengths)},
        },
        "baseline_reference": {
            "model": "BART-large on XSum",
            "rouge1": 45.14,
            "rouge2": 22.27,
            "rougeL": 37.25,
            "source": BASELINE_SOURCE_URL,
        },
    }


def metric_table(metrics: dict[str, Any]) -> str:
    rows = [
        ("ROUGE-1", metrics["rouge"]["rouge1"]),
        ("ROUGE-2", metrics["rouge"]["rouge2"]),
        ("ROUGE-L", metrics["rouge"]["rougeL"]),
        ("ROUGE-Lsum", metrics["rouge"]["rougeLsum"]),
        ("BERTScore precision mean", metrics["bertscore"]["precision"]["mean"]),
        ("BERTScore precision std", metrics["bertscore"]["precision"]["std"]),
        ("BERTScore recall mean", metrics["bertscore"]["recall"]["mean"]),
        ("BERTScore recall std", metrics["bertscore"]["recall"]["std"]),
        ("BERTScore F1 mean", metrics["bertscore"]["f1"]["mean"]),
        ("BERTScore F1 std", metrics["bertscore"]["f1"]["std"]),
        (
            "Reference summary length mean",
            metrics["summary_lengths"]["reference"]["mean"],
        ),
        (
            "Generated summary length mean",
            metrics["summary_lengths"]["generated"]["mean"],
        ),
    ]
    table = ["| Metric | Value |", "| --- | ---: |"]
    table.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(table)


def examples_section(title: str, examples: list[dict[str, Any]]) -> str:
    lines = [f"## {title}"]
    for example in examples:
        lines.extend(
            [
                f"### Example {example['index']} - ROUGE-L {example['rougeL']}",
                f"Article excerpt: {example['article_excerpt']}",
                "",
                f"Reference: {example['reference_summary']}",
                "",
                f"Generated: {example['generated_summary']}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def write_report(
    path: Path,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    strongest: list[dict[str, Any]],
    weakest: list[dict[str, Any]],
    length_plot_path: Path,
) -> None:
    notes = observations(metrics)
    report = f"""# XSum BART Evaluation Report

## Context
- Model checkpoint: `{args.model_dir}`
- Dataset split: `{args.split}`
- Test examples: {metrics["test_size"]}
- Device: `{metrics["device"]["device"]}`
- Generation: {args.num_beams} beams, max length {args.generation_max_length}
- Checkpoint note: this is the 1-epoch fine-tuned `facebook/bart-base` model.
- Article excerpt note: decoded article excerpts may already be truncated to 512 BART tokens by design from preprocessing.
- Summary length plot: `{length_plot_path}`

## Metrics
ROUGE is computed with `use_stemmer=True`.

{metric_table(metrics)}

## Baseline Comparison
Published BART-large XSum reference scores are ROUGE-1 0.4514, ROUGE-2 0.2227, and ROUGE-L 0.3725.
This is a reference point, not an exact apples-to-apples comparison, because this project uses `facebook/bart-base` and the current checkpoint was fine-tuned for 1 epoch.

Source: {BASELINE_SOURCE_URL}

## Observations
{chr(10).join(f"- {note}" for note in notes)}

{examples_section("Strongest Examples", strongest)}

{examples_section("Weakest Examples", weakest)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_split(args.dataset_dir, args.split, args.max_test_samples)
    tokenizer = BartTokenizer.from_pretrained(args.model_dir)
    model = BartForConditionalGeneration.from_pretrained(args.model_dir)
    trainer = build_trainer(args, model, tokenizer)

    device = device_summary()
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Test examples: {len(dataset)}")
    print(f"Model: {args.model_dir}")

    prediction_output = trainer.predict(dataset)
    predictions = decode_token_ids(prediction_output.predictions, tokenizer)
    references = decode_token_ids(prediction_output.label_ids, tokenizer)
    articles = decoded_inputs(dataset, tokenizer)

    rouge = compute_rouge(predictions, references)
    rouge_l_scores = compute_per_example_rouge_l(predictions, references)
    bertscore = compute_bertscore(predictions, references, args)
    reference_lengths = token_lengths(references, tokenizer)
    generated_lengths = token_lengths(predictions, tokenizer)

    metrics = build_metrics(
        args=args,
        device=device,
        test_size=len(dataset),
        rouge=rouge,
        bertscore=bertscore,
        reference_lengths=reference_lengths,
        generated_lengths=generated_lengths,
    )
    strongest, weakest = qualitative_examples(
        articles,
        references,
        predictions,
        rouge_l_scores,
    )

    predictions_path = args.output_dir / "test_predictions.jsonl"
    metrics_path = args.output_dir / "evaluation_metrics.json"
    length_plot_path = args.output_dir / "summary_length_distribution.png"

    write_predictions(
        predictions_path,
        articles,
        references,
        predictions,
        rouge_l_scores,
        reference_lengths,
        generated_lengths,
    )
    write_metrics(metrics_path, metrics)
    save_length_plot(length_plot_path, reference_lengths, generated_lengths)
    write_report(args.report_path, args, metrics, strongest, weakest, length_plot_path)

    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Length plot: {length_plot_path}")
    print(f"Report: {args.report_path}")


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
