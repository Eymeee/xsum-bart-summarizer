"""Exploratory data analysis for the XSum summarization dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "facebook/bart-base"
DATASET_NAME = "EdinburghNLP/xsum"
DEFAULT_OUTPUT_DIR = Path("data/eda")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA for XSum and write local plots/reports."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated EDA artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of train examples used for length and novelty statistics. "
        "Use 0 or a negative value for the full train split.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of representative examples to save.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling.",
    )
    return parser.parse_args()


def load_xsum() -> DatasetDict:
    dataset = load_dataset(DATASET_NAME)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected XSum to load as a DatasetDict.")
    return dataset


def select_sample(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    if sample_size <= 0 or sample_size >= len(dataset):
        return dataset

    return dataset.shuffle(seed=seed).select(range(sample_size))


def token_lengths(
    texts: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 128,
) -> list[int]:
    text_list = list(texts)
    lengths: list[int] = []

    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(input_ids) for input_ids in tokenized["input_ids"])

    return lengths


def describe_lengths(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        raise ValueError("Cannot describe an empty length list.")

    sorted_lengths = sorted(lengths)
    return {
        "count": len(lengths),
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(mean(sorted_lengths), 2),
        "median": round(median(sorted_lengths), 2),
        "p90": percentile(sorted_lengths, 0.90),
        "p95": percentile(sorted_lengths, 0.95),
        "p99": percentile(sorted_lengths, 0.99),
    }


def percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        raise ValueError("Cannot compute percentile for an empty list.")

    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


def percentage_above(values: list[int], threshold: int) -> float:
    if not values:
        return 0.0

    return round(sum(value > threshold for value in values) * 100 / len(values), 2)


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()

    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def novelty_ratio(document: str, summary: str, n: int) -> float | None:
    document_ngrams = ngrams(tokenize_words(document), n)
    summary_ngrams = ngrams(tokenize_words(summary), n)

    if not summary_ngrams:
        return None

    novel_count = len(summary_ngrams - document_ngrams)
    return novel_count / len(summary_ngrams)


def compute_novelty(records: Dataset) -> dict[str, float]:
    novelty_scores: dict[str, list[float]] = {"1gram": [], "2gram": [], "3gram": []}

    for record in records:
        document = str(record["document"])
        summary = str(record["summary"])
        for n in (1, 2, 3):
            score = novelty_ratio(document, summary, n)
            if score is not None:
                novelty_scores[f"{n}gram"].append(score)

    return {
        key: round(mean(scores) * 100, 2) if scores else 0.0
        for key, scores in novelty_scores.items()
    }


def save_histogram(
    values: list[int],
    title: str,
    xlabel: str,
    output_path: Path,
    threshold: int | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, color="#2563eb", edgecolor="white")
    if threshold is not None:
        plt.axvline(
            threshold,
            color="#dc2626",
            linestyle="--",
            linewidth=2,
            label=f"Phase 3 max: {threshold}",
        )
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Example count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def sample_records(dataset: Dataset, num_examples: int, seed: int) -> list[dict[str, str]]:
    if num_examples <= 0:
        return []

    sample = select_sample(dataset, min(num_examples, len(dataset)), seed)
    return [
        {
            "id": str(record["id"]),
            "document": str(record["document"]),
            "summary": str(record["summary"]),
        }
        for record in sample
    ]


def truncate(text: str, limit: int = 500) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed

    return f"{collapsed[:limit].rstrip()}..."


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def build_report(
    split_sizes: dict[str, int],
    sample_size: int,
    article_stats: dict[str, float | int],
    summary_stats: dict[str, float | int],
    article_over_512: float,
    summary_over_128: float,
    novelty: dict[str, float],
    examples: list[dict[str, str]],
) -> str:
    split_rows = [[split, size] for split, size in split_sizes.items()]
    stats_headers = ["metric", "article tokens", "summary tokens"]
    stats_rows = [
        [metric, article_stats[metric], summary_stats[metric]]
        for metric in ("count", "min", "max", "mean", "median", "p90", "p95", "p99")
    ]
    novelty_rows = [[key, f"{value}%"] for key, value in novelty.items()]
    example_sections = []

    for index, example in enumerate(examples, start=1):
        example_sections.append(
            "\n".join(
                [
                    f"### Example {index}",
                    f"- ID: `{example['id']}`",
                    f"- Summary: {example['summary']}",
                    f"- Document excerpt: {truncate(example['document'])}",
                ]
            )
        )

    return "\n\n".join(
        [
            "# XSum EDA Report",
            f"Dataset: `{DATASET_NAME}`",
            f"Model tokenizer: `{MODEL_NAME}`",
            f"Statistics sample size: `{sample_size}` train examples",
            "## Dataset Splits",
            markdown_table(["split", "examples"], split_rows),
            "## Token Length Statistics",
            markdown_table(stats_headers, stats_rows),
            "## Phase 3 Length Threshold Checks",
            markdown_table(
                ["threshold", "percentage above threshold"],
                [
                    ["articles > 512 tokens", f"{article_over_512}%"],
                    ["summaries > 128 tokens", f"{summary_over_128}%"],
                ],
            ),
            "## Summary Abstractiveness",
            "Novelty is the mean percentage of unique summary n-grams not present in the source document.",
            markdown_table(["n-gram", "mean novelty"], novelty_rows),
            "## Sample Records",
            "\n\n".join(example_sections) if example_sections else "No examples requested.",
            "",
        ]
    )


def run_eda(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_xsum()
    train_sample = select_sample(dataset["train"], args.sample_size, args.seed)
    effective_sample_size = len(train_sample)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 10**12

    article_lengths = token_lengths(train_sample["document"], tokenizer)
    summary_lengths = token_lengths(train_sample["summary"], tokenizer)

    article_stats = describe_lengths(article_lengths)
    summary_stats = describe_lengths(summary_lengths)
    novelty = compute_novelty(train_sample)
    examples = sample_records(dataset["train"], args.num_examples, args.seed)
    split_sizes = {split: len(split_dataset) for split, split_dataset in dataset.items()}

    article_plot_path = args.output_dir / "article_length_distribution.png"
    summary_plot_path = args.output_dir / "summary_length_distribution.png"
    report_path = args.output_dir / "xsum_eda_report.md"
    samples_path = args.output_dir / "sample_records.json"

    save_histogram(
        article_lengths,
        "XSum Article Token Lengths",
        "BART token count",
        article_plot_path,
        threshold=512,
    )
    save_histogram(
        summary_lengths,
        "XSum Summary Token Lengths",
        "BART token count",
        summary_plot_path,
        threshold=128,
    )

    report = build_report(
        split_sizes=split_sizes,
        sample_size=effective_sample_size,
        article_stats=article_stats,
        summary_stats=summary_stats,
        article_over_512=percentage_above(article_lengths, 512),
        summary_over_128=percentage_above(summary_lengths, 128),
        novelty=novelty,
        examples=examples,
    )
    report_path.write_text(report, encoding="utf-8")
    samples_path.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("XSum EDA complete")
    print(f"Split sizes: {split_sizes}")
    print(f"Sample size: {effective_sample_size}")
    print(f"Article plot: {article_plot_path}")
    print(f"Summary plot: {summary_plot_path}")
    print(f"Report: {report_path}")
    print(f"Samples: {samples_path}")


def main() -> None:
    run_eda(parse_args())


if __name__ == "__main__":
    main()
