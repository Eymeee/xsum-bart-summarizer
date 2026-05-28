"""Preprocess XSum for BART sequence-to-sequence fine-tuning."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import BartTokenizer, DataCollatorForSeq2Seq


DATASET_NAME = "EdinburghNLP/xsum"
MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
DEFAULT_OUTPUT_DIR = Path("data/processed/xsum_bart_base")
DEFAULT_VERIFICATION_REPORT = Path("data/processed/preprocess_verification.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize XSum documents and summaries for BART fine-tuning."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the tokenized DatasetDict and tokenizer are saved.",
    )
    parser.add_argument(
        "--verification-report",
        type=Path,
        default=DEFAULT_VERIFICATION_REPORT,
        help="Markdown report validating tokenization and dynamic padding.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=MAX_INPUT_LENGTH,
        help="Maximum BART token length for source documents.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=MAX_TARGET_LENGTH,
        help="Maximum BART token length for summaries.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Per-split sample size for smoke tests. Use 0 for full splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sample subsets.",
    )
    return parser.parse_args()


def load_xsum() -> DatasetDict:
    dataset = load_dataset(DATASET_NAME)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected XSum to load as a DatasetDict.")
    return dataset


def load_tokenizer(model_name: str = MODEL_NAME) -> BartTokenizer:
    tokenizer = BartTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 10**12
    return tokenizer


def select_sample(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    if sample_size <= 0 or sample_size >= len(dataset):
        return dataset

    return dataset.shuffle(seed=seed).select(range(sample_size))


def maybe_sample_dataset(dataset: DatasetDict, sample_size: int, seed: int) -> DatasetDict:
    if sample_size <= 0:
        return dataset

    return DatasetDict(
        {
            split: select_sample(split_dataset, sample_size, seed)
            for split, split_dataset in dataset.items()
        }
    )


def preprocess_function(
    examples: dict[str, list[str]],
    tokenizer: BartTokenizer,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
) -> dict[str, list[list[int]]]:
    tokenized = tokenizer(
        examples["document"],
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=max_target_length,
        truncation=True,
    )
    tokenized["labels"] = labels["input_ids"]
    return tokenized


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: BartTokenizer,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
) -> DatasetDict:
    columns_to_remove = dataset["train"].column_names

    return dataset.map(
        lambda examples: preprocess_function(
            examples,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        ),
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing XSum",
    )


def build_data_collator(tokenizer: BartTokenizer) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)


def token_lengths(
    texts: Iterable[str],
    tokenizer: BartTokenizer,
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


def truncation_coverage_by_split(
    dataset: DatasetDict,
    tokenizer: BartTokenizer,
    max_input_length: int,
) -> dict[str, float]:
    coverage: dict[str, float] = {}

    for split, split_dataset in dataset.items():
        lengths = token_lengths(split_dataset["document"], tokenizer)
        truncated_count = sum(length > max_input_length for length in lengths)
        coverage[split] = round(truncated_count * 100 / len(lengths), 2)

    return coverage


def decode_labels(labels: list[int], tokenizer: BartTokenizer) -> str:
    label_ids = [
        tokenizer.pad_token_id if label_id == -100 else label_id for label_id in labels
    ]
    return tokenizer.decode(label_ids, skip_special_tokens=True)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def truncate_text(text: str, limit: int = 900) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit].rstrip()}..."


def analyze_collated_batch(
    tokenized_dataset: DatasetDict,
    tokenizer: BartTokenizer,
) -> dict[str, Any]:
    collator = build_data_collator(tokenizer)
    batch_examples = select_padding_check_examples(tokenized_dataset["train"])
    batch = collator(batch_examples)

    labels = batch["labels"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    has_masked_labels = bool((labels == -100).any().item())

    return {
        "input_shape": tuple(input_ids.shape),
        "attention_mask_shape": tuple(attention_mask.shape),
        "labels_shape": tuple(labels.shape),
        "has_masked_labels": has_masked_labels,
        "decoded_input": tokenizer.decode(
            tokenized_dataset["train"][0]["input_ids"], skip_special_tokens=True
        ),
        "decoded_label": decode_labels(
            tokenized_dataset["train"][0]["labels"], tokenizer
        ),
    }


def select_padding_check_examples(dataset: Dataset) -> list[dict[str, list[int]]]:
    if len(dataset) < 2:
        raise ValueError("Need at least two examples to verify dynamic padding.")

    first = dataset[0]
    first_label_length = len(first["labels"])

    for index in range(1, len(dataset)):
        candidate = dataset[index]
        if len(candidate["labels"]) != first_label_length:
            return [first, candidate]

    return [first, dataset[1]]


def build_verification_report(
    tokenized_dataset: DatasetDict,
    tokenizer: BartTokenizer,
    output_dir: Path,
    tokenizer_dir: Path,
    max_input_length: int,
    max_target_length: int,
    truncation_coverage: dict[str, float],
) -> str:
    split_rows = [
        [split, len(split_dataset)] for split, split_dataset in tokenized_dataset.items()
    ]
    feature_names = tokenized_dataset["train"].column_names
    removed_columns = all(
        column not in feature_names for column in ("document", "summary", "id")
    )
    batch_info = analyze_collated_batch(tokenized_dataset, tokenizer)
    coverage_rows = [
        [split, f"{coverage}%"] for split, coverage in truncation_coverage.items()
    ]

    return "\n\n".join(
        [
            "# Preprocessing Verification Report",
            f"Dataset: `{DATASET_NAME}`",
            f"Tokenizer: `{MODEL_NAME}`",
            f"Max input length: `{max_input_length}`",
            f"Max target length: `{max_target_length}`",
            f"Saved tokenized dataset: `{output_dir}`",
            f"Saved tokenizer: `{tokenizer_dir}`",
            "## Tokenized Split Sizes",
            markdown_table(["split", "examples"], split_rows),
            "## Tokenized Features",
            f"Features: `{', '.join(feature_names)}`",
            (
                "Original columns removed: `yes`"
                if removed_columns
                else "Original columns removed: `no`"
            ),
            "## Decoded Sample",
            f"Input article: {truncate_text(batch_info['decoded_input'])}",
            f"Target summary: {batch_info['decoded_label']}",
            "## Dynamic Padding Check",
            markdown_table(
                ["check", "value"],
                [
                    ["input_ids tensor shape", batch_info["input_shape"]],
                    ["attention_mask tensor shape", batch_info["attention_mask_shape"]],
                    ["labels tensor shape", batch_info["labels_shape"]],
                    ["labels contain -100 mask", batch_info["has_masked_labels"]],
                ],
            ),
            "## Article Truncation Coverage",
            markdown_table(["split", f"articles > {max_input_length} tokens"], coverage_rows),
            "",
        ]
    )


def save_tokenized_outputs(
    tokenized_dataset: DatasetDict,
    tokenizer: BartTokenizer,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    tokenized_dataset.save_to_disk(output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer_dir


def run_preprocessing(args: argparse.Namespace) -> None:
    dataset = maybe_sample_dataset(load_xsum(), args.sample_size, args.seed)
    tokenizer = load_tokenizer()
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    truncation_coverage = truncation_coverage_by_split(
        dataset,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
    )
    tokenizer_dir = save_tokenized_outputs(tokenized_dataset, tokenizer, args.output_dir)

    args.verification_report.parent.mkdir(parents=True, exist_ok=True)
    args.verification_report.write_text(
        build_verification_report(
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            tokenizer_dir=tokenizer_dir,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            truncation_coverage=truncation_coverage,
        ),
        encoding="utf-8",
    )

    split_sizes = {
        split: len(split_dataset) for split, split_dataset in tokenized_dataset.items()
    }
    print("XSum preprocessing complete")
    print(f"Split sizes: {split_sizes}")
    print(f"Tokenized dataset: {args.output_dir}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"Verification report: {args.verification_report}")


def main() -> None:
    run_preprocessing(parse_args())


if __name__ == "__main__":
    main()
