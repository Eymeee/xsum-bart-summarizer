"""Fine-tune BART on the preprocessed XSum dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


MODEL_NAME = "facebook/bart-base"
DEFAULT_DATASET_DIR = Path("data/processed/xsum_bart_base")
DEFAULT_TOKENIZER_DIR = DEFAULT_DATASET_DIR / "tokenizer"
DEFAULT_OUTPUT_DIR = Path("outputs/bart_xsum")
DEFAULT_FINAL_MODEL_DIR = Path("models/bart_xsum_finetuned")
DEFAULT_WANDB_PROJECT = "xsum-bart-summarizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune facebook/bart-base on preprocessed XSum."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--final-model-dir", type=Path, default=DEFAULT_FINAL_MODEL_DIR
    )
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "bf16", "fp16", "none"),
        default="auto",
        help="Mixed precision mode. 'auto' prefers bf16 on supported CUDA GPUs.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to reduce VRAM usage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_tokenized_dataset(dataset_dir: Path) -> DatasetDict:
    dataset = load_from_disk(dataset_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {dataset_dir}.")
    return dataset


def load_tokenizer(tokenizer_dir: Path) -> BartTokenizer:
    return BartTokenizer.from_pretrained(tokenizer_dir)


def load_model(
    model_name: str,
    gradient_checkpointing: bool,
) -> BartForConditionalGeneration:
    model = BartForConditionalGeneration.from_pretrained(model_name)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def select_samples(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def prepare_datasets(
    dataset: DatasetDict,
    max_train_samples: int,
    max_eval_samples: int,
) -> tuple[Dataset, Dataset]:
    train_dataset = select_samples(dataset["train"], max_train_samples)
    eval_dataset = select_samples(dataset["validation"], max_eval_samples)
    return train_dataset, eval_dataset


def build_data_collator(
    tokenizer: BartTokenizer,
    model: BartForConditionalGeneration,
) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )


def as_list(value: Any) -> list[Any]:
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def replace_label_mask(labels: list[list[int]], pad_token_id: int) -> list[list[int]]:
    return [
        [pad_token_id if token_id == -100 else token_id for token_id in label]
        for label in labels
    ]


def build_compute_metrics(tokenizer: BartTokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        predictions = as_list(eval_prediction.predictions)
        labels = as_list(eval_prediction.label_ids)
        labels = replace_label_mask(labels, tokenizer.pad_token_id)

        decoded_predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )
        generated_lengths = [
            sum(token_id != tokenizer.pad_token_id for token_id in prediction)
            for prediction in predictions
        ]
        result["gen_len"] = sum(generated_lengths) / max(len(generated_lengths), 1)
        return {key: round(value, 4) for key, value in result.items()}

    return compute_metrics


def resolve_mixed_precision(mode: str) -> tuple[bool, bool]:
    if mode == "none" or not torch.cuda.is_available():
        return False, False
    if mode == "bf16":
        return False, True
    if mode == "fp16":
        return True, False
    if torch.cuda.is_bf16_supported():
        return False, True
    return True, False


def report_to_value(report_to: str) -> list[str]:
    if report_to == "none":
        return []
    return [report_to]


def step_interval(value: int, max_steps: int) -> int:
    if max_steps > 0:
        return max(1, min(value, max_steps))
    return value


def compute_warmup_steps(args: argparse.Namespace, train_size: int) -> int:
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        update_steps_per_epoch = math.ceil(
            train_size
            / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        )
        total_steps = math.ceil(update_steps_per_epoch * args.num_train_epochs)
    return max(0, round(total_steps * args.warmup_ratio))


def build_training_args(
    args: argparse.Namespace,
    train_size: int,
) -> Seq2SeqTrainingArguments:
    eval_steps = step_interval(args.eval_steps, args.max_steps)
    save_steps = step_interval(args.save_steps, args.max_steps)
    logging_steps = step_interval(args.logging_steps, args.max_steps)
    fp16, bf16 = resolve_mixed_precision(args.mixed_precision)
    warmup_steps = compute_warmup_steps(args, train_size)

    return Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.num_beams,
        fp16=fp16,
        bf16=bf16,
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=report_to_value(args.report_to),
        seed=args.seed,
    )


def build_trainer(
    model: BartForConditionalGeneration,
    training_args: Seq2SeqTrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: BartTokenizer,
) -> Seq2SeqTrainer:
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_data_collator(tokenizer, model),
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )


def device_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda_available": False}

    properties = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "cuda_available": True,
        "gpu_name": properties.name,
        "gpu_total_memory_gb": round(properties.total_memory / 1024**3, 2),
        "cuda_capability": torch.cuda.get_device_capability(0),
    }


def write_training_summary(
    output_dir: Path,
    final_model_dir: Path,
    device: dict[str, Any],
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    train_size: int,
    eval_size: int,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"
    summary = {
        "device": device,
        "train_size": train_size,
        "eval_size": eval_size,
        "effective_train_batch_size": (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        ),
        "mixed_precision": args.mixed_precision,
        "warmup_steps": compute_warmup_steps(args, train_size),
        "output_dir": str(output_dir),
        "final_model_dir": str(final_model_dir),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def run_training(args: argparse.Namespace) -> None:
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    dataset = load_tokenized_dataset(args.dataset_dir)
    train_dataset, eval_dataset = prepare_datasets(
        dataset,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = load_model(args.model_name, args.gradient_checkpointing)
    training_args = build_training_args(args, train_size=len(train_dataset))
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    device = device_summary()
    print(f"Device: {device}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    print(
        "Effective train batch size: "
        f"{args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    eval_metrics = trainer.evaluate()
    summary_path = write_training_summary(
        output_dir=args.output_dir,
        final_model_dir=args.final_model_dir,
        device=device,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
        args=args,
    )

    print(f"Checkpoints: {args.output_dir}")
    print(f"Final model: {args.final_model_dir}")
    print(f"Training summary: {summary_path}")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()