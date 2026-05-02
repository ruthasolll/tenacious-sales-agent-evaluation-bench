"""DPO fine-tuning for Tenacious-Bench preference data with Unsloth.

Run from the repository root:
    python training/train_dpo.py

This script is sized for a Google Colab Tesla T4 GPU with 16GB VRAM.
It trains a 4-bit LoRA adapter with a tiny batch and gradient accumulation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict


# Avoid CUDA allocator fragmentation on small Colab GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported


# Patch TRL before importing DPOTrainer so Unsloth can use its optimized kernels.
PatchDPOTrainer()

from transformers import TrainingArguments  # noqa: E402
from trl import DPOTrainer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "training_data" / "dpo_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

MODEL_NAME = "unsloth/qwen2.5-1.5b"
MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 512


def parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides while keeping repo defaults runnable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_cuda() -> None:
    """Fail early with a clear message if Colab did not attach a GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this Unsloth DPO run. In Colab, set Runtime > Change runtime type > T4 GPU."
        )
    device_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Using CUDA device: {device_name} ({total_gb:.1f}GB VRAM)")
    torch.cuda.empty_cache()


def load_dpo_dataset(dataset_path: Path) -> Dataset:
    """Load and validate prompt/chosen/rejected JSONL for TRL DPOTrainer."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing DPO dataset: {dataset_path}")

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"DPO dataset is missing required column(s): {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError(f"DPO dataset is empty: {dataset_path}")

    def normalize_row(row: Dict[str, Any]) -> Dict[str, str]:
        prompt = str(row["prompt"]).strip()
        chosen = str(row["chosen"]).strip()
        rejected = str(row["rejected"]).strip()
        if not prompt or not chosen or not rejected:
            raise ValueError("Every DPO row must have non-empty prompt, chosen, and rejected strings.")
        if chosen == rejected:
            raise ValueError("DPO row has identical chosen and rejected strings.")
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    dataset = dataset.map(normalize_row, remove_columns=[c for c in dataset.column_names if c not in required_columns])
    print(f"Loaded {len(dataset)} DPO examples from {dataset_path}")
    print("First row character lengths:", {key: len(dataset[0][key]) for key in ("prompt", "chosen", "rejected")})
    return dataset


def load_model_and_tokenizer(model_name: str, max_seq_length: int):
    """Load a 4-bit Qwen model and attach LoRA adapters for T4-safe training."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )
    return model, tokenizer


def make_training_args(output_dir: Path, seed: int) -> TrainingArguments:
    """Create T4-safe training arguments."""
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        max_grad_norm=0.3,
        seed=seed,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )
    return training_args


def attach_dpo_args(training_args: TrainingArguments, max_seq_length: int, max_prompt_length: int) -> None:
    """Attach DPO-specific args for TRL versions that expect DPOConfig-like attributes."""
    dpo_values = {
        "beta": 0.1,
        "max_length": max_seq_length,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max(128, max_seq_length - max_prompt_length),
        "loss_type": "sigmoid",
        "label_smoothing": 0.0,
        "reference_free": False,
        "precompute_ref_log_probs": False,
    }
    for key, value in dpo_values.items():
        if not hasattr(training_args, key):
            setattr(training_args, key, value)


def build_dpo_trainer(
    model,
    tokenizer,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    max_seq_length: int,
    max_prompt_length: int,
) -> DPOTrainer:
    """Instantiate DPOTrainer across common TRL versions."""
    attach_dpo_args(training_args, max_seq_length=max_seq_length, max_prompt_length=max_prompt_length)
    trainer_signature = inspect.signature(DPOTrainer.__init__)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    # Newer TRL versions use processing_class; older versions use tokenizer.
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    # Older Unsloth/TRL examples pass these directly to DPOTrainer.
    optional_kwargs = {
        "beta": 0.1,
        "max_length": max_seq_length,
        "max_prompt_length": max_prompt_length,
    }
    for key, value in optional_kwargs.items():
        if key in trainer_signature.parameters:
            trainer_kwargs[key] = value

    return DPOTrainer(**trainer_kwargs)


def print_memory(prefix: str) -> None:
    """Print current CUDA memory usage for Colab debugging."""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{prefix} CUDA memory: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def main() -> None:
    """Run one-epoch DPO fine-tuning and save the trained adapter."""
    args = parse_args()
    require_cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dpo_dataset(args.dataset_path)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_seq_length)
    training_args = make_training_args(args.output_dir, args.seed)
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )

    print_memory("Before training")
    print("Starting DPO training...")
    train_result = trainer.train()
    print("Training complete.")
    print_memory("After training")

    metrics = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Saving trained model and tokenizer to:", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    summary = {
        "model_name": args.model_name,
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "train_examples": len(train_dataset),
        "epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "train_loss": metrics.get("train_loss"),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
