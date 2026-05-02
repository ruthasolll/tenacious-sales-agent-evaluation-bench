"""DPO fine-tuning for Tenacious-Bench using Unsloth + TRL.

Run from the repository root:
    python training/train_dpo.py

Expected dataset:
    training_data/dpo_dataset.jsonl

Each JSONL row must contain:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

This script is configured for Google Colab T4 GPUs with 16GB VRAM.
"""
import unsloth
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from transformers import TrainingArguments
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


# Keep Colab CUDA memory fragmentation low on small GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported


# Unsloth must patch TRL before DPOTrainer is constructed.
PatchDPOTrainer()

from trl import DPOConfig, DPOTrainer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "training_data" / "dpo_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

MODEL_NAME = "unsloth/qwen2.5-1.5b"
MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 512


class LossPrinterCallback(TrainerCallback):
    """Print compact loss logs during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")
        if loss is not None:
            if learning_rate is None:
                print(f"[step {state.global_step}] loss={loss:.4f}")
            else:
                print(f"[step {state.global_step}] loss={loss:.4f} lr={learning_rate:.2e}")


def parse_args() -> argparse.Namespace:
    """Parse optional overrides while keeping the requested default command runnable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_cuda() -> None:
    """Fail early if Colab is not using a GPU runtime."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. In Colab, choose Runtime > Change runtime type > T4 GPU, then rerun."
        )

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"CUDA device: {device_name} ({total_vram_gb:.1f}GB VRAM)")
    torch.cuda.empty_cache()


def load_dpo_dataset(dataset_path: Path) -> Dataset:
    """Load and validate prompt/chosen/rejected JSONL."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required DPO field(s): {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    def clean_row(row: Dict[str, Any]) -> Dict[str, str]:
        prompt = str(row["prompt"]).strip()
        chosen = str(row["chosen"]).strip()
        rejected = str(row["rejected"]).strip()
        if not prompt or not chosen or not rejected:
            raise ValueError("Every row must have non-empty prompt, chosen, and rejected fields.")
        if chosen == rejected:
            raise ValueError("A DPO row has identical chosen and rejected text.")
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    extra_columns = [column for column in dataset.column_names if column not in required_columns]
    dataset = dataset.map(clean_row, remove_columns=extra_columns)

    print(f"Loaded {len(dataset)} DPO examples from {dataset_path}")
    print("Example field lengths:", {key: len(dataset[0][key]) for key in ("prompt", "chosen", "rejected")})
    return dataset


def load_model_and_tokenizer(model_name: str, max_seq_length: int):
    """Load the 4-bit base model and attach trainable LoRA adapters."""
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


def make_dpo_config(output_dir: Path, max_seq_length: int, max_prompt_length: int, seed: int) -> DPOConfig:
    """Create a latest-TRL DPOConfig instead of deprecated TrainingArguments."""
    return DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
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
        beta=0.1,
        loss_type="sigmoid",
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        truncation_mode="keep_end",
        precompute_ref_log_probs=False,
        reference_free=False,
        torch_empty_cache_steps=10,
    )


def print_cuda_memory(label: str) -> None:
    """Print CUDA memory usage for quick Colab debugging."""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{label}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def main() -> None:
    """Train one DPO epoch and save the resulting LoRA adapter."""
    args = parse_args()
    require_cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dpo_dataset(args.dataset_path)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_seq_length)
    dpo_config = make_dpo_config(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[LossPrinterCallback()],
    )

    print_cuda_memory("Before training")
    print("Starting DPO training...")
    train_result = trainer.train()
    print("DPO training finished.")
    print_cuda_memory("After training")

    metrics = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    summary = {
        "model_name": args.model_name,
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "train_examples": len(train_dataset),
        "epochs": dpo_config.num_train_epochs,
        "per_device_train_batch_size": dpo_config.per_device_train_batch_size,
        "gradient_accumulation_steps": dpo_config.gradient_accumulation_steps,
        "learning_rate": dpo_config.learning_rate,
        "train_loss": metrics.get("train_loss"),
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
