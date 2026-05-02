"""Compare base vs DPO fine-tuned model outputs on Tenacious-Bench tasks.

Run from the repository root:
    python training/evaluate_trained_model.py

The script checks that the LoRA adapter in outputs/ is actually loaded, then
generates and scores base-model and fine-tuned outputs side by side.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import unsloth
from peft import PeftModel
from unsloth import FastLanguageModel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring_evaluator import evaluate_task  # noqa: E402


BASE_MODEL_NAME = "unsloth/qwen2.5-1.5b"
DEFAULT_ADAPTER_DIR = ROOT / "outputs"
DEFAULT_TEST_PATH = ROOT / "data" / "splits" / "test.json"
DEFAULT_FALLBACK_TEST_PATH = ROOT / "data" / "splits" / "held_out.json"
DEFAULT_RESULTS_PATH = ROOT / "outputs" / "evaluation_comparison.json"
MAX_SEQ_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=3, help="Max tasks to evaluate. Use 0 for all tasks.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--side-by-side", type=int, default=3)
    return parser.parse_args()


def load_tasks(path: Path) -> List[Mapping[str, Any]]:
    """Load tasks from JSON list, {'tasks': [...]}, or JSONL."""
    if not path.exists():
        if path == DEFAULT_TEST_PATH and DEFAULT_FALLBACK_TEST_PATH.exists():
            print(f"WARNING: {path} not found. Falling back to {DEFAULT_FALLBACK_TEST_PATH}.")
            path = DEFAULT_FALLBACK_TEST_PATH
        else:
            raise FileNotFoundError(f"Evaluation task file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, Mapping):
            data = data.get("tasks", [])
        if not isinstance(data, list):
            raise ValueError(f"Expected a task list or tasks object in {path}")
        return [task for task in data if isinstance(task, Mapping)]

    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"JSONL row {line_number} in {path} is not an object")
        rows.append(row)
    return rows


def build_prompt(task: Mapping[str, Any]) -> str:
    """Use the same prompt shape as the DPO dataset builder."""
    task_input = get_mapping(task, "input")
    prospect = get_mapping(task_input, "prospect")
    signal_brief = get_mapping(task_input, "signal_brief")
    requested_capacity = get_mapping(task_input, "requested_capacity")
    bench_summary = get_mapping(task_input, "bench_summary")
    pricing_scope = get_mapping(task_input, "pricing_scope")

    lines = [
        "Write a concise Tenacious outreach email using only the task input.",
        "",
        "Prospect:",
        f"- Company: {prospect.get('company', 'Unknown')}",
        f"- Contact: {prospect.get('contact_name', 'Unknown')}",
        f"- Title: {prospect.get('title', 'Unknown')}",
        f"- Segment: {prospect.get('segment', 'Unknown')}",
        "",
        "Signal:",
        f"- Summary: {signal_brief.get('summary', '')}",
        f"- Confidence: {signal_brief.get('confidence', '')}",
        f"- Source window: {signal_brief.get('source_window', '')}",
    ]

    signals = signal_brief.get("signals", [])
    if isinstance(signals, Sequence) and not isinstance(signals, str):
        for signal in signals:
            if isinstance(signal, Mapping):
                lines.append(f"- Signal detail: {signal.get('text', '')} ({signal.get('confidence', '')})")

    lines.extend(
        [
            "",
            "Requested capacity:",
            f"- Headcount: {requested_capacity.get('headcount', '')}",
            f"- Role: {requested_capacity.get('role', '')}",
            f"- Stack: {requested_capacity.get('stack', '')}",
            f"- Seniority: {requested_capacity.get('seniority', '')}",
            f"- Timeline days: {requested_capacity.get('timeline_days', '')}",
            f"- Timezone overlap hours: {requested_capacity.get('timezone_overlap_hours', '')}",
            "",
            "Available capacity summary:",
        ]
    )

    capacities = bench_summary.get("available_capacity", [])
    if isinstance(capacities, Sequence) and not isinstance(capacities, str):
        for capacity in capacities:
            if isinstance(capacity, Mapping):
                lines.append(
                    "- "
                    f"{capacity.get('available_headcount', '')} "
                    f"{capacity.get('seniority', '')} "
                    f"{capacity.get('stack', '')} "
                    f"{capacity.get('role', '')}, earliest start "
                    f"{capacity.get('earliest_start_days', '')} days, "
                    f"{capacity.get('timezone_overlap_hours', '')} hours overlap"
                )
    if bench_summary.get("notes"):
        lines.append(f"- Notes: {bench_summary.get('notes')}")

    lines.extend(
        [
            "",
            "Pricing scope:",
            f"- Allowed amounts: {pricing_scope.get('allowed_amounts', [])}",
            f"- Allowed phrases: {pricing_scope.get('allowed_phrases', [])}",
            f"- Can quote total contract value: {pricing_scope.get('can_quote_total_contract_value', False)}",
            "",
            f"Prior thread: {task_input.get('prior_thread', '')}",
            f"Channel: {task_input.get('channel', '')}",
            f"Goal: {task_input.get('outreach_goal', '')}",
            "",
            "Return only the email. Keep it under 120 words. Include a subject line.",
            "Email:",
        ]
    )
    return "\n".join(line.rstrip() for line in lines).strip()


def load_base_model(model_name: str, max_seq_length: int):
    """Load the base model through Unsloth."""
    print(f"Loading base model for generation: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    prepare_tokenizer(tokenizer)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model(base_model_name: str, adapter_dir: Path, max_seq_length: int):
    """Load the base model plus the trained LoRA adapter from outputs/."""
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    adapter_config = adapter_dir / "adapter_config.json"
    if adapter_config.exists():
        print(f"Loading fine-tuned model: base={base_model_name}, adapter={adapter_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        prepare_tokenizer(tokenizer)
        try:
            FastLanguageModel.for_inference(model)
        except Exception as exc:  # Unsloth may not patch every PEFT wrapper version.
            print(f"WARNING: FastLanguageModel.for_inference failed on adapter model: {exc}")
            model.eval()
        print("Confirmed adapter_config.json found and PeftModel.from_pretrained was used.")
        return model, tokenizer

    print(f"adapter_config.json not found in {adapter_dir}; trying to load as a merged/local model.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    prepare_tokenizer(tokenizer)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def prepare_tokenizer(tokenizer: Any) -> None:
    """Normalize tokenizer padding for generation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def generate_outputs(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    max_seq_length: int,
    label: str,
) -> List[str]:
    """Generate one output per prompt."""
    outputs = []
    model.eval()
    total = len(prompts)
    for index, prompt in enumerate(prompts, start=1):
        print(f"Generating {label} output {index}/{total}...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(output)
        if index == 1:
            print(f"\nFirst generated output from {label}:\n{output}\n")
    return outputs


def score_outputs(tasks: Sequence[Mapping[str, Any]], outputs: Sequence[str]) -> List[Dict[str, Any]]:
    """Score generated outputs with scoring_evaluator.evaluate_task using overall."""
    results = []
    for task, output in zip(tasks, outputs):
        task_for_scoring = dict(task)
        task_for_scoring["candidate_output"] = output_to_candidate_output(output)
        score = evaluate_task(task_for_scoring, candidate_field="candidate_output")
        results.append(
            {
                "task_id": score.get("task_id", task.get("task_id", "")),
                "overall": float(score.get("overall", 0.0)),
                "scores": score.get("scores", {}),
                "passed": bool(score.get("passed", False)),
                "hard_fail_conditions": score.get("hard_fail_conditions", []),
            }
        )
    return results


def output_to_candidate_output(text: str) -> Dict[str, str]:
    """Convert generated text into scoring_evaluator's subject/body shape."""
    subject = ""
    body = text.strip()
    lines = body.splitlines()
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
    return {"subject": subject, "body": body}


def summarize_scores(label: str, results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return and print aggregate score summary without empty-list division."""
    if not results:
        summary = {"label": label, "count": 0, "mean_overall": 0.0, "pass_rate": 0.0}
    else:
        summary = {
            "label": label,
            "count": len(results),
            "mean_overall": round(mean(float(result["overall"]) for result in results), 3),
            "pass_rate": round(sum(1 for result in results if result["passed"]) / len(results), 4),
        }
    print(f"{label.upper()} mean overall: {summary['mean_overall']} pass_rate: {summary['pass_rate']}")
    return summary


def print_side_by_side(
    tasks: Sequence[Mapping[str, Any]],
    base_outputs: Sequence[str],
    finetuned_outputs: Sequence[str],
    count: int,
) -> None:
    """Print side-by-side comparison for the first few tasks."""
    print("\nSIDE-BY-SIDE OUTPUT CHECK")
    for index, (task, base_output, finetuned_output) in enumerate(
        zip(tasks[:count], base_outputs[:count], finetuned_outputs[:count]),
        start=1,
    ):
        print("=" * 90)
        print(f"Example {index}: {task.get('task_id', '')}")
        print("- BASE OUTPUT -")
        print(base_output)
        print("- FINE-TUNED OUTPUT -")
        print(finetuned_output)
        print(f"outputs_differ={normalize_space(base_output) != normalize_space(finetuned_output)}")


def unload_model(model: Any, tokenizer: Any) -> None:
    """Free GPU memory between base and fine-tuned evaluation."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Safely fetch nested mappings."""
    value = mapping.get(key) if isinstance(mapping, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def normalize_space(text: str) -> str:
    """Normalize whitespace for output comparisons."""
    return " ".join(str(text).split())


def main() -> None:
    """Run base-vs-fine-tuned generation and scoring."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. In Colab, set runtime type to T4 GPU.")

    tasks = load_tasks(args.tasks)
    loaded_count = len(tasks)
    if args.limit == 0:
        selected_tasks = tasks
    else:
        selected_tasks = tasks[: args.limit]
    tasks = selected_tasks
    print(f"Evaluation dataset size loaded: {loaded_count}")
    print(f"Evaluation dataset size selected: {len(tasks)}")
    if args.limit != 0 and loaded_count > len(tasks):
        print("Using a short debug subset by default. Pass --limit 0 to evaluate all tasks.")
    if not tasks:
        raise ValueError("No evaluation tasks loaded; cannot compare models.")

    prompts = [build_prompt(task) for task in tasks]
    print(f"Base model name used during generation: {args.base_model}")
    print(f"Fine-tuned adapter/model path used during generation: {args.adapter_dir}")

    base_model, base_tokenizer = load_base_model(args.base_model, args.max_seq_length)
    base_outputs = generate_outputs(
        base_model,
        base_tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        label="base model",
    )
    unload_model(base_model, base_tokenizer)

    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        args.base_model,
        args.adapter_dir,
        args.max_seq_length,
    )
    finetuned_outputs = generate_outputs(
        finetuned_model,
        finetuned_tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        label="fine-tuned model",
    )
    unload_model(finetuned_model, finetuned_tokenizer)

    base_scores = score_outputs(tasks, base_outputs)
    finetuned_scores = score_outputs(tasks, finetuned_outputs)
    base_summary = summarize_scores("base", base_scores)
    finetuned_summary = summarize_scores("fine_tuned", finetuned_scores)
    delta = round(finetuned_summary["mean_overall"] - base_summary["mean_overall"], 3)
    print(f"SCORE DELTA fine_tuned - base: {delta}")

    print_side_by_side(
        tasks=tasks,
        base_outputs=base_outputs,
        finetuned_outputs=finetuned_outputs,
        count=min(args.side_by_side, len(tasks)),
    )

    payload = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "task_file": str(args.tasks),
        "dataset_size": len(tasks),
        "base_summary": base_summary,
        "fine_tuned_summary": finetuned_summary,
        "score_delta": delta,
        "examples": [
            {
                "task_id": task.get("task_id", ""),
                "base_output": base_output,
                "fine_tuned_output": finetuned_output,
                "base_score": base_score,
                "fine_tuned_score": finetuned_score,
            }
            for task, base_output, finetuned_output, base_score, finetuned_score in zip(
                tasks,
                base_outputs,
                finetuned_outputs,
                base_scores,
                finetuned_scores,
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote detailed comparison to {args.output}")


if __name__ == "__main__":
    main()
