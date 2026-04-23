from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

SYSTEM_PROMPT = "Translate the following Sarkaz cipher text into Chinese."


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Minimal SFT training entry for Sarkaz translation.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--train-jsonl", type=Path, default=root / "corpus" / "skz_parallel" / "base" / "train.jsonl")
    parser.add_argument("--valid-jsonl", type=Path, default=root / "corpus" / "skz_parallel" / "base" / "valid.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "models" / "base_model" / "qwen3_0_6b_verify")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-valid-samples", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def _format_example(example: dict[str, Any]) -> str:
    if example.get("messages"):
        role_to_content = {item["role"]: item["content"] for item in example["messages"]}
        system = role_to_content.get("system", SYSTEM_PROMPT)
        user = role_to_content.get("user", example.get("skz_text", ""))
        assistant = role_to_content.get("assistant", example.get("zh_text", ""))
    else:
        system = SYSTEM_PROMPT
        user = example.get("skz_text", "")
        assistant = example.get("zh_text", "")

    return f"System: {system}\nUser: {user}\nAssistant: {assistant}"


def _prepare_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
) -> Dataset:
    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        text = _format_example(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = [
            tid if tid != tokenizer.pad_token_id else -100
            for tid in tokenized["input_ids"]
        ]
        return tokenized

    columns = dataset.column_names
    return dataset.map(preprocess, remove_columns=columns)


def _guess_lora_targets(model: torch.nn.Module) -> list[str]:
    candidate_names: set[str] = set()

    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf == "lm_head":
            continue

        if isinstance(module, torch.nn.Linear):
            candidate_names.add(leaf)
            continue

        if module.__class__.__name__ == "Conv1D":
            candidate_names.add(leaf)

    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
        "W_pack",
    ]
    selected = [name for name in preferred if name in candidate_names]

    if selected:
        return selected
    return sorted(candidate_names)[:8]


def _load_datasets(train_jsonl: Path, valid_jsonl: Path) -> DatasetDict:
    if not train_jsonl.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_jsonl}")

    data_files: dict[str, str] = {"train": str(train_jsonl)}
    if valid_jsonl.exists():
        data_files["validation"] = str(valid_jsonl)

    return load_dataset("json", data_files=data_files)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    use_lora = args.use_lora and not args.disable_lora
    lora_targets: list[str] = []
    if use_lora:
        lora_targets = _guess_lora_targets(model)
        if lora_targets:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_targets,
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()
        else:
            print("No suitable LoRA target modules found, fallback to full-parameter training.")

    raw_datasets = _load_datasets(args.train_jsonl, args.valid_jsonl)

    train_dataset = raw_datasets["train"].shuffle(seed=args.seed)
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))
    train_dataset = _prepare_dataset(train_dataset, tokenizer, args.max_length)

    eval_dataset = None
    if "validation" in raw_datasets:
        eval_dataset = raw_datasets["validation"].shuffle(seed=args.seed)
        if args.max_valid_samples > 0:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_valid_samples)))
        eval_dataset = _prepare_dataset(eval_dataset, tokenizer, args.max_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.logging_steps if eval_dataset is not None else None,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model_name": args.model_name,
        "tokenizer_path": tokenizer_source,
        "train_samples": len(train_dataset),
        "valid_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_steps": args.max_steps,
        "use_lora": bool(lora_targets),
        "lora_targets": lora_targets,
        "output_dir": str(args.output_dir),
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
