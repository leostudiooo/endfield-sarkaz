from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Run single-pass Sarkaz decoding with optional AC hints.")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model-path", type=Path, default=root / "models" / "base_model" / "tiny_verify")
    parser.add_argument("--fallback-model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--automaton", type=Path, default=root / "models" / "trie" / "sarkaz_automaton.pkl")
    parser.add_argument("--max-hints", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def load_automaton(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def collect_hints(skz_text: str, automaton, max_hints: int) -> list[dict[str, Any]]:
    if automaton is None:
        return []

    hints: list[dict[str, Any]] = []
    for end_index, payload in automaton.iter(skz_text):
        key = payload.get("skz", "")
        values = payload.get("words", [])
        key_len = len(key)
        start_index = end_index - key_len + 1 if key_len > 0 else end_index
        hints.append({"start": start_index, "end": end_index, "candidates": values, "skz": key})
        if len(hints) >= max_hints:
            break
    return hints


def build_prompt(skz_text: str, hints: list[dict[str, Any]]) -> str:
    hint_lines = []
    if hints:
        hint_lines.append("Detected candidate terms:")
        for item in hints:
            candidates = ", ".join(item["candidates"])
            hint_lines.append(f"- {item['skz']} ({item['start']}:{item['end']}) -> {candidates}")
    else:
        hint_lines.append("No trie hints were detected.")

    hint_text = "\n".join(hint_lines)
    return (
        "System: You are an expert Sarkaz translator. Translate Sarkaz cipher text into Chinese.\n"
        f"{hint_text}\n"
        f"User: {skz_text}\n"
        "Assistant:"
    )


def main() -> None:
    args = parse_args()

    model_source = str(args.model_path) if args.model_path.exists() else args.fallback_model
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_source, trust_remote_code=True)
    device = _detect_device()
    model.to(device)
    model.eval()

    automaton = load_automaton(args.automaton)
    hints = collect_hints(args.text, automaton, args.max_hints)
    prompt = build_prompt(args.text, hints)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    result = {
        "model_source": model_source,
        "input": args.text,
        "hint_count": len(hints),
        "hints": hints,
        "translation": translation,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
