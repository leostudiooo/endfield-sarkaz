from __future__ import annotations

import argparse
import re
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.common import convert_chars_to_skz, read_non_empty_lines


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    mcp_dir = root / "vendors" / "sarkaz_tools" / "LLMTools" / "MCPServer"
    parser = argparse.ArgumentParser(description="Merge custom SPM tokens into a base HF tokenizer.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--spm-model", type=Path, default=root / "models" / "tokenizer" / "custom_sarkaz.model")
    parser.add_argument("--endfield-words", type=Path, default=mcp_dir / "endfield_words.txt")
    parser.add_argument("--output-dir", type=Path, default=root / "models" / "tokenizer" / "merged")
    parser.add_argument("--min-latin-len", type=int, default=3)
    parser.add_argument("--init-embeddings", action="store_true")
    parser.add_argument("--model-output-dir", type=Path, default=root / "models" / "base_model" / "resized")
    return parser.parse_args()


def _normalize_spm_piece(piece: str) -> str:
    if piece.startswith("<") and piece.endswith(">"):
        return ""
    return piece.replace("▁", "").strip()


def _is_candidate_token(token: str, min_latin_len: int) -> bool:
    if not token:
        return False
    if re.fullmatch(r"[\W_]+", token):
        return False
    if re.fullmatch(r"\d+", token):
        return False
    if re.fullmatch(r"[A-Za-z]+", token) and len(token) < min_latin_len:
        return False
    return True


def _collect_spm_tokens(spm_model: Path, min_latin_len: int) -> list[str]:
    processor = spm.SentencePieceProcessor(model_file=str(spm_model))
    tokens: list[str] = []
    for index in range(processor.get_piece_size()):
        token = _normalize_spm_piece(processor.id_to_piece(index))
        if _is_candidate_token(token, min_latin_len):
            tokens.append(token)
    return tokens


def _collect_endfield_tokens(path: Path, min_latin_len: int) -> list[str]:
    tokens: list[str] = []
    for word in read_non_empty_lines(path):
        tokens.append(word)
        skz = convert_chars_to_skz(word)
        if _is_candidate_token(skz, min_latin_len):
            tokens.append(skz)
    return tokens


def _init_new_embeddings(
    model: AutoModelForCausalLM,
    tokenizer,
    reference_tokenizer,
    new_tokens: list[str],
    old_vocab_size: int,
) -> None:
    input_embed = model.get_input_embeddings().weight.data
    output_embed = model.get_output_embeddings().weight.data if model.get_output_embeddings() is not None else None

    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id < old_vocab_size:
            continue

        sub_ids = reference_tokenizer.encode(token, add_special_tokens=False)
        sub_ids = [item for item in sub_ids if item < old_vocab_size]
        if not sub_ids:
            continue

        mean_vector = input_embed[sub_ids].mean(dim=0)
        input_embed[token_id] = mean_vector
        if output_embed is not None and output_embed.shape[0] == input_embed.shape[0]:
            output_embed[token_id] = mean_vector


def main() -> None:
    args = parse_args()

    if not args.spm_model.exists():
        raise FileNotFoundError(f"SPM model not found: {args.spm_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    reference_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    existing = set(tokenizer.get_vocab().keys())
    candidates = _collect_spm_tokens(args.spm_model, args.min_latin_len)
    candidates.extend(_collect_endfield_tokens(args.endfield_words, args.min_latin_len))

    deduped_new_tokens: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        if token in seen or token in existing:
            continue
        seen.add(token)
        deduped_new_tokens.append(token)

    added_count = tokenizer.add_tokens(deduped_new_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Added tokens: {added_count}")
    print(f"Merged tokenizer saved to: {args.output_dir}")

    if args.init_embeddings:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
        old_vocab_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))

        _init_new_embeddings(
            model=model,
            tokenizer=tokenizer,
            reference_tokenizer=reference_tokenizer,
            new_tokens=deduped_new_tokens,
            old_vocab_size=old_vocab_size,
        )

        args.model_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.model_output_dir)
        tokenizer.save_pretrained(args.model_output_dir)
        print(f"Resized model saved to: {args.model_output_dir}")


if __name__ == "__main__":
    main()
