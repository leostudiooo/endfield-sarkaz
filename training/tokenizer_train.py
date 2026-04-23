from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_input = root / "corpus" / "skz_parallel" / "base" / "tokenizer_mix.txt"
    default_output_dir = root / "models" / "tokenizer"

    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer on mixed Chinese+Sarkaz corpus.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--model-prefix", type=str, default="custom_sarkaz")
    parser.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"])
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input corpus not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = args.output_dir / args.model_prefix

    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=str(model_prefix),
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        split_digits=True,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )

    print(f"Saved model: {model_prefix}.model")
    print(f"Saved vocab: {model_prefix}.vocab")


if __name__ == "__main__":
    main()
