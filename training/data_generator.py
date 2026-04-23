from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from training.common import convert_chars_to_skz, read_non_empty_lines

SYSTEM_PROMPT = "Translate the following Sarkaz cipher text into Chinese."


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> dict[str, Path]:
    root = _repo_root()
    mcp_dir = root / "vendors" / "sarkaz_tools" / "LLMTools" / "MCPServer"
    return {
        "wordlist": mcp_dir / "wordlist.txt",
        "single_char": mcp_dir / "single_char.txt",
        "endfield_words": mcp_dir / "endfield_words.txt",
        "train_jsonl": root / "corpus" / "skz_parallel" / "base" / "train.jsonl",
        "valid_jsonl": root / "corpus" / "skz_parallel" / "base" / "valid.jsonl",
        "tokenizer_mix": root / "corpus" / "skz_parallel" / "base" / "tokenizer_mix.txt",
    }


def _filter_words(lines: list[str], min_len: int = 1, max_len: int = 24) -> list[str]:
    result = []
    for line in lines:
        if min_len <= len(line) <= max_len:
            result.append(line)
    return result


def _build_sentence(
    rng: random.Random,
    common_words: list[str],
    long_phrases: list[str],
    endfield_words: list[str],
    min_words: int,
    max_words: int,
    min_chars: int,
    max_chars: int,
    endfield_ratio: float,
) -> str:
    for _ in range(64):
        if long_phrases and rng.random() < 0.25:
            sentence = rng.choice(long_phrases)
        else:
            token_count = rng.randint(min_words, max_words)
            picked: list[str] = []
            for _ in range(token_count):
                if endfield_words and rng.random() < endfield_ratio:
                    picked.append(rng.choice(endfield_words))
                else:
                    picked.append(rng.choice(common_words))
            sentence = "".join(picked)

        if min_chars <= len(sentence) <= max_chars:
            return sentence

    fallback = sentence[:max_chars]
    if len(fallback) < min_chars:
        fallback = fallback + rng.choice(common_words)
    return fallback


def _make_record(index: int, zh_text: str) -> dict[str, Any]:
    skz_text = convert_chars_to_skz(zh_text)
    return {
        "id": index,
        "zh_text": zh_text,
        "skz_text": skz_text,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": skz_text},
            {"role": "assistant", "content": zh_text},
        ],
    }


def build_dataset(
    num_samples: int,
    min_words: int,
    max_words: int,
    min_chars: int,
    max_chars: int,
    endfield_ratio: float,
    seed: int,
    wordlist_path: Path,
    single_char_path: Path,
    endfield_words_path: Path,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    wordlist = _filter_words(read_non_empty_lines(wordlist_path), min_len=1, max_len=24)
    single_chars = _filter_words(read_non_empty_lines(single_char_path), min_len=1, max_len=1)
    endfield_words = _filter_words(read_non_empty_lines(endfield_words_path), min_len=1, max_len=24)

    if not wordlist:
        raise ValueError(f"No words found in {wordlist_path}")

    common_words = wordlist + single_chars
    long_phrases = [item for item in wordlist if 12 <= len(item) <= max_chars]

    records: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()

    while len(records) < num_samples:
        sentence = _build_sentence(
            rng=rng,
            common_words=common_words,
            long_phrases=long_phrases,
            endfield_words=endfield_words,
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
            max_chars=max_chars,
            endfield_ratio=endfield_ratio,
        )
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        records.append(_make_record(len(records), sentence))

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tokenizer_mix(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record["zh_text"] + "\n")
            handle.write(record["skz_text"] + "\n")


def split_train_valid(records: list[dict[str, Any]], valid_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_size = max(1, int(len(records) * valid_ratio))
    valid = records[:valid_size]
    train = records[valid_size:]
    return train, valid


def build_from_corpus(
    corpus_path: Path,
    num_samples: int,
    min_chars: int,
    max_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    lines = read_non_empty_lines(corpus_path)
    lines = [l for l in lines if min_chars <= len(l) <= max_chars]
    print(f"Corpus: {len(lines)} eligible lines (from {corpus_path})")

    if num_samples <= 0 or num_samples >= len(lines):
        sampled = rng.sample(lines, len(lines)) if num_samples < len(lines) else lines
    else:
        sampled = rng.sample(lines, num_samples)

    return [_make_record(i, sentence) for i, sentence in enumerate(sampled)]


def parse_args() -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Generate Sarkaz parallel data.")
    parser.add_argument("--wordlist", type=Path, default=defaults["wordlist"])
    parser.add_argument("--single-char", type=Path, default=defaults["single_char"])
    parser.add_argument("--endfield-words", type=Path, default=defaults["endfield_words"])
    parser.add_argument("--train-jsonl", type=Path, default=defaults["train_jsonl"])
    parser.add_argument("--valid-jsonl", type=Path, default=defaults["valid_jsonl"])
    parser.add_argument("--tokenizer-mix", type=Path, default=defaults["tokenizer_mix"])
    parser.add_argument("--corpus", type=Path, default="")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=96)
    parser.add_argument("--endfield-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.corpus:
        records = build_from_corpus(
            corpus_path=args.corpus,
            num_samples=args.num_samples,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            seed=args.seed,
        )
    else:
        records = build_dataset(
            num_samples=args.num_samples,
            min_words=args.min_words,
            max_words=args.max_words,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            endfield_ratio=args.endfield_ratio,
            seed=args.seed,
            wordlist_path=args.wordlist,
            single_char_path=args.single_char,
            endfield_words_path=args.endfield_words,
        )

    train_records, valid_records = split_train_valid(records, args.valid_ratio)

    write_jsonl(args.train_jsonl, train_records)
    write_jsonl(args.valid_jsonl, valid_records)
    write_tokenizer_mix(args.tokenizer_mix, records)

    print(f"Generated total records: {len(records)}")
    print(f"Train records: {len(train_records)} -> {args.train_jsonl}")
    print(f"Valid records: {len(valid_records)} -> {args.valid_jsonl}")
    print(f"Tokenizer mix file: {args.tokenizer_mix}")


if __name__ == "__main__":
    main()
