from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import ahocorasick

from training.common import convert_chars_to_skz, read_non_empty_lines


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    mcp_dir = root / "vendors" / "sarkaz_tools" / "LLMTools" / "MCPServer"

    parser = argparse.ArgumentParser(description="Build Aho-Corasick automaton for Sarkaz hints.")
    parser.add_argument("--word-files", nargs="+", default=[str(mcp_dir / "endfield_words.txt"), str(mcp_dir / "favorite_words.txt")])
    parser.add_argument("--output", type=Path, default=root / "models" / "trie" / "sarkaz_automaton.pkl")
    parser.add_argument("--metadata", type=Path, default=root / "models" / "trie" / "sarkaz_automaton_meta.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mapping: dict[str, set[str]] = {}
    total_words = 0
    for item in args.word_files:
        path = Path(item)
        if not path.exists():
            continue
        words = read_non_empty_lines(path)
        total_words += len(words)
        for word in words:
            skz = convert_chars_to_skz(word)
            mapping.setdefault(skz, set()).add(word)

    automaton: ahocorasick.Automaton = ahocorasick.Automaton()
    for skz, words in mapping.items():
        automaton.add_word(skz, {"skz": skz, "words": sorted(words)})
    automaton.make_automaton()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(automaton, handle)

    metadata = {
        "input_files": args.word_files,
        "total_input_words": total_words,
        "unique_cipher_keys": len(mapping),
        "output": str(args.output),
    }
    with args.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
