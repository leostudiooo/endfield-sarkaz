from __future__ import annotations

import argparse
import json
from pathlib import Path


def _edit_distance(a: str, b: str) -> int:
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_terms(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CER and term hit rate.")
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--target-key", type=str, default="zh_text")
    parser.add_argument("--pred-key", type=str, default="pred_text")
    parser.add_argument("--term-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_jsonl(args.pred_jsonl)
    if not records:
        raise ValueError("No records found.")

    total_chars = 0
    total_distance = 0

    for item in records:
        target = item.get(args.target_key, "")
        pred = item.get(args.pred_key, "")
        total_chars += max(1, len(target))
        total_distance += _edit_distance(target, pred)

    cer = total_distance / total_chars

    terms = _load_terms(args.term_file)
    term_target_total = 0
    term_hit_total = 0
    if terms:
        for item in records:
            target = item.get(args.target_key, "")
            pred = item.get(args.pred_key, "")
            for term in terms:
                if term in target:
                    term_target_total += 1
                    if term in pred:
                        term_hit_total += 1

    result = {
        "samples": len(records),
        "cer": cer,
        "term_targets": term_target_total,
        "term_hits": term_hit_total,
        "term_hit_rate": (term_hit_total / term_target_total) if term_target_total > 0 else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
