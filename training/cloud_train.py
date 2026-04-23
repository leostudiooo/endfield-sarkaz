from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud training entry placeholder.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--train-jsonl", type=str, default="corpus/skz_parallel/base/train.jsonl")
    parser.add_argument("--valid-jsonl", type=str, default="corpus/skz_parallel/base/valid.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/base_model_qwen3_4b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Cloud training placeholder, configure your distributed launcher here.")
    print(f"model={args.model_name}")
    print(f"train={args.train_jsonl}")
    print(f"valid={args.valid_jsonl}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
