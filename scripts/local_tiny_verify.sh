#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES="${NUM_SAMPLES:-2000}"
MAX_STEPS="${MAX_STEPS:-40}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
SAMPLE_TEXT="${SAMPLE_TEXT:-ytqqrvqbsbtjyrernx}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[1/4] Generate tiny parallel corpus..."
HF_ENDPOINT="${HF_ENDPOINT}" uv run skz-generate-data \
  --num-samples "${NUM_SAMPLES}" \
  --valid-ratio 0.05

echo "[2/4] Run tiny SFT smoke training..."
HF_ENDPOINT="${HF_ENDPOINT}" uv run skz-train-base \
  --model-name "${MODEL_NAME}" \
  --max-train-samples "${NUM_SAMPLES}" \
  --max-valid-samples 128 \
  --max-steps "${MAX_STEPS}"

echo "[3/4] Build trie hints..."
HF_ENDPOINT="${HF_ENDPOINT}" uv run skz-build-trie

echo "[4/4] Decode one sample..."
HF_ENDPOINT="${HF_ENDPOINT}" uv run skz-decode --text "${SAMPLE_TEXT}"

echo "Tiny local verification finished."
