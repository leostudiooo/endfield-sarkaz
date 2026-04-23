#!/usr/bin/env bash
set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
MAX_STEPS="${MAX_STEPS:-40}"
SAMPLE_TEXT="${SAMPLE_TEXT:-ytqqrvqbsbtjyrernx}"

HF_ENDPOINT="${HF_ENDPOINT}" \
MODEL_NAME="${MODEL_NAME}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
MAX_STEPS="${MAX_STEPS}" \
SAMPLE_TEXT="${SAMPLE_TEXT}" \
bash scripts/local_tiny_verify.sh
