#!/usr/bin/env bash
set -euo pipefail

# 16GB kill-test protocol for Qwen3-4B -> Qwen3-1.7B.
# E0 capture uses controlled offload, then reclaims the GPU for CUDA fitting.
# Inference does the opposite: first try fully resident BF16 to use the card well and
# sweep batch=1/2/4.  Only fall back to controlled offload if resident model loading
# or the short E2 pilot genuinely cannot fit.

: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to a path with sufficient free disk space}"
: "${EVAL_TEXT:?Set EVAL_TEXT to a held-out JSONL/raw-text file disjoint from E0 calibration}"

MAPPER="checkpoints/qwen3_4b_to_1p7b_matched_16gb.pt"
CALIB="artifacts/e0_qwen3_4b_to_1p7b_matched_16gb_calibration"

python bench/fit_ridge_calibration.py \
  --target Qwen/Qwen3-4B --draft Qwen/Qwen3-1.7B \
  --head-mode matched --fit-profile 16gb --low-vram \
  --sequences 128 --selection-sequences 32 \
  --seq-len 1024 --stride 4 --k 8 --lambda 0.01 \
  --layer-selection r2 \
  --calibration-dir "$CALIB" \
  --output "$MAPPER"

RESIDENT_OK=0
if python bench/benchmark_prefill_bridge.py \
  --mapper "$MAPPER" --memory-profile 16gb \
  --lengths 512,1024,2048,4096,8192 \
  --batch-sizes 1,2,4 \
  --warmup 5 --repetitions 20 --continue-on-oom \
  --mapper-dtype bfloat16 \
  --output results/e1_matched_16gb_resident.json; then
  RESIDENT_OK=1
else
  echo "Resident 16GB E1 could not load/run; recording offload fallback instead." >&2
  python bench/benchmark_prefill_bridge.py \
    --mapper "$MAPPER" --memory-profile 16gb --low-vram \
    --lengths 512,1024,2048,4096,8192 \
    --batch-sizes 1,2,4 \
    --warmup 5 --repetitions 20 --continue-on-oom \
    --mapper-dtype bfloat16 \
    --output results/e1_matched_16gb_offload.json
fi

# Many independent short generations are more useful for deciding G1/G2 than the
# previous one-prompt smoke. Prefer resident inference if the models loaded in E1.
if [[ "$RESIDENT_OK" == "1" ]] && python bench/eval_acceptance_pilot.py \
  --mapper "$MAPPER" --memory-profile 16gb \
  --text-file "$EVAL_TEXT" \
  --bootstrap-samples 5000 --mapper-dtype bfloat16 \
  --output results/e2_matched_16gb_resident.json; then
  echo "Resident 16GB E2 completed."
else
  echo "Using controlled-offload E2 fallback on 16GB." >&2
  python bench/eval_acceptance_pilot.py \
    --mapper "$MAPPER" --memory-profile 16gb --low-vram \
    --text-file "$EVAL_TEXT" \
    --bootstrap-samples 5000 --mapper-dtype bfloat16 \
    --output results/e2_matched_16gb_offload.json
fi

# Only run long-generation drift after the 64-prompt E2 gate shows refresh > init.
# Example:
# python bench/eval_generation_drift.py --mapper "$MAPPER" \
#   --text-file "$EVAL_TEXT" --prompts 16 --new-tokens 256 --gamma 4 \
#   --output results/e2_drift_matched_16gb.json
