#!/usr/bin/env bash
set -euo pipefail

# 16GB kill-test protocol for Qwen3-4B -> Qwen3-1.7B.
# The capture step uses controlled offload; after capture, both LLMs are deleted and
# E0 fitting moves back to CUDA.  E1 intentionally sweeps batch=1,2,4 and records
# OOM as the capacity boundary rather than aborting completed measurements.

: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to a path with sufficient free disk space}"

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

python bench/benchmark_prefill_bridge.py \
  --mapper "$MAPPER" --memory-profile 16gb --low-vram \
  --lengths 512,1024,2048,4096,8192 \
  --warmup 5 --repetitions 20 --continue-on-oom \
  --mapper-dtype bfloat16 \
  --output results/e1_matched_16gb.json

# Many independent short generations give a much more useful acceptance estimate on
# a 16GB machine than the previous one-prompt smoke.  This is the first G1/G2 gate.
python bench/eval_acceptance_pilot.py \
  --mapper "$MAPPER" --memory-profile 16gb --low-vram \
  --bootstrap-samples 5000 --mapper-dtype bfloat16 \
  --output results/e2_matched_16gb.json

# Only run long-generation drift after the 64-prompt E2 gate shows refresh > init.
# Example:
# python bench/eval_generation_drift.py --mapper "$MAPPER" \
#   --prompts 16 --new-tokens 256 --gamma 4 --low-vram \
#   --output results/e2_drift_matched_16gb.json
