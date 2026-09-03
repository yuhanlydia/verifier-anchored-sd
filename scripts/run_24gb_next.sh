#!/usr/bin/env bash
set -euo pipefail

# 24GB confirmatory protocol.  Full-head is the original closed-form baseline;
# matched-head is the strong efficiency baseline.  Both use the same model pair and
# benchmark prompts.  Do not start phase-2 acceptance training until G0/G1/G2 pass.

: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to a path with sufficient free disk space}"
: "${EVAL_TEXT:?Set EVAL_TEXT to a held-out JSONL/raw-text file disjoint from E0 calibration}"

FULL="checkpoints/qwen3_4b_to_1p7b_full_24gb.pt"
MATCHED="checkpoints/qwen3_4b_to_1p7b_matched_24gb.pt"
CALIB="artifacts/e0_qwen3_4b_to_1p7b_24gb_calibration"

# Original Full-Head baseline: retain the full 500-sequence paper-style calibration.
python bench/fit_ridge_calibration.py \
  --target Qwen/Qwen3-4B --draft Qwen/Qwen3-1.7B \
  --head-mode full --fit-profile 24gb \
  --sequences 500 --selection-sequences 500 \
  --seq-len 1024 --stride 4 --k 8 --lambda 0.01 \
  --layer-selection r2 \
  --calibration-dir "$CALIB" \
  --kvbridge-artifact artifacts/e0_qwen3_4b_to_1p7b_full_24gb \
  --output "$FULL" --overwrite-artifact

# Same calibration shards and R² regime, but matched-head affine support.
python bench/fit_ridge_calibration.py \
  --target Qwen/Qwen3-4B --draft Qwen/Qwen3-1.7B \
  --head-mode matched --fit-profile 24gb \
  --sequences 500 --selection-sequences 64 \
  --seq-len 1024 --stride 4 --k 8 --lambda 0.01 \
  --layer-selection r2 \
  --calibration-dir "$CALIB" \
  --output "$MATCHED"

for NAME in full matched; do
  if [[ "$NAME" == "full" ]]; then MAPPER="$FULL"; else MAPPER="$MATCHED"; fi

  python bench/benchmark_prefill_bridge.py \
    --mapper "$MAPPER" --memory-profile 24gb \
    --lengths 512,1024,2048,4096,8192,16384 \
    --warmup 20 --repetitions 100 --continue-on-oom \
    --mapper-dtype bfloat16 \
    --output "results/e1_${NAME}_24gb.json"

  python bench/eval_acceptance_pilot.py \
    --mapper "$MAPPER" --memory-profile 24gb \
    --text-file "$EVAL_TEXT" --bootstrap-samples 10000 \
    --mapper-dtype bfloat16 \
    --output "results/e2_${NAME}_24gb.json"
done
