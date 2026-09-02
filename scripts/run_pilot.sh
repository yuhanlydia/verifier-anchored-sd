#!/usr/bin/env bash
set -euo pipefail

# Edit these paths once on the execution host.  The script does not download or
# vendor benchmark data into git.
PROMPTS="${PROMPTS:?set PROMPTS to held-out JSONL/raw prompt file}"
MAPPER="${MAPPER:-checkpoints/qwen3_4b_to_1p7b_ridge.pt}"

python bench/fit_ridge_calibration.py --output "$MAPPER"
python bench/benchmark_prefill_bridge.py --mapper "$MAPPER"
python bench/eval_acceptance_pilot.py --mapper "$MAPPER" --text-file "$PROMPTS"
python bench/eval_generation_drift.py --mapper "$MAPPER" --text-file "$PROMPTS"

