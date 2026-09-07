#!/usr/bin/env bash
set -euo pipefail

: "${CALIBRATION_TEXT:?Set CALIBRATION_TEXT to frozen calibration JSONL/raw text}"
: "${EVAL_TEXT:?Set EVAL_TEXT to a disjoint frozen evaluation JSONL/raw text}"

PYTHON="${PYTHON:-.venv/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/pair_screen_2026-09-07}"
RESULT_ROOT="${RESULT_ROOT:-results/pair_screen_2026-09-07}"
GPU_MEMORY_GIB="${GPU_MEMORY_GIB:-14}"
SEQUENCES="${SEQUENCES:-128}"
SELECTION_SEQUENCES="${SELECTION_SEQUENCES:-32}"
PROMPTS="${PROMPTS:-128}"
SEQ_LEN="${SEQ_LEN:-1024}"
PREFIX_TOKENS="${PREFIX_TOKENS:-1024}"
STRIDE="${STRIDE:-4}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT"

run_candidate() {
  local name="$1"
  local target="$2"
  local target_revision="$3"
  local draft="$4"
  local draft_revision="$5"
  local pair_dir="$ARTIFACT_ROOT/$name/calibration"
  local mapper="$ARTIFACT_ROOT/$name/mapper.pt"
  local screen_dir="$ARTIFACT_ROOT/$name/screen"
  local result="$RESULT_ROOT/$name.json"

  "$PYTHON" bench/capture_sequential_calibration.py \
    --pair-dir "$pair_dir" --role source \
    --target "$target" --draft "$draft" \
    --target-revision "$target_revision" --draft-revision "$draft_revision" \
    --text-file "$CALIBRATION_TEXT" --sequences "$SEQUENCES" \
    --seq-len "$SEQ_LEN" --stride "$STRIDE" \
    --device cuda --dtype bfloat16 --gpu-memory-gib "$GPU_MEMORY_GIB"

  "$PYTHON" bench/capture_sequential_calibration.py \
    --pair-dir "$pair_dir" --role draft \
    --target "$target" --draft "$draft" \
    --target-revision "$target_revision" --draft-revision "$draft_revision" \
    --text-file "$CALIBRATION_TEXT" --sequences "$SEQUENCES" \
    --seq-len "$SEQ_LEN" --stride "$STRIDE" \
    --device cuda --dtype bfloat16 --gpu-memory-gib "$GPU_MEMORY_GIB"

  "$PYTHON" bench/fit_sequential_mapper.py \
    --pair-dir "$pair_dir" --output "$mapper" \
    --k 8 --lambda 0.01 --selection-ridge 0.000001 \
    --selection-sequences "$SELECTION_SEQUENCES" \
    --accumulation-device cuda --selection-layer-block 1 --fit-layer-block 4

  "$PYTHON" bench/capture_screen_prefixes.py \
    --screen-dir "$screen_dir" --mapper-metadata "$mapper.json" \
    --text-file "$EVAL_TEXT" --prompts "$PROMPTS" \
    --prefix-tokens "$PREFIX_TOKENS" --device cuda --dtype bfloat16 \
    --gpu-memory-gib "$GPU_MEMORY_GIB"

  "$PYTHON" bench/eval_pair_transfer.py \
    --screen-dir "$screen_dir" --mapper "$mapper" --output "$result" \
    --prompts "$PROMPTS" --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --threshold 0.95 --device cuda --mapper-device cuda --dtype bfloat16 \
    --gpu-memory-gib "$GPU_MEMORY_GIB" --attention-cosine
}

run_candidate \
  qwen3_8b_to_4b \
  Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 \
  Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c

run_candidate \
  qwen3_4b_to_1p7b \
  Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c \
  Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

"$PYTHON" - "$RESULT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("*.json")):
    result = json.loads(path.read_text())
    print(path.name, result["summary"]["a_transfer"], result["gate"])
PY
