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
PRUNE_COMPLETED_SHARDS="${PRUNE_COMPLETED_SHARDS:-1}"

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

  if [[ -f "$result" ]] && "$PYTHON" - "$result" "$target" "$draft" "$PROMPTS" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
expected_pair = {"target": sys.argv[2], "draft": sys.argv[3]}
complete = result.get("completed_rows") == int(sys.argv[4])
raise SystemExit(0 if complete and result.get("pair") == expected_pair else 1)
PY
  then
    echo "Reusing completed pair-screen result: $result"
    return
  fi

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

  "$PYTHON" - "$pair_dir" "$screen_dir" "$mapper" "$result" \
    "$SEQUENCES" "$PROMPTS" "$PRUNE_COMPLETED_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file

pair_dir, screen_dir, mapper, result = map(Path, sys.argv[1:5])
sequences, prompts = map(int, sys.argv[5:7])
prune = sys.argv[7] == "1"
files = [
    pair_dir / "source/manifest.json",
    pair_dir / "source/complete.json",
    pair_dir / "draft/manifest.json",
    pair_dir / "draft/complete.json",
    pair_dir / "tokens.pt",
    mapper,
    Path(f"{mapper}.json"),
    screen_dir / "manifest.json",
    screen_dir / "complete.json",
    screen_dir / "tokens.pt",
    result,
]
inventory = {
    "schema_version": 1,
    "files": {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    },
}
inventory_path = mapper.parent / "artifact_inventory.json"
atomic_write_json(inventory_path, inventory)
if prune:
    for root, count in (
        (pair_dir / "source/shards", sequences),
        (pair_dir / "draft/shards", sequences),
        (screen_dir / "shards", prompts),
    ):
        paths = sorted(root.glob("*.pt"))
        if len(paths) != count or any(path.parent != root for path in paths):
            raise RuntimeError(f"refusing to prune unexpected shard set: {root}")
        for path in paths:
            path.unlink()
        root.rmdir()
    print(f"Recorded {inventory_path} and pruned completed reconstructible shards")
PY
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
    print(path.name, result["summary"]["mean_a_transfer"], result["gate"])
PY
