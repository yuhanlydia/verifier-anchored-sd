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
CONTEXT_PROMPTS="${CONTEXT_PROMPTS:-32}"
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
  local effective_prompts="$PROMPTS"

  if [[ -f "$result" && -f "$mapper" && -f "$mapper.json" ]] && "$PYTHON" - \
    "$result" "$mapper" "$mapper.json" "$target" "$target_revision" \
    "$draft" "$draft_revision" "$CALIBRATION_TEXT" "$EVAL_TEXT" \
    "$SEQUENCES" "$SEQ_LEN" "$STRIDE" "$SELECTION_SEQUENCES" \
    "$PROMPTS" "$PREFIX_TOKENS" "$BOOTSTRAP_SAMPLES" "$CONTEXT_PROMPTS" \
    "$GPU_MEMORY_GIB" <<'PY'
import json
import sys
from pathlib import Path

from verifier_anchored_sd.experiment_artifacts import sha256_file

(
    result_path, mapper_path, metadata_path, target, target_revision,
    draft, draft_revision, calibration_text, eval_text,
    sequences, seq_len, stride, selection_sequences,
    initial_prompts, prefix_tokens, bootstrap_samples, context_prompts,
    gpu_memory_gib,
) = sys.argv[1:]
result = json.loads(Path(result_path).read_text())
metadata = json.loads(Path(metadata_path).read_text())
protocol = result.get("protocol_contract", {})
actual_prompts = result.get("requested_rows")
allowed_prompts = {int(initial_prompts), 512}
checks = [
    result.get("completed_rows") == actual_prompts,
    actual_prompts in allowed_prompts,
    not (
        actual_prompts == int(initial_prompts)
        and int(initial_prompts) < 512
        and result.get("gate", {}).get("status") == "inconclusive"
    ),
    protocol.get("pair") == {"target": target, "draft": draft},
    protocol.get("target_revision") == target_revision,
    protocol.get("draft_revision") == draft_revision,
    protocol.get("dtype") == "bfloat16",
    protocol.get("calibration_input_sha256") == sha256_file(calibration_text),
    protocol.get("evaluation_input_sha256") == sha256_file(eval_text),
    protocol.get("calibration_capture") == {
        "count": int(sequences), "seq_len": int(seq_len), "stride": int(stride)
    },
    protocol.get("mapper") == metadata.get("mapper"),
    result.get("source_model") == metadata.get("source_model"),
    result.get("draft_model") == metadata.get("draft_model"),
    result.get("mapper_metadata_sha256") == sha256_file(metadata_path),
    result.get("mapper_checkpoint_sha256") == sha256_file(mapper_path),
    metadata.get("mapper", {}).get("k") == 8,
    metadata.get("mapper", {}).get("lambda") == 0.01,
    metadata.get("mapper", {}).get("selection_ridge") == 0.000001,
    metadata.get("mapper", {}).get("selection_sequences") == int(selection_sequences),
    protocol.get("mapper_checkpoint_sha256") == sha256_file(mapper_path),
    protocol.get("screen_capture") == {
        "count": actual_prompts, "prefix_tokens": int(prefix_tokens), "stride": 1
    },
    protocol.get("prompts") == actual_prompts,
    protocol.get("bootstrap_samples") == int(bootstrap_samples),
    protocol.get("threshold") == 0.95,
    protocol.get("attention_cosine") is True,
    protocol.get("device") == "cuda",
    protocol.get("mapper_device") == "cuda",
    protocol.get("gpu_memory_gib") == int(gpu_memory_gib),
]
if result.get("gate", {}).get("status") == "pass":
    base = Path(result_path)
    for context_length in (2048, 8192):
        diagnostic_path = base.with_name(f"{base.stem}_context{context_length}.json")
        if not diagnostic_path.exists():
            checks.append(False)
            continue
        diagnostic = json.loads(diagnostic_path.read_text())
        diagnostic_protocol = diagnostic.get("protocol_contract", {})
        checks.extend([
            diagnostic.get("completed_rows") == int(context_prompts),
            diagnostic_protocol.get("screen_capture") == {
                "count": int(context_prompts),
                "prefix_tokens": context_length,
                "stride": 1,
            },
            all(
                diagnostic_protocol.get(key) == protocol.get(key)
                for key in (
                    "pair", "target_revision", "draft_revision", "dtype",
                    "calibration_input_sha256", "evaluation_input_sha256",
                    "calibration_capture", "mapper", "mapper_checkpoint_sha256",
                    "bootstrap_samples", "threshold", "attention_cosine",
                    "device", "mapper_device", "gpu_memory_gib",
                )
            ),
        ])
raise SystemExit(0 if all(checks) else 1)
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

  run_screen_stage() {
    local stage_screen="$1"
    local stage_result="$2"
    local stage_prompts="$3"
    local stage_prefix="$4"
    "$PYTHON" bench/capture_screen_prefixes.py \
      --screen-dir "$stage_screen" --mapper-metadata "$mapper.json" \
      --text-file "$EVAL_TEXT" --prompts "$stage_prompts" \
      --prefix-tokens "$stage_prefix" --device cuda --dtype bfloat16 \
      --gpu-memory-gib "$GPU_MEMORY_GIB"

    "$PYTHON" bench/eval_pair_transfer.py \
      --screen-dir "$stage_screen" --mapper "$mapper" --output "$stage_result" \
      --prompts "$stage_prompts" --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --threshold 0.95 --device cuda --mapper-device cuda --dtype bfloat16 \
      --gpu-memory-gib "$GPU_MEMORY_GIB" --attention-cosine

    if [[ "$PRUNE_COMPLETED_SHARDS" == "1" ]]; then
      "$PYTHON" - "$stage_screen" "$stage_prompts" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "shards"
count = int(sys.argv[2])
paths = sorted(root.glob("*.pt"))
if len(paths) != count or any(path.parent != root for path in paths):
    raise RuntimeError(f"refusing to prune unexpected screen shard set: {root}")
for path in paths:
    path.unlink()
root.rmdir()
PY
    fi
  }

  run_screen_stage "$screen_dir" "$result" "$PROMPTS" "$PREFIX_TOKENS"

  local gate_status
  gate_status="$($PYTHON - "$result" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["gate"]["status"])
PY
)"
  if [[ "$gate_status" == "inconclusive" && "$PROMPTS" -lt 512 ]]; then
    mv "$result" "$RESULT_ROOT/${name}_n${PROMPTS}.json"
    screen_dir="$ARTIFACT_ROOT/$name/screen_n512"
    effective_prompts=512
    run_screen_stage "$screen_dir" "$result" 512 "$PREFIX_TOKENS"
    gate_status="$($PYTHON - "$result" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["gate"]["status"])
PY
)"
  fi

  if [[ "$gate_status" == "pass" ]]; then
    for context_length in 2048 8192; do
      run_screen_stage \
        "$ARTIFACT_ROOT/$name/screen_context${context_length}" \
        "$RESULT_ROOT/${name}_context${context_length}.json" \
        "$CONTEXT_PROMPTS" "$context_length"
    done
  fi

  "$PYTHON" - "$pair_dir" "$screen_dir" "$mapper" "$result" \
    "$SEQUENCES" "$effective_prompts" "$PRUNE_COMPLETED_SHARDS" <<'PY'
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
    pair_dir / "tokens/manifest.json",
    pair_dir / "tokens.pt",
    mapper,
    Path(f"{mapper}.json"),
    screen_dir / "manifest.json",
    screen_dir / "complete.json",
    screen_dir / "tokens/manifest.json",
    screen_dir / "tokens.pt",
    result,
]
inventory = {
    "schema_version": 2,
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
