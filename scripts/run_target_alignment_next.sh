#!/usr/bin/env bash
set -euo pipefail

# Stage A: re-evaluate the EXISTING Qwen3-8B -> Qwen3-4B mapper against the
# verifier distribution.  This script intentionally does not refit the mapper.

: "${EVAL_TEXT:?Set EVAL_TEXT to the frozen held-out pair-screen text file}"

PYTHON="${PYTHON:-.venv/bin/python}"
MAPPER="${MAPPER:-artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt}"
MAPPER_METADATA="${MAPPER_METADATA:-${MAPPER}.json}"
GPU_MEMORY_GIB="${GPU_MEMORY_GIB:-28}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/target_alignment_2026-09-07/qwen3_8b_to_4b}"
RESULT_ROOT="${RESULT_ROOT:-results/target_alignment_2026-09-07}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
RUN_E2_ON_SUPPORT="${RUN_E2_ON_SUPPORT:-1}"
EXPAND_ON_INCONCLUSIVE="${EXPAND_ON_INCONCLUSIVE:-1}"

if [[ ! -f "$MAPPER" || ! -f "$MAPPER_METADATA" ]]; then
  cat >&2 <<EOF
Stage A requires the EXISTING audited 8B->4B mapper so the translator is held fixed.
Missing:
  mapper:          $MAPPER
  mapper metadata: $MAPPER_METADATA
Do not silently refit for this experiment.  Restore the prior artifact on this host,
or run scripts/run_32b_to_14b_pair_screen.sh as the next independent candidate.
EOF
  exit 3
fi

if [[ "$RUN_E2_ON_SUPPORT" != "0" && "$RUN_E2_ON_SUPPORT" != "1" ]]; then
  echo "RUN_E2_ON_SUPPORT must be 0 or 1" >&2
  exit 2
fi
if [[ "$EXPAND_ON_INCONCLUSIVE" != "0" && "$EXPAND_ON_INCONCLUSIVE" != "1" ]]; then
  echo "EXPAND_ON_INCONCLUSIVE must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT/smoke" "$RESULT_ROOT"

run_screen() {
  local screen_dir="$1"
  local output="$2"
  local prompts="$3"
  local prefix_tokens="$4"
  local bootstrap_samples="$5"

  "$PYTHON" bench/capture_screen_prefixes.py \
    --screen-dir "$screen_dir" \
    --mapper-metadata "$MAPPER_METADATA" \
    --text-file "$EVAL_TEXT" \
    --prompts "$prompts" --prefix-tokens "$prefix_tokens" \
    --device cuda --dtype bfloat16 --gpu-memory-gib "$GPU_MEMORY_GIB"

  "$PYTHON" bench/eval_pair_transfer.py \
    --screen-dir "$screen_dir" \
    --mapper "$MAPPER" --mapper-metadata "$MAPPER_METADATA" \
    --output "$output" \
    --prompts "$prompts" --bootstrap-samples "$bootstrap_samples" \
    --threshold 0.95 --device cuda --mapper-device cuda --dtype bfloat16 \
    --gpu-memory-gib "$GPU_MEMORY_GIB" --attention-cosine
}

# Non-scientific integration smoke.  It only verifies cache/probability binding and
# the three-distribution evaluator.  Never quote its gate as a paper result.
SMOKE_DIR="$ARTIFACT_ROOT/smoke/screen"
SMOKE_RESULT="$RESULT_ROOT/smoke/qwen3_8b_to_4b_target_alignment.json"
run_screen "$SMOKE_DIR" "$SMOKE_RESULT" 4 256 200

"$PYTHON" - "$SMOKE_RESULT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"native_fidelity", "target_alignment", "decision", "rows"}
missing = required - set(r)
if missing or r.get("completed_rows") != 4:
    raise SystemExit(f"target-alignment smoke incomplete or malformed: missing={sorted(missing)}")
print("Smoke completed; scientific=false")
PY

# Scientific Stage A: same audited mapper, new verifier-probability artifacts.
SCREEN_DIR="$ARTIFACT_ROOT/screen_n128_l1024"
RESULT="$RESULT_ROOT/qwen3_8b_to_4b_target_alignment.json"
run_screen "$SCREEN_DIR" "$RESULT" 128 1024 "$BOOTSTRAP_SAMPLES"

DECISION="$($PYTHON - "$RESULT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print(r["decision"]["status"])
print(json.dumps({
    "native_fidelity": r["native_fidelity"]["summary"],
    "target_alignment": r["target_alignment"],
    "decision": r["decision"],
}, indent=2), file=sys.stderr)
PY
)"

echo "Stage A decision: $DECISION"

if [[ "$DECISION" == "expand" && "$EXPAND_ON_INCONCLUSIVE" == "1" ]]; then
  SCREEN_DIR_512="$ARTIFACT_ROOT/screen_n512_l1024"
  RESULT_512="$RESULT_ROOT/qwen3_8b_to_4b_target_alignment_n512.json"
  run_screen "$SCREEN_DIR_512" "$RESULT_512" 512 1024 "$BOOTSTRAP_SAMPLES"
  RESULT="$RESULT_512"
  DECISION="$($PYTHON - "$RESULT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"]["status"])
PY
)"
  echo "Expanded Stage A decision: $DECISION"
fi

case "$DECISION" in
  go_sd)
    echo "GO: mapping significantly improves verifier alignment."
    if [[ "$RUN_E2_ON_SUPPORT" == "1" ]]; then
      if [[ -z "${E2_TEXT:-}" ]]; then
        cat >&2 <<EOF
Target alignment supports SD, but E2_TEXT is not set.
Use a THIRD frozen prompt source, disjoint from calibration and pair-selection data:
  export SCREEN_RESULT=$RESULT
  export MAPPER=$MAPPER
  export EVAL_TEXT=/path/to/disjoint_e2.jsonl
  export LOW_VRAM=0   # set 1 only if simultaneous model loading requires offload
  bash scripts/run_native_frontier_e2.sh
EOF
      else
        export SCREEN_RESULT="$RESULT"
        export MAPPER
        export MAPPER_METADATA
        export EVAL_TEXT="$E2_TEXT"
        export OUTPUT="${E2_OUTPUT:-$RESULT_ROOT/qwen3_8b_to_4b_native_frontier_e2.json}"
        bash scripts/run_native_frontier_e2.sh
      fi
    fi
    ;;
  stop_pair)
    cat >&2 <<'EOF'
STOP 8B->4B: mapped state significantly moves the draft away from the verifier.
Do not tune refresh, HellaSwag, or acceptance residuals on this pair.
Next permitted experiment: scripts/run_32b_to_14b_pair_screen.sh
EOF
    ;;
  expand)
    echo "INCONCLUSIVE after requested expansion. Do not run E2; add more held-out clusters." >&2
    ;;
  incomplete)
    echo "INCOMPLETE Stage A. Inspect failure/progress artifacts; do not reinterpret partial rows." >&2
    exit 4
    ;;
  *)
    echo "Unknown target-alignment decision: $DECISION" >&2
    exit 5
    ;;
esac
