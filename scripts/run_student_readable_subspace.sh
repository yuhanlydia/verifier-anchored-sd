#!/usr/bin/env bash
set -euo pipefail

# Direct 32GB/48GB runner for the student-readable KV subspace kill test.
# The existing 8B->4B mapper is held fixed. No automatic quantization, dtype
# change, model substitution, rank reduction, or low-VRAM fallback is allowed.

: "${SUBSPACE_TEXT:?Set SUBSPACE_TEXT to a frozen basis-fit text/JSONL file}"
: "${EVAL_TEXT:?Set EVAL_TEXT to a disjoint frozen intervention-eval file}"

PYTHON="${PYTHON:-.venv/bin/python}"
MAPPER="${MAPPER:-artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt}"
MAPPER_METADATA="${MAPPER_METADATA:-${MAPPER}.json}"
GPU_MEMORY_GIB="${GPU_MEMORY_GIB:-28}"
METHOD_BATCH_SIZE="${METHOD_BATCH_SIZE:-8}"
SUBSPACE_PROMPTS="${SUBSPACE_PROMPTS:-64}"
SUBSPACE_PREFIX_TOKENS="${SUBSPACE_PREFIX_TOKENS:-512}"
EVAL_PROMPTS="${EVAL_PROMPTS:-128}"
EVAL_PREFIX_TOKENS="${EVAL_PREFIX_TOKENS:-1024}"
E2_PROMPTS="${E2_PROMPTS:-64}"
E2_PROMPT_TOKENS="${E2_PROMPT_TOKENS:-512}"
E2_NEW_TOKENS="${E2_NEW_TOKENS:-128}"
E2_GAMMA="${E2_GAMMA:-4}"
E2_LOW_VRAM="${E2_LOW_VRAM:-0}"
MAX_RANK="${MAX_RANK:-64}"
RANKS="${RANKS:-4,8,16,32,64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
GRADIENT_SMOKE_PREFIXES="${GRADIENT_SMOKE_PREFIXES:-4}"
PRUNE_KV_SHARDS="${PRUNE_KV_SHARDS:-1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/student_readable_subspace_2026-09-08}"
RESULT_ROOT="${RESULT_ROOT:-results/student_readable_subspace_2026-09-08}"

if [[ ! -f "$MAPPER" || ! -f "$MAPPER_METADATA" ]]; then
  cat >&2 <<EOF
The primary experiment must reuse the existing audited Qwen3-8B -> Qwen3-4B mapper.
Missing:
  $MAPPER
  $MAPPER_METADATA
Do not silently refit it for this experiment.
EOF
  exit 3
fi

for value in "$GPU_MEMORY_GIB" "$METHOD_BATCH_SIZE" "$SUBSPACE_PROMPTS" \
             "$SUBSPACE_PREFIX_TOKENS" "$EVAL_PROMPTS" "$EVAL_PREFIX_TOKENS" \
             "$E2_PROMPTS" "$E2_PROMPT_TOKENS" "$E2_NEW_TOKENS" "$E2_GAMMA" \
             "$MAX_RANK" "$BOOTSTRAP_SAMPLES" "$GRADIENT_SMOKE_PREFIXES"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    echo "numeric experiment settings must be positive integers (got $value)" >&2
    exit 2
  fi
done

if [[ "$PRUNE_KV_SHARDS" != "0" && "$PRUNE_KV_SHARDS" != "1" ]]; then
  echo "PRUNE_KV_SHARDS must be 0 or 1" >&2
  exit 2
fi
if [[ "$E2_LOW_VRAM" != "0" && "$E2_LOW_VRAM" != "1" ]]; then
  echo "E2_LOW_VRAM must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT" "$RESULT_ROOT/smoke"

# CPU/contract verification first. GPU integration is still a separate claim.
"$PYTHON" -m compileall -q src bench training
"$PYTHON" -m pytest -q

capture_screen() {
  local root="$1"
  local text="$2"
  local prompts="$3"
  local prefix_tokens="$4"
  "$PYTHON" bench/capture_screen_prefixes.py \
    --screen-dir "$root" \
    --mapper-metadata "$MAPPER_METADATA" \
    --text-file "$text" \
    --prompts "$prompts" \
    --prefix-tokens "$prefix_tokens" \
    --device cuda --dtype bfloat16 \
    --gpu-memory-gib "$GPU_MEMORY_GIB"
}

# -----------------------------------------------------------------------------
# S0: non-scientific 4-prefix gradient/integration smoke on two disjoint files.
# -----------------------------------------------------------------------------
SMOKE_FIT_DIR="$ARTIFACT_ROOT/smoke/fit_screen"
SMOKE_EVAL_DIR="$ARTIFACT_ROOT/smoke/eval_screen"
SMOKE_BASIS="$ARTIFACT_ROOT/smoke/basis.pt"
SMOKE_RESULT="$RESULT_ROOT/smoke/interventions.json"

capture_screen "$SMOKE_FIT_DIR" "$SUBSPACE_TEXT" 4 256
"$PYTHON" bench/fit_student_readable_subspace.py \
  --screen-dir "$SMOKE_FIT_DIR" \
  --mapper "$MAPPER" --mapper-metadata "$MAPPER_METADATA" \
  --output "$SMOKE_BASIS" \
  --prompts 4 --gradient-smoke-prefixes 4 --max-rank 8 \
  --device cuda --mapper-device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB"

capture_screen "$SMOKE_EVAL_DIR" "$EVAL_TEXT" 4 256
"$PYTHON" bench/eval_subspace_interventions.py \
  --screen-dir "$SMOKE_EVAL_DIR" \
  --mapper "$MAPPER" --mapper-metadata "$MAPPER_METADATA" \
  --subspace-artifact "$SMOKE_BASIS" \
  --output "$SMOKE_RESULT" \
  --prompts 4 --ranks 4,8 --method-batch-size "$METHOD_BATCH_SIZE" \
  --bootstrap-samples 200 --device cuda --mapper-device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB"

echo "S0 smoke complete; scientific=false"

# -----------------------------------------------------------------------------
# S1: scientific basis fit on B only.
# -----------------------------------------------------------------------------
FIT_DIR="$ARTIFACT_ROOT/fit_screen_n${SUBSPACE_PROMPTS}_l${SUBSPACE_PREFIX_TOKENS}"
BASIS="$ARTIFACT_ROOT/student_readable_basis.pt"
capture_screen "$FIT_DIR" "$SUBSPACE_TEXT" "$SUBSPACE_PROMPTS" "$SUBSPACE_PREFIX_TOKENS"

"$PYTHON" bench/fit_student_readable_subspace.py \
  --screen-dir "$FIT_DIR" \
  --mapper "$MAPPER" --mapper-metadata "$MAPPER_METADATA" \
  --output "$BASIS" \
  --prompts "$SUBSPACE_PROMPTS" \
  --gradient-smoke-prefixes "$GRADIENT_SMOKE_PREFIXES" \
  --max-rank "$MAX_RANK" \
  --benefit-eigenvalue-tol 1e-10 --random-seed 0 \
  --device cuda --mapper-device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB"

# -----------------------------------------------------------------------------
# S2: scientific intervention matrix on disjoint C.
# -----------------------------------------------------------------------------
EVAL_DIR="$ARTIFACT_ROOT/eval_screen_n${EVAL_PROMPTS}_l${EVAL_PREFIX_TOKENS}"
RESULT="$RESULT_ROOT/qwen3_8b_to_4b_subspace_interventions.json"
capture_screen "$EVAL_DIR" "$EVAL_TEXT" "$EVAL_PROMPTS" "$EVAL_PREFIX_TOKENS"

"$PYTHON" bench/eval_subspace_interventions.py \
  --screen-dir "$EVAL_DIR" \
  --mapper "$MAPPER" --mapper-metadata "$MAPPER_METADATA" \
  --subspace-artifact "$BASIS" \
  --output "$RESULT" \
  --prompts "$EVAL_PROMPTS" --ranks "$RANKS" \
  --method-batch-size "$METHOD_BATCH_SIZE" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --device cuda --mapper-device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB"

"$PYTHON" - "$RESULT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "decision": r["decision"],
    "winner": r.get("winner"),
    "native": r["summary"]["native"],
    "full_mapped": r["summary"]["full_mapped"],
    "method_count": r["method_count"],
}, indent=2, sort_keys=True))
PY

# Provenance inventory: retain small probability/basis/result artifacts; KV shards
# are reconstructible from frozen text + exact model revisions and may be pruned.
"$PYTHON" - "$MAPPER" "$MAPPER_METADATA" "$BASIS" "$FIT_DIR" "$EVAL_DIR" "$RESULT" <<'PY'
import json, sys
from pathlib import Path
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file

mapper, metadata, basis, fit_dir, eval_dir, result = map(Path, sys.argv[1:])
files = [
    mapper,
    metadata,
    basis,
    Path(f"{basis}.json"),
    fit_dir / "manifest.json",
    fit_dir / "tokens.pt",
    eval_dir / "manifest.json",
    eval_dir / "tokens.pt",
    result,
]
for root in (fit_dir / "target_probs", eval_dir / "target_probs"):
    files.extend(sorted(root.glob("*.pt")))
inventory = {
    "schema_version": 1,
    "files": {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    },
}
atomic_write_json(result.parent / "artifact_inventory.json", inventory)
PY

DECISION="$($PYTHON - "$RESULT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"]["status"])
PY
)"

case "$DECISION" in
  go_e2)
    echo "GO_E2: a deployment-valid subspace winner beat BOTH native and full_mapped."
    echo "Winner is frozen in: $RESULT"
    if [[ -n "${E2_TEXT:-}" ]]; then
      E2_OUTPUT="$RESULT_ROOT/qwen3_8b_to_4b_subspace_block_acceptance.json"
      LOW_VRAM_ARGS=()
      if [[ "$E2_LOW_VRAM" == "1" ]]; then
        LOW_VRAM_ARGS+=(--low-vram)
      fi
      "$PYTHON" bench/eval_subspace_acceptance.py \
        --winner-result "$RESULT" \
        --selection-screen-dir "$EVAL_DIR" \
        --mapper "$MAPPER" --mapper-metadata "$MAPPER_METADATA" \
        --subspace-artifact "$BASIS" \
        --text-file "$E2_TEXT" --output "$E2_OUTPUT" \
        --prompts "$E2_PROMPTS" --prompt-tokens "$E2_PROMPT_TOKENS" \
        --new-tokens "$E2_NEW_TOKENS" --gamma "$E2_GAMMA" \
        --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
        --device cuda --dtype bfloat16 "${LOW_VRAM_ARGS[@]}"
      "$PYTHON" - "$E2_OUTPUT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"gates": r["gates"], "summary": r["summary"]}, indent=2, sort_keys=True))
PY
    else
      cat >&2 <<EOF
One-step subspace winner is frozen, but E2_TEXT is not set.
Use a THIRD frozen data source D and rerun with:
  export E2_TEXT=/path/to/disjoint_block_eval.jsonl
The runner will verify A/B/C/D token-row disjointness before block evaluation.
EOF
    fi
    ;;
  no_deployment_winner)
    cat >&2 <<'EOF'
STOP before E2: no mapped-only subspace intervention beat BOTH native 4B and full-mapped 8B->4B in target overlap with paired 95% CI low > 0.
Do not select a delta upper bound as a deployment method. Inspect whether benefit+/grad beat random/PCA to decide mechanism vs generic compression.
EOF
    ;;
  *)
    echo "Unknown intervention decision: $DECISION" >&2
    exit 5
    ;;
esac

# Prune reconstructible large KV shards only after the optional D run has recovered
# C's exact row provenance from the retained manifest.
if [[ "$PRUNE_KV_SHARDS" == "1" ]]; then
  "$PYTHON" - "$FIT_DIR" "$EVAL_DIR" <<'PY'
import shutil, sys
from pathlib import Path
for root in map(Path, sys.argv[1:]):
    shard_root = root / "shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
PY
fi
