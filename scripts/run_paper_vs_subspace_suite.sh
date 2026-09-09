#!/usr/bin/env bash
set -euo pipefail

# Five-split causal suite:
# A mapper fit -> B mapper/k selection -> C subspace fit -> D fair final comparison -> E SD.

: "${CALIBRATION_TEXT:?Set CALIBRATION_TEXT (split A)}"
: "${MAPPER_EVAL_TEXT:?Set MAPPER_EVAL_TEXT (split B)}"
: "${SUBSPACE_TEXT:?Set SUBSPACE_TEXT (split C)}"
: "${EVAL_TEXT:?Set EVAL_TEXT (split D)}"

PYTHON="${PYTHON:-.venv/bin/python}"
SUITE_PROFILE="${SUITE_PROFILE:-pilot}"
GPU_MEMORY_GIB="${GPU_MEMORY_GIB:-28}"
METHOD_BATCH_SIZE="${METHOD_BATCH_SIZE:-8}"
FULL_KS="${FULL_KS:-8,12,16,20}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/paper_vs_subspace_2026-09-09}"
RESULT_ROOT="${RESULT_ROOT:-results/paper_vs_subspace_2026-09-09}"
TARGET="Qwen/Qwen3-8B"
DRAFT="Qwen/Qwen3-4B"
TARGET_REV="b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_REV="1cfa9a7208912126459214e8b04321603b3df60c"

case "$SUITE_PROFILE" in
  pilot)
    CALIBRATION_SEQUENCES="${CALIBRATION_SEQUENCES:-128}"
    SELECTION_SEQUENCES="${SELECTION_SEQUENCES:-32}"
    MLP_PROFILE="${MLP_PROFILE:-pilot}"
    MLP_SEQUENCES="${MLP_SEQUENCES:-32}"
    ;;
  paper)
    CALIBRATION_SEQUENCES="${CALIBRATION_SEQUENCES:-500}"
    SELECTION_SEQUENCES="${SELECTION_SEQUENCES:-64}"
    MLP_PROFILE="${MLP_PROFILE:-paper}"
    MLP_SEQUENCES="${MLP_SEQUENCES:-500}"
    ;;
  *)
    echo "SUITE_PROFILE must be pilot or paper" >&2
    exit 2
    ;;
esac

CALIBRATION_TOKENS="${CALIBRATION_TOKENS:-1024}"
CALIBRATION_STRIDE="${CALIBRATION_STRIDE:-4}"
MAPPER_EVAL_PROMPTS="${MAPPER_EVAL_PROMPTS:-128}"
MAPPER_EVAL_TOKENS="${MAPPER_EVAL_TOKENS:-1024}"
SUBSPACE_PROMPTS="${SUBSPACE_PROMPTS:-64}"
SUBSPACE_PREFIX_TOKENS="${SUBSPACE_PREFIX_TOKENS:-512}"
SUBSPACE_EVAL_PROMPTS="${SUBSPACE_EVAL_PROMPTS:-128}"
SUBSPACE_EVAL_TOKENS="${SUBSPACE_EVAL_TOKENS:-1024}"

mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT"

# Fail before GPU if the user accidentally aliases scientific splits.
"$PYTHON" - "$CALIBRATION_TEXT" "$MAPPER_EVAL_TEXT" "$SUBSPACE_TEXT" "$EVAL_TEXT" "${E2_TEXT:-}" <<'PY'
import sys
from pathlib import Path
from verifier_anchored_sd.experiment_artifacts import sha256_file
paths = [Path(x) for x in sys.argv[1:] if x]
missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen input files: {missing}")
hashes = [sha256_file(path) for path in paths]
if len(hashes) != len(set(hashes)):
    raise SystemExit("A/B/C/D/E frozen input files must have distinct SHA256 digests")
print("input split hashes are distinct")
PY

"$PYTHON" -m compileall -q src bench training
"$PYTHON" -m pytest -q

PAIR_DIR="$ARTIFACT_ROOT/calibration"
# A: exact-weight sequential capture once, reused by every ridge/MLP baseline.
for ROLE in source draft; do
  "$PYTHON" bench/capture_sequential_calibration.py \
    --pair-dir "$PAIR_DIR" --role "$ROLE" \
    --target "$TARGET" --draft "$DRAFT" \
    --target-revision "$TARGET_REV" --draft-revision "$DRAFT_REV" \
    --text-file "$CALIBRATION_TEXT" \
    --sequences "$CALIBRATION_SEQUENCES" --seq-len "$CALIBRATION_TOKENS" \
    --stride "$CALIBRATION_STRIDE" --device cuda --dtype bfloat16 \
    --gpu-memory-gib "$GPU_MEMORY_GIB"
done

IFS=',' read -r -a K_VALUES <<< "$FULL_KS"
if [[ "${#K_VALUES[@]}" -eq 0 ]]; then
  echo "FULL_KS must contain at least one k" >&2
  exit 2
fi

MATCHED_DIR="$ARTIFACT_ROOT/matched_ridge"
FULL_DIR="$ARTIFACT_ROOT/full_head_ridge"
mkdir -p "$MATCHED_DIR" "$FULL_DIR"

# Fit both support patterns with identical A, k, lambda and selection criterion.
for K in "${K_VALUES[@]}"; do
  if ! [[ "$K" =~ ^[0-9]+$ ]] || [[ "$K" -le 0 ]]; then
    echo "invalid k in FULL_KS: $K" >&2
    exit 2
  fi
  "$PYTHON" bench/fit_sequential_mapper.py \
    --pair-dir "$PAIR_DIR" --output "$MATCHED_DIR/k${K}.pt" \
    --k "$K" --lambda 0.01 --selection-ridge 0.000001 \
    --selection-sequences "$SELECTION_SEQUENCES" \
    --accumulation-device cuda --selection-layer-block 1 --fit-layer-block 4

  "$PYTHON" bench/fit_paper_fullhead_from_sequential.py \
    --pair-dir "$PAIR_DIR" --output "$FULL_DIR/k${K}.pt" \
    --kvbridge-artifact "$FULL_DIR/k${K}_kvbridge" \
    --k "$K" --lambda 0.01 --selection-ridge 0.000001 \
    --accumulation-device cuda --accumulation-dtype float32 \
    --target-layer-block 1 --selection-layer-block 1 --storage-dtype bfloat16
done

# B: capture exact verifier state/probabilities once. Any full-head metadata has the
# same pair/revisions/calibration rows, so use the first k only as capture contract.
FIRST_K="${K_VALUES[0]}"
MAPPER_SCREEN="$ARTIFACT_ROOT/mapper_eval_screen"
"$PYTHON" bench/capture_screen_prefixes.py \
  --screen-dir "$MAPPER_SCREEN" --mapper-metadata "$FULL_DIR/k${FIRST_K}.pt.json" \
  --text-file "$MAPPER_EVAL_TEXT" --prompts "$MAPPER_EVAL_PROMPTS" \
  --prefix-tokens "$MAPPER_EVAL_TOKENS" --device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB"

MAPPER_RESULT_DIR="$RESULT_ROOT/mapper_selection_B"
mkdir -p "$MAPPER_RESULT_DIR"
for K in "${K_VALUES[@]}"; do
  for SUPPORT in matched full; do
    if [[ "$SUPPORT" == "matched" ]]; then
      MAP="$MATCHED_DIR/k${K}.pt"
      NAME="matched_k${K}"
    else
      MAP="$FULL_DIR/k${K}.pt"
      NAME="full_k${K}"
    fi
    "$PYTHON" bench/eval_mapper_alignment.py \
      --screen-dir "$MAPPER_SCREEN" --mapper "$MAP" --mapper-metadata "$MAP.json" \
      --output "$MAPPER_RESULT_DIR/${NAME}.json" \
      --prompts "$MAPPER_EVAL_PROMPTS" --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --device cuda --mapper-device cuda --dtype bfloat16 \
      --gpu-memory-gib "$GPU_MEMORY_GIB"
  done
done

SELECTION_JSON="$RESULT_ROOT/linear_mapper_selection.json"
"$PYTHON" - "$MAPPER_RESULT_DIR" "$MATCHED_DIR" "$FULL_DIR" "$SELECTION_JSON" <<'PY'
import json, re, sys
from pathlib import Path
from verifier_anchored_sd.experiment_artifacts import atomic_write_json
result_root, matched_root, full_root, output = map(Path, sys.argv[1:])
rows = []
for path in sorted(result_root.glob("*.json")):
    value = json.loads(path.read_text())
    summary = value["target_alignment"]["summary"]
    match = re.fullmatch(r"(matched|full)_k(\d+)", path.stem)
    if match is None:
        continue
    support, k = match.group(1), int(match.group(2))
    mapper_root = matched_root if support == "matched" else full_root
    rows.append({
        "support": support,
        "k": k,
        "result": str(path),
        "mapper": str(mapper_root / f"k{k}.pt"),
        "mapper_metadata": str(mapper_root / f"k{k}.pt.json"),
        "a_target": float(summary["mean_a_target_mapped"]),
        "delta_target": float(summary["mean_delta_target_alignment"]),
        "kl_target": float(summary.get("mean_kl_target_mapped", float("inf"))),
        "top1_target": float(summary.get("top1_target_mapped", float("nan"))),
    })
if not rows:
    raise SystemExit("no completed linear mapper results")
key = lambda row: (row["a_target"], -row["kl_target"], -row["k"])
best_linear = max(rows, key=key)
full_rows = [row for row in rows if row["support"] == "full"]
if not full_rows:
    raise SystemExit("paper MLP requires at least one full-head ridge result")
best_full = max(full_rows, key=key)
payload = {
    "schema_version": 1,
    "selection_split": "B",
    "selection_metric": "max mean verifier-proposal overlap; tie lower target KL; tie lower k",
    "rows": rows,
    "best_linear": best_linear,
    "best_full_head": best_full,
}
atomic_write_json(output, payload)
print(json.dumps(payload, indent=2))
PY

readarray -t SELECTED < <("$PYTHON" - "$SELECTION_JSON" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
for key in ("best_linear", "best_full_head"):
    print(r[key]["mapper"])
    print(r[key]["mapper_metadata"])
PY
)
BEST_LINEAR="${SELECTED[0]}"
BEST_LINEAR_META="${SELECTED[1]}"
BEST_FULL="${SELECTED[2]}"
BEST_FULL_META="${SELECTED[3]}"

# Train the paper nonlinear baseline only on the selected full-head k.
MLP_DIR="$ARTIFACT_ROOT/paper_mlp"
mkdir -p "$MLP_DIR"
MLP="$MLP_DIR/selected.pt"
"$PYTHON" bench/fit_paper_mlp_from_sequential.py \
  --pair-dir "$PAIR_DIR" --ridge-metadata "$BEST_FULL_META" \
  --output "$MLP" --profile "$MLP_PROFILE" --sequences "$MLP_SEQUENCES" \
  --device cuda --storage-dtype bfloat16 --seed 0

"$PYTHON" bench/eval_mapper_alignment.py \
  --screen-dir "$MAPPER_SCREEN" --mapper "$MLP" --mapper-metadata "$MLP.json" \
  --output "$MAPPER_RESULT_DIR/paper_mlp_selected_k.json" \
  --prompts "$MAPPER_EVAL_PROMPTS" --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --device cuda --mapper-device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB"

# C/D/E: reuse the already-audited subspace runner, but on the strongest LINEAR
# mapper selected on B. Keep D KV shards until MLP/full-head are evaluated on D too.
SUBSPACE_ARTIFACT_ROOT="$ARTIFACT_ROOT/subspace"
SUBSPACE_RESULT_ROOT="$RESULT_ROOT/subspace"
export MAPPER="$BEST_LINEAR"
export MAPPER_METADATA="$BEST_LINEAR_META"
export SUBSPACE_TEXT EVAL_TEXT
export GPU_MEMORY_GIB METHOD_BATCH_SIZE BOOTSTRAP_SAMPLES SUBSPACE_PROMPTS
export SUBSPACE_PREFIX_TOKENS
export EVAL_PROMPTS="$SUBSPACE_EVAL_PROMPTS"
export EVAL_PREFIX_TOKENS="$SUBSPACE_EVAL_TOKENS"
export ARTIFACT_ROOT="$SUBSPACE_ARTIFACT_ROOT"
export RESULT_ROOT="$SUBSPACE_RESULT_ROOT"
export PRUNE_KV_SHARDS=0
if [[ -n "${E2_TEXT:-}" ]]; then export E2_TEXT; fi
bash scripts/run_student_readable_subspace.sh

D_SCREEN="$SUBSPACE_ARTIFACT_ROOT/eval_screen_n${SUBSPACE_EVAL_PROMPTS}_l${SUBSPACE_EVAL_TOKENS}"
SUBSPACE_RESULT="$SUBSPACE_RESULT_ROOT/qwen3_8b_to_4b_subspace_interventions.json"
FINAL_DIR="$RESULT_ROOT/final_D"
mkdir -p "$FINAL_DIR"

# D is the fair final comparison: paper full-head, paper MLP and subspace winner all
# see exactly the same held-out verifier artifacts.
"$PYTHON" bench/eval_mapper_alignment.py \
  --screen-dir "$D_SCREEN" --mapper "$BEST_FULL" --mapper-metadata "$BEST_FULL_META" \
  --output "$FINAL_DIR/paper_full_head.json" --prompts "$SUBSPACE_EVAL_PROMPTS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" --device cuda --mapper-device cuda \
  --dtype bfloat16 --gpu-memory-gib "$GPU_MEMORY_GIB"

"$PYTHON" bench/eval_mapper_alignment.py \
  --screen-dir "$D_SCREEN" --mapper "$MLP" --mapper-metadata "$MLP.json" \
  --output "$FINAL_DIR/paper_mlp.json" --prompts "$SUBSPACE_EVAL_PROMPTS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" --device cuda --mapper-device cuda \
  --dtype bfloat16 --gpu-memory-gib "$GPU_MEMORY_GIB"

FINAL_SUMMARY="$RESULT_ROOT/paper_vs_subspace_final.json"
"$PYTHON" - "$SUBSPACE_RESULT" "$FINAL_DIR/paper_full_head.json" \
  "$FINAL_DIR/paper_mlp.json" "$FINAL_SUMMARY" "$BOOTSTRAP_SAMPLES" <<'PY'
import json, sys
from pathlib import Path
from verifier_anchored_sd.evaluation import paired_bootstrap_mean_difference
from verifier_anchored_sd.experiment_artifacts import atomic_write_json
sub_path, full_path, mlp_path, output = map(Path, sys.argv[1:5])
samples = int(sys.argv[5])
sub, full, mlp = (json.loads(path.read_text()) for path in (sub_path, full_path, mlp_path))
winner = sub.get("winner")
summary = {
    "schema_version": 1,
    "split": "D",
    "subspace_decision": sub.get("decision"),
    "subspace_winner": winner,
    "paper_full_head": full["target_alignment"]["summary"],
    "paper_mlp": mlp["target_alignment"]["summary"],
    "comparisons": {},
}

def generic_rows(result):
    return {int(r["prompt"]): r for r in result["rows"]}
full_rows, mlp_rows = generic_rows(full), generic_rows(mlp)
prompts = sorted(set(full_rows) & set(mlp_rows))
clusters = [int(mlp_rows[p]["document_id"]) for p in prompts]
summary["comparisons"]["mlp_minus_full_head_a_target"] = paired_bootstrap_mean_difference(
    [float(mlp_rows[p]["a_target_mapped"]) for p in prompts],
    [float(full_rows[p]["a_target_mapped"]) for p in prompts],
    samples=samples, seed=10, cluster_ids=clusters,
)
if winner is not None:
    method = winner["method"] if isinstance(winner, dict) else str(winner)
    sub_rows = {int(r["prompt"]): r for r in sub["rows"] if r["method"] == method}
    prompts2 = sorted(set(sub_rows) & set(mlp_rows) & set(full_rows))
    if len(prompts2) != int(sub["requested_prompts"]):
        raise RuntimeError("final D comparison does not have complete paired winner rows")
    clusters2 = [int(sub_rows[p]["document_id"]) for p in prompts2]
    summary["comparisons"]["subspace_minus_mlp_a_target"] = paired_bootstrap_mean_difference(
        [float(sub_rows[p]["a_target"]) for p in prompts2],
        [float(mlp_rows[p]["a_target_mapped"]) for p in prompts2],
        samples=samples, seed=11, cluster_ids=clusters2,
    )
    summary["comparisons"]["subspace_minus_full_head_a_target"] = paired_bootstrap_mean_difference(
        [float(sub_rows[p]["a_target"]) for p in prompts2],
        [float(full_rows[p]["a_target_mapped"]) for p in prompts2],
        samples=samples, seed=12, cluster_ids=clusters2,
    )
atomic_write_json(output, summary)
print(json.dumps(summary, indent=2))
PY

# Keep all small provenance and probability artifacts. Large A/C/D KV shards are
# reconstructible and may be removed manually after the result inventory is saved.
echo "Suite complete. Final decision artifact: $FINAL_SUMMARY"
