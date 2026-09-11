#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
RESULT_ROOT="results/paper_vs_subspace_2026-09-09"
ARTIFACT_ROOT="artifacts/paper_vs_subspace_2026-09-09"
SUBSPACE_RESULT="$RESULT_ROOT/subspace/qwen3_8b_to_4b_subspace_interventions.json"
ACCEPTANCE_RESULT="$RESULT_ROOT/subspace/qwen3_8b_to_4b_subspace_block_acceptance.json"
D_SCREEN="$ARTIFACT_ROOT/subspace/eval_screen_n128_l1024"
BEST_FULL="$ARTIFACT_ROOT/full_head_ridge/k8.pt"
BEST_FULL_META="$BEST_FULL.json"
MLP="$ARTIFACT_ROOT/paper_mlp/selected.pt"
FINAL_DIR="$RESULT_ROOT/final_D"
BOOTSTRAP_SAMPLES=10000

for required in "$SUBSPACE_RESULT" "$D_SCREEN/manifest.json" "$BEST_FULL" \
  "$BEST_FULL_META" "$MLP" "$MLP.json" \
  "data/paper_vs_subspace_2026-09-11/E.jsonl"; do
  test -e "$required" || { echo "Missing required artifact: $required" >&2; exit 2; }
done

# Exact BF16 weights are retained. CPU offload only changes model residency so the
# frozen 8B + 4B pair and the ridge mapper fit on a 24 GiB GPU.
"$PYTHON" bench/eval_subspace_acceptance.py \
  --winner-result "$SUBSPACE_RESULT" \
  --selection-screen-dir "$D_SCREEN" \
  --mapper "$BEST_FULL" --mapper-metadata "$BEST_FULL_META" \
  --subspace-artifact "$ARTIFACT_ROOT/subspace/student_readable_basis.pt" \
  --text-file data/paper_vs_subspace_2026-09-11/E.jsonl \
  --output "$ACCEPTANCE_RESULT" \
  --prompts 64 --prompt-tokens 512 --new-tokens 128 --gamma 4 \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" --device cuda --dtype bfloat16 --low-vram \
  --target-gpu-memory-gib 12 --draft-gpu-memory-gib 8

mkdir -p "$FINAL_DIR"
"$PYTHON" bench/eval_mapper_alignment.py \
  --screen-dir "$D_SCREEN" --mapper "$BEST_FULL" --mapper-metadata "$BEST_FULL_META" \
  --output "$FINAL_DIR/paper_full_head.json" --prompts 128 \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" --device cuda --mapper-device cuda \
  --dtype bfloat16 --gpu-memory-gib 20

"$PYTHON" bench/eval_mapper_alignment.py \
  --screen-dir "$D_SCREEN" --mapper "$MLP" --mapper-metadata "$MLP.json" \
  --output "$FINAL_DIR/paper_mlp.json" --prompts 128 \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" --device cuda --mapper-device cuda \
  --dtype bfloat16 --gpu-memory-gib 20

"$PYTHON" - "$SUBSPACE_RESULT" "$FINAL_DIR/paper_full_head.json" \
  "$FINAL_DIR/paper_mlp.json" "$RESULT_ROOT/paper_vs_subspace_final.json" \
  "$BOOTSTRAP_SAMPLES" <<'PY'
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
    return {int(row["prompt"]): row for row in result["rows"]}

full_rows, mlp_rows = generic_rows(full), generic_rows(mlp)
prompts = sorted(set(full_rows) & set(mlp_rows))
if len(prompts) != 128:
    raise RuntimeError("final D paper baselines do not have 128 complete paired rows")
clusters = [int(mlp_rows[p]["document_id"]) for p in prompts]
summary["comparisons"]["mlp_minus_full_head_a_target"] = paired_bootstrap_mean_difference(
    [float(mlp_rows[p]["a_target_mapped"]) for p in prompts],
    [float(full_rows[p]["a_target_mapped"]) for p in prompts],
    samples=samples, seed=10, cluster_ids=clusters,
)
if winner is not None:
    method = winner["method"] if isinstance(winner, dict) else str(winner)
    sub_rows = {int(row["prompt"]): row for row in sub["rows"] if row["method"] == method}
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

echo "Resume complete: E acceptance and final D comparisons finished."
