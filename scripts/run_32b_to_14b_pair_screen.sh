#!/usr/bin/env bash
set -euo pipefail

# Stage B: independent Qwen3-32B verifier -> Qwen3-14B draft screen.
# Scientific rule: exact BF16 weights, sequential model loading, no automatic
# quantization, and target-alignment (not native reconstruction) decides eligibility.

: "${CALIBRATION_TEXT:?Set CALIBRATION_TEXT to frozen calibration JSONL/raw text}"
: "${EVAL_TEXT:?Set EVAL_TEXT to a disjoint frozen pair-selection JSONL/raw text}"

PYTHON="${PYTHON:-.venv/bin/python}"
GPU_MEMORY_GIB="${GPU_MEMORY_GIB:-44}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/target_alignment_2026-09-07/qwen3_32b_to_14b}"
RESULT_ROOT="${RESULT_ROOT:-results/target_alignment_2026-09-07}"
SEQUENCES="${SEQUENCES:-128}"
SELECTION_SEQUENCES="${SELECTION_SEQUENCES:-32}"
PROMPTS="${PROMPTS:-128}"
SEQ_LEN="${SEQ_LEN:-1024}"
PREFIX_TOKENS="${PREFIX_TOKENS:-1024}"
STRIDE="${STRIDE:-4}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
PRUNE_COMPLETED_SHARDS="${PRUNE_COMPLETED_SHARDS:-1}"

TARGET="Qwen/Qwen3-32B"
DRAFT="Qwen/Qwen3-14B"
PAIR_DIR="$ARTIFACT_ROOT/calibration"
MAPPER="$ARTIFACT_ROOT/mapper.pt"
SCREEN_DIR="$ARTIFACT_ROOT/screen_n${PROMPTS}_l${PREFIX_TOKENS}"
RESULT="$RESULT_ROOT/qwen3_32b_to_14b_target_alignment.json"

mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT"

resolve_revision() {
  local model="$1"
  "$PYTHON" - "$model" <<'PY'
import sys
try:
    from huggingface_hub import model_info
except ImportError as exc:
    raise SystemExit("huggingface_hub is required; install project HF extras") from exc
info = model_info(sys.argv[1])
if not info.sha:
    raise SystemExit(f"could not resolve immutable Hub revision for {sys.argv[1]}")
print(info.sha)
PY
}

TARGET_REVISION="${TARGET_REVISION:-$(resolve_revision "$TARGET")}"
DRAFT_REVISION="${DRAFT_REVISION:-$(resolve_revision "$DRAFT")}"

echo "Scientific pair: $TARGET@$TARGET_REVISION -> $DRAFT@$DRAFT_REVISION"
echo "GPU cap per sequential model process: ${GPU_MEMORY_GIB}GiB"

# A 32B BF16 verifier will offload on a 32/48GB GPU.  Fail instead of changing
# precision if host RAM/offload capacity is insufficient.
if [[ -r /proc/meminfo ]]; then
  MEM_GIB="$($PYTHON - <<'PY'
with open('/proc/meminfo') as f:
    values = {line.split(':')[0]: int(line.split()[1]) for line in f if ':' in line}
print(round(values.get('MemTotal', 0) / 1024 / 1024, 1))
PY
)"
  echo "Host RAM: ${MEM_GIB} GiB (96+ GiB recommended; 128 GiB preferred)"
fi

"$PYTHON" bench/capture_sequential_calibration.py \
  --pair-dir "$PAIR_DIR" --role source \
  --target "$TARGET" --draft "$DRAFT" \
  --target-revision "$TARGET_REVISION" --draft-revision "$DRAFT_REVISION" \
  --text-file "$CALIBRATION_TEXT" --sequences "$SEQUENCES" \
  --seq-len "$SEQ_LEN" --stride "$STRIDE" \
  --device cuda --dtype bfloat16 --gpu-memory-gib "$GPU_MEMORY_GIB"

"$PYTHON" bench/capture_sequential_calibration.py \
  --pair-dir "$PAIR_DIR" --role draft \
  --target "$TARGET" --draft "$DRAFT" \
  --target-revision "$TARGET_REVISION" --draft-revision "$DRAFT_REVISION" \
  --text-file "$CALIBRATION_TEXT" --sequences "$SEQUENCES" \
  --seq-len "$SEQ_LEN" --stride "$STRIDE" \
  --device cuda --dtype bfloat16 --gpu-memory-gib "$GPU_MEMORY_GIB"

"$PYTHON" bench/fit_sequential_mapper.py \
  --pair-dir "$PAIR_DIR" --output "$MAPPER" \
  --k 8 --lambda 0.01 --selection-ridge 0.000001 \
  --selection-sequences "$SELECTION_SEQUENCES" \
  --accumulation-device cuda --selection-layer-block 1 --fit-layer-block 4

"$PYTHON" bench/capture_screen_prefixes.py \
  --screen-dir "$SCREEN_DIR" --mapper-metadata "$MAPPER.json" \
  --text-file "$EVAL_TEXT" --prompts "$PROMPTS" \
  --prefix-tokens "$PREFIX_TOKENS" --device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB"

"$PYTHON" bench/eval_pair_transfer.py \
  --screen-dir "$SCREEN_DIR" --mapper "$MAPPER" --mapper-metadata "$MAPPER.json" \
  --output "$RESULT" --prompts "$PROMPTS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" --threshold 0.95 \
  --device cuda --mapper-device cuda --dtype bfloat16 \
  --gpu-memory-gib "$GPU_MEMORY_GIB" --attention-cosine

DECISION="$($PYTHON - "$RESULT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding='utf-8'))
print(r['decision']['status'])
print(json.dumps({
    'native_fidelity': r['native_fidelity'],
    'target_alignment': r['target_alignment'],
    'decision': r['decision'],
}, indent=2), file=sys.stderr)
PY
)"

echo "32B->14B target-alignment decision: $DECISION"

# Record hashes before deleting reconstructible large cache shards.
"$PYTHON" - "$ARTIFACT_ROOT" "$PAIR_DIR" "$SCREEN_DIR" "$MAPPER" "$RESULT" <<'PY'
import json
import sys
from pathlib import Path
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file

artifact_root, pair_dir, screen_dir, mapper, result = map(Path, sys.argv[1:6])
files = [
    pair_dir / 'source/manifest.json',
    pair_dir / 'source/complete.json',
    pair_dir / 'draft/manifest.json',
    pair_dir / 'draft/complete.json',
    pair_dir / 'tokens/manifest.json',
    pair_dir / 'tokens.pt',
    mapper,
    Path(f'{mapper}.json'),
    screen_dir / 'manifest.json',
    screen_dir / 'complete.json',
    screen_dir / 'tokens/manifest.json',
    screen_dir / 'tokens.pt',
    result,
]
missing = [str(p) for p in files if not p.exists()]
if missing:
    raise RuntimeError(f'refusing to write inventory with missing files: {missing}')
atomic_write_json(
    artifact_root / 'artifact_inventory.json',
    {
        'schema_version': 1,
        'files': {str(p): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files},
    },
)
PY

if [[ "$PRUNE_COMPLETED_SHARDS" == "1" ]]; then
  "$PYTHON" - "$PAIR_DIR" "$SCREEN_DIR" "$SEQUENCES" "$PROMPTS" <<'PY'
import sys
from pathlib import Path

pair_dir = Path(sys.argv[1])
screen_dir = Path(sys.argv[2])
sequences = int(sys.argv[3])
prompts = int(sys.argv[4])
for root, expected in (
    (pair_dir / 'source/shards', sequences),
    (pair_dir / 'draft/shards', sequences),
    (screen_dir / 'shards', prompts),
):
    paths = sorted(root.glob('*.pt'))
    if len(paths) != expected:
        raise RuntimeError(f'refusing to prune unexpected shard set {root}: {len(paths)} != {expected}')
    for path in paths:
        path.unlink()
    root.rmdir()
# Target probability shards are small and retained for audit/re-analysis.
PY
fi

case "$DECISION" in
  go_sd)
    cat <<EOF
GO: 32B->14B mapped draft significantly improves verifier alignment.
Do NOT use 32/48GB single-GPU wall-clock as a paper systems number yet: exact 32B+14B
simultaneous SD requires substantial offload. First preserve this pair-quality result.
For a mechanism-only E2 on a sufficiently provisioned host, use the same SCREEN_RESULT
and MAPPER with scripts/run_native_frontier_e2.sh after configuring an appropriate
simultaneous exact-weight memory profile.
EOF
    ;;
  stop_pair)
    echo "STOP: 32B->14B is harmed by this translator. Do not tune refresh on this pair." >&2
    ;;
  expand)
    echo "INCONCLUSIVE: expand the held-out target-alignment screen before E2." >&2
    ;;
  incomplete)
    echo "INCOMPLETE: inspect failure artifacts; do not change precision or quantize automatically." >&2
    exit 4
    ;;
  *)
    echo "Unknown decision: $DECISION" >&2
    exit 5
    ;;
esac
