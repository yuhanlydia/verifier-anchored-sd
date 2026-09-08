#!/usr/bin/env bash
set -euo pipefail

: "${SCREEN_RESULT:?Set SCREEN_RESULT to a completed target-alignment screen JSON}"
: "${MAPPER:?Set MAPPER to the mapper checkpoint used by SCREEN_RESULT}"
: "${EVAL_TEXT:?Set EVAL_TEXT to the held-out E2 prompt file}"

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT="${OUTPUT:-results/e2_native_frontier.json}"
MAPPER_METADATA="${MAPPER_METADATA:-${MAPPER}.json}"
PROMPTS="${PROMPTS:-64}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
NEW_TOKENS="${NEW_TOKENS:-64}"
LOW_VRAM="${LOW_VRAM:-0}"

"$PYTHON" - "$SCREEN_RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
target_status = result.get("target_alignment", {}).get("gate", {}).get("status")
decision = result.get("decision", {}).get("status")
if target_status != "support" or decision != "go_sd":
    raise SystemExit(
        "target alignment must support E2 "
        f"(target_status={target_status!r}, decision={decision!r})"
    )
PY

readarray -t PAIR < <("$PYTHON" - "$SCREEN_RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print(result["pair"]["target"])
print(result["pair"]["draft"])
PY
)

LOW_VRAM_ARGS=()
if [[ "$LOW_VRAM" == "1" ]]; then
  LOW_VRAM_ARGS+=(--low-vram)
elif [[ "$LOW_VRAM" != "0" ]]; then
  echo "LOW_VRAM must be 0 or 1" >&2
  exit 2
fi

"$PYTHON" bench/eval_acceptance_pilot.py \
  --screen-result "$SCREEN_RESULT" --mapper "$MAPPER" \
  --mapper-metadata "$MAPPER_METADATA" \
  --target "${PAIR[0]}" --draft "${PAIR[1]}" \
  --text-file "$EVAL_TEXT" --prompts "$PROMPTS" \
  --prompt-tokens "$PROMPT_TOKENS" --new-tokens "$NEW_TOKENS" --gamma 4 \
  --bootstrap-samples 10000 --device cuda --dtype bfloat16 \
  --mapper-dtype bfloat16 "${LOW_VRAM_ARGS[@]}" --output "$OUTPUT"

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(result["gates"], indent=2, sort_keys=True))
PY
