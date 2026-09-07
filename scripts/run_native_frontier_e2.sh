#!/usr/bin/env bash
set -euo pipefail

: "${SCREEN_RESULT:?Set SCREEN_RESULT to a completed passing pair-screen JSON}"
: "${MAPPER:?Set MAPPER to the mapper checkpoint used by SCREEN_RESULT}"
: "${EVAL_TEXT:?Set EVAL_TEXT to the held-out E2 prompt file}"

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT="${OUTPUT:-results/e2_native_frontier.json}"
PROMPTS="${PROMPTS:-64}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
NEW_TOKENS="${NEW_TOKENS:-64}"

"$PYTHON" - "$SCREEN_RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
status = result.get("gate", {}).get("status")
if status != "pass":
    raise SystemExit(f"pair screen must pass before E2 (status={status!r})")
PY

readarray -t PAIR < <("$PYTHON" - "$SCREEN_RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
print(result["pair"]["target"])
print(result["pair"]["draft"])
PY
)

"$PYTHON" bench/eval_acceptance_pilot.py \
  --mapper "$MAPPER" --target "${PAIR[0]}" --draft "${PAIR[1]}" \
  --text-file "$EVAL_TEXT" --prompts "$PROMPTS" \
  --prompt-tokens "$PROMPT_TOKENS" --new-tokens "$NEW_TOKENS" --gamma 4 \
  --bootstrap-samples 10000 --device cuda --dtype bfloat16 \
  --mapper-dtype bfloat16 --low-vram --output "$OUTPUT"

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
print(json.dumps(result["gates"], indent=2, sort_keys=True))
PY
