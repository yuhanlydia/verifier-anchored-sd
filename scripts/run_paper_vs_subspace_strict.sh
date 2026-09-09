#!/usr/bin/env bash
set -euo pipefail

SUITE_PROFILE="${SUITE_PROFILE:-pilot}"
case "$SUITE_PROFILE" in
  pilot)
    export CALIBRATION_SEQUENCES="${CALIBRATION_SEQUENCES:-128}"
    export SELECTION_SEQUENCES="${SELECTION_SEQUENCES:-128}"
    export MLP_PROFILE="${MLP_PROFILE:-pilot}"
    export MLP_SEQUENCES="${MLP_SEQUENCES:-32}"
    ;;
  paper)
    export CALIBRATION_SEQUENCES="${CALIBRATION_SEQUENCES:-500}"
    export SELECTION_SEQUENCES="${SELECTION_SEQUENCES:-500}"
    export MLP_PROFILE="${MLP_PROFILE:-paper}"
    export MLP_SEQUENCES="${MLP_SEQUENCES:-500}"
    ;;
  *)
    echo "SUITE_PROFILE must be pilot or paper" >&2
    exit 2
    ;;
esac

if [[ "$SELECTION_SEQUENCES" != "$CALIBRATION_SEQUENCES" ]]; then
  cat >&2 <<EOF
Strict paper-vs-subspace entrypoint requires matched-head and Full-Head layer selection
to see the same complete A calibration set.
CALIBRATION_SEQUENCES=$CALIBRATION_SEQUENCES
SELECTION_SEQUENCES=$SELECTION_SEQUENCES
Use scripts/run_paper_vs_subspace_suite.sh directly only for an explicitly diagnostic run.
EOF
  exit 3
fi

exec bash scripts/run_paper_vs_subspace_suite.sh
