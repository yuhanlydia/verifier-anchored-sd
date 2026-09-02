#!/usr/bin/env python3
"""Initialize the phase-2 low-rank acceptance adapter.

The requested first pass is E0--E2; this command intentionally only creates a
zero-gated adapter checkpoint.  Actual block-objective training should start only
after the E2 gates pass, using disjoint prefixes and the frozen-draft forward path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    mapper = RidgeKVMapper.load(args.mapper)
    mapper.add_low_rank(args.rank)
    mapper.save(args.output)
    print(f"initialized zero-gated rank-{args.rank} residual at {args.output}; no LLM training was run")


if __name__ == "__main__":
    main()
