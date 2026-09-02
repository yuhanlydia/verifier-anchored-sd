#!/usr/bin/env python3
"""Evaluate a SPEED-Bench-style JSONL prompt file with real wall-clock metrics.

Expected input rows contain ``prompt`` (string); optional ``id`` is retained.  The
script stays dataset-format agnostic so the benchmark checkout is not vendored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import load_hf_pair

from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-jsonl", required=True)
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output", default="results/speedbench.jsonl")
    args = ap.parse_args()
    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device)
    rows = []
    for line in Path(args.prompts_jsonl).read_text().splitlines():
        row = json.loads(line)
        ids = tokenizer(row["prompt"], add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        runtime = QwenPairRuntime(target, draft, mapper, seed=len(rows), init_mode="mapped", refresh=True)
        start = time.perf_counter()
        output = runtime.generate(ids.tolist(), args.new_tokens, args.gamma)
        elapsed = time.perf_counter() - start
        rows.append({"id": row.get("id", len(rows)), "input_tokens": len(ids), "output_tokens": len(output),
                     "elapsed_s": elapsed, "output_tokens_per_s": len(output) / max(elapsed, 1e-9),
                     "mean_accepted_length": sum(runtime.accepted_lengths) / max(len(runtime.accepted_lengths), 1),
                     "acceptance_rate": sum(runtime.accepted_lengths) / max(len(runtime.accepted_lengths) * args.gamma, 1)})
        print(json.dumps(rows[-1]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(json.dumps(x) for x in rows) + "\n")


if __name__ == "__main__":
    main()

