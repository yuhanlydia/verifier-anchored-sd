#!/usr/bin/env python3
"""Long-generation drift curve: MAL by output position for three cache policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import iter_texts, load_hf_pair

from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--new-tokens", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output", default="results/generation_drift.json")
    args = ap.parse_args()
    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device)
    texts = list(iter_texts(args.text_file, limit=args.prompts * 2))
    buckets = ((1, 64), (65, 128), (129, 256), (257, 512))
    result = []
    for method, options in (("ridge_init_only", ("mapped", False)), ("ridge_refresh", ("mapped", True))):
        values = {f"{lo}-{hi}": [] for lo, hi in buckets}
        for i, text in enumerate(texts[:args.prompts]):
            ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            runtime = QwenPairRuntime(target, draft, mapper, seed=i, init_mode=options[0], refresh=options[1])
            runtime.generate(ids.tolist(), args.new_tokens, args.gamma)
            cursor = 0
            for length, emitted in zip(runtime.accepted_lengths, runtime.block_emitted_lengths):
                for lo, hi in buckets:
                    overlap = max(0, min(cursor + emitted, hi) - max(cursor + 1, lo))
                    if overlap:
                        values[f"{lo}-{hi}"].append(length)
                cursor += emitted
        result.append({"method": method, "mal_by_bucket": {k: sum(v) / max(len(v), 1) for k, v in values.items()}})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"config": vars(args), "results": result}, indent=2))


if __name__ == "__main__":
    main()
