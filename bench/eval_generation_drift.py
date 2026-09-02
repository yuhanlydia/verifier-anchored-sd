#!/usr/bin/env python3
"""Long-generation MAL curve by output position for the three cache policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from common import iter_texts, load_hf_pair

from verifier_anchored_sd.evaluation import block_bucket_overlap
from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--mapper-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--output", default="results/generation_drift.json")
    args = ap.parse_args()
    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    map_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.mapper_dtype]
    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device).to(args.device, dtype=map_dtype)
    texts = list(iter_texts(args.text_file, limit=args.prompts * 2))
    buckets = ((1, 64), (65, 128), (129, 256), (257, 512))
    methods = (
        ("native_sd", ("native", False)),
        ("ridge_init_only", ("mapped", False)),
        ("ridge_refresh", ("mapped", True)),
    )
    result = []
    for method, options in methods:
        weighted_sum = {f"{lo}-{hi}": 0.0 for lo, hi in buckets}
        weighted_count = {f"{lo}-{hi}": 0 for lo, hi in buckets}
        prompt_count = 0
        for i, text in enumerate(texts[: args.prompts]):
            ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0][
                : args.prompt_tokens
            ]
            if ids.numel() < 2:
                continue
            runtime = QwenPairRuntime(
                target,
                draft,
                mapper,
                seed=i,
                init_mode=options[0],
                refresh=options[1],
            )
            runtime.generate(ids.tolist(), args.new_tokens, args.gamma)
            prompt_count += 1
            cursor = 0
            for accepted, emitted in zip(
                runtime.accepted_lengths, runtime.block_emitted_lengths, strict=True
            ):
                for lo, hi in buckets:
                    key = f"{lo}-{hi}"
                    overlap = block_bucket_overlap(cursor=cursor, emitted=emitted, lo=lo, hi=hi)
                    if overlap:
                        weighted_sum[key] += float(accepted) * overlap
                        weighted_count[key] += overlap
                cursor += emitted
        result.append(
            {
                "method": method,
                "prompts_evaluated": prompt_count,
                "mal_by_bucket": {
                    key: (weighted_sum[key] / weighted_count[key] if weighted_count[key] else None)
                    for key in weighted_sum
                },
                "positions_per_bucket": weighted_count,
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"config": vars(args), "results": result}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
