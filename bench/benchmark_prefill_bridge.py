#!/usr/bin/env python3
"""E1 / KT-A: compare native draft prefill with target-prefill + KV mapping."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from common import load_hf_pair

from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def timed(fn, *, warmup: int, repetitions: int, device: torch.device):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repetitions):
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); fn(); end.record(); end.synchronize()
            values.append(start.elapsed_time(end) / 1000.0)
        else:
            start = time.perf_counter(); fn(); values.append(time.perf_counter() - start)
    x = torch.tensor(values)
    return {"median_s": x.median().item(), "p95_s": torch.quantile(x, 0.95).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repetitions", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--output", default="results/e1_prefill_bridge.json")
    args = ap.parse_args()
    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device)
    device = next(target.parameters()).device
    lengths = [int(x) for x in args.lengths.split(",")]
    rows = []
    for length in lengths:
        ids = torch.randint(0, tokenizer.vocab_size, (1, length), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        target_time = timed(lambda ids=ids: forward_incremental(target, ids), warmup=args.warmup, repetitions=args.repetitions, device=device)
        draft_time = timed(lambda ids=ids: forward_incremental(draft, ids), warmup=args.warmup, repetitions=args.repetitions, device=device)
        def bridge(ids=ids):
            out = forward_incremental(target, ids)
            mapper.map(out.cache)
        map_only_cache = forward_incremental(target, ids).cache
        map_time = timed(lambda cache=map_only_cache: mapper.map(cache), warmup=args.warmup, repetitions=args.repetitions, device=device)
        bridge_time = timed(bridge, warmup=args.warmup, repetitions=args.repetitions, device=device)
        native_total = target_time["median_s"] + draft_time["median_s"]
        bridge_total = bridge_time["median_s"]
        rows.append({"length": length, "target_prefill": target_time, "draft_prefill": draft_time,
                     "mapper_only": map_time, "bridge_total": bridge_total, "native_total": native_total,
                     "speedup": native_total / bridge_total, "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        print(json.dumps(rows[-1]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
