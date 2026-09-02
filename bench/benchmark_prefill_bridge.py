#!/usr/bin/env python3
"""E1 / KT-A: native draft prefill versus verifier prefill + KV bridge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from common import load_hf_pair

from verifier_anchored_sd.spec_decode.hf_runtime import capture_rotary_factors, forward_incremental
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def timed(fn, *, warmup: int, repetitions: int, device: torch.device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    for _ in range(repetitions):
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / 1000.0)
        else:
            start = time.perf_counter()
            fn()
            values.append(time.perf_counter() - start)
    x = torch.tensor(values)
    return {
        "median_s": x.median().item(),
        "p95_s": torch.quantile(x, 0.95).item(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


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
    ap.add_argument(
        "--mapper-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="live mapper precision; BF16 is the recommended 24GB serving setting",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="16GB feasibility profile; reduces default lengths and permits model offload",
    )
    ap.add_argument("--output", default="results/e1_prefill_bridge.json")
    args = ap.parse_args()
    if args.low_vram and args.lengths == "512,1024,2048,4096,8192,16384":
        args.lengths = "256,512,1024,2048,4096"
    tokenizer, target, draft = load_hf_pair(
        args.target, args.draft, args.device, args.dtype, low_vram=args.low_vram
    )
    map_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.mapper_dtype]
    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device).to(
        args.device, dtype=map_dtype
    )
    device = next(target.parameters()).device
    lengths = [int(x) for x in args.lengths.split(",")]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    for length in lengths:
        ids = torch.randint(0, tokenizer.vocab_size, (1, length), device=device, generator=generator)
        positions = torch.arange(length, device=next(draft.parameters()).device).unsqueeze(0)

        # Native SD does not need explicit RoPE capture, so do not charge it for our
        # bridge instrumentation.
        target_time = timed(
            lambda ids=ids: forward_incremental(target, ids, capture_rotary=False),
            warmup=args.warmup,
            repetitions=args.repetitions,
            device=device,
        )
        draft_time = timed(
            lambda ids=ids: forward_incremental(draft, ids, capture_rotary=False),
            warmup=args.warmup,
            repetitions=args.repetitions,
            device=device,
        )

        target_capture = forward_incremental(target, ids)
        draft_rotary = capture_rotary_factors(draft, positions)
        map_time = timed(
            lambda cache=target_capture.cache, rope=draft_rotary: mapper.map(
                cache, draft_rotary=rope
            ),
            warmup=args.warmup,
            repetitions=args.repetitions,
            device=device,
        )

        def bridge_full(ids=ids, positions=positions):
            out = forward_incremental(target, ids)
            receiver_rope = capture_rotary_factors(draft, positions)
            mapper.map(out.cache, draft_rotary=receiver_rope)

        bridge_time = timed(
            bridge_full,
            warmup=args.warmup,
            repetitions=args.repetitions,
            device=device,
        )
        native_total = target_time["median_s"] + draft_time["median_s"]
        bridge_total = bridge_time["median_s"]
        rows.append(
            {
                "length": length,
                "target_prefill": target_time,
                "draft_prefill": draft_time,
                "mapper_only": map_time,
                "bridge_total": bridge_time,
                "native_total_median_s": native_total,
                "speedup": native_total / bridge_total,
                "draft_prefill_over_mapper": draft_time["median_s"] / map_time["median_s"],
            }
        )
        print(json.dumps(rows[-1]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
