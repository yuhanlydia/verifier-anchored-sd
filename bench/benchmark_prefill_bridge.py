#!/usr/bin/env python3
"""E1 / KT-A: native draft prefill versus verifier prefill + KV bridge.

Batch=1 remains the latency/TTFT result.  Larger batches deliberately probe GPU
utilization and the real 16GB/24GB capacity frontier.  With ``--continue-on-oom``
an OOM becomes a recorded boundary row instead of invalidating earlier measurements.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from common import load_hf_pair

from verifier_anchored_sd.resource_profiles import e1_batch_sizes
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
    ap.add_argument(
        "--batch-sizes",
        default="",
        help="comma-separated batch sweep; empty uses the selected memory profile or batch=1",
    )
    ap.add_argument(
        "--memory-profile",
        choices=["manual", "16gb", "24gb"],
        default="manual",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repetitions", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--mapper-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="controlled model offload; useful as a 16GB fallback but slower than resident inference",
    )
    ap.add_argument(
        "--continue-on-oom",
        action="store_true",
        help="record OOM rows and continue to other length/batch combinations",
    )
    ap.add_argument("--output", default="results/e1_prefill_bridge.json")
    args = ap.parse_args()

    if args.low_vram and args.lengths == "512,1024,2048,4096,8192,16384":
        args.lengths = "256,512,1024,2048,4096"
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    elif args.memory_profile != "manual":
        batch_sizes = e1_batch_sizes(args.memory_profile)
    else:
        batch_sizes = [1]
    if not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")

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
    target_device = target.get_input_embeddings().weight.device
    draft_device = draft.get_input_embeddings().weight.device
    lengths = [int(x) for x in args.lengths.split(",")]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = []

    for length in lengths:
        for batch_size in batch_sizes:
            try:
                ids_cpu = torch.randint(
                    0,
                    tokenizer.vocab_size,
                    (batch_size, length),
                    generator=generator,
                )
                target_ids = ids_cpu.to(target_device)
                draft_ids = ids_cpu.to(draft_device)
                target_positions = torch.arange(length, device=target_device).unsqueeze(0).expand(
                    batch_size, -1
                )
                draft_positions = target_positions.to(draft_device)

                target_time = timed(
                    lambda: forward_incremental(target, target_ids, capture_rotary=False),
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )
                draft_time = timed(
                    lambda: forward_incremental(draft, draft_ids, capture_rotary=False),
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )

                def native_full():
                    forward_incremental(target, target_ids, capture_rotary=False)
                    forward_incremental(draft, draft_ids, capture_rotary=False)

                native_time = timed(
                    native_full,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )

                target_capture = forward_incremental(target, target_ids)
                draft_rotary = capture_rotary_factors(draft, draft_positions)
                map_time = timed(
                    lambda: mapper.map(target_capture.cache, draft_rotary=draft_rotary),
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )

                def bridge_full():
                    out = forward_incremental(target, target_ids)
                    receiver_rope = capture_rotary_factors(draft, draft_positions)
                    mapper.map(out.cache, draft_rotary=receiver_rope)

                bridge_time = timed(
                    bridge_full,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    device=device,
                )
                native_total = native_time["median_s"]
                bridge_total = bridge_time["median_s"]
                tokens = batch_size * length
                row = {
                    "length": length,
                    "batch_size": batch_size,
                    "oom": False,
                    "target_prefill": target_time,
                    "draft_prefill": draft_time,
                    "native_initialization": native_time,
                    "mapper_only": map_time,
                    "bridge_total": bridge_time,
                    "native_total_median_s": native_total,
                    "component_sum_native_median_s": target_time["median_s"]
                    + draft_time["median_s"],
                    "speedup": native_total / bridge_total,
                    "draft_prefill_over_mapper": draft_time["median_s"] / map_time["median_s"],
                    "native_context_tokens_per_s": tokens / native_total,
                    "bridge_context_tokens_per_s": tokens / bridge_total,
                }
                rows.append(row)
                print(json.dumps(row))
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if not is_oom or not args.continue_on_oom:
                    raise
                row = {
                    "length": length,
                    "batch_size": batch_size,
                    "oom": True,
                    "error": str(exc).splitlines()[0][:500],
                }
                rows.append(row)
                print(json.dumps(row))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    result = {
        "config": {**vars(args), "resolved_batch_sizes": batch_sizes},
        "rows": rows,
        "interpretation": {
            "latency_rows": "batch_size=1",
            "throughput_rows": "batch_size>1; OOM is a capacity boundary, not a failed run",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
