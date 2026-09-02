#!/usr/bin/env python3
"""E2 / KT-B: Native SD vs Ridge Init-only vs Ridge + Verifier Refresh."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from common import iter_texts, load_hf_pair

from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=512)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument(
        "--mapper-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument("--output", default="results/e2_acceptance_pilot.json")
    args = ap.parse_args()

    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    map_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.mapper_dtype]
    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device).to(
        args.device, dtype=map_dtype
    )
    texts = list(iter_texts(args.text_file, limit=args.prompts * 2))
    rows = []
    methods = {
        "native_sd": {"init_mode": "native", "refresh": False},
        "ridge_init_only": {"init_mode": "mapped", "refresh": False},
        "ridge_refresh": {"init_mode": "mapped", "refresh": True},
    }
    cuda_device = next(target.parameters()).device

    for method, options in methods.items():
        for idx, text in enumerate(texts[: args.prompts]):
            ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0][
                : args.prompt_tokens
            ]
            if ids.numel() < 2:
                continue
            runtime = QwenPairRuntime(
                target,
                draft,
                mapper,
                seed=idx,
                init_mode=options["init_mode"],
                refresh=options["refresh"],
            )
            if cuda_device.type == "cuda":
                torch.cuda.synchronize(cuda_device)
                torch.cuda.reset_peak_memory_stats(cuda_device)
            start = time.perf_counter()
            runtime.generate(ids.tolist(), args.new_tokens, args.gamma)
            if cuda_device.type == "cuda":
                torch.cuda.synchronize(cuda_device)
            elapsed = time.perf_counter() - start
            lengths = runtime.accepted_lengths
            bonus_count = sum(kind == "bonus" for kind in runtime.frontier_kinds)
            rows.append(
                {
                    "method": method,
                    "prompt": idx,
                    "prompt_tokens": int(ids.numel()),
                    "mean_accepted_length": sum(lengths) / max(len(lengths), 1),
                    "acceptance_rate": sum(lengths) / max(len(lengths) * args.gamma, 1),
                    "blocks": len(lengths),
                    "bonus_rate": bonus_count / max(len(lengths), 1),
                    "elapsed_s": elapsed,
                    "output_tokens_per_s": args.new_tokens / elapsed,
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated(cuda_device)
                        if cuda_device.type == "cuda"
                        else None
                    ),
                }
            )
            if (idx + 1) % 10 == 0:
                print(method, idx + 1)

    summary = {}
    metrics = (
        "mean_accepted_length",
        "acceptance_rate",
        "bonus_rate",
        "elapsed_s",
        "output_tokens_per_s",
    )
    for method in methods:
        subset = [x for x in rows if x["method"] == method]
        summary[method] = {
            key: sum(x[key] for x in subset) / max(len(subset), 1) for key in metrics
        }
        summary[method]["prompts_evaluated"] = len(subset)
        summary[method]["peak_vram_bytes"] = max(
            (x["peak_vram_bytes"] or 0 for x in subset), default=0
        )

    result = {"config": vars(args), "summary": summary, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
