#!/usr/bin/env python3
"""Speed-only MATH-500 prompt comparison for pure draft and mapped decoding."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from common import load_hf_model, load_hf_pair, load_hf_tokenizer
from eval_math_accuracy import _greedy, _load_math500, _prompt

from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--gpu-memory-gib", type=int, default=12)
    ap.add_argument("--target-device", default="cuda")
    ap.add_argument("--draft-device", default="cuda")
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--mapper-metadata")
    ap.add_argument("--output", default="results/debug_batch_2026-09-07/math_speed.json")
    args = ap.parse_args()
    if args.limit <= 0 or args.offset < 0 or args.max_new_tokens <= 0 or args.gamma <= 0:
        raise ValueError("limit/max-new-tokens/gamma must be positive; offset non-negative")

    mapper_path = Path(args.mapper)
    metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_id = metadata["pair"]["target"]
    draft_id = metadata["pair"]["draft"]
    target_revision = metadata["source_model"]["revision"]
    draft_revision = metadata["draft_model"]["revision"]
    problems = _load_math500(args.limit, args.offset)
    if len(problems) != args.limit:
        raise RuntimeError("requested rows are not available")
    tokenizer = load_hf_tokenizer(target_id, revision=target_revision)
    prompts = [_prompt(tokenizer, row["problem"]) for row in problems]
    result = {
        "scientific": False,
        "dataset": "HuggingFaceH4/MATH-500",
        "offset": args.offset,
        "evaluated": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "gamma": args.gamma,
        "target_device": args.target_device,
        "draft_device": args.draft_device,
        "methods": {},
    }

    model = load_hf_model(
        draft_id,
        args.draft_device,
        "bfloat16",
        revision=draft_revision,
        gpu_memory_gib=(args.gpu_memory_gib if args.target_device == args.draft_device else None),
        offload_folder=Path(args.output).parent / "speed_pure_4b_offload",
    )
    rows = []
    for index, prompt_ids in enumerate(prompts):
        generated, elapsed, peak = _greedy(model, prompt_ids, args.max_new_tokens)
        rows.append({"index": index, "tokens": len(generated), "elapsed_s": elapsed,
                     "tokens_per_s": len(generated) / max(elapsed, 1e-9), "peak_vram_bytes": peak})
        print("pure_4b", index + 1, rows[-1], flush=True)
    result["methods"]["pure_4b"] = rows
    del model
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer, target, draft = load_hf_pair(
        target_id, draft_id, "cuda", "bfloat16", low_vram=True,
        target_device=args.target_device, draft_device=args.draft_device,
        target_revision=target_revision, draft_revision=draft_revision,
    )
    mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
        draft.get_input_embeddings().weight.device, dtype=torch.bfloat16
    )
    rows = []
    for index, prompt_ids in enumerate(prompts):
        runtime = QwenPairRuntime(
            target, draft, mapper, temperature=1e-5, seed=index,
            init_mode="mapped_native_frontier", refresh_policy="accepted_only",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        generated = runtime.generate(prompt_ids, args.max_new_tokens, args.gamma)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append({"index": index, "tokens": len(generated), "elapsed_s": elapsed,
                     "tokens_per_s": len(generated) / max(elapsed, 1e-9),
                     "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                     "mean_accepted_length": sum(runtime.accepted_lengths) / max(len(runtime.accepted_lengths), 1)})
        print("mapped_accepted_only", index + 1, rows[-1], flush=True)
    result["methods"]["mapped_accepted_only"] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for method, values in result["methods"].items():
        print(method, "mean_tok_s", sum(x["tokens_per_s"] for x in values) / len(values))


if __name__ == "__main__":
    main()
