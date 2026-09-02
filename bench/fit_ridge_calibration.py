#!/usr/bin/env python3
"""E0: fit the target->draft affine ridge mapper from disjoint text calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from common import iter_texts, load_hf_pair, model_dims, token_windows

from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import (
    default_layer_selection,
    fit_ridge_mapper_from_cache_pairs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--text-file", help="JSONL with a text field, or one raw document per line")
    ap.add_argument("--sequences", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--output", default="checkpoints/qwen3_4b_to_1p7b_ridge.pt")
    args = ap.parse_args()
    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    tl, heads, dim = model_dims(target)
    dl, draft_heads, draft_dim = model_dims(draft)
    if (heads, dim) != (draft_heads, draft_dim):
        raise RuntimeError(f"matched-KV pilot requires equal draft/target heads and dimensions: {(heads, dim)} vs {(draft_heads, draft_dim)}")
    selection = default_layer_selection(tl, dl, args.k)
    device = next(target.parameters()).device
    windows = token_windows(tokenizer, iter_texts(args.text_file, limit=args.sequences * 4), seq_len=args.seq_len, count=args.sequences)

    def pairs():
        for completed, ids in enumerate(windows, start=1):
            ids = ids.unsqueeze(0).to(device)
            with torch.inference_mode():
                target_cache = forward_incremental(target, ids).cache.to("cpu")
                draft_cache = forward_incremental(draft, ids).cache.to("cpu")
            if completed % 10 == 0:
                print(f"calibration sequences: {completed}/{args.sequences}")
            yield target_cache, draft_cache

    mapper = fit_ridge_mapper_from_cache_pairs(
        pairs(), target_layers=tl, draft_layers=dl, kv_heads=heads, head_dim=dim,
        layer_selection=selection, lambda_=args.lambda_, stride=args.stride,
    )
    mapper.save(args.output)
    metadata = {"args": vars(args), "target_dims": [tl, heads, dim], "draft_dims": [dl, draft_heads, draft_dim], "layer_selection": selection}
    Path(args.output + ".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
