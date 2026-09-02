#!/usr/bin/env python3
"""E0: paper-faithful target->draft ridge calibration.

This script deliberately delegates fitting to the pinned KVBridge backend rather
than maintaining a second, simplified implementation.  KVBridge performs:

* model-produced inverse/forward RoPE handling,
* head-averaged single-source R² layer selection,
* top-k cross-layer features containing *all* source KV heads, and
* bounded-memory centered ridge fitting.

Calibration caches are stride-sampled before being written to SafeTensors shards;
for 500x1024, stride=4 this is about 128k token observations while keeping the
out-of-core working set roughly four times smaller than full-cache storage.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from common import iter_texts, load_hf_pair, token_windows

from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def _require_kvbridge():
    try:
        from kvbridge.config import FitConfig
        from kvbridge.fit import CalibrationPair, fit_mapper
        from kvbridge.huggingface import capture_cache, model_signature
        from kvbridge.io import calibration_shard_factory, save_calibration_shard
    except ImportError as exc:
        raise RuntimeError(
            "E0 requires the pinned paper-faithful backend: pip install -e '.[hf,kvbridge]'"
        ) from exc
    return (
        FitConfig,
        CalibrationPair,
        fit_mapper,
        capture_cache,
        model_signature,
        calibration_shard_factory,
        save_calibration_shard,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--target-revision", default="main")
    ap.add_argument("--draft-revision", default="main")
    ap.add_argument("--text-file", help="JSONL with a text field, or one raw document per line")
    ap.add_argument("--sequences", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--accumulation-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--accumulation-dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--selection-layer-block", type=int, default=4)
    ap.add_argument("--fit-layer-block", type=int, default=1)
    ap.add_argument("--calibration-dir", default="artifacts/e0_qwen3_4b_to_1p7b_calibration")
    ap.add_argument("--kvbridge-artifact", default="artifacts/e0_qwen3_4b_to_1p7b_kvbridge")
    ap.add_argument("--output", default="checkpoints/qwen3_4b_to_1p7b_ridge.pt")
    ap.add_argument("--overwrite-artifact", action="store_true")
    args = ap.parse_args()
    if args.sequences <= 0 or args.seq_len <= 0 or args.stride <= 0:
        raise ValueError("sequences, seq-len, and stride must be positive")

    (
        FitConfig,
        CalibrationPair,
        fit_mapper,
        capture_cache,
        model_signature,
        calibration_shard_factory,
        save_calibration_shard,
    ) = _require_kvbridge()

    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    source_signature = model_signature(
        target, tokenizer, revision=args.target_revision, attention_kind="dense"
    )
    draft_signature = model_signature(
        draft, tokenizer, revision=args.draft_revision, attention_kind="dense"
    )
    source_signature.validate_pair(draft_signature, require_matched_kv=True)

    root = Path(args.calibration_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in root.glob("*.safetensors")}
    windows = token_windows(
        tokenizer,
        iter_texts(args.text_file, limit=max(args.sequences * 8, args.sequences)),
        seq_len=args.seq_len,
        count=args.sequences,
    )
    captured = 0
    for index, ids in enumerate(windows):
        shard = root / f"{index:05d}.safetensors"
        if shard.name in existing:
            captured += 1
            continue
        ids = ids.unsqueeze(0).to(next(target.parameters()).device)
        with torch.inference_mode():
            source_cache = capture_cache(target, ids).sample_tokens(args.stride).detach()
            draft_cache = capture_cache(draft, ids).sample_tokens(args.stride).detach()
        save_calibration_shard(
            shard,
            CalibrationPair(source_cache, draft_cache),
            sequence_id=f"{index:05d}",
        )
        captured += 1
        del source_cache, draft_cache
        if (index + 1) % 10 == 0:
            print(f"calibration shards: {index + 1}/{args.sequences}")
    if captured != args.sequences:
        raise RuntimeError(
            f"only {captured}/{args.sequences} calibration windows were available; provide more text"
        )

    # Fitting no longer needs the LLM weights.  Free the 24GB GPU before allocating
    # the large k=8 cross-head normal equations.
    del target, draft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = FitConfig(
        top_k=args.k,
        ridge_alpha=args.lambda_,
        content_space=True,
        selection_alpha=1e-6,
        accumulation_dtype=args.accumulation_dtype,
        accumulation_device=args.accumulation_device,
        require_matched_kv=True,
        target_layer_block_size=args.fit_layer_block,
        selection_target_layer_block_size=args.selection_layer_block,
        token_stride=1,  # shards were already sampled before persistence
    )
    external = fit_mapper(
        calibration_shard_factory(root), source_signature, draft_signature, config
    )
    artifact = Path(args.kvbridge_artifact)
    external.save(artifact, overwrite=args.overwrite_artifact, storage_dtype="bfloat16")

    # Convert to the trainable/runtime checkpoint used by this repo.  Keep FP32 base
    # weights for scientific baselines; E1 can cast the live mapper to BF16 for a
    # 24GB serving measurement via --mapper-dtype.
    runtime_mapper = RidgeKVMapper.from_kvbridge_artifact(artifact, dtype=torch.float32)
    runtime_mapper.save(args.output)
    metadata = {
        "args": vars(args),
        "source_signature": source_signature.to_dict(),
        "draft_signature": draft_signature.to_dict(),
        "selected_layers": runtime_mapper.metadata.layer_selection,
        "observations": args.sequences * ((args.seq_len + args.stride - 1) // args.stride),
        "backend": "kvbridge@0d75f31dcde6eeceaa609d3affed6ca1401deb77",
    }
    Path(args.output + ".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved paper-faithful runtime mapper: {args.output}")


if __name__ == "__main__":
    main()
