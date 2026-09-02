#!/usr/bin/env python3
"""E0: paper-faithful target->draft ridge calibration.

The production fit is delegated to the pinned KVBridge backend rather than keeping
a second simplified mapper implementation. KVBridge supplies exact RoPE
strip/reapply, calibration-R² source-layer selection, all-source-head features, and
bounded-memory centered ridge fitting.
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
        from kvbridge.io import load_calibration_shard, save_calibration_shard
    except ImportError as exc:
        raise RuntimeError(
            "E0 requires the pinned paper-faithful backend: "
            "pip install -e '.[hf,kvbridge]'"
        ) from exc
    return (
        FitConfig,
        CalibrationPair,
        fit_mapper,
        capture_cache,
        model_signature,
        load_calibration_shard,
        save_calibration_shard,
    )


def _model_revision(model, fallback: str) -> str:
    return str(getattr(model.config, "_commit_hash", None) or fallback)


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
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument("--accumulation-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument(
        "--accumulation-dtype", default="float32", choices=["float32", "float64"]
    )
    ap.add_argument("--selection-layer-block", type=int, default=4)
    ap.add_argument("--fit-layer-block", type=int, default=1)
    ap.add_argument(
        "--calibration-dir", default="artifacts/e0_qwen3_4b_to_1p7b_calibration"
    )
    ap.add_argument(
        "--kvbridge-artifact", default="artifacts/e0_qwen3_4b_to_1p7b_kvbridge"
    )
    ap.add_argument("--output", default="checkpoints/qwen3_4b_to_1p7b_ridge.pt")
    ap.add_argument("--overwrite-artifact", action="store_true")
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="16GB feasibility profile; keeps seq-len 1024 but enables controlled model offload",
    )
    args = ap.parse_args()
    if args.sequences <= 0 or args.seq_len <= 0 or args.stride <= 0:
        raise ValueError("sequences, seq-len, and stride must be positive")

    (
        FitConfig,
        CalibrationPair,
        fit_mapper,
        capture_cache,
        model_signature,
        load_calibration_shard,
        save_calibration_shard,
    ) = _require_kvbridge()

    tokenizer, target, draft = load_hf_pair(
        args.target, args.draft, args.device, args.dtype, low_vram=args.low_vram
    )
    target_revision = _model_revision(target, args.target_revision)
    draft_revision = _model_revision(draft, args.draft_revision)
    source_signature = model_signature(
        target, tokenizer, revision=target_revision, attention_kind="dense"
    )
    draft_signature = model_signature(
        draft, tokenizer, revision=draft_revision, attention_kind="dense"
    )
    source_signature.validate_pair(draft_signature, require_matched_kv=True)

    root = Path(args.calibration_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    calibration_contract = {
        "target": args.target,
        "draft": args.draft,
        "target_revision": target_revision,
        "draft_revision": draft_revision,
        "sequences": args.sequences,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "tokenizer_hash": source_signature.tokenizer_hash,
    }
    if manifest_path.exists():
        old_contract = json.loads(manifest_path.read_text())
        if old_contract != calibration_contract:
            raise RuntimeError(
                "calibration directory contains shards from a different contract; "
                "choose a fresh --calibration-dir or remove the old directory"
            )
    else:
        manifest_path.write_text(json.dumps(calibration_contract, indent=2))

    required_paths = [root / f"{index:05d}.safetensors" for index in range(args.sequences)]
    missing_indices = [index for index, path in enumerate(required_paths) if not path.exists()]
    if missing_indices:
        windows = token_windows(
            tokenizer,
            iter_texts(args.text_file, limit=max(args.sequences * 8, args.sequences)),
            seq_len=args.seq_len,
            count=args.sequences,
        )
        available = {index: ids for index, ids in enumerate(windows) if index in set(missing_indices)}
        if len(available) != len(missing_indices):
            raise RuntimeError(
                f"only {len(available)}/{len(missing_indices)} missing calibration windows were "
                "available; provide more text"
            )
        for done, index in enumerate(missing_indices, start=1):
            ids = available[index].unsqueeze(0).to(next(target.parameters()).device)
            with torch.inference_mode():
                source_cache = capture_cache(target, ids).sample_tokens(args.stride).detach()
                draft_cache = capture_cache(draft, ids).sample_tokens(args.stride).detach()
            save_calibration_shard(
                required_paths[index],
                CalibrationPair(source_cache, draft_cache),
                sequence_id=f"{index:05d}",
            )
            del source_cache, draft_cache
            if done % 10 == 0 or done == len(missing_indices):
                print(f"new calibration shards: {done}/{len(missing_indices)}")

    # Fitting no longer needs LLM weights. Free the 24GB GPU before the k=8
    # cross-head normal equations are allocated.
    del target, draft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def exact_factory():
        # Consume exactly this run's requested N shards, ignoring unrelated files.
        for path in required_paths:
            yield load_calibration_shard(path)

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
    external = fit_mapper(exact_factory, source_signature, draft_signature, config)
    artifact = Path(args.kvbridge_artifact)
    external.save(
        artifact, overwrite=args.overwrite_artifact, storage_dtype="bfloat16"
    )

    # Convert to this repo's trainable/runtime checkpoint. Preserve FP32 base weights
    # for science baselines; serving scripts cast the live mapper to BF16 on 24GB.
    runtime_mapper = RidgeKVMapper.from_kvbridge_artifact(artifact, dtype=torch.float32)
    runtime_mapper.save(args.output)
    metadata = {
        "args": vars(args),
        "source_signature": source_signature.to_dict(),
        "draft_signature": draft_signature.to_dict(),
        "selected_layers": runtime_mapper.metadata.layer_selection,
        "observations": args.sequences
        * ((args.seq_len + args.stride - 1) // args.stride),
        "backend": "kvbridge@0d75f31dcde6eeceaa609d3affed6ca1401deb77",
        "content_space": True,
        "cross_head": True,
    }
    Path(args.output + ".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved paper-faithful runtime mapper: {args.output}")


if __name__ == "__main__":
    main()
