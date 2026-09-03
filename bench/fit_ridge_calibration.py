#!/usr/bin/env python3
"""E0: verifier->draft ridge calibration with full-head and matched-head backends.

``--head-mode full`` reproduces the audited KVBridge Full-Head mapper.
``--head-mode matched`` is an architecture-indexed head-local ridge baseline.  It
shares the same calibration shards, content-space RoPE treatment, and R²-selected
source layers, but reduces each affine input from ``k * H * d`` to ``k * d``.

The 16GB/24GB fit profiles apply *after* calibration capture.  At that point both
LLMs are deleted, so the otherwise idle CUDA device is used for R² selection and
normal-equation accumulation instead of falling back to a very slow CPU fit.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from common import iter_texts, load_hf_pair, token_windows

from verifier_anchored_sd.resource_profiles import e0_fit_profile
from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV, RotaryFactors
from verifier_anchored_sd.spec_decode.head_local_fit import (
    fit_matched_head_mapper_from_cache_pairs,
)
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


def _depth_selection(source_layers: int, target_layers: int, top_k: int) -> list[list[int]]:
    """Deterministic layer-neighbour selection restricted to explicit smoke runs."""
    if source_layers <= 0 or target_layers <= 0 or top_k <= 0:
        raise ValueError("layer counts and top_k must be positive")
    width = min(top_k, source_layers)
    result = []
    for target_layer in range(target_layers):
        center = round(target_layer * (source_layers - 1) / max(target_layers - 1, 1))
        start = center - (width - 1) // 2
        start = max(0, min(start, source_layers - width))
        result.append(list(range(start, start + width)))
    return result


def _to_runtime_cache(cache) -> CacheState:
    rotary = None
    if cache.rotary is not None:
        rotary = RotaryFactors(
            cache.rotary.cos,
            cache.rotary.sin,
            cache.rotary.interleaved,
        )
    return CacheState(
        (LayerKV(k, v) for k, v in zip(cache.keys, cache.values, strict=True)),
        rotary=rotary,
        keys_are_content=cache.keys_are_content,
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
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument(
        "--head-mode",
        choices=["full", "matched"],
        default="full",
        help="full=original all-source-head mapper; matched=head-local efficiency baseline",
    )
    ap.add_argument(
        "--fit-profile",
        choices=["manual", "16gb", "24gb"],
        default="manual",
        help="post-capture normal-equation profile; 16gb also enables controlled model offload",
    )
    ap.add_argument("--accumulation-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument(
        "--accumulation-dtype", default="float32", choices=["float32", "float64"]
    )
    ap.add_argument("--selection-layer-block", type=int, default=4)
    ap.add_argument("--fit-layer-block", type=int, default=1)
    ap.add_argument(
        "--selection-sequences",
        type=int,
        default=0,
        help="number of calibration sequences used only for R² layer selection; 0=all",
    )
    ap.add_argument(
        "--layer-selection",
        choices=["r2", "depth"],
        default="r2",
        help="depth is a non-paper smoke-only shortcut",
    )
    ap.add_argument(
        "--calibration-dir", default="artifacts/e0_qwen3_4b_to_1p7b_calibration"
    )
    ap.add_argument(
        "--kvbridge-artifact",
        default="artifacts/e0_qwen3_4b_to_1p7b_kvbridge",
        help="full-head artifact directory; matched-head writes the runtime checkpoint directly",
    )
    ap.add_argument("--output", default="checkpoints/qwen3_4b_to_1p7b_ridge.pt")
    ap.add_argument("--overwrite-artifact", action="store_true")
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="controlled Qwen weight offload during calibration capture",
    )
    args = ap.parse_args()

    if args.fit_profile == "16gb":
        args.low_vram = True
    if args.sequences <= 0 or args.seq_len <= 0 or args.stride <= 0 or args.k <= 0:
        raise ValueError("sequences, seq-len, stride, and k must be positive")
    if args.selection_sequences < 0 or args.selection_sequences > args.sequences:
        raise ValueError("selection-sequences must lie in [0, sequences]")
    if args.layer_selection == "depth" and not args.low_vram:
        raise ValueError("--layer-selection depth is restricted to explicit low-VRAM smoke runs")

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
        missing_set = set(missing_indices)
        windows = token_windows(
            tokenizer,
            iter_texts(args.text_file, limit=max(args.sequences * 8, args.sequences)),
            seq_len=args.seq_len,
            count=args.sequences,
        )
        available = {
            index: ids for index, ids in enumerate(windows) if index in missing_set
        }
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

    # Model residency is a capture problem, not a fitting problem.  Reclaim CUDA
    # before allocating R² / normal-equation statistics.
    del target, draft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.fit_profile != "manual":
        profile = e0_fit_profile(
            args.fit_profile,
            head_mode=args.head_mode,
            draft_layers=draft_signature.num_layers,
        )
        args.accumulation_device = profile.accumulation_device
        args.selection_layer_block = profile.selection_layer_block
        args.fit_layer_block = profile.fit_layer_block
    if args.accumulation_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA E0 fitting requested but CUDA is unavailable")

    selection_count = args.selection_sequences or args.sequences

    def external_factory(limit: int | None = None):
        use = required_paths if limit is None else required_paths[:limit]
        for path in use:
            yield load_calibration_shard(path)

    selection_config = FitConfig(
        top_k=args.k,
        ridge_alpha=args.lambda_,
        content_space=True,
        selection_alpha=1e-6,
        accumulation_dtype=args.accumulation_dtype,
        accumulation_device=args.accumulation_device,
        require_matched_kv=True,
        target_layer_block_size=args.fit_layer_block,
        selection_target_layer_block_size=args.selection_layer_block,
        token_stride=1,
    )

    if args.layer_selection == "depth":
        selected = _depth_selection(
            source_signature.num_layers,
            draft_signature.num_layers,
            args.k,
        )
        selection_scores = [[] for _ in selected]
    else:
        # KVBridge's audited R² selector is reused for both feature supports so the
        # full-head vs matched-head experiment changes only the head support.
        import kvbridge.fit as kvbridge_fit

        dtype = torch.float64 if args.accumulation_dtype == "float64" else torch.float32
        selected, selection_scores = kvbridge_fit._select_layers(
            lambda: external_factory(selection_count),
            source_signature,
            draft_signature,
            selection_config,
            dtype,
            torch.device(args.accumulation_device),
        )

    if args.head_mode == "full":
        import kvbridge.fit as kvbridge_fit

        # Fit on all requested calibration shards, but do not pay for R² selection
        # a second time.  The pinned backend remains responsible for centered stats.
        original_selector = kvbridge_fit._select_layers

        def fixed_selector(*_args, **_kwargs):
            return selected, selection_scores

        kvbridge_fit._select_layers = fixed_selector
        try:
            external = fit_mapper(
                lambda: external_factory(None),
                source_signature,
                draft_signature,
                selection_config,
            )
        finally:
            kvbridge_fit._select_layers = original_selector
        artifact = Path(args.kvbridge_artifact)
        external.save(
            artifact,
            overwrite=args.overwrite_artifact,
            storage_dtype="bfloat16",
        )
        runtime_mapper = RidgeKVMapper.from_kvbridge_artifact(
            artifact, dtype=torch.float32
        )
        backend = "kvbridge-full-head"
    else:
        def runtime_factory():
            for pair in external_factory(None):
                yield _to_runtime_cache(pair.source), _to_runtime_cache(pair.target)

        runtime_mapper = fit_matched_head_mapper_from_cache_pairs(
            runtime_factory,
            target_layers=source_signature.num_layers,
            draft_layers=draft_signature.num_layers,
            kv_heads=source_signature.num_kv_heads,
            head_dim=source_signature.head_dim,
            layer_selection=selected,
            lambda_=args.lambda_,
            accumulation_device=args.accumulation_device,
            layer_block_size=args.fit_layer_block,
            content_space=True,
        )
        backend = "matched-head-centered-ridge"

    runtime_mapper.save(args.output)
    metadata = {
        "args": vars(args),
        "source_signature": source_signature.to_dict(),
        "draft_signature": draft_signature.to_dict(),
        "selected_layers": runtime_mapper.metadata.layer_selection,
        "observations": args.sequences
        * ((args.seq_len + args.stride - 1) // args.stride),
        "selection_observations": selection_count
        * ((args.seq_len + args.stride - 1) // args.stride),
        "backend": backend,
        "content_space": True,
        "head_mode": args.head_mode,
        "feature_width": runtime_mapper.in_dim,
        "layer_selection": args.layer_selection,
        "selection_scores": selection_scores,
    }
    Path(args.output + ".json").write_text(json.dumps(metadata, indent=2))
    print(
        f"saved {args.head_mode} runtime mapper: {args.output}; "
        f"feature_width={runtime_mapper.in_dim}, fit_device={args.accumulation_device}, "
        f"selection_sequences={selection_count}"
    )


if __name__ == "__main__":
    main()
