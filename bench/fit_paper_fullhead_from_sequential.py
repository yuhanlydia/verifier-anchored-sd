#!/usr/bin/env python3
"""Fit the paper-faithful full-head ridge mapper from sequential cache shards.

The source and receiver LLMs are never loaded by this script. It reuses exact-BF16
sequential calibration shards, invokes the pinned kvbridge two-pass fitter, converts
the resulting full-head mapper into the local runtime format, and emits modern
provenance metadata compatible with target-alignment and subspace evaluators.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from verifier_anchored_sd.cache_artifacts import (
    exact_shard_paths,
    load_cache_shard,
    load_manifest,
    validate_capture_pair,
)
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file
from verifier_anchored_sd.paper_mapper_baselines import (
    cache_state_to_kvbridge,
    model_signature_from_manifest,
)
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--kvbridge-artifact", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    ap.add_argument("--selection-ridge", type=float, default=1e-6)
    ap.add_argument(
        "--accumulation-device", choices=["cpu", "cuda"], default="cuda"
    )
    ap.add_argument(
        "--accumulation-dtype", choices=["float32", "float64"], default="float32"
    )
    ap.add_argument("--target-layer-block", type=int, default=1)
    ap.add_argument("--selection-layer-block", type=int, default=1)
    ap.add_argument(
        "--storage-dtype", choices=["float32", "bfloat16"], default="bfloat16"
    )
    args = ap.parse_args()
    if args.k <= 0 or args.lambda_ < 0 or args.selection_ridge < 0:
        raise ValueError("k must be positive and ridge penalties non-negative")
    if args.target_layer_block <= 0 or args.selection_layer_block <= 0:
        raise ValueError("layer block sizes must be positive")

    try:
        from kvbridge.config import FitConfig
        from kvbridge.fit import CalibrationPair, fit_mapper
    except ImportError as exc:
        raise RuntimeError("install the pinned kvbridge extra: pip install -e '.[kvbridge]'") from exc

    pair_root = Path(args.pair_dir)
    source_root, draft_root = pair_root / "source", pair_root / "draft"
    source_manifest = load_manifest(source_root)
    draft_manifest = load_manifest(draft_root)
    geometry = validate_capture_pair(source_manifest, draft_manifest)
    count = int(source_manifest["capture"]["count"])
    if count <= 0:
        raise RuntimeError("sequential calibration capture is empty")
    source_paths = exact_shard_paths(source_root / "shards", count)
    draft_paths = exact_shard_paths(draft_root / "shards", count)

    source_signature = model_signature_from_manifest(source_manifest)
    draft_signature = model_signature_from_manifest(draft_manifest)
    source_signature.validate_pair(draft_signature, require_matched_kv=True)
    config = FitConfig(
        top_k=args.k,
        ridge_alpha=args.lambda_,
        content_space=True,
        selection_alpha=args.selection_ridge,
        accumulation_dtype=args.accumulation_dtype,
        accumulation_device=args.accumulation_device,
        require_matched_kv=True,
        target_layer_block_size=args.target_layer_block,
        selection_target_layer_block_size=args.selection_layer_block,
        # capture_sequential_calibration already sampled positions using capture.stride
        token_stride=1,
    )

    def pair_factory():
        for index, (source_path, draft_path) in enumerate(
            zip(source_paths, draft_paths, strict=True)
        ):
            source_cache, source_meta = load_cache_shard(source_path)
            draft_cache, draft_meta = load_cache_shard(draft_path)
            for field in ("sequence_id", "token_digest", "dtype"):
                if source_meta.get(field) != draft_meta.get(field):
                    raise RuntimeError(
                        f"sequential calibration shard {index} differs on {field}"
                    )
            if source_meta.get("model_revision") != source_manifest["model"]["revision"]:
                raise RuntimeError("source calibration shard revision differs from manifest")
            if draft_meta.get("model_revision") != draft_manifest["model"]["revision"]:
                raise RuntimeError("draft calibration shard revision differs from manifest")
            if source_cache.seq_len != draft_cache.seq_len:
                raise RuntimeError("source/draft sampled cache lengths differ")
            yield CalibrationPair(
                cache_state_to_kvbridge(source_cache),
                cache_state_to_kvbridge(draft_cache),
            )

    output = Path(args.output)
    kvbridge_artifact = Path(args.kvbridge_artifact)
    try:
        external = fit_mapper(pair_factory, source_signature, draft_signature, config)
        external.save(
            kvbridge_artifact,
            overwrite=True,
            storage_dtype=args.storage_dtype,
        )
        runtime = RidgeKVMapper.from_kvbridge_artifact(
            kvbridge_artifact,
            dtype=torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        runtime.save(output)
    except torch.cuda.OutOfMemoryError as exc:
        atomic_write_json(
            f"{output}.failure.json",
            {
                "schema_version": 1,
                "status": "incomplete",
                "phase": "paper_full_head_ridge_fit",
                "pair_dir": str(pair_root.resolve()),
                "k": args.k,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "rule": "do_not_auto_change_k_dtype_model_or_pair",
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    metadata_path = Path(f"{output}.json")
    metadata = {
        "schema_version": 3,
        "git_commit": _git_commit(),
        "pair": source_manifest["pair"],
        "source_model": source_manifest["model"],
        "draft_model": draft_manifest["model"],
        "capture": source_manifest["capture"],
        "dtype": source_manifest["dtype"],
        "calibration_input_sha256": source_manifest["input_sha256"],
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "draft_manifest_sha256": sha256_file(draft_root / "manifest.json"),
        "token_rows_digest": source_manifest["token_rows_digest"],
        "token_row_digests": source_manifest["token_row_digests"],
        "mapper": {
            "kind": "ridge_full_head",
            "head_mode": "full",
            "content_space": True,
            "k": args.k,
            "lambda": args.lambda_,
            "selection_ridge": args.selection_ridge,
            "selected_layers": [list(row) for row in external.selected_layers],
            "selection_scores": [list(row) for row in external.selection_scores],
            "fit_key_r2": list(external.fit_key_r2),
            "fit_value_r2": list(external.fit_value_r2),
            "feature_width_max": max(
                len(row) * geometry["kv_heads"] * geometry["head_dim"]
                for row in external.selected_layers
            ),
            "backend": "pinned_kvbridge_full_head_ridge",
            "kvbridge_revision": "0d75f31dcde6eeceaa609d3affed6ca1401deb77",
            "accumulation_device": args.accumulation_device,
            "accumulation_dtype": args.accumulation_dtype,
        },
        "kvbridge_artifact": str(kvbridge_artifact),
        "kvbridge_manifest_sha256": sha256_file(kvbridge_artifact / "manifest.json"),
        "checkpoint_sha256": sha256_file(output),
    }
    atomic_write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "output": str(output),
                "k": args.k,
                "feature_width_max": metadata["mapper"]["feature_width_max"],
                "mean_key_r2": sum(external.fit_key_r2) / len(external.fit_key_r2),
                "mean_value_r2": sum(external.fit_value_r2) / len(external.fit_value_r2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        raise
