#!/usr/bin/env python3
"""Fit a matched-head mapper from separately captured source/draft shards."""

from __future__ import annotations

import argparse
import subprocess
from itertools import islice
from pathlib import Path

from verifier_anchored_sd.cache_artifacts import (
    exact_shard_paths,
    load_cache_shard,
    load_manifest,
    validate_capture_pair,
)
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file
from verifier_anchored_sd.spec_decode.head_local_fit import (
    fit_matched_head_mapper_from_cache_pairs,
    select_source_layers_by_r2,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    parser.add_argument("--selection-ridge", type=float, default=1e-6)
    parser.add_argument("--selection-sequences", type=int, default=32)
    parser.add_argument("--accumulation-device", default="cuda")
    parser.add_argument("--selection-layer-block", type=int, default=4)
    parser.add_argument("--fit-layer-block", type=int, default=8)
    args = parser.parse_args()
    if min(
        args.k,
        args.selection_sequences,
        args.selection_layer_block,
        args.fit_layer_block,
    ) <= 0:
        raise ValueError("k, selection sequences, and block sizes must be positive")
    if args.lambda_ < 0 or args.selection_ridge < 0:
        raise ValueError("ridge coefficients must be non-negative")

    pair_root = Path(args.pair_dir)
    source_manifest = load_manifest(pair_root / "source")
    draft_manifest = load_manifest(pair_root / "draft")
    geometry = validate_capture_pair(source_manifest, draft_manifest)
    count = int(source_manifest["capture"]["count"])
    if args.selection_sequences > count:
        raise ValueError("selection-sequences cannot exceed captured sequence count")
    source_paths = exact_shard_paths(pair_root / "source" / "shards", count)
    draft_paths = exact_shard_paths(pair_root / "draft" / "shards", count)
    path_pairs = list(zip(source_paths, draft_paths, strict=True))

    def pair_factory(limit: int | None = None):
        use = path_pairs if limit is None else islice(path_pairs, limit)
        for source_path, draft_path in use:
            source, source_meta = load_cache_shard(source_path)
            draft, draft_meta = load_cache_shard(draft_path)
            if source_meta["sequence_id"] != draft_meta["sequence_id"]:
                raise RuntimeError(f"calibration sequence mismatch: {source_path}")
            if source_meta["token_digest"] != draft_meta["token_digest"]:
                raise RuntimeError(f"calibration token mismatch: {source_path}")
            yield source, draft

    selected, selection_scores = select_source_layers_by_r2(
        lambda: pair_factory(args.selection_sequences),
        **geometry,
        top_k=args.k,
        device=args.accumulation_device,
        layer_block_size=args.selection_layer_block,
        content_space=True,
        ridge=args.selection_ridge,
    )
    mapper = fit_matched_head_mapper_from_cache_pairs(
        lambda: pair_factory(None),
        **geometry,
        layer_selection=selected,
        lambda_=args.lambda_,
        accumulation_device=args.accumulation_device,
        layer_block_size=args.fit_layer_block,
        content_space=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    mapper.save(temporary)
    temporary.replace(output)
    atomic_write_json(
        output.with_suffix(output.suffix + ".json"),
        {
            "schema_version": 1,
            "git_commit": _git_commit(),
            "pair": source_manifest["pair"],
            "source_model": source_manifest["model"],
            "draft_model": draft_manifest["model"],
            "capture": source_manifest["capture"],
            "token_rows_digest": source_manifest["token_rows_digest"],
            "mapper": {
                "head_mode": "matched",
                "content_space": True,
                "k": args.k,
                "lambda": args.lambda_,
                "selection_ridge": args.selection_ridge,
                "selection_sequences": args.selection_sequences,
                "selected_layers": selected,
                "selection_scores": selection_scores,
            },
            "checkpoint_sha256": sha256_file(output),
        },
    )


if __name__ == "__main__":
    main()
