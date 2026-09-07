#!/usr/bin/env python3
"""Capture held-out verifier prefix KV for sequential pair screening."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from common import (
    iter_texts,
    load_hf_model,
    load_hf_tokenizer,
    token_windows_with_sources,
)

from verifier_anchored_sd.cache_artifacts import (
    load_cache_shard,
    load_token_rows,
    save_cache_shard,
    save_token_rows,
    write_or_validate_manifest,
)
from verifier_anchored_sd.experiment_artifacts import (
    atomic_write_json,
    sha256_file,
    token_rows_digest,
)
from verifier_anchored_sd.model_contracts import model_metadata, validate_tokenizer_pair
from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--mapper-metadata", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=14)
    args = parser.parse_args()
    if args.prompts <= 0 or args.prefix_tokens <= 1:
        raise ValueError("prompts must be positive and prefix-tokens must exceed one")

    mapper_metadata = json.loads(Path(args.mapper_metadata).read_text(encoding="utf-8"))
    if args.dtype != mapper_metadata.get("dtype"):
        raise RuntimeError("screen dtype differs from mapper calibration")
    pair = mapper_metadata["pair"]
    target_revision = mapper_metadata["source_model"]["revision"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    source_tokenizer = load_hf_tokenizer(pair["target"], revision=target_revision)
    draft_tokenizer = load_hf_tokenizer(pair["draft"], revision=draft_revision)
    tokenizer_hash = validate_tokenizer_pair(source_tokenizer, draft_tokenizer)
    if tokenizer_hash != mapper_metadata["source_model"]["tokenizer_hash"]:
        raise RuntimeError("current tokenizer contract differs from mapper calibration")

    root = Path(args.screen_dir)
    token_path = root / "tokens.pt"
    input_digest = sha256_file(args.text_file)
    if token_path.exists():
        rows, token_metadata = load_token_rows(token_path)
    else:
        windows = list(
            token_windows_with_sources(
                source_tokenizer,
                iter_texts(args.text_file, limit=max(args.prompts * 8, args.prompts)),
                seq_len=args.prefix_tokens + 1,
                count=args.prompts,
            )
        )
        if len(windows) != args.prompts:
            raise RuntimeError(f"only {len(windows)}/{args.prompts} held-out windows were available")
        rows = [[int(token) for token in window.token_ids.tolist()] for window in windows]
        prefix_rows = [row[:-1] for row in rows]
        window_sources = [
            {"document_id": window.document_id, "chunk_id": window.chunk_id}
            for window in windows
        ]
        token_metadata = {
            "schema_version": 2,
            "pair": pair,
            "input_sha256": input_digest,
            "count": args.prompts,
            "prefix_tokens": args.prefix_tokens,
            "tokenizer_hash": tokenizer_hash,
            "token_rows_digest": token_rows_digest(prefix_rows),
            "token_row_digests": [token_rows_digest([row]) for row in prefix_rows],
            "window_sources": window_sources,
        }
        save_token_rows(token_path, rows, token_metadata)
    expected_token_metadata = {
        "schema_version": 2,
        "pair": pair,
        "input_sha256": input_digest,
        "count": args.prompts,
        "prefix_tokens": args.prefix_tokens,
        "tokenizer_hash": tokenizer_hash,
        "token_rows_digest": token_rows_digest([row[:-1] for row in rows]),
        "token_row_digests": [token_rows_digest([row[:-1]]) for row in rows],
        "window_sources": token_metadata.get("window_sources"),
    }
    if token_metadata != expected_token_metadata:
        raise RuntimeError(f"frozen screen tokens use a different contract: {token_path}")
    write_or_validate_manifest(root / "tokens", token_metadata)

    target = load_hf_model(
        pair["target"],
        args.device,
        args.dtype,
        revision=target_revision,
        gpu_memory_gib=args.gpu_memory_gib,
        offload_folder=root / "offload",
    )
    model_info = model_metadata(target, pair["target"], tokenizer_hash)
    if model_info["revision"] != target_revision:
        raise RuntimeError("loaded verifier revision differs from mapper calibration")
    manifest = {
        "schema_version": 2,
        "role": "screen_source",
        "pair": pair,
        "model": model_info,
        "capture": {"count": args.prompts, "prefix_tokens": args.prefix_tokens, "stride": 1},
        "dtype": args.dtype,
        "input_sha256": input_digest,
        "window_sources": token_metadata["window_sources"],
        "token_rows_digest": token_metadata["token_rows_digest"],
        "token_row_digests": token_metadata["token_row_digests"],
    }
    write_or_validate_manifest(root, manifest)
    shard_root = root / "shards"
    for index, row in enumerate(rows):
        prefix = row[:-1]
        path = shard_root / f"{index:05d}.pt"
        metadata = {
            "sequence_id": f"{index:05d}",
            "token_digest": token_rows_digest([prefix]),
            "next_token_id": int(row[-1]),
            "model_revision": model_info["revision"],
            "dtype": args.dtype,
        }
        if path.exists():
            existing, existing_metadata = load_cache_shard(path)
            if existing_metadata != metadata or existing.seq_len != args.prefix_tokens:
                raise RuntimeError(f"existing screen shard has a different contract: {path}")
            continue
        ids = torch.tensor([prefix], device=_input_device(target), dtype=torch.long)
        cache = forward_incremental(target, ids).cache
        save_cache_shard(path, cache, metadata)
        del cache, ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"screen verifier shards: {index + 1}/{len(rows)}", flush=True)

    del target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    atomic_write_json(root / "complete.json", {"complete": True, "count": len(rows)})


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        root = Path(sys.argv[sys.argv.index("--screen-dir") + 1])
        requested = (
            int(sys.argv[sys.argv.index("--prompts") + 1])
            if "--prompts" in sys.argv
            else 128
        )
        completed = len(list((root / "shards").glob("*.pt")))
        atomic_write_json(
            root / "failure.json",
            {
                "status": "incomplete",
                "requested_rows": requested,
                "completed_rows": completed,
                "failure": {
                    "phase": "model_load_or_capture",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )
        raise
