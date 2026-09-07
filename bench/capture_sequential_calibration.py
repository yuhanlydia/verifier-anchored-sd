#!/usr/bin/env python3
"""Capture one side of a calibration pair without co-loading both LLMs."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from common import (
    iter_texts,
    load_hf_model,
    load_hf_tokenizer,
    token_windows,
)

from verifier_anchored_sd.cache_artifacts import (
    load_cache_shard,
    load_token_rows,
    sample_cache_tokens,
    save_cache_shard,
    save_token_rows,
    write_or_validate_manifest,
)
from verifier_anchored_sd.experiment_artifacts import (
    atomic_write_json,
    sha256_file,
    token_rows_digest,
)
from verifier_anchored_sd.model_contracts import (
    model_metadata,
    validate_tokenizer_pair,
)
from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental


def _input_device(model) -> torch.device:
    device = model.get_input_embeddings().weight.device
    if device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("model has no materialized input device")


def _token_contract(args, tokenizer_hash: str, input_digest: str, rows) -> dict:
    return {
        "schema_version": 1,
        "pair": {"target": args.target, "draft": args.draft},
        "requested_revisions": {
            "target": args.target_revision,
            "draft": args.draft_revision,
        },
        "input_sha256": input_digest,
        "count": args.sequences,
        "seq_len": args.seq_len,
        "tokenizer_hash": tokenizer_hash,
        "token_rows_digest": token_rows_digest(rows),
    }


def _load_or_freeze_rows(args, tokenizer, tokenizer_hash: str) -> tuple[list[list[int]], dict]:
    root = Path(args.pair_dir)
    token_path = root / "tokens.pt"
    input_digest = sha256_file(args.text_file)
    if token_path.exists():
        rows, stored = load_token_rows(token_path)
        expected = _token_contract(args, tokenizer_hash, input_digest, rows)
        if stored != expected:
            raise RuntimeError(f"frozen token windows use a different contract: {token_path}")
        write_or_validate_manifest(root / "tokens", expected)
        return rows, stored

    windows = list(
        token_windows(
            tokenizer,
            iter_texts(args.text_file, limit=max(args.sequences * 8, args.sequences)),
            seq_len=args.seq_len,
            count=args.sequences,
        )
    )
    if len(windows) != args.sequences:
        raise RuntimeError(
            f"only {len(windows)}/{args.sequences} calibration windows were available"
        )
    rows = [[int(token) for token in window.tolist()] for window in windows]
    contract = _token_contract(args, tokenizer_hash, input_digest, rows)
    write_or_validate_manifest(root / "tokens", contract)
    save_token_rows(token_path, rows, contract)
    return rows, contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--role", required=True, choices=["source", "draft"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target-revision", default="main")
    parser.add_argument("--draft-revision", default="main")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        default=14,
        help="exact-weight GPU cap; remaining weights are offloaded to CPU",
    )
    args = parser.parse_args()
    if args.sequences <= 0 or args.seq_len <= 0 or args.stride <= 0:
        raise ValueError("sequences, seq-len, and stride must be positive")

    source_tokenizer = load_hf_tokenizer(args.target, revision=args.target_revision)
    draft_tokenizer = load_hf_tokenizer(args.draft, revision=args.draft_revision)
    tokenizer_hash = validate_tokenizer_pair(source_tokenizer, draft_tokenizer)
    rows, token_contract = _load_or_freeze_rows(args, source_tokenizer, tokenizer_hash)

    model_id = args.target if args.role == "source" else args.draft
    revision = args.target_revision if args.role == "source" else args.draft_revision
    role_root = Path(args.pair_dir) / args.role
    model = load_hf_model(
        model_id,
        args.device,
        args.dtype,
        revision=revision,
        gpu_memory_gib=args.gpu_memory_gib,
        offload_folder=role_root / "offload",
    )
    model_info = model_metadata(model, model_id, tokenizer_hash)
    manifest = {
        "schema_version": 1,
        "role": args.role,
        "pair": {"target": args.target, "draft": args.draft},
        "model": model_info,
        "capture": {
            "count": args.sequences,
            "seq_len": args.seq_len,
            "stride": args.stride,
        },
        "token_rows_digest": token_contract["token_rows_digest"],
    }
    write_or_validate_manifest(role_root, manifest)
    shard_root = role_root / "shards"
    sampled_tokens = (args.seq_len + args.stride - 1) // args.stride
    input_device = _input_device(model)
    for index, row in enumerate(rows):
        path = shard_root / f"{index:05d}.pt"
        row_digest = token_rows_digest([row])
        metadata = {
            "sequence_id": f"{index:05d}",
            "token_digest": row_digest,
            "role": args.role,
            "model_revision": model_info["revision"],
        }
        if path.exists():
            existing, existing_metadata = load_cache_shard(path)
            if existing_metadata != metadata or existing.seq_len != sampled_tokens:
                raise RuntimeError(f"existing cache shard has a different contract: {path}")
            continue
        ids = torch.tensor([row], device=input_device, dtype=torch.long)
        captured = forward_incremental(model, ids).cache
        sampled = sample_cache_tokens(captured, args.stride)
        save_cache_shard(path, sampled, metadata)
        del captured, sampled, ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"{args.role} cache shards: {index + 1}/{len(rows)}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    atomic_write_json(
        role_root / "complete.json",
        {
            "complete": True,
            "count": len(rows),
            "model_revision": model_info["revision"],
            "token_rows_digest": token_contract["token_rows_digest"],
        },
    )


if __name__ == "__main__":
    main()
