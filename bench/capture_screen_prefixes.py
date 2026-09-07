#!/usr/bin/env python3
"""Capture held-out verifier KV and next-token distributions for pair screening."""

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
    exact_shard_paths,
    load_cache_shard,
    load_token_rows,
    save_cache_shard,
    save_token_rows,
    write_or_validate_manifest,
)
from verifier_anchored_sd.distribution_artifacts import (
    exact_probability_paths,
    load_probability_shard,
    save_probability_shard,
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


def _probability_metadata(cache_metadata: dict, vocab_size: int) -> dict:
    return {
        **cache_metadata,
        "storage_dtype": "float32",
        "temperature": 1.0,
        "vocab_size": int(vocab_size),
    }


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
    output_embeddings = target.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("verifier does not expose an output embedding head")
    vocab_size = int(output_embeddings.weight.shape[0])
    manifest = {
        "schema_version": 3,
        "role": "screen_source_with_distribution",
        "pair": pair,
        "model": model_info,
        "capture": {"count": args.prompts, "prefix_tokens": args.prefix_tokens, "stride": 1},
        "dtype": args.dtype,
        "target_distribution": {
            "storage_dtype": "float32",
            "temperature": 1.0,
            "vocab_size": vocab_size,
        },
        "input_sha256": input_digest,
        "window_sources": token_metadata["window_sources"],
        "token_rows_digest": token_metadata["token_rows_digest"],
        "token_row_digests": token_metadata["token_row_digests"],
    }
    write_or_validate_manifest(root, manifest)
    shard_root = root / "shards"
    probability_root = root / "target_probs"

    for index, row in enumerate(rows):
        prefix = row[:-1]
        cache_path = shard_root / f"{index:05d}.pt"
        probability_path = probability_root / f"{index:05d}.pt"
        cache_metadata = {
            "sequence_id": f"{index:05d}",
            "token_digest": token_rows_digest([prefix]),
            "next_token_id": int(row[-1]),
            "model_revision": model_info["revision"],
            "dtype": args.dtype,
        }
        probability_metadata = _probability_metadata(cache_metadata, vocab_size)

        cache_ok = False
        if cache_path.exists():
            existing, existing_metadata = load_cache_shard(cache_path)
            if existing_metadata != cache_metadata or existing.seq_len != args.prefix_tokens:
                raise RuntimeError(
                    f"existing screen cache shard has a different contract: {cache_path}"
                )
            cache_ok = True
            del existing

        probability_ok = False
        if probability_path.exists():
            existing_probs, existing_probability_metadata = load_probability_shard(probability_path)
            if existing_probability_metadata != probability_metadata:
                raise RuntimeError(
                    "existing target probability shard has a different contract: "
                    f"{probability_path}"
                )
            if existing_probs.numel() != vocab_size:
                raise RuntimeError(
                    f"existing target probability vocabulary differs: {probability_path}"
                )
            probability_ok = True
            del existing_probs

        if cache_ok and probability_ok:
            continue

        ids = torch.tensor([prefix], device=_input_device(target), dtype=torch.long)
        out = forward_incremental(target, ids)
        probabilities = torch.softmax(out.logits[:, -1, :].float(), dim=-1)[0]
        if probabilities.numel() != vocab_size:
            raise RuntimeError("verifier logits vocabulary differs from output embedding vocabulary")
        if not cache_ok:
            save_cache_shard(cache_path, out.cache, cache_metadata)
        if not probability_ok:
            save_probability_shard(probability_path, probabilities, probability_metadata)
        del out, probabilities, ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"screen verifier cache+distribution: {index + 1}/{len(rows)}", flush=True)

    exact_shard_paths(shard_root, len(rows))
    exact_probability_paths(probability_root, len(rows))
    del target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    atomic_write_json(
        root / "complete.json",
        {
            "complete": True,
            "count": len(rows),
            "schema_version": 3,
            "target_distribution": manifest["target_distribution"],
        },
    )


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        root = Path(sys.argv[sys.argv.index("--screen-dir") + 1])
        requested = int(sys.argv[sys.argv.index("--prompts") + 1]) if "--prompts" in sys.argv else 128
        cache_completed = len(list((root / "shards").glob("*.pt")))
        probability_completed = len(list((root / "target_probs").glob("*.pt")))
        atomic_write_json(
            root / "failure.json",
            {
                "status": "incomplete",
                "requested_rows": requested,
                "completed_cache_rows": cache_completed,
                "completed_probability_rows": probability_completed,
                "completed_rows": min(cache_completed, probability_completed),
                "failure": {
                    "phase": "model_load_or_capture",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )
        raise
