#!/usr/bin/env python3
"""Compare any declared runtime mapper against native 4B on verifier alignment.

Matched-head ridge, paper full-head ridge, and paper full-head MLP all run through
this exact evaluator.  The verifier is not loaded: the script consumes immutable
verifier KV + FP32 next-token probability artifacts, loads only the frozen draft,
and keeps the newest causal frontier native to the draft.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from common import load_hf_model, load_hf_tokenizer, resolve_dtype

from verifier_anchored_sd.cache_artifacts import (
    exact_shard_paths,
    load_cache_shard,
    load_manifest,
    load_token_rows,
)
from verifier_anchored_sd.distribution_artifacts import (
    exact_probability_paths,
    load_probability_shard,
    validate_probability_binding,
)
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file
from verifier_anchored_sd.mapper_io import load_runtime_mapper
from verifier_anchored_sd.model_contracts import model_metadata, tokenizer_contract_hash
from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental
from verifier_anchored_sd.transfer_metrics import (
    distribution_transfer_rows,
    finalize_screen,
    summarize_target_alignment,
    target_alignment_decision,
    target_alignment_rows,
    validate_screen_inputs,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits[:, -1, :].float(), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--mapper-metadata")
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--native-fidelity-threshold", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mapper-device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu-memory-gib", type=int, default=28)
    args = ap.parse_args()
    if args.prompts <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("prompts and bootstrap-samples must be positive")

    screen_root = Path(args.screen_dir)
    mapper_path = Path(args.mapper)
    mapper_metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    mapper_metadata = json.loads(mapper_metadata_path.read_text(encoding="utf-8"))
    mapper_sha = sha256_file(mapper_path)
    if mapper_sha != mapper_metadata.get("checkpoint_sha256"):
        raise RuntimeError("mapper checkpoint digest differs from mapper metadata")
    screen_manifest = load_manifest(screen_root)
    validate_screen_inputs(mapper_metadata, screen_manifest)
    captured = int(screen_manifest["capture"]["count"])
    if args.prompts > captured:
        raise ValueError("requested prompts exceed captured mapper-evaluation rows")

    token_rows, token_metadata = load_token_rows(screen_root / "tokens.pt")
    if token_metadata.get("token_rows_digest") != screen_manifest.get("token_rows_digest"):
        raise RuntimeError("mapper-evaluation token artifact differs from manifest")
    sources = token_metadata.get("window_sources")
    if not isinstance(sources, list) or len(sources) < args.prompts:
        raise RuntimeError("mapper-evaluation artifact lacks source-document provenance")
    cache_paths = exact_shard_paths(screen_root / "shards", captured)[: args.prompts]
    probability_paths = exact_probability_paths(screen_root / "target_probs", captured)[
        : args.prompts
    ]

    pair = mapper_metadata["pair"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    tokenizer = load_hf_tokenizer(pair["draft"], revision=draft_revision)
    tokenizer_hash = tokenizer_contract_hash(tokenizer)
    if tokenizer_hash != mapper_metadata["draft_model"]["tokenizer_hash"]:
        raise RuntimeError("loaded draft tokenizer differs from mapper provenance")
    draft = load_hf_model(
        pair["draft"],
        args.device,
        args.dtype,
        revision=draft_revision,
        gpu_memory_gib=args.gpu_memory_gib,
        offload_folder=screen_root / "generic_mapper_draft_offload",
    )
    if model_metadata(draft, pair["draft"], tokenizer_hash) != mapper_metadata["draft_model"]:
        raise RuntimeError("loaded draft differs from mapper provenance")
    for parameter in draft.parameters():
        parameter.requires_grad_(False)
    draft.eval()
    draft_vocab = int(draft.get_output_embeddings().weight.shape[0])

    mapper = load_runtime_mapper(mapper_path, mapper_metadata_path).to(
        args.mapper_device,
        dtype=resolve_dtype(args.dtype),
    )
    input_device = _input_device(draft)
    rows: list[dict] = []
    start = time.perf_counter()
    output_path = Path(args.output)

    try:
        for index, (token_row, cache_path, probability_path, source) in enumerate(
            zip(
                token_rows[: args.prompts],
                cache_paths,
                probability_paths,
                sources[: args.prompts],
                strict=True,
            )
        ):
            prefix = [int(token) for token in token_row[:-1]]
            next_id = int(token_row[-1])
            if len(prefix) < 2:
                raise RuntimeError("mapper evaluation prefix needs history plus native frontier")
            history_ids, frontier = prefix[:-1], prefix[-1]
            target_cache, cache_metadata = load_cache_shard(cache_path)
            target_probs, probability_metadata = load_probability_shard(probability_path)
            validate_probability_binding(
                cache_metadata,
                probability_metadata,
                target_probs,
                expected_vocab_size=draft_vocab,
            )
            if target_cache.seq_len != len(prefix):
                raise RuntimeError("verifier cache length differs from frozen prefix")
            if cache_metadata.get("next_token_id") != next_id:
                raise RuntimeError("verifier cache next-token binding differs")

            history_tensor = torch.tensor([history_ids], device=input_device, dtype=torch.long)
            frontier_tensor = torch.tensor([[frontier]], device=input_device, dtype=torch.long)
            with torch.no_grad():
                native_history = forward_incremental(draft, history_tensor).cache.to(input_device)
                native_step = forward_incremental(draft, frontier_tensor, native_history)
                target_history = target_cache.slice(0, len(history_ids)).to(
                    mapper.device, dtype=resolve_dtype(args.dtype)
                )
                mapped_history = mapper.map(
                    target_history,
                    draft_rotary=native_history.rotary.to(mapper.device),
                ).to(input_device)
                mapped_step = forward_incremental(draft, frontier_tensor, mapped_history)
            native_probs = _probs(native_step.logits).cpu()
            mapped_probs = _probs(mapped_step.logits).cpu()
            row = distribution_transfer_rows(
                native_probs, mapped_probs, next_ids=[next_id]
            )[0]
            row.update(
                target_alignment_rows(
                    target_probs.unsqueeze(0),
                    native_probs,
                    mapped_probs,
                    next_ids=[next_id],
                )[0]
            )
            row.update(
                {
                    "prompt": index,
                    "sequence_id": cache_metadata["sequence_id"],
                    "document_id": int(source["document_id"]),
                    "chunk_id": int(source["chunk_id"]),
                    "prefix_tokens": len(prefix),
                }
            )
            rows.append(row)
            atomic_write_json(
                f"{output_path}.progress.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "requested_rows": args.prompts,
                    "completed_rows": len(rows),
                    "mapper_checkpoint_sha256": mapper_sha,
                    "rows": rows,
                },
            )
            del (
                target_cache,
                target_probs,
                native_history,
                mapped_history,
                native_step,
                mapped_step,
                native_probs,
                mapped_probs,
                history_tensor,
                frontier_tensor,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (index + 1) % 10 == 0 or index + 1 == args.prompts:
                print(f"generic-mapper alignment rows: {index + 1}/{args.prompts}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        atomic_write_json(
            f"{output_path}.failure.json",
            {
                "schema_version": 1,
                "status": "incomplete",
                "phase": "generic_mapper_alignment",
                "requested_rows": args.prompts,
                "completed_rows": len(rows),
                "mapper_kind": mapper_metadata.get("mapper", {}).get("kind", "ridge"),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "rule": "do_not_auto_change_mapper_dtype_model_k_or_hidden_size",
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    clusters = [row["document_id"] for row in rows]
    native_fidelity = finalize_screen(
        rows,
        requested=args.prompts,
        samples=args.bootstrap_samples,
        seed=0,
        threshold=args.native_fidelity_threshold,
        cluster_ids=clusters,
    )
    target_alignment = summarize_target_alignment(
        rows,
        requested=args.prompts,
        samples=args.bootstrap_samples,
        seed=1,
        cluster_ids=clusters,
    )
    decision = target_alignment_decision(
        target_alignment["gate"]["status"], native_fidelity["gate"]["status"]
    )
    result = {
        "schema_version": 1,
        "experiment": "generic_mapper_target_alignment",
        "git_commit": _git_commit(),
        "status": "complete",
        "pair": pair,
        "source_model": mapper_metadata["source_model"],
        "draft_model": mapper_metadata["draft_model"],
        "mapper": mapper_metadata.get("mapper", {}),
        "mapper_checkpoint_sha256": mapper_sha,
        "mapper_metadata_sha256": sha256_file(mapper_metadata_path),
        "screen_manifest_sha256": sha256_file(screen_root / "manifest.json"),
        "evaluation_input_sha256": screen_manifest["input_sha256"],
        "requested_rows": args.prompts,
        "completed_rows": len(rows),
        "native_fidelity": native_fidelity,
        "target_alignment": target_alignment,
        "decision": {
            "status": decision,
            "primary_metric": "mean_delta_target_alignment",
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
        },
        "elapsed_s": time.perf_counter() - start,
        "rows": rows,
    }
    atomic_write_json(output_path, result)
    Path(f"{output_path}.progress.json").unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "mapper": result["mapper"],
                "native_fidelity": native_fidelity["summary"],
                "target_alignment": target_alignment,
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
