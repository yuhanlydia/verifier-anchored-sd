#!/usr/bin/env python3
"""Evaluate native fidelity and verifier-target alignment for a mapped draft state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from verifier_anchored_sd.experiment_artifacts import (
    atomic_write_json,
    sha256_file,
    validate_protocol_contract,
)
from verifier_anchored_sd.model_contracts import model_metadata, tokenizer_contract_hash
from verifier_anchored_sd.spec_decode.hf_runtime import (
    capture_rotary_factors,
    forward_incremental,
)
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper
from verifier_anchored_sd.transfer_metrics import (
    attention_output_cosines,
    distribution_transfer_rows,
    finalize_screen,
    summarize_target_alignment,
    target_alignment_decision,
    target_alignment_rows,
    validate_screen_inputs,
)


def _probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits[:, -1, :].float(), dim=-1)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _gpu_description() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.cuda.current_device())


def _attention_modules(model) -> list:
    body = getattr(model, "model", model)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise RuntimeError("draft model does not expose model.layers for attention diagnostics")
    return [layer.self_attn for layer in layers]


def _incomplete_result(protocol_contract: dict, requested: int, completed: int, failure: dict) -> dict:
    return {
        "schema_version": 3,
        "experiment": "target_alignment_pair_screen",
        "status": "incomplete",
        "protocol_contract": protocol_contract,
        "requested_rows": requested,
        "completed_rows": completed,
        "native_fidelity": {"gate": {"status": "incomplete"}},
        "target_alignment": {"gate": {"status": "incomplete"}},
        "decision": {"status": "incomplete"},
        "failure": failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--mapper-metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="native-fidelity diagnostic threshold only; target alignment drives SD eligibility",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mapper-device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=14)
    parser.add_argument("--attention-cosine", action="store_true")
    args = parser.parse_args()
    if args.prompts <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("prompts and bootstrap samples must be positive")

    mapper_path = Path(args.mapper)
    mapper_metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    mapper_metadata = json.loads(mapper_metadata_path.read_text(encoding="utf-8"))
    mapper_sha = sha256_file(mapper_path)
    if mapper_sha != mapper_metadata["checkpoint_sha256"]:
        raise RuntimeError("mapper checkpoint digest differs from its metadata")
    if args.dtype != mapper_metadata.get("dtype"):
        raise RuntimeError("evaluation dtype differs from mapper calibration")

    screen_root = Path(args.screen_dir)
    screen_manifest = load_manifest(screen_root)
    validate_screen_inputs(mapper_metadata, screen_manifest)
    captured_count = int(screen_manifest["capture"]["count"])
    if args.prompts > captured_count:
        raise ValueError("requested prompts exceed the held-out capture count")
    token_rows, token_metadata = load_token_rows(screen_root / "tokens.pt")
    if token_metadata["token_rows_digest"] != screen_manifest["token_rows_digest"]:
        raise RuntimeError("screen token artifact differs from its manifest")
    window_sources = token_metadata.get("window_sources")
    if not isinstance(window_sources, list) or len(window_sources) != len(token_rows):
        raise RuntimeError("screen token artifact lacks per-document window provenance")
    shard_paths = exact_shard_paths(screen_root / "shards", captured_count)[: args.prompts]
    probability_paths = exact_probability_paths(
        screen_root / "target_probs", captured_count
    )[: args.prompts]

    pair = mapper_metadata["pair"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    protocol_contract = {
        "pair": pair,
        "target_revision": mapper_metadata["source_model"]["revision"],
        "draft_revision": draft_revision,
        "dtype": args.dtype,
        "calibration_input_sha256": mapper_metadata["calibration_input_sha256"],
        "evaluation_input_sha256": screen_manifest["input_sha256"],
        "calibration_capture": mapper_metadata["capture"],
        "mapper": mapper_metadata["mapper"],
        "mapper_checkpoint_sha256": mapper_sha,
        "screen_capture": screen_manifest["capture"],
        "target_distribution": screen_manifest["target_distribution"],
        "prompts": args.prompts,
        "bootstrap_samples": args.bootstrap_samples,
        "native_fidelity_threshold": args.threshold,
        "primary_gate": "verifier_target_alignment_delta",
        "attention_cosine": bool(args.attention_cosine),
        "device": args.device,
        "mapper_device": args.mapper_device,
        "gpu_memory_gib": args.gpu_memory_gib,
    }
    progress_path = Path(f"{args.output}.progress.json")
    rows = []
    prior_elapsed = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        validate_protocol_contract(progress, protocol_contract)
        rows = progress.get("rows", [])
        prior_elapsed = float(progress.get("elapsed_s", 0.0))
        if not isinstance(rows, list) or len(rows) > args.prompts:
            raise RuntimeError("invalid target-alignment progress artifact")

    def record_startup_oom(exc: torch.cuda.OutOfMemoryError) -> None:
        atomic_write_json(
            args.output,
            _incomplete_result(
                protocol_contract,
                args.prompts,
                len(rows),
                {
                    "phase": "draft_or_mapper_load",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            ),
        )

    tokenizer = load_hf_tokenizer(pair["draft"], revision=draft_revision)
    if tokenizer_contract_hash(tokenizer) != mapper_metadata["draft_model"]["tokenizer_hash"]:
        raise RuntimeError("draft tokenizer differs from mapper calibration")
    try:
        draft = load_hf_model(
            pair["draft"],
            args.device,
            args.dtype,
            revision=draft_revision,
            gpu_memory_gib=args.gpu_memory_gib,
            offload_folder=screen_root / "draft_offload",
        )
    except torch.cuda.OutOfMemoryError as exc:
        record_startup_oom(exc)
        raise
    loaded_draft = model_metadata(draft, pair["draft"], tokenizer_contract_hash(tokenizer))
    if loaded_draft != mapper_metadata["draft_model"]:
        raise RuntimeError("loaded draft contract differs from mapper calibration")
    output_embeddings = draft.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("draft model does not expose an output embedding head")
    expected_vocab_size = int(output_embeddings.weight.shape[0])
    if int(screen_manifest["target_distribution"].get("vocab_size", -1)) != expected_vocab_size:
        raise RuntimeError("verifier probability vocabulary differs from draft output vocabulary")

    map_dtype = resolve_dtype(args.dtype)
    try:
        mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
            args.mapper_device, dtype=map_dtype
        )
    except torch.cuda.OutOfMemoryError as exc:
        record_startup_oom(exc)
        raise
    input_device = draft.get_input_embeddings().weight.device

    attention_sink: list[torch.Tensor] = []
    handles = []
    if args.attention_cosine:

        def capture_attention(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            attention_sink.append(tensor[:, -1].detach().float().cpu())

        handles = [
            module.register_forward_hook(capture_attention)
            for module in _attention_modules(draft)
        ]

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        iterator = zip(
            token_rows[len(rows) : args.prompts],
            shard_paths[len(rows) :],
            probability_paths[len(rows) :],
            window_sources[len(rows) : args.prompts],
            strict=True,
        )
        for index, (token_row, shard_path, probability_path, source) in enumerate(
            iterator, start=len(rows)
        ):
            prefix, next_id = token_row[:-1], int(token_row[-1])
            target_cache, shard_metadata = load_cache_shard(shard_path)
            target_probs, probability_metadata = load_probability_shard(probability_path)
            validate_probability_binding(
                shard_metadata,
                probability_metadata,
                target_probs,
                expected_vocab_size=expected_vocab_size,
            )
            if shard_metadata["next_token_id"] != next_id:
                raise RuntimeError(f"next-token mismatch in screen shard: {shard_path}")
            if shard_metadata["token_digest"] != token_metadata["token_row_digests"][index]:
                raise RuntimeError(f"token-digest mismatch in screen shard: {shard_path}")

            ids = torch.tensor([prefix], device=input_device, dtype=torch.long)
            attention_sink.clear()
            native = forward_incremental(draft, ids)
            native_attention = list(attention_sink)

            history_length = len(prefix) - 1
            positions = torch.arange(history_length, device=input_device).unsqueeze(0)
            draft_rotary = capture_rotary_factors(draft, positions).to(mapper.device)
            target_history = target_cache.slice(0, history_length).to(
                mapper.device, dtype=map_dtype
            )
            mapped_history = mapper.map(target_history, draft_rotary=draft_rotary)
            attention_sink.clear()
            mapped = forward_incremental(draft, ids[:, -1:], mapped_history)
            mapped_attention = list(attention_sink)
            native_probs = _probs(native.logits).cpu()
            mapped_probs = _probs(mapped.logits).cpu()
            if native_probs.shape[-1] != expected_vocab_size or mapped_probs.shape[-1] != expected_vocab_size:
                raise RuntimeError("draft logits vocabulary changed during evaluation")

            row = distribution_transfer_rows(
                native_probs,
                mapped_probs,
                next_ids=[next_id],
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
                    "prefix_tokens": len(prefix),
                    "next_token_id": next_id,
                    "sequence_id": shard_metadata["sequence_id"],
                    "document_id": int(source["document_id"]),
                    "chunk_id": int(source["chunk_id"]),
                }
            )
            if args.attention_cosine:
                cosines = attention_output_cosines(native_attention, mapped_attention)
                row["attention_output_cosines"] = cosines
                row["mean_attention_output_cosine"] = sum(cosines) / len(cosines)
            rows.append(row)
            atomic_write_json(
                progress_path,
                {
                    "schema_version": 2,
                    "protocol_contract": protocol_contract,
                    "rows": rows,
                    "elapsed_s": prior_elapsed + time.perf_counter() - start,
                },
            )
            del (
                target_cache,
                target_probs,
                target_history,
                mapped_history,
                native,
                mapped,
                native_probs,
                mapped_probs,
                ids,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (index + 1) % 10 == 0 or index + 1 == args.prompts:
                print(f"target-alignment rows: {index + 1}/{args.prompts}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        failure = _incomplete_result(
            protocol_contract,
            args.prompts,
            len(rows),
            {"phase": "evaluation", "type": type(exc).__name__, "message": str(exc)},
        )
        failure["rows"] = rows
        failure["elapsed_s"] = prior_elapsed + time.perf_counter() - start
        atomic_write_json(
            progress_path,
            {
                "schema_version": 2,
                "protocol_contract": protocol_contract,
                "elapsed_s": failure["elapsed_s"],
                "rows": rows,
            },
        )
        atomic_write_json(args.output, failure)
        raise
    finally:
        for handle in handles:
            handle.remove()

    cluster_ids = [row["document_id"] for row in rows]
    native_fidelity = finalize_screen(
        rows,
        requested=args.prompts,
        samples=args.bootstrap_samples,
        seed=0,
        threshold=args.threshold,
        cluster_ids=cluster_ids,
    )
    target_alignment = summarize_target_alignment(
        rows,
        requested=args.prompts,
        samples=args.bootstrap_samples,
        seed=1,
        cluster_ids=cluster_ids,
    )
    decision_status = target_alignment_decision(
        target_alignment["gate"]["status"], native_fidelity["gate"]["status"]
    )
    result = {
        "schema_version": 3,
        "experiment": "target_alignment_pair_screen",
        "git_commit": _git_commit(),
        "status": "complete",
        "hardware": {
            "gpu": _gpu_description(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
        },
        "config": vars(args),
        "protocol_contract": protocol_contract,
        "pair": pair,
        "source_model": mapper_metadata["source_model"],
        "draft_model": mapper_metadata["draft_model"],
        "mapper_checkpoint_sha256": mapper_sha,
        "mapper_metadata_sha256": sha256_file(mapper_metadata_path),
        "calibration_input_sha256": mapper_metadata["calibration_input_sha256"],
        "evaluation_input_sha256": screen_manifest["input_sha256"],
        "source_capture_manifest_sha256": mapper_metadata["source_manifest_sha256"],
        "draft_capture_manifest_sha256": mapper_metadata["draft_manifest_sha256"],
        "screen_manifest_sha256": sha256_file(screen_root / "manifest.json"),
        "calibration_token_rows_digest": mapper_metadata["token_rows_digest"],
        "evaluation_token_rows_digest": screen_manifest["token_rows_digest"],
        "requested_rows": args.prompts,
        "completed_rows": len(rows),
        "native_fidelity": native_fidelity,
        "target_alignment": target_alignment,
        "decision": {
            "status": decision_status,
            "primary_metric": "mean_delta_target_alignment",
            "rule": "support->go_sd; harm->stop_pair; inconclusive->expand",
            "native_fidelity_is_diagnostic": True,
        },
        "elapsed_s": prior_elapsed + time.perf_counter() - start,
        "rows": rows,
    }
    atomic_write_json(args.output, result)
    progress_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "native_fidelity": native_fidelity,
                "target_alignment": target_alignment,
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        atomic_write_json(
            f"{output}.failure.json",
            {
                "status": "incomplete",
                "failure": {
                    "phase": "startup_or_evaluation",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )
        raise
