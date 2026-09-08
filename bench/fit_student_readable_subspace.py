#!/usr/bin/env python3
"""Fit student-readable KV subspaces using a frozen draft decoder.

The verifier model is not loaded by this program.  It consumes a sequential screen
artifact containing exact verifier KV and FP32 next-token distributions, then uses
only the frozen draft model plus the existing target->draft mapper.

For each prefix it accumulates three small per-layer/head statistics:

* mapped-state gradient covariance E[g g^T] (SPD-style sensitivity baseline),
* native-gradient / teacher-delta signed benefit matrix,
* centered mapped-activation covariance (PCA control).

No per-token gradients are persisted after each prefix.
"""

from __future__ import annotations

import argparse
import gc
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
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file
from verifier_anchored_sd.kv_subspace import (
    SubspaceBasisArtifact,
    deterministic_random_basis,
)
from verifier_anchored_sd.model_contracts import model_metadata, tokenizer_contract_hash
from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper
from verifier_anchored_sd.subspace_fit import (
    assert_gradient_coverage,
    cache_content_rows,
    cache_gradient_rows,
    gradient_norms,
    make_gradient_cache,
    target_kl_loss,
    teacher_delta_rows,
    validate_subspace_fit_inputs,
)
from verifier_anchored_sd.subspace_stats import (
    ActivationCovarianceStats,
    BenefitCrossStats,
    GradientCovarianceStats,
    benefit_basis,
    pca_basis,
    sensitivity_basis,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _input_device(model) -> torch.device:
    device = model.get_input_embeddings().weight.device
    if device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("draft model has no materialized input device")


def _freeze_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _family_payload(solution):
    return solution.vectors, solution.eigenvalues, solution.valid_ranks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--mapper-metadata")
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--max-rank", type=int, default=64)
    ap.add_argument("--gradient-smoke-prefixes", type=int, default=4)
    ap.add_argument("--benefit-eigenvalue-tol", type=float, default=1e-10)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mapper-device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu-memory-gib", type=int, default=28)
    args = ap.parse_args()

    if args.prompts <= 0:
        raise ValueError("prompts must be positive")
    if args.gradient_smoke_prefixes <= 0 or args.gradient_smoke_prefixes > args.prompts:
        raise ValueError("gradient-smoke-prefixes must lie in [1, prompts]")
    if args.max_rank <= 0:
        raise ValueError("max-rank must be positive")
    if args.benefit_eigenvalue_tol < 0:
        raise ValueError("benefit-eigenvalue-tol must be non-negative")

    screen_root = Path(args.screen_dir)
    mapper_path = Path(args.mapper)
    mapper_metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    output_path = Path(args.output)
    mapper_metadata = json.loads(mapper_metadata_path.read_text(encoding="utf-8"))
    mapper_sha = sha256_file(mapper_path)
    screen_manifest = load_manifest(screen_root)
    geometry = validate_subspace_fit_inputs(
        mapper_metadata,
        screen_manifest,
        mapper_sha256=mapper_sha,
        requested_prompts=args.prompts,
    )
    if args.max_rank > geometry["head_dim"]:
        raise ValueError("max-rank cannot exceed draft KV head dimension")
    if args.dtype != mapper_metadata.get("dtype"):
        raise RuntimeError("subspace fitting dtype differs from mapper calibration")

    token_rows, token_metadata = load_token_rows(screen_root / "tokens.pt")
    if token_metadata.get("token_rows_digest") != screen_manifest.get("token_rows_digest"):
        raise RuntimeError("subspace screen token artifact differs from its manifest")
    if len(token_rows) < args.prompts:
        raise RuntimeError("subspace screen contains fewer token rows than requested")
    cache_paths = exact_shard_paths(
        screen_root / "shards", int(screen_manifest["capture"]["count"])
    )[: args.prompts]
    probability_paths = exact_probability_paths(
        screen_root / "target_probs", int(screen_manifest["capture"]["count"])
    )[: args.prompts]

    pair = mapper_metadata["pair"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    tokenizer = load_hf_tokenizer(pair["draft"], revision=draft_revision)
    tokenizer_hash = tokenizer_contract_hash(tokenizer)
    if tokenizer_hash != mapper_metadata["draft_model"]["tokenizer_hash"]:
        raise RuntimeError("loaded draft tokenizer differs from mapper calibration")

    draft = load_hf_model(
        pair["draft"],
        args.device,
        args.dtype,
        revision=draft_revision,
        gpu_memory_gib=args.gpu_memory_gib,
        offload_folder=screen_root / "subspace_draft_offload",
    )
    if model_metadata(draft, pair["draft"], tokenizer_hash) != mapper_metadata["draft_model"]:
        raise RuntimeError("loaded draft model differs from mapper calibration")
    _freeze_model(draft)
    output_embeddings = draft.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("draft model does not expose an output head")
    draft_vocab = int(output_embeddings.weight.shape[0])
    if draft_vocab != geometry["vocab_size"]:
        raise RuntimeError("draft output vocabulary differs from verifier probability artifact")

    mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
        args.mapper_device, dtype=resolve_dtype(args.dtype)
    )
    if mapper.metadata.draft_layers != geometry["draft_layers"]:
        raise RuntimeError("mapper draft layer count differs from artifact geometry")
    if mapper.metadata.draft_kv_heads != geometry["kv_heads"]:
        raise RuntimeError("mapper draft KV-head count differs from artifact geometry")
    if mapper.metadata.head_dim != geometry["head_dim"]:
        raise RuntimeError("mapper head dimension differs from artifact geometry")

    prefix_shape = (geometry["draft_layers"], 2, geometry["kv_heads"])
    grad_stats = GradientCovarianceStats(prefix_shape=prefix_shape, dim=geometry["head_dim"])
    benefit_stats = BenefitCrossStats(prefix_shape=prefix_shape, dim=geometry["head_dim"])
    pca_stats = ActivationCovarianceStats(prefix_shape=prefix_shape, dim=geometry["head_dim"])
    native_norm_sums = torch.zeros(prefix_shape, dtype=torch.float64)
    mapped_norm_sums = torch.zeros(prefix_shape, dtype=torch.float64)
    native_losses: list[float] = []
    mapped_losses: list[float] = []
    input_device = _input_device(draft)

    progress_path = Path(f"{output_path}.progress.json")
    start = time.perf_counter()
    completed = 0
    try:
        for index, (token_row, cache_path, probability_path) in enumerate(
            zip(token_rows[: args.prompts], cache_paths, probability_paths, strict=True)
        ):
            prefix = [int(token) for token in token_row[:-1]]
            if len(prefix) < 2:
                raise RuntimeError("subspace fitting prefix must contain history plus frontier")
            history_ids = prefix[:-1]
            frontier = int(prefix[-1])

            target_cache, cache_metadata = load_cache_shard(cache_path)
            target_probs, probability_metadata = load_probability_shard(probability_path)
            validate_probability_binding(
                cache_metadata,
                probability_metadata,
                target_probs,
                expected_vocab_size=draft_vocab,
            )
            if cache_metadata.get("next_token_id") != int(token_row[-1]):
                raise RuntimeError("subspace screen next-token metadata differs from frozen token row")
            if target_cache.seq_len != len(prefix):
                raise RuntimeError("verifier screen cache length differs from frozen prefix")

            history_tensor = torch.tensor([history_ids], device=input_device, dtype=torch.long)
            with torch.no_grad():
                native_history = forward_incremental(draft, history_tensor).cache
            if native_history.rotary is None:
                raise RuntimeError("native draft history lacks rotary provenance")
            target_history = target_cache.slice(0, len(history_ids)).to(
                mapper.device, dtype=resolve_dtype(args.dtype)
            )
            with torch.no_grad():
                mapped_history = mapper.map(
                    target_history,
                    draft_rotary=native_history.rotary.to(mapper.device),
                ).to(input_device)
            native_history = native_history.to(input_device)

            delta_rows = teacher_delta_rows(native_history, mapped_history).detach().cpu()
            mapped_rows = cache_content_rows(mapped_history).detach().cpu()
            pca_stats.update(mapped_rows)

            frontier_ids = torch.tensor([[frontier]], device=input_device, dtype=torch.long)

            native_leaf = make_gradient_cache(native_history)
            native_step = forward_incremental(
                draft,
                frontier_ids,
                native_leaf,
                inference=False,
                capture_rotary=False,
            )
            native_loss = target_kl_loss(native_step.logits, target_probs)
            native_loss.backward()
            native_grad = cache_gradient_rows(native_leaf).detach().cpu()
            benefit_stats.update(native_grad, delta_rows)
            native_norm_sums += gradient_norms(native_grad).to(torch.float64)
            native_losses.append(float(native_loss.detach().cpu()))
            del native_step, native_loss, native_leaf

            mapped_leaf = make_gradient_cache(mapped_history)
            mapped_step = forward_incremental(
                draft,
                frontier_ids,
                mapped_leaf,
                inference=False,
                capture_rotary=False,
            )
            mapped_loss = target_kl_loss(mapped_step.logits, target_probs)
            mapped_loss.backward()
            mapped_grad = cache_gradient_rows(mapped_leaf).detach().cpu()
            grad_stats.update(mapped_grad)
            mapped_norm_sums += gradient_norms(mapped_grad).to(torch.float64)
            mapped_losses.append(float(mapped_loss.detach().cpu()))
            del mapped_step, mapped_loss, mapped_leaf

            completed = index + 1
            if completed == args.gradient_smoke_prefixes:
                assert_gradient_coverage(native_norm_sums, prefixes=completed)
                assert_gradient_coverage(mapped_norm_sums, prefixes=completed)

            atomic_write_json(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "running",
                    "completed_prompts": completed,
                    "requested_prompts": args.prompts,
                    "mapper_checkpoint_sha256": mapper_sha,
                    "screen_manifest_sha256": sha256_file(screen_root / "manifest.json"),
                    "mean_native_target_kl": sum(native_losses) / len(native_losses),
                    "mean_mapped_target_kl": sum(mapped_losses) / len(mapped_losses),
                    "elapsed_s": time.perf_counter() - start,
                },
            )

            del (
                target_cache,
                target_probs,
                target_history,
                native_history,
                mapped_history,
                delta_rows,
                mapped_rows,
                native_grad,
                mapped_grad,
                history_tensor,
                frontier_ids,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if completed % 8 == 0 or completed == args.prompts:
                print(f"subspace-fit prefixes: {completed}/{args.prompts}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        atomic_write_json(
            f"{output_path}.failure.json",
            {
                "schema_version": 1,
                "status": "incomplete",
                "phase": "student_gradient_fit",
                "requested_prompts": args.prompts,
                "completed_prompts": completed,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "rule": "do_not_auto_change_model_dtype_rank_or_pair",
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    assert_gradient_coverage(native_norm_sums, prefixes=args.prompts)
    assert_gradient_coverage(mapped_norm_sums, prefixes=args.prompts)

    grad_solution = sensitivity_basis(grad_stats, max_rank=args.max_rank)
    benefit_positive = benefit_basis(
        benefit_stats,
        max_rank=args.max_rank,
        sign="positive",
        eigenvalue_tol=args.benefit_eigenvalue_tol,
    )
    benefit_negative = benefit_basis(
        benefit_stats,
        max_rank=args.max_rank,
        sign="negative",
        eigenvalue_tol=args.benefit_eigenvalue_tol,
    )
    pca_solution = pca_basis(pca_stats, max_rank=args.max_rank)
    random_vectors = deterministic_random_basis(
        layers=geometry["draft_layers"],
        kinds=2,
        heads=geometry["kv_heads"],
        head_dim=geometry["head_dim"],
        max_rank=args.max_rank,
        seed=args.random_seed,
    )
    random_values = torch.zeros(
        geometry["draft_layers"], 2, geometry["kv_heads"], args.max_rank, dtype=torch.float64
    )
    random_ranks = torch.full(
        (geometry["draft_layers"], 2, geometry["kv_heads"]),
        args.max_rank,
        dtype=torch.int64,
    )

    families = {
        "grad": _family_payload(grad_solution),
        "benefit_positive": _family_payload(benefit_positive),
        "benefit_negative": _family_payload(benefit_negative),
        "pca": _family_payload(pca_solution),
        "random": (random_vectors, random_values, random_ranks),
    }
    metadata = {
        "schema_version": 1,
        "experiment": "student_readable_kv_subspace_fit",
        "git_commit": _git_commit(),
        "pair": pair,
        "source_model": mapper_metadata["source_model"],
        "draft_model": mapper_metadata["draft_model"],
        "dtype": args.dtype,
        "objective": "kl_p_target_q_student",
        "draft_layers": geometry["draft_layers"],
        "kv_heads": geometry["kv_heads"],
        "head_dim": geometry["head_dim"],
        "max_rank": args.max_rank,
        "requested_prompts": args.prompts,
        "completed_prompts": completed,
        "gradient_smoke_prefixes": args.gradient_smoke_prefixes,
        "benefit_eigenvalue_tol": args.benefit_eigenvalue_tol,
        "random_seed": args.random_seed,
        "mapper_checkpoint_sha256": mapper_sha,
        "mapper_metadata_sha256": sha256_file(mapper_metadata_path),
        "screen_manifest_sha256": sha256_file(screen_root / "manifest.json"),
        "fit_input_sha256": screen_manifest["input_sha256"],
        "fit_token_rows_digest": screen_manifest["token_rows_digest"],
        "fit_token_row_digests": screen_manifest["token_row_digests"][: args.prompts],
        "mean_native_target_kl": sum(native_losses) / len(native_losses),
        "mean_mapped_target_kl": sum(mapped_losses) / len(mapped_losses),
        "native_gradient_norm_min": float(native_norm_sums.min()),
        "native_gradient_norm_max": float(native_norm_sums.max()),
        "mapped_gradient_norm_min": float(mapped_norm_sums.min()),
        "mapped_gradient_norm_max": float(mapped_norm_sums.max()),
        "positive_benefit_rank_min": int(benefit_positive.valid_ranks.min()),
        "positive_benefit_rank_max": int(benefit_positive.valid_ranks.max()),
        "negative_benefit_rank_min": int(benefit_negative.valid_ranks.min()),
        "negative_benefit_rank_max": int(benefit_negative.valid_ranks.max()),
        "elapsed_s": time.perf_counter() - start,
    }
    artifact = SubspaceBasisArtifact(
        metadata=metadata,
        bases={name: value[0] for name, value in families.items()},
        eigenvalues={name: value[1] for name, value in families.items()},
        valid_ranks={name: value[2] for name, value in families.items()},
    )
    artifact.save(output_path)
    artifact_sha = sha256_file(output_path)
    atomic_write_json(
        f"{output_path}.json",
        {
            **metadata,
            "artifact_sha256": artifact_sha,
            "families": {
                name: {
                    "valid_rank_min": int(value[2].min()),
                    "valid_rank_max": int(value[2].max()),
                    "leading_eigenvalue_mean": float(value[1][..., 0].mean()),
                }
                for name, value in families.items()
            },
        },
    )
    progress_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "artifact_sha256": artifact_sha,
                "mean_native_target_kl": metadata["mean_native_target_kl"],
                "mean_mapped_target_kl": metadata["mean_mapped_target_kl"],
                "positive_benefit_rank": [
                    metadata["positive_benefit_rank_min"],
                    metadata["positive_benefit_rank_max"],
                ],
            },
            indent=2,
        )
    )

    del draft, mapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        raise
