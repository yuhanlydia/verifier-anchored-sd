#!/usr/bin/env python3
"""Evaluate a broad KV subspace intervention matrix on held-out prefixes.

The verifier is not loaded. The program consumes held-out verifier KV/probability
artifacts, one frozen 8B->4B mapper, and one frozen student-readable basis artifact.
Different intervention caches for the same prefix are materialized lazily in small
method batches for the single native 4B frontier forward, so peak memory scales with
``method_batch_size`` rather than the total method matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
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
    validate_no_row_overlap,
)
from verifier_anchored_sd.kv_subspace import (
    InterventionSpec,
    SubspaceBasisArtifact,
    apply_intervention,
)
from verifier_anchored_sd.model_contracts import model_metadata, tokenizer_contract_hash
from verifier_anchored_sd.spec_decode.cache_state import CacheState, LayerKV
from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper
from verifier_anchored_sd.subspace_metrics import (
    paired_method_difference,
    select_deployment_winner,
    summarize_methods,
)
from verifier_anchored_sd.transfer_metrics import validate_screen_inputs


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _stack_caches(caches: list[CacheState]) -> CacheState:
    if not caches:
        raise ValueError("cannot stack an empty cache list")
    reference = caches[0]
    if reference.keys_are_content or reference.rotary is None:
        raise ValueError("method caches must be position-space with rotary provenance")
    for cache in caches[1:]:
        if (
            cache.num_layers != reference.num_layers
            or cache.kv_heads != reference.kv_heads
            or cache.head_dim != reference.head_dim
            or cache.seq_len != reference.seq_len
            or cache.keys_are_content
            or cache.rotary is None
        ):
            raise ValueError("method caches cannot be batched because their geometry differs")
    layers = []
    for layer_index in range(reference.num_layers):
        layers.append(
            LayerKV(
                torch.cat([cache.layers[layer_index].key for cache in caches], dim=0),
                torch.cat([cache.layers[layer_index].value for cache in caches], dim=0),
            )
        )
    return CacheState(layers, rotary=reference.rotary, keys_are_content=False)


def _method_id(spec: InterventionSpec) -> str:
    parts = [spec.family, spec.mode, f"r{spec.rank}"]
    if spec.beta is not None:
        parts.append(f"b{spec.beta:g}")
    if spec.alpha is not None:
        parts.append(f"a{spec.alpha:g}")
    return "_".join(parts)


def _method_matrix(ranks: list[int]) -> list[dict]:
    """Return the preregistered broad screen without duplicate endpoint forwards."""
    methods: list[dict] = [
        {"method": "native", "baseline": "native", "deployment_valid": True, "rank": 0},
        {
            "method": "full_mapped",
            "baseline": "full_mapped",
            "deployment_valid": True,
            "rank": 128,
        },
    ]
    mapped_soft_betas = (0.0, 0.25, 0.5, 0.75)
    delta_alphas = (0.25, 0.5, 1.0, 1.5)
    shrink_betas = (0.0, 0.25, 0.5, 0.75)
    for rank in ranks:
        for family in ("grad", "benefit_positive"):
            for beta in mapped_soft_betas:
                spec = InterventionSpec(
                    mode="mapped_soft", family=family, rank=rank, beta=beta
                )
                methods.append(
                    {
                        "method": _method_id(spec),
                        "spec": spec,
                        "deployment_valid": True,
                        "rank": rank,
                    }
                )
            for alpha in delta_alphas:
                spec = InterventionSpec(
                    mode="delta_projected", family=family, rank=rank, alpha=alpha
                )
                methods.append(
                    {
                        "method": _method_id(spec),
                        "spec": spec,
                        "deployment_valid": False,
                        "rank": rank,
                        "mechanistic_upper_bound": True,
                    }
                )
            for beta in shrink_betas:
                spec = InterventionSpec(
                    mode="delta_shrink", family=family, rank=rank, beta=beta
                )
                methods.append(
                    {
                        "method": _method_id(spec),
                        "spec": spec,
                        "deployment_valid": False,
                        "rank": rank,
                        "mechanistic_upper_bound": True,
                    }
                )

        for family in ("pca", "random", "benefit_negative"):
            spec = InterventionSpec(mode="mapped_soft", family=family, rank=rank, beta=0.0)
            methods.append(
                {
                    "method": _method_id(spec),
                    "spec": spec,
                    "deployment_valid": family in {"pca", "random"},
                    "rank": rank,
                    "causal_control": family in {"random", "benefit_negative"},
                }
            )
        random_delta = InterventionSpec(
            mode="delta_projected", family="random", rank=rank, alpha=1.0
        )
        methods.append(
            {
                "method": _method_id(random_delta),
                "spec": random_delta,
                "deployment_valid": False,
                "rank": rank,
                "causal_control": True,
                "mechanistic_upper_bound": True,
            }
        )
        orthogonal = InterventionSpec(mode="orthogonal", family="grad", rank=rank)
        methods.append(
            {
                "method": _method_id(orthogonal),
                "spec": orthogonal,
                "deployment_valid": True,
                "rank": rank,
                "causal_control": True,
            }
        )
    ids = [method["method"] for method in methods]
    if len(ids) != len(set(ids)):
        raise RuntimeError("subspace method matrix contains duplicate method IDs")
    return methods


def _target_metrics(target: torch.Tensor, student: torch.Tensor, next_id: int) -> dict:
    p = target.float()
    q = student.float()
    p = p / p.sum().clamp_min(1e-12)
    q = q / q.sum().clamp_min(1e-12)
    overlap = float((1.0 - 0.5 * (p - q).abs().sum()).clamp(0.0, 1.0))
    kl = float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum())
    return {
        "a_target": overlap,
        "kl_target": kl,
        "top1_target": int(p.argmax() == q.argmax()),
        "next_token_nll": float(-q[next_id].clamp_min(1e-12).log()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--mapper-metadata")
    ap.add_argument("--subspace-artifact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--ranks", default="4,8,16,32,64")
    ap.add_argument("--method-batch-size", type=int, default=8)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mapper-device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu-memory-gib", type=int, default=28)
    args = ap.parse_args()

    if args.prompts <= 0 or args.method_batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("prompts, method-batch-size and bootstrap-samples must be positive")
    ranks = sorted({int(value) for value in args.ranks.split(",") if value.strip()})
    if not ranks or min(ranks) <= 0:
        raise ValueError("ranks must contain positive integers")

    screen_root = Path(args.screen_dir)
    mapper_path = Path(args.mapper)
    mapper_metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    subspace_path = Path(args.subspace_artifact)
    output_path = Path(args.output)
    mapper_metadata = json.loads(mapper_metadata_path.read_text(encoding="utf-8"))
    mapper_sha = sha256_file(mapper_path)
    if mapper_sha != mapper_metadata.get("checkpoint_sha256"):
        raise RuntimeError("mapper checkpoint digest differs from metadata")
    screen_manifest = load_manifest(screen_root)
    validate_screen_inputs(mapper_metadata, screen_manifest)
    captured = int(screen_manifest["capture"]["count"])
    if args.prompts > captured:
        raise ValueError("requested evaluation prompts exceed captured screen rows")

    artifact = SubspaceBasisArtifact.load(subspace_path)
    if artifact.metadata.get("mapper_checkpoint_sha256") != mapper_sha:
        raise RuntimeError("subspace artifact was fit with a different mapper")
    if artifact.metadata.get("pair") != mapper_metadata.get("pair"):
        raise RuntimeError("subspace artifact describes a different model pair")
    if artifact.metadata.get("draft_model") != mapper_metadata.get("draft_model"):
        raise RuntimeError("subspace artifact draft revision differs from mapper")
    if artifact.metadata.get("source_model") != mapper_metadata.get("source_model"):
        raise RuntimeError("subspace artifact verifier revision differs from mapper")
    if max(ranks) > artifact.max_rank:
        raise ValueError("requested evaluation rank exceeds fitted max-rank")
    validate_no_row_overlap(
        artifact.metadata.get("fit_token_row_digests", []),
        screen_manifest.get("token_row_digests", [])[: args.prompts],
    )
    c_input_sha = str(screen_manifest.get("input_sha256", ""))
    if not c_input_sha:
        raise RuntimeError("intervention evaluation screen lacks input-file provenance")
    prior_file_shas = {
        str(mapper_metadata.get("calibration_input_sha256", "")),
        str(artifact.metadata.get("fit_input_sha256", "")),
    }
    if "" in prior_file_shas:
        raise RuntimeError("mapper or subspace artifact lacks input-file provenance")
    if c_input_sha in prior_file_shas:
        raise RuntimeError("intervention evaluation reuses the same frozen input file as A or B")

    token_rows, token_metadata = load_token_rows(screen_root / "tokens.pt")
    if token_metadata.get("token_rows_digest") != screen_manifest.get("token_rows_digest"):
        raise RuntimeError("evaluation token artifact differs from screen manifest")
    sources = token_metadata.get("window_sources")
    if not isinstance(sources, list) or len(sources) < args.prompts:
        raise RuntimeError("evaluation token artifact lacks document provenance")
    cache_paths = exact_shard_paths(screen_root / "shards", captured)[: args.prompts]
    probability_paths = exact_probability_paths(screen_root / "target_probs", captured)[: args.prompts]

    pair = mapper_metadata["pair"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    tokenizer = load_hf_tokenizer(pair["draft"], revision=draft_revision)
    tokenizer_hash = tokenizer_contract_hash(tokenizer)
    if tokenizer_hash != mapper_metadata["draft_model"]["tokenizer_hash"]:
        raise RuntimeError("evaluation tokenizer differs from mapper")
    draft = load_hf_model(
        pair["draft"],
        args.device,
        args.dtype,
        revision=draft_revision,
        gpu_memory_gib=args.gpu_memory_gib,
        offload_folder=screen_root / "subspace_eval_draft_offload",
    )
    if model_metadata(draft, pair["draft"], tokenizer_hash) != mapper_metadata["draft_model"]:
        raise RuntimeError("loaded draft differs from mapper calibration")
    for parameter in draft.parameters():
        parameter.requires_grad_(False)
    draft.eval()
    draft_vocab = int(draft.get_output_embeddings().weight.shape[0])
    mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
        args.mapper_device, dtype=resolve_dtype(args.dtype)
    )
    input_device = _input_device(draft)
    methods = _method_matrix(ranks)

    protocol = {
        "pair": pair,
        "mapper_checkpoint_sha256": mapper_sha,
        "mapper_metadata_sha256": sha256_file(mapper_metadata_path),
        "subspace_artifact_sha256": sha256_file(subspace_path),
        "screen_manifest_sha256": sha256_file(screen_root / "manifest.json"),
        "evaluation_input_sha256": screen_manifest["input_sha256"],
        "evaluation_token_rows_digest": screen_manifest["token_rows_digest"],
        "evaluation_token_row_digests": screen_manifest["token_row_digests"][: args.prompts],
        "prompts": args.prompts,
        "ranks": ranks,
        "method_batch_size": args.method_batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "dtype": args.dtype,
        "methods": [
            {
                **{k: v for k, v in method.items() if k != "spec"},
                **({"spec": asdict(method["spec"])} if "spec" in method else {}),
            }
            for method in methods
        ],
    }
    rows: list[dict] = []
    progress_path = Path(f"{output_path}.progress.json")
    start = time.perf_counter()

    for prompt_index, (token_row, cache_path, probability_path, source) in enumerate(
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
        if target_cache.seq_len != len(prefix):
            raise RuntimeError("verifier evaluation cache length differs from frozen prefix")
        history_tensor = torch.tensor([history_ids], device=input_device, dtype=torch.long)
        with torch.no_grad():
            native_history = forward_incremental(draft, history_tensor).cache.to(input_device)
            target_history = target_cache.slice(0, len(history_ids)).to(
                mapper.device, dtype=resolve_dtype(args.dtype)
            )
            mapped_history = mapper.map(
                target_history,
                draft_rotary=native_history.rotary.to(mapper.device),
            ).to(input_device)

        for start_index in range(0, len(methods), args.method_batch_size):
            batch_methods = methods[start_index : start_index + args.method_batch_size]
            batch: list[tuple[dict, CacheState]] = []
            for method in batch_methods:
                baseline = method.get("baseline")
                if baseline == "native":
                    cache = native_history
                elif baseline == "full_mapped":
                    cache = mapped_history
                else:
                    cache = apply_intervention(
                        native_history,
                        mapped_history,
                        artifact,
                        method["spec"],
                    )
                batch.append((method, cache))

            batch_cache = _stack_caches([cache for _, cache in batch])
            frontier_ids = torch.full(
                (len(batch), 1), frontier, device=input_device, dtype=torch.long
            )
            batch_start = time.perf_counter()
            with torch.no_grad():
                step = forward_incremental(
                    draft,
                    frontier_ids,
                    batch_cache,
                    inference=True,
                    capture_rotary=False,
                )
                probs = torch.softmax(step.logits[:, -1].float(), dim=-1).cpu()
            elapsed = time.perf_counter() - batch_start
            for row_index, (method, _) in enumerate(batch):
                metrics = _target_metrics(target_probs, probs[row_index], next_id)
                rows.append(
                    {
                        "method": method["method"],
                        "prompt": prompt_index,
                        "document_id": int(source["document_id"]),
                        "chunk_id": int(source["chunk_id"]),
                        "rank": int(method.get("rank", 0)),
                        "deployment_valid": bool(method.get("deployment_valid", False)),
                        "mechanistic_upper_bound": bool(
                            method.get("mechanistic_upper_bound", False)
                        ),
                        "causal_control": bool(method.get("causal_control", False)),
                        "elapsed_s": elapsed / len(batch),
                        **(
                            {"spec": asdict(method["spec"])}
                            if "spec" in method
                            else {}
                        ),
                        **metrics,
                    }
                )
            del batch_cache, frontier_ids, step, probs, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        atomic_write_json(
            progress_path,
            {
                "schema_version": 1,
                "protocol": protocol,
                "completed_prompts": prompt_index + 1,
                "rows": rows,
                "elapsed_s": time.perf_counter() - start,
            },
        )
        del target_cache, target_probs, target_history, native_history, mapped_history
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (prompt_index + 1) % 8 == 0 or prompt_index + 1 == args.prompts:
            print(f"subspace-eval prefixes: {prompt_index + 1}/{args.prompts}", flush=True)

    expected_rows = len(methods) * args.prompts
    if len(rows) != expected_rows:
        raise RuntimeError(f"subspace evaluator completed {len(rows)}/{expected_rows} rows")
    summary = summarize_methods(rows, expected_prompts=args.prompts)

    paired: dict[str, dict] = {}
    candidates: dict[str, dict] = {}
    for method in methods:
        name = method["method"]
        if name in {"native", "full_mapped"}:
            continue
        vs_native = paired_method_difference(
            rows,
            method=name,
            baseline="native",
            metric="a_target",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=0,
        )
        vs_full = paired_method_difference(
            rows,
            method=name,
            baseline="full_mapped",
            metric="a_target",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=1,
        )
        kl_vs_native = paired_method_difference(
            rows,
            method=name,
            baseline="native",
            metric="kl_target",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=2,
        )
        paired[f"{name}__vs_native__a_target"] = vs_native
        paired[f"{name}__vs_full_mapped__a_target"] = vs_full
        paired[f"{name}__vs_native__kl_target"] = kl_vs_native
        candidates[name] = {
            "method": name,
            "deployment_valid": bool(method.get("deployment_valid", False)),
            "mechanistic_upper_bound": bool(method.get("mechanistic_upper_bound", False)),
            "causal_control": bool(method.get("causal_control", False)),
            "rank": int(method.get("rank", 0)),
            "mean_a_target": summary[name]["mean_a_target"],
            "mean_kl_target": summary[name]["mean_kl_target"],
            "vs_native": vs_native,
            "vs_full_mapped": vs_full,
            "kl_vs_native": kl_vs_native,
            **({"spec": asdict(method["spec"])} if "spec" in method else {}),
        }
    winner = select_deployment_winner(candidates)

    result = {
        "schema_version": 1,
        "experiment": "student_readable_kv_subspace_interventions",
        "git_commit": _git_commit(),
        "status": "complete",
        "protocol": protocol,
        "pair": pair,
        "source_model": mapper_metadata["source_model"],
        "draft_model": mapper_metadata["draft_model"],
        "subspace_metadata": artifact.metadata,
        "requested_prompts": args.prompts,
        "completed_prompts": args.prompts,
        "method_count": len(methods),
        "summary": summary,
        "paired_bootstrap": paired,
        "candidates": candidates,
        "winner": winner,
        "decision": {
            "status": "go_e2" if winner is not None else "no_deployment_winner",
            "rule": "primary grad/benefit mapped-only winner must beat native and full_mapped target overlap with paired 95% CI low > 0",
        },
        "elapsed_s": time.perf_counter() - start,
        "rows": rows,
    }
    atomic_write_json(output_path, result)
    progress_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "winner": winner,
                "native": summary["native"],
                "full_mapped": summary["full_mapped"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
