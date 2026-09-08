#!/usr/bin/env python3
"""Block-level speculative-decoding validation for one frozen subspace winner."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import torch
from common import iter_texts, load_hf_pair, load_hf_tokenizer

from verifier_anchored_sd.cache_artifacts import load_manifest
from verifier_anchored_sd.device_utils import timing_device_for_models
from verifier_anchored_sd.evaluation import paired_bootstrap_mean_difference
from verifier_anchored_sd.experiment_artifacts import (
    atomic_write_json,
    sha256_file,
    token_rows_digest,
)
from verifier_anchored_sd.experiment_data import collect_tokenized_prompts
from verifier_anchored_sd.kv_subspace import SubspaceBasisArtifact
from verifier_anchored_sd.model_contracts import model_metadata, tokenizer_contract_hash
from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime
from verifier_anchored_sd.spec_decode.subspace_mapper import SubspaceMappedKVMapper
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper
from verifier_anchored_sd.subspace_acceptance import (
    acceptance_method_matrix,
    classify_block_delta,
    validate_subspace_winner,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _paired(
    rows: list[dict],
    *,
    method: str,
    baseline: str,
    metric: str,
    expected_prompts: int,
    samples: int,
    seed: int,
) -> dict:
    left = {int(row["prompt"]): row for row in rows if row["method"] == method}
    right = {int(row["prompt"]): row for row in rows if row["method"] == baseline}
    prompts = sorted(set(left) & set(right))
    if len(prompts) != expected_prompts:
        raise RuntimeError(
            f"paired block metric has {len(prompts)}/{expected_prompts} complete prompts"
        )
    result = paired_bootstrap_mean_difference(
        [float(left[p][metric]) for p in prompts],
        [float(right[p][metric]) for p in prompts],
        samples=samples,
        seed=seed,
    )
    result.update({"method": method, "baseline": baseline, "metric": metric})
    return result


def _summary(rows: list[dict], methods: dict, expected_prompts: int) -> dict:
    result = {}
    metrics = (
        "mean_accepted_length",
        "conditional_expected_accepted_length",
        "acceptance_rate",
        "verifier_calls_per_output_token",
        "bonus_rate",
        "elapsed_s",
        "output_tokens_per_s",
    )
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        prompts = {int(row["prompt"]) for row in subset}
        if len(subset) != expected_prompts or len(prompts) != expected_prompts:
            raise RuntimeError(
                f"block method {method!r} has {len(prompts)}/{expected_prompts} complete prompts"
            )
        result[method] = {
            metric: sum(float(row[metric]) for row in subset) / expected_prompts
            for metric in metrics
        }
        result[method]["prompts"] = expected_prompts
        result[method]["peak_vram_bytes"] = max(
            int(row.get("peak_vram_bytes") or 0) for row in subset
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner-result", required=True)
    ap.add_argument("--selection-screen-dir", required=True)
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--mapper-metadata")
    ap.add_argument("--subspace-artifact", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--low-vram", action="store_true")
    args = ap.parse_args()

    if min(args.prompts, args.prompt_tokens, args.new_tokens, args.gamma, args.bootstrap_samples) <= 0:
        raise ValueError("block acceptance counts and lengths must be positive")

    winner_path = Path(args.winner_result)
    selection_root = Path(args.selection_screen_dir)
    mapper_path = Path(args.mapper)
    mapper_metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    subspace_path = Path(args.subspace_artifact)
    output_path = Path(args.output)

    winner_result = json.loads(winner_path.read_text(encoding="utf-8"))
    mapper_metadata = json.loads(mapper_metadata_path.read_text(encoding="utf-8"))
    mapper_sha = sha256_file(mapper_path)
    subspace_sha = sha256_file(subspace_path)
    if mapper_sha != mapper_metadata.get("checkpoint_sha256"):
        raise RuntimeError("block acceptance mapper checkpoint differs from mapper metadata")

    # Recover C's per-row provenance from its immutable manifest. C KV shards may
    # already have been pruned; the manifest/tokens remain small and sufficient.
    selection_manifest = load_manifest(selection_root)
    expected_selection_sha = winner_result.get("protocol", {}).get("screen_manifest_sha256")
    if sha256_file(selection_root / "manifest.json") != expected_selection_sha:
        raise RuntimeError("selection screen manifest differs from the frozen winner result")
    selection_rows = selection_manifest.get("token_row_digests")
    if not isinstance(selection_rows, list) or not selection_rows:
        raise RuntimeError("selection screen lacks exact token-row provenance")
    enriched = copy.deepcopy(winner_result)
    enriched.setdefault("protocol", {})["evaluation_token_row_digests"] = selection_rows[
        : int(winner_result.get("requested_prompts", len(selection_rows)))
    ]
    winner_contract = validate_subspace_winner(
        enriched,
        mapper_sha256=mapper_sha,
        subspace_sha256=subspace_sha,
    )
    if winner_contract["pair"] != mapper_metadata.get("pair"):
        raise RuntimeError("frozen subspace winner describes a different mapper pair")

    artifact = SubspaceBasisArtifact.load(subspace_path)
    if artifact.metadata.get("mapper_checkpoint_sha256") != mapper_sha:
        raise RuntimeError("subspace artifact was fit with a different mapper")
    if artifact.metadata.get("source_model") != mapper_metadata.get("source_model"):
        raise RuntimeError("subspace artifact verifier revision differs from mapper")
    if artifact.metadata.get("draft_model") != mapper_metadata.get("draft_model"):
        raise RuntimeError("subspace artifact draft revision differs from mapper")

    pair = mapper_metadata["pair"]
    target_revision = mapper_metadata["source_model"]["revision"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    tokenizer = load_hf_tokenizer(pair["target"], revision=target_revision)
    tokenizer_hash = tokenizer_contract_hash(tokenizer)
    if tokenizer_hash != mapper_metadata["source_model"]["tokenizer_hash"]:
        raise RuntimeError("block acceptance tokenizer differs from mapper calibration")

    prompt_rows = collect_tokenized_prompts(
        tokenizer,
        iter_texts(args.text_file, limit=args.prompts * 8),
        max_tokens=args.prompt_tokens,
        count=args.prompts,
    )
    if len(prompt_rows) != args.prompts:
        raise RuntimeError(f"only {len(prompt_rows)}/{args.prompts} block prompts were available")
    prompt_row_digests = [token_rows_digest([row]) for row in prompt_rows]
    mapper_calibration_rows = {str(value) for value in mapper_metadata.get("token_row_digests", [])}
    forbidden = set(winner_contract["forbidden_token_row_digests"]) | mapper_calibration_rows
    overlap = forbidden & set(prompt_row_digests)
    if overlap:
        raise RuntimeError(
            f"block-level E2 token rows overlap mapper/basis/selection data: {sorted(overlap)}"
        )

    tokenizer, target, draft = load_hf_pair(
        pair["target"],
        pair["draft"],
        args.device,
        args.dtype,
        low_vram=args.low_vram,
        target_revision=target_revision,
        draft_revision=draft_revision,
    )
    tokenizer_hash = tokenizer_contract_hash(tokenizer)
    if model_metadata(target, pair["target"], tokenizer_hash) != mapper_metadata["source_model"]:
        raise RuntimeError("loaded verifier differs from mapper provenance")
    if model_metadata(draft, pair["draft"], tokenizer_hash) != mapper_metadata["draft_model"]:
        raise RuntimeError("loaded draft differs from mapper provenance")

    map_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    base_mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
        args.device, dtype=map_dtype
    )
    subspace_mapper = SubspaceMappedKVMapper(
        base_mapper,
        artifact,
        winner_contract["spec"],
    )
    methods = acceptance_method_matrix()
    cuda_device = timing_device_for_models(target, draft)

    protocol = {
        "pair": pair,
        "target_revision": target_revision,
        "draft_revision": draft_revision,
        "dtype": args.dtype,
        "winner_result_sha256": sha256_file(winner_path),
        "selection_screen_manifest_sha256": sha256_file(selection_root / "manifest.json"),
        "mapper_checkpoint_sha256": mapper_sha,
        "mapper_metadata_sha256": sha256_file(mapper_metadata_path),
        "subspace_artifact_sha256": subspace_sha,
        "winner_method": winner_contract["method"],
        "winner_spec": {
            "mode": winner_contract["spec"].mode,
            "family": winner_contract["spec"].family,
            "rank": winner_contract["spec"].rank,
            "beta": winner_contract["spec"].beta,
            "alpha": winner_contract["spec"].alpha,
        },
        "evaluation_input_sha256": sha256_file(args.text_file),
        "prompt_token_rows_digest": token_rows_digest(prompt_rows),
        "prompt_token_row_digests": prompt_row_digests,
        "prompts": args.prompts,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "gamma": args.gamma,
        "bootstrap_samples": args.bootstrap_samples,
        "low_vram": bool(args.low_vram),
        "methods": methods,
    }

    rows: list[dict] = []
    progress_path = Path(f"{output_path}.progress.json")
    start_all = time.perf_counter()
    try:
        for method, options in methods.items():
            mapper = base_mapper if options["mapper"] == "base" else subspace_mapper
            for prompt_index, prompt_ids in enumerate(prompt_rows):
                runtime = QwenPairRuntime(
                    target,
                    draft,
                    mapper,
                    seed=prompt_index,
                    init_mode=options["init_mode"],
                    refresh_policy=options["refresh_policy"],
                )
                if cuda_device.type == "cuda":
                    torch.cuda.synchronize(cuda_device)
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                start = time.perf_counter()
                runtime.generate(prompt_ids, args.new_tokens, args.gamma)
                if cuda_device.type == "cuda":
                    torch.cuda.synchronize(cuda_device)
                elapsed = time.perf_counter() - start
                lengths = runtime.accepted_lengths
                expected = runtime.expected_accepted_lengths
                if not lengths:
                    raise RuntimeError("speculative runtime emitted no verifier blocks")
                blocks = len(lengths)
                bonus_count = sum(kind == "bonus" for kind in runtime.frontier_kinds)
                rows.append(
                    {
                        "method": method,
                        "prompt": prompt_index,
                        "prompt_tokens": len(prompt_ids),
                        "mean_accepted_length": sum(lengths) / blocks,
                        "conditional_expected_accepted_length": sum(expected) / max(len(expected), 1),
                        "acceptance_rate": sum(lengths) / max(blocks * args.gamma, 1),
                        "blocks": blocks,
                        "verifier_calls_per_output_token": blocks / args.new_tokens,
                        "bonus_rate": bonus_count / blocks,
                        "elapsed_s": elapsed,
                        "output_tokens_per_s": args.new_tokens / elapsed,
                        "peak_vram_bytes": (
                            torch.cuda.max_memory_allocated(cuda_device)
                            if cuda_device.type == "cuda"
                            else None
                        ),
                    }
                )
                atomic_write_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "protocol": protocol,
                        "completed_rows": len(rows),
                        "requested_rows": len(methods) * args.prompts,
                        "rows": rows,
                    },
                )
                if (prompt_index + 1) % 8 == 0 or prompt_index + 1 == args.prompts:
                    print(f"{method}: {prompt_index + 1}/{args.prompts}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        atomic_write_json(
            f"{output_path}.failure.json",
            {
                "schema_version": 1,
                "status": "incomplete",
                "phase": "subspace_block_acceptance",
                "protocol": protocol,
                "completed_rows": len(rows),
                "requested_rows": len(methods) * args.prompts,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "rule": "do_not_auto_change_model_dtype_pair_or_winner",
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    if len(rows) != len(methods) * args.prompts:
        raise RuntimeError("block acceptance did not complete every method/prompt row")
    summary = _summary(rows, methods, args.prompts)

    paired = {
        "subspace_init_minus_full_init_mal": _paired(
            rows,
            method="subspace_mapped_init_only",
            baseline="full_mapped_init_only",
            metric="mean_accepted_length",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=0,
        ),
        "subspace_init_minus_native_mal": _paired(
            rows,
            method="subspace_mapped_init_only",
            baseline="native_sd",
            metric="mean_accepted_length",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=1,
        ),
        "subspace_init_minus_full_init_acceptance": _paired(
            rows,
            method="subspace_mapped_init_only",
            baseline="full_mapped_init_only",
            metric="acceptance_rate",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=2,
        ),
        "subspace_refresh_minus_subspace_init_mal": _paired(
            rows,
            method="subspace_mapped_accepted_only",
            baseline="subspace_mapped_init_only",
            metric="mean_accepted_length",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=3,
        ),
        "subspace_refresh_minus_subspace_init_throughput": _paired(
            rows,
            method="subspace_mapped_accepted_only",
            baseline="subspace_mapped_init_only",
            metric="output_tokens_per_s",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=4,
        ),
        "full_refresh_minus_full_init_mal": _paired(
            rows,
            method="full_mapped_accepted_only",
            baseline="full_mapped_init_only",
            metric="mean_accepted_length",
            expected_prompts=args.prompts,
            samples=args.bootstrap_samples,
            seed=5,
        ),
    }
    primary = paired["subspace_init_minus_full_init_mal"]
    vs_native = paired["subspace_init_minus_native_mal"]
    refresh = paired["subspace_refresh_minus_subspace_init_mal"]
    block_status = classify_block_delta(primary["ci_low"], primary["ci_high"])
    native_status = classify_block_delta(vs_native["ci_low"], vs_native["ci_high"])
    refresh_status = classify_block_delta(refresh["ci_low"], refresh["ci_high"])

    if block_status == "support" and native_status == "support":
        final_status = "strong_block_support"
    elif block_status == "support":
        final_status = "mapped_filter_support"
    elif block_status == "stop":
        final_status = "one_step_only_stop"
    else:
        final_status = "block_inconclusive"

    result = {
        "schema_version": 1,
        "experiment": "student_readable_kv_subspace_block_acceptance",
        "git_commit": _git_commit(),
        "status": "complete",
        "protocol": protocol,
        "summary": summary,
        "paired_bootstrap": paired,
        "gates": {
            "subspace_init_vs_full_mapped": block_status,
            "subspace_init_vs_native": native_status,
            "subspace_refresh_vs_subspace_init": refresh_status,
            "final_status": final_status,
            "primary_rule": "realized MAL paired CI for subspace_mapped_init_only - full_mapped_init_only",
        },
        "elapsed_s": time.perf_counter() - start_all,
        "rows": rows,
    }
    atomic_write_json(output_path, result)
    progress_path.unlink(missing_ok=True)
    print(json.dumps({"gates": result["gates"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
