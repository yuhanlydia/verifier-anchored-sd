#!/usr/bin/env python3
"""E2 / KT-B: Native SD vs Ridge Init-only vs Ridge + Verifier Refresh.

Besides realized MAL, E2 reports conditional expected acceptance for each sampled
proposal path and paired-bootstrap confidence intervals.
This is especially important on a 16GB pilot, where many short independent prompts
are statistically more useful than one extremely long stochastic generation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from common import iter_texts, load_hf_pair, load_hf_tokenizer

from verifier_anchored_sd.device_utils import timing_device_for_models
from verifier_anchored_sd.e2_contract import validate_e2_artifacts
from verifier_anchored_sd.evaluation import (
    acceptance_methods,
    classify_mapper_retention,
    classify_refresh_delta,
    paired_bootstrap_mean_difference,
)
from verifier_anchored_sd.experiment_artifacts import (
    atomic_write_json,
    sha256_file,
    token_rows_digest,
)
from verifier_anchored_sd.experiment_data import collect_tokenized_prompts
from verifier_anchored_sd.model_contracts import model_metadata, tokenizer_contract_hash
from verifier_anchored_sd.resource_profiles import e2_profile
from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _paired(
    rows,
    a: str,
    b: str,
    metric: str,
    *,
    samples: int,
    seed: int,
    expected_prompts: int,
):
    a_rows = {row["prompt"]: row for row in rows if row["method"] == a}
    b_rows = {row["prompt"]: row for row in rows if row["method"] == b}
    prompts = sorted(set(a_rows) & set(b_rows))
    if len(prompts) != expected_prompts:
        raise RuntimeError(
            f"paired E2 metric has {len(prompts)}/{expected_prompts} complete prompts"
        )
    result = paired_bootstrap_mean_difference(
        [a_rows[p][metric] for p in prompts],
        [b_rows[p][metric] for p in prompts],
        samples=samples,
        seed=seed,
    )
    result.update({"method_a": a, "method_b": b, "metric": metric})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-result", required=True)
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--mapper-metadata")
    ap.add_argument("--target")
    ap.add_argument("--draft")
    ap.add_argument(
        "--text-file",
        required=True,
        help="frozen JSONL/raw-text E2 prompt source",
    )
    ap.add_argument("--prompts", type=int)
    ap.add_argument("--prompt-tokens", type=int)
    ap.add_argument("--new-tokens", type=int)
    ap.add_argument("--gamma", type=int)
    ap.add_argument(
        "--memory-profile",
        choices=["manual", "16gb", "24gb"],
        default="manual",
    )
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument(
        "--mapper-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument("--output", default="results/e2_acceptance_pilot.json")
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="controlled model offload; use only if resident 16GB inference cannot load",
    )
    args = ap.parse_args()

    profile = (
        e2_profile(args.memory_profile)
        if args.memory_profile != "manual"
        else e2_profile("24gb")
    )
    args.prompts = profile.prompts if args.prompts is None else args.prompts
    args.prompt_tokens = (
        profile.prompt_tokens if args.prompt_tokens is None else args.prompt_tokens
    )
    args.new_tokens = profile.new_tokens if args.new_tokens is None else args.new_tokens
    args.gamma = profile.gamma if args.gamma is None else args.gamma
    if min(args.prompts, args.prompt_tokens, args.new_tokens, args.gamma) <= 0:
        raise ValueError("E2 counts and lengths must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    screen_path = Path(args.screen_result)
    mapper_path = Path(args.mapper)
    mapper_metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    screen_result = json.loads(screen_path.read_text(encoding="utf-8"))
    mapper_metadata = json.loads(mapper_metadata_path.read_text(encoding="utf-8"))
    mapper_sha = sha256_file(mapper_path)
    contract = validate_e2_artifacts(
        screen_result, mapper_metadata, mapper_sha256=mapper_sha
    )
    pair = mapper_metadata["pair"]
    if args.target is not None and args.target != pair["target"]:
        raise RuntimeError("requested target differs from the screened pair")
    if args.draft is not None and args.draft != pair["draft"]:
        raise RuntimeError("requested draft differs from the screened pair")
    args.target, args.draft = pair["target"], pair["draft"]
    if args.dtype != contract["dtype"] or args.mapper_dtype != contract["dtype"]:
        raise RuntimeError("E2 model and mapper dtypes must match the screened dtype")

    tokenizer = load_hf_tokenizer(args.target, revision=contract["target_revision"])
    tokenizer_hash = tokenizer_contract_hash(tokenizer)
    if tokenizer_hash != contract["tokenizer_hash"]:
        raise RuntimeError("loaded tokenizer differs from the screened pair")
    prompt_rows = collect_tokenized_prompts(
        tokenizer,
        iter_texts(args.text_file, limit=args.prompts * 8),
        max_tokens=args.prompt_tokens,
        count=args.prompts,
    )
    methods = acceptance_methods()
    protocol_contract = {
        "pair": pair,
        "target_revision": contract["target_revision"],
        "draft_revision": contract["draft_revision"],
        "dtype": args.dtype,
        "mapper_dtype": args.mapper_dtype,
        "screen_result_sha256": sha256_file(screen_path),
        "mapper_metadata_sha256": sha256_file(mapper_metadata_path),
        "mapper_checkpoint_sha256": mapper_sha,
        "evaluation_input_sha256": sha256_file(args.text_file),
        "prompt_token_rows_digest": token_rows_digest(prompt_rows),
        "prompts": args.prompts,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "gamma": args.gamma,
        "bootstrap_samples": args.bootstrap_samples,
        "methods": methods,
    }
    progress_path = Path(f"{args.output}.progress.json")
    rows = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("protocol_contract") != protocol_contract:
            raise RuntimeError("E2 progress uses a different protocol contract")
        rows = progress.get("rows", [])
        if not isinstance(rows, list):
            raise RuntimeError("invalid E2 progress rows")
    completed = {(str(row["method"]), int(row["prompt"])) for row in rows}

    try:
        tokenizer, target, draft = load_hf_pair(
            args.target,
            args.draft,
            args.device,
            args.dtype,
            low_vram=args.low_vram,
            target_revision=contract["target_revision"],
            draft_revision=contract["draft_revision"],
        )
        if model_metadata(target, args.target, tokenizer_hash) != mapper_metadata["source_model"]:
            raise RuntimeError("loaded target differs from the screened revision")
        if model_metadata(draft, args.draft, tokenizer_hash) != mapper_metadata["draft_model"]:
            raise RuntimeError("loaded draft differs from the screened revision")
        map_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.mapper_dtype]
        mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
            args.device, dtype=map_dtype
        )
    except torch.cuda.OutOfMemoryError as exc:
        atomic_write_json(
            args.output,
            {
                "schema_version": 2,
                "experiment": "native_frontier_e2",
                "status": "incomplete",
                "protocol_contract": protocol_contract,
                "requested_rows": len(methods) * args.prompts,
                "completed_rows": len(rows),
                "failure": {"phase": "model_or_mapper_load", "type": type(exc).__name__, "message": str(exc)},
                "rows": rows,
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    cuda_device = timing_device_for_models(target, draft)

    try:
        for method, options in methods.items():
            for idx, prompt_ids in enumerate(prompt_rows):
                if (method, idx) in completed:
                    continue
                ids = torch.tensor(prompt_ids, dtype=torch.long)
                runtime = QwenPairRuntime(
                    target,
                    draft,
                    mapper,
                    seed=idx,
                    init_mode=options["init_mode"],
                    refresh_policy=options["refresh_policy"],
                )
                if cuda_device.type == "cuda":
                    torch.cuda.synchronize(cuda_device)
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                start = time.perf_counter()
                runtime.generate(ids.tolist(), args.new_tokens, args.gamma)
                if cuda_device.type == "cuda":
                    torch.cuda.synchronize(cuda_device)
                elapsed = time.perf_counter() - start
                lengths = runtime.accepted_lengths
                expected = runtime.expected_accepted_lengths
                bonus_count = sum(kind == "bonus" for kind in runtime.frontier_kinds)
                rows.append(
                    {
                        "method": method,
                        "prompt": idx,
                        "prompt_tokens": int(ids.numel()),
                        "mean_accepted_length": sum(lengths) / max(len(lengths), 1),
                        "expected_accepted_length": sum(expected) / max(len(expected), 1),
                        "acceptance_rate": sum(lengths) / max(len(lengths) * args.gamma, 1),
                        "blocks": len(lengths),
                        "bonus_rate": bonus_count / max(len(lengths), 1),
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
                        "protocol_contract": protocol_contract,
                        "rows": rows,
                    },
                )
                if (idx + 1) % 10 == 0:
                    print(method, idx + 1)
    except torch.cuda.OutOfMemoryError as exc:
        atomic_write_json(
            args.output,
            {
                "schema_version": 2,
                "experiment": "native_frontier_e2",
                "status": "incomplete",
                "protocol_contract": protocol_contract,
                "requested_rows": len(methods) * args.prompts,
                "completed_rows": len(rows),
                "failure": {"phase": "evaluation", "type": type(exc).__name__, "message": str(exc)},
                "rows": rows,
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    if len(rows) != len(methods) * args.prompts:
        raise RuntimeError("E2 did not complete every requested method/prompt row")

    summary = {}
    metrics = (
        "mean_accepted_length",
        "expected_accepted_length",
        "acceptance_rate",
        "bonus_rate",
        "elapsed_s",
        "output_tokens_per_s",
    )
    for method in methods:
        subset = [x for x in rows if x["method"] == method]
        summary[method] = {
            key: sum(x[key] for x in subset) / max(len(subset), 1) for key in metrics
        }
        summary[method]["prompts_evaluated"] = len(subset)
        summary[method]["peak_vram_bytes"] = max(
            (x["peak_vram_bytes"] or 0 for x in subset), default=0
        )

    paired = {
        "accepted_only_minus_init_expected_mal": _paired(
            rows,
            "mapped_accepted_only",
            "mapped_init_only",
            "expected_accepted_length",
            samples=args.bootstrap_samples,
            seed=0,
            expected_prompts=args.prompts,
        ),
        "accepted_only_minus_init_realized_mal": _paired(
            rows,
            "mapped_accepted_only",
            "mapped_init_only",
            "mean_accepted_length",
            samples=args.bootstrap_samples,
            seed=1,
            expected_prompts=args.prompts,
        ),
        "legacy_full_minus_legacy_init_expected_mal": _paired(
            rows,
            "legacy_full_refresh",
            "legacy_mapped_init_only",
            "expected_accepted_length",
            samples=args.bootstrap_samples,
            seed=2,
            expected_prompts=args.prompts,
        ),
        "legacy_full_minus_legacy_init_realized_mal": _paired(
            rows,
            "legacy_full_refresh",
            "legacy_mapped_init_only",
            "mean_accepted_length",
            samples=args.bootstrap_samples,
            seed=3,
            expected_prompts=args.prompts,
        ),
        "accepted_only_minus_native_throughput": _paired(
            rows,
            "mapped_accepted_only",
            "native_sd",
            "output_tokens_per_s",
            samples=args.bootstrap_samples,
            seed=4,
            expected_prompts=args.prompts,
        ),
    }
    native_expected = summary["native_sd"]["expected_accepted_length"]
    init_expected = summary["mapped_init_only"]["expected_accepted_length"]
    retention = init_expected / max(native_expected, 1e-12)
    mapper_status = classify_mapper_retention(retention)
    refresh_ci = paired["accepted_only_minus_init_expected_mal"]
    refresh_status = classify_refresh_delta(refresh_ci["ci_low"], refresh_ci["ci_high"])
    gates = {
        "mapped_native_frontier_expected_mal_retention": retention,
        "mapper_status": mapper_status,
        "accepted_only_expected_mal_delta": refresh_ci["mean_difference"],
        "accepted_only_status": refresh_status,
        "structural_decision": (
            refresh_status if mapper_status == "confirmatory" else "not_eligible"
        ),
        "legacy_full_expected_mal_delta": paired[
            "legacy_full_minus_legacy_init_expected_mal"
        ]["mean_difference"],
    }

    result = {
        "schema_version": 2,
        "experiment": "native_frontier_e2",
        "git_commit": _git_commit(),
        "config": vars(args),
        "status": "complete",
        "protocol_contract": protocol_contract,
        "pair": pair,
        "source_model": mapper_metadata["source_model"],
        "draft_model": mapper_metadata["draft_model"],
        "tokenizer_hash": tokenizer_hash,
        "screen_result_sha256": sha256_file(screen_path),
        "mapper_metadata_sha256": sha256_file(mapper_metadata_path),
        "mapper_checkpoint_sha256": mapper_sha,
        "evaluation_input_sha256": sha256_file(args.text_file),
        "prompt_token_rows_digest": token_rows_digest(prompt_rows),
        "requested_prompts": args.prompts,
        "completed_prompts_per_method": {
            method: summary[method]["prompts_evaluated"] for method in methods
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(cuda_device) if cuda_device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "timing_device": str(cuda_device),
        "summary": summary,
        "paired_bootstrap": paired,
        "gates": gates,
        "rows": rows,
    }
    atomic_write_json(args.output, result)
    progress_path.unlink(missing_ok=True)
    print(json.dumps({"summary": summary, "gates": gates}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        output = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" in sys.argv else Path("results/e2_acceptance_pilot.json")
        sidecar = Path(f"{output}.failure.json")
        atomic_write_json(
            sidecar,
            {
                "status": "incomplete",
                "failure": {"phase": "startup_or_evaluation", "type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
