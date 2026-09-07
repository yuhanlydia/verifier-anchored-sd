#!/usr/bin/env python3
"""Evaluate native-draft versus mapped-history/native-frontier distributions."""

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
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--mapper-metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.95)
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
    if sha256_file(mapper_path) != mapper_metadata["checkpoint_sha256"]:
        raise RuntimeError("mapper checkpoint digest differs from its metadata")
    screen_root = Path(args.screen_dir)
    screen_manifest = load_manifest(screen_root)
    validate_screen_inputs(mapper_metadata, screen_manifest)
    captured_count = int(screen_manifest["capture"]["count"])
    if args.prompts > captured_count:
        raise ValueError("requested prompts exceed the held-out capture count")
    token_rows, token_metadata = load_token_rows(screen_root / "tokens.pt")
    if token_metadata["token_rows_digest"] != screen_manifest["token_rows_digest"]:
        raise RuntimeError("screen token artifact differs from its manifest")
    shard_paths = exact_shard_paths(screen_root / "shards", captured_count)[: args.prompts]

    pair = mapper_metadata["pair"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    tokenizer = load_hf_tokenizer(pair["draft"], revision=draft_revision)
    if tokenizer_contract_hash(tokenizer) != mapper_metadata["draft_model"]["tokenizer_hash"]:
        raise RuntimeError("draft tokenizer differs from mapper calibration")
    draft = load_hf_model(
        pair["draft"],
        args.device,
        args.dtype,
        revision=draft_revision,
        gpu_memory_gib=args.gpu_memory_gib,
        offload_folder=screen_root / "draft_offload",
    )
    loaded_draft = model_metadata(draft, pair["draft"], tokenizer_contract_hash(tokenizer))
    if loaded_draft["revision"] != draft_revision:
        raise RuntimeError("loaded draft revision differs from mapper calibration")
    map_dtype = resolve_dtype(args.dtype)
    mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
        args.mapper_device, dtype=map_dtype
    )
    input_device = draft.get_input_embeddings().weight.device

    attention_sink: list[torch.Tensor] = []
    handles = []
    if args.attention_cosine:
        def capture_attention(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            attention_sink.append(tensor[:, -1].detach().float().cpu())

        handles = [module.register_forward_hook(capture_attention) for module in _attention_modules(draft)]

    rows = []
    start = time.perf_counter()
    try:
        for index, (token_row, shard_path) in enumerate(
            zip(token_rows[: args.prompts], shard_paths, strict=True)
        ):
            prefix, next_id = token_row[:-1], int(token_row[-1])
            target_cache, shard_metadata = load_cache_shard(shard_path)
            if shard_metadata["next_token_id"] != next_id:
                raise RuntimeError(f"next-token mismatch in screen shard: {shard_path}")
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
            row = distribution_transfer_rows(
                _probs(native.logits).cpu(),
                _probs(mapped.logits).cpu(),
                next_ids=[next_id],
            )[0]
            row.update(
                {
                    "prompt": index,
                    "prefix_tokens": len(prefix),
                    "next_token_id": next_id,
                }
            )
            if args.attention_cosine:
                cosines = attention_output_cosines(native_attention, mapped_attention)
                row["attention_output_cosines"] = cosines
                row["mean_attention_output_cosine"] = sum(cosines) / len(cosines)
            rows.append(row)
            del target_cache, target_history, mapped_history, native, mapped, ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (index + 1) % 10 == 0 or index + 1 == args.prompts:
                print(f"pair-screen rows: {index + 1}/{args.prompts}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    aggregate = finalize_screen(
        rows,
        requested=args.prompts,
        samples=args.bootstrap_samples,
        seed=0,
        threshold=args.threshold,
    )
    result = {
        "schema_version": 1,
        "experiment": "pair_screen",
        "git_commit": _git_commit(),
        "hardware": {
            "gpu": _gpu_description(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": vars(args),
        "pair": pair,
        "source_model": mapper_metadata["source_model"],
        "draft_model": mapper_metadata["draft_model"],
        "calibration_token_rows_digest": mapper_metadata["token_rows_digest"],
        "evaluation_token_rows_digest": screen_manifest["token_rows_digest"],
        **aggregate,
        "elapsed_s": time.perf_counter() - start,
        "rows": rows,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"summary": result["summary"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
