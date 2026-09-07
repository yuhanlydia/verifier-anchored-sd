#!/usr/bin/env python3
"""Confirm task retention for a pair that passed the distribution screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from common import load_hf_model, load_hf_tokenizer, resolve_dtype

from verifier_anchored_sd.cache_artifacts import (
    load_cache_shard,
    save_cache_shard,
    write_or_validate_manifest,
)
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file
from verifier_anchored_sd.model_contracts import model_metadata, tokenizer_contract_hash
from verifier_anchored_sd.multiple_choice import (
    choice_nll,
    normalized_retention,
    validate_screen_gate,
)
from verifier_anchored_sd.spec_decode.hf_runtime import (
    Forward,
    capture_rotary_factors,
    forward_incremental,
)
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _raw_examples(data_file: str | None):
    if data_file:
        with Path(data_file).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional real-run dependency
        raise RuntimeError("install the HF extra or provide --data-file") from exc
    yield from load_dataset("Rowan/hellaswag", split="validation")


def _freeze_examples(tokenizer, data_file: str | None, count: int) -> list[dict]:
    frozen = []
    for row in _raw_examples(data_file):
        context = str(row.get("ctx") or f"{row['ctx_a']} {row['ctx_b']}")
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        ending_ids = [
            tokenizer(" " + str(ending).lstrip(), add_special_tokens=False)["input_ids"]
            for ending in row["endings"]
        ]
        if len(context_ids) < 1 or len(ending_ids) != 4 or any(not ids for ids in ending_ids):
            continue
        frozen.append(
            {
                "context_ids": [int(token) for token in context_ids],
                "ending_ids": [[int(token) for token in ids] for ids in ending_ids],
                "label": int(row["label"]),
            }
        )
        if len(frozen) == count:
            break
    if len(frozen) != count:
        raise RuntimeError(f"only {len(frozen)}/{count} valid HellaSwag examples were available")
    return frozen


def _load_or_freeze_examples(
    path: Path, tokenizer, data_file: str | None, count: int, tokenizer_hash: str
) -> tuple[list[dict], str]:
    input_hash = sha256_file(data_file) if data_file else "Rowan/hellaswag:validation"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "dataset": input_hash,
            "count": count,
            "tokenizer_hash": tokenizer_hash,
            "examples_digest": _json_digest(payload["examples"]),
        }
        if payload.get("metadata") != expected:
            raise RuntimeError(f"frozen HellaSwag examples use a different contract: {path}")
        return payload["examples"], expected["examples_digest"]

    examples = _freeze_examples(tokenizer, data_file, count)
    digest = _json_digest(examples)
    value = {
        "metadata": {
            "schema_version": 1,
            "dataset": input_hash,
            "count": count,
            "tokenizer_hash": tokenizer_hash,
            "examples_digest": digest,
        },
        "examples": examples,
    }
    atomic_write_json(path, value)
    return examples, digest


def _score_ending(model, base: Forward, ending_ids: list[int]) -> float:
    prediction_steps = [base.logits[:, -1, :]]
    if len(ending_ids) > 1:
        device = model.get_input_embeddings().weight.device
        teacher_ids = torch.tensor([ending_ids[:-1]], device=device, dtype=torch.long)
        continuation = forward_incremental(model, teacher_ids, base.cache)
        prediction_steps.append(continuation.logits[0])
    predictions = torch.cat(prediction_steps, dim=0)
    # A one-token dummy context aligns causal position zero with the first ending token.
    labels = torch.tensor([0, *ending_ids], dtype=torch.long)
    return choice_nll(predictions, labels, context_length=1)


def _mapped_context(
    draft,
    mapper: RidgeKVMapper,
    source_cache,
    context_ids: list[int],
    map_dtype: torch.dtype,
) -> Forward:
    input_device = draft.get_input_embeddings().weight.device
    history_length = len(context_ids) - 1
    positions = torch.arange(history_length, device=input_device).unsqueeze(0)
    rotary = capture_rotary_factors(draft, positions).to(mapper.device)
    source_history = source_cache.slice(0, history_length).to(
        mapper.device, dtype=map_dtype
    )
    mapped_history = mapper.map(source_history, draft_rotary=rotary)
    frontier = torch.tensor([[context_ids[-1]]], device=input_device, dtype=torch.long)
    return forward_incremental(draft, frontier, mapped_history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-result", required=True)
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--mapper-metadata")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-file")
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mapper-device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=14)
    parser.add_argument("--allow-failed-screen", action="store_true")
    args = parser.parse_args()
    if args.examples <= 0:
        raise ValueError("examples must be positive")

    started = time.perf_counter()
    screen_result = json.loads(Path(args.screen_result).read_text(encoding="utf-8"))
    validate_screen_gate(screen_result, allow_failed=args.allow_failed_screen)
    mapper_path = Path(args.mapper)
    metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    mapper_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(mapper_path) != mapper_metadata["checkpoint_sha256"]:
        raise RuntimeError("mapper checkpoint digest differs from its metadata")
    if screen_result.get("pair") != mapper_metadata["pair"]:
        raise RuntimeError("screen result and mapper describe different pairs")

    pair = mapper_metadata["pair"]
    target_revision = mapper_metadata["source_model"]["revision"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    tokenizer = load_hf_tokenizer(pair["target"], revision=target_revision)
    tokenizer_hash = tokenizer_contract_hash(tokenizer)
    if tokenizer_hash != mapper_metadata["source_model"]["tokenizer_hash"]:
        raise RuntimeError("tokenizer differs from mapper calibration")

    artifact_root = Path(args.artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    examples, examples_digest = _load_or_freeze_examples(
        artifact_root / "examples.json",
        tokenizer,
        args.data_file,
        args.examples,
        tokenizer_hash,
    )
    capture_contract = {
        "schema_version": 1,
        "role": "hellaswag_source",
        "pair": pair,
        "model": mapper_metadata["source_model"],
        "examples": args.examples,
        "examples_digest": examples_digest,
    }
    write_or_validate_manifest(artifact_root, capture_contract)

    shard_root = artifact_root / "source_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    missing = [
        index
        for index in range(args.examples)
        if not (shard_root / f"{index:05d}.pt").exists()
    ]
    if missing:
        target = load_hf_model(
            pair["target"],
            args.device,
            args.dtype,
            revision=target_revision,
            gpu_memory_gib=args.gpu_memory_gib,
            offload_folder=artifact_root / "source_offload",
        )
        loaded = model_metadata(target, pair["target"], tokenizer_hash)
        if loaded != mapper_metadata["source_model"]:
            raise RuntimeError("loaded verifier differs from mapper calibration")
        input_device = target.get_input_embeddings().weight.device
        for progress, index in enumerate(missing, start=1):
            context_ids = examples[index]["context_ids"]
            ids = torch.tensor([context_ids], device=input_device, dtype=torch.long)
            cache = forward_incremental(target, ids).cache
            save_cache_shard(
                shard_root / f"{index:05d}.pt",
                cache,
                {
                    "sequence_id": f"{index:05d}",
                    "context_digest": _json_digest(context_ids),
                    "model_revision": target_revision,
                },
            )
            del ids, cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress % 10 == 0 or progress == len(missing):
                print(f"HellaSwag verifier shards: {progress}/{len(missing)}", flush=True)
        del target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    draft = load_hf_model(
        pair["draft"],
        args.device,
        args.dtype,
        revision=draft_revision,
        gpu_memory_gib=args.gpu_memory_gib,
        offload_folder=artifact_root / "draft_offload",
    )
    loaded_draft = model_metadata(draft, pair["draft"], tokenizer_hash)
    if loaded_draft != mapper_metadata["draft_model"]:
        raise RuntimeError("loaded draft differs from mapper calibration")
    map_dtype = resolve_dtype(args.dtype)
    mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
        args.mapper_device, dtype=map_dtype
    )
    input_device = draft.get_input_embeddings().weight.device

    rows = []
    for index, example in enumerate(examples):
        source_cache, shard_metadata = load_cache_shard(
            shard_root / f"{index:05d}.pt"
        )
        if shard_metadata["context_digest"] != _json_digest(example["context_ids"]):
            raise RuntimeError(f"HellaSwag context shard mismatch at row {index}")
        context = torch.tensor(
            [example["context_ids"]], device=input_device, dtype=torch.long
        )
        native = forward_incremental(draft, context)
        mapped = _mapped_context(
            draft, mapper, source_cache, example["context_ids"], map_dtype
        )
        native_scores = [
            _score_ending(draft, native, ending) for ending in example["ending_ids"]
        ]
        mapped_scores = [
            _score_ending(draft, mapped, ending) for ending in example["ending_ids"]
        ]
        rows.append(
            {
                "example": index,
                "label": example["label"],
                "native_prediction": min(range(4), key=native_scores.__getitem__),
                "mapped_prediction": min(range(4), key=mapped_scores.__getitem__),
                "native_choice_nll": native_scores,
                "mapped_choice_nll": mapped_scores,
            }
        )
        del source_cache, context, native, mapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (index + 1) % 10 == 0 or index + 1 == args.examples:
            print(f"HellaSwag scored: {index + 1}/{args.examples}", flush=True)

    native_accuracy = sum(row["native_prediction"] == row["label"] for row in rows) / len(rows)
    mapped_accuracy = sum(row["mapped_prediction"] == row["label"] for row in rows) / len(rows)
    retention = normalized_retention(native_accuracy, mapped_accuracy, random_floor=0.25)
    result = {
        "schema_version": 1,
        "experiment": "mapped_hellaswag_confirmation",
        "git_commit": _git_commit(),
        "pair": pair,
        "screen_result": str(Path(args.screen_result).resolve()),
        "screen_gate": screen_result["gate"],
        "diagnostic_override": bool(args.allow_failed_screen),
        "examples": args.examples,
        "examples_digest": examples_digest,
        "native_accuracy": native_accuracy,
        "mapped_accuracy": mapped_accuracy,
        "random_floor": 0.25,
        "floor_normalized_retention": retention,
        "gate": {"threshold": 0.95, "status": "pass" if retention >= 0.95 else "fail"},
        "elapsed_s": time.perf_counter() - started,
        "rows": rows,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"native_accuracy": native_accuracy, "mapped_accuracy": mapped_accuracy, "floor_normalized_retention": retention, "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
