#!/usr/bin/env python3
"""Small MATH-500 accuracy pilot for native and verifier-anchored decoding."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch
from common import load_hf_model, load_hf_pair, load_hf_tokenizer

from verifier_anchored_sd.spec_decode.hf_runtime import QwenPairRuntime, forward_incremental
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _prompt(tokenizer, problem: str) -> list[int]:
    content = (
        "Solve the following mathematics problem. Give only the final answer in "
        "\\boxed{...}.\n\n" + problem
    )
    messages = [{"role": "user", "content": content}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    if hasattr(encoded, "__getitem__") and not isinstance(encoded, list):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not encoded or not all(isinstance(token, int) for token in encoded):
        raise TypeError("chat template did not return a flat integer token-id list")
    return encoded


def _correct(text: str, answer: str) -> bool:
    from math_verify import parse, verify

    prediction = parse(text)
    gold = parse(answer)
    if prediction is None or gold is None:
        return False
    return bool(verify(gold, prediction, raise_on_error=False))


def _greedy(model, prompt_ids: list[int], max_new_tokens: int) -> tuple[list[int], float, int]:
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    first = forward_incremental(model, ids)
    cache, logits = first.cache, first.logits
    output: list[int] = []
    eos = getattr(model.config, "eos_token_id", None)
    eos_ids = {int(eos)} if isinstance(eos, int) else set(eos or [])
    for _ in range(max_new_tokens):
        token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        output.append(token)
        if token in eos_ids:
            break
        step = forward_incremental(
            model, torch.tensor([[token]], device=device, dtype=torch.long), cache
        )
        cache = cache.clone()
        cache.append(step.cache)
        logits = step.logits
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return output, elapsed, peak


def _load_math500(limit: int, offset: int):
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [dataset[i] for i in range(offset, min(offset + limit, len(dataset)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--gpu-memory-gib", type=int, default=12)
    parser.add_argument(
        "--target-device", default="cuda", help="verifier device, e.g. cuda:0"
    )
    parser.add_argument(
        "--draft-device", default="cuda", help="draft device, e.g. cuda:1"
    )
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--mapper-metadata")
    parser.add_argument("--output", default="results/debug_batch_2026-09-07/math500_pilot.json")
    args = parser.parse_args()
    if args.limit <= 0 or args.offset < 0 or args.max_new_tokens <= 0 or args.gamma <= 0:
        raise ValueError("limit, max-new-tokens, and gamma must be positive; offset non-negative")

    mapper_path = Path(args.mapper)
    mapper_metadata_path = Path(args.mapper_metadata or f"{mapper_path}.json")
    mapper_metadata = json.loads(mapper_metadata_path.read_text(encoding="utf-8"))
    target_id = mapper_metadata["pair"]["target"]
    draft_id = mapper_metadata["pair"]["draft"]
    target_revision = mapper_metadata["source_model"]["revision"]
    draft_revision = mapper_metadata["draft_model"]["revision"]
    problems = _load_math500(args.limit, args.offset)
    if len(problems) != args.limit:
        raise RuntimeError(f"requested {args.limit} MATH-500 rows, found {len(problems)}")
    tokenizer = load_hf_tokenizer(target_id, revision=target_revision)
    prompts = [_prompt(tokenizer, row["problem"]) for row in problems]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "experiment": "math500_accuracy_pilot",
        "scientific": False,
        "git_commit": _git_commit(),
        "dataset": {"name": "HuggingFaceH4/MATH-500", "split": "test", "offset": args.offset},
        "pair": {"target": target_id, "draft": draft_id},
        "revisions": {"target": target_revision, "draft": draft_revision},
        "dtype": "bfloat16",
        "gpu_memory_gib": args.gpu_memory_gib,
        "target_device": args.target_device,
        "draft_device": args.draft_device,
        "max_new_tokens": args.max_new_tokens,
        "gamma": args.gamma,
        "methods": {},
    }

    for method, model_id, revision in (
        ("pure_8b", target_id, target_revision),
        ("pure_4b", draft_id, draft_revision),
    ):
        model = load_hf_model(
            model_id,
            args.target_device if method == "pure_8b" else args.draft_device,
            "bfloat16",
            revision=revision,
            gpu_memory_gib=(
                args.gpu_memory_gib
                if args.target_device == args.draft_device
                else None
            ),
            offload_folder=output_path.parent / f"{method}_offload",
        )
        rows = []
        for index, (prompt_ids, problem) in enumerate(zip(prompts, problems, strict=True)):
            generated, elapsed, peak = _greedy(model, prompt_ids, args.max_new_tokens)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            rows.append(
                {
                    "index": index,
                    "unique_id": problem["unique_id"],
                    "correct": _correct(text, problem["answer"]),
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": len(generated),
                    "elapsed_s": elapsed,
                    "tokens_per_s": len(generated) / max(elapsed, 1e-9),
                    "peak_vram_bytes": peak,
                    "output": text,
                }
            )
            print(method, index + 1, rows[-1]["correct"], flush=True)
        result["methods"][method] = {
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "correct": sum(row["correct"] for row in rows),
            "evaluated": len(rows),
            "mean_tokens_per_s": sum(row["tokens_per_s"] for row in rows) / len(rows),
            "rows": rows,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    tokenizer, target, draft = load_hf_pair(
        target_id,
        draft_id,
        "cuda",
        "bfloat16",
        low_vram=True,
        target_device=args.target_device,
        draft_device=args.draft_device,
        target_revision=target_revision,
        draft_revision=draft_revision,
    )
    mapper = RidgeKVMapper.load(mapper_path, map_location="cpu").to(
        draft.get_input_embeddings().weight.device, dtype=torch.bfloat16
    )
    rows = []
    for index, (prompt_ids, problem) in enumerate(zip(prompts, problems, strict=True)):
        runtime = QwenPairRuntime(
            target,
            draft,
            mapper,
            temperature=1e-5,
            seed=index,
            init_mode="mapped_native_frontier",
            refresh_policy="accepted_only",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        generated = runtime.generate(prompt_ids, args.max_new_tokens, args.gamma)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        text = tokenizer.decode(generated, skip_special_tokens=True)
        rows.append(
            {
                "index": index,
                "unique_id": problem["unique_id"],
                "correct": _correct(text, problem["answer"]),
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(generated),
                "elapsed_s": elapsed,
                "tokens_per_s": len(generated) / max(elapsed, 1e-9),
                "mean_accepted_length": sum(runtime.accepted_lengths)
                / max(len(runtime.accepted_lengths), 1),
                "peak_vram_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0,
                "output": text,
            }
        )
        print("mapped_accepted_only", index + 1, rows[-1]["correct"], flush=True)
    result["methods"]["mapped_accepted_only"] = {
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "correct": sum(row["correct"] for row in rows),
        "evaluated": len(rows),
        "mean_tokens_per_s": sum(row["tokens_per_s"] for row in rows) / len(rows),
        "mean_accepted_length": sum(row["mean_accepted_length"] for row in rows) / len(rows),
        "rows": rows,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in value.items() if k != "rows"} for name, value in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
