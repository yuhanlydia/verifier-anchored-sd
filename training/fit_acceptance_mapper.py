#!/usr/bin/env python3
"""Train a low-rank residual on top of the audited ridge KV mapper.

``--steps`` always means optimizer updates. Target and draft LLM weights remain
frozen. Target forwards use ordinary ``torch.no_grad`` tensors (not inference-mode
tensors) because the mapper must save source KV for gradients with respect to its
residual parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import torch
from common import iter_texts, load_hf_pair
from torch.nn.utils import clip_grad_norm_

from verifier_anchored_sd.spec_decode.hf_runtime import (
    capture_rotary_factors,
    forward_incremental,
)
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper
from verifier_anchored_sd.training.block_acceptance_loss import (
    block_acceptance_loss,
    one_step_acceptance_loss,
)
from verifier_anchored_sd.training.schedule import optimizer_microbatch_schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument(
        "--text-file",
        help="disjoint JSONL/raw text prefixes; defaults to streaming FineWeb-Edu",
    )
    ap.add_argument(
        "--objective", choices=["block", "one_step_tv"], default="block"
    )
    ap.add_argument(
        "--steps", type=int, default=500, help="optimizer updates, not microbatches"
    )
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--context-lengths", default="512,1024,2048")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lambda-reg", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument(
        "--merge-output", help="optional final checkpoint with W0+gUV^T merged"
    )
    ap.add_argument("--initialize-only", action="store_true")
    args = ap.parse_args()
    if args.steps > args.max_steps:
        raise ValueError("--steps may not exceed --max-steps")

    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device)
    if mapper.u is None:
        mapper.add_low_rank(args.rank)
    mapper.save(args.output)
    if args.initialize_only:
        print(
            f"initialized zero-gated rank-{args.rank} residual at {args.output}; "
            "no LLM training was run"
        )
        return

    torch.manual_seed(args.seed)
    tokenizer, target, draft = load_hf_pair(
        args.target, args.draft, args.device, args.dtype
    )
    for model in (target, draft):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    device = next(target.parameters()).device
    optimizer = torch.optim.AdamW(
        mapper.residual_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lengths = [int(x) for x in args.context_lengths.split(",")]
    if not lengths or min(lengths) < 2:
        raise ValueError("context lengths must be >=2")
    text_limit = max(args.max_steps * args.grad_accum * 8, 4000)
    text_stream = iter_texts(args.text_file, limit=text_limit)
    token_buffer = torch.empty(0, dtype=torch.long)

    def next_prefix(length: int) -> torch.Tensor:
        nonlocal token_buffer
        while token_buffer.numel() < length:
            try:
                text = next(text_stream)
            except StopIteration as exc:
                raise RuntimeError(
                    "not enough training text; provide a larger disjoint --text-file"
                ) from exc
            encoded = tokenizer(
                text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            token_buffer = torch.cat((token_buffer, encoded))
        result, token_buffer = token_buffer[:length], token_buffer[length:]
        return result

    def probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits[:, -1, :].float(), dim=-1)

    def train_one(ids: torch.Tensor):
        ids = ids.unsqueeze(0).to(device)
        with torch.no_grad():
            target_full = forward_incremental(target, ids, inference=False)
            target_next = probs(target_full.logits)

        positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        draft_rotary = capture_rotary_factors(draft, positions)
        with torch.no_grad():
            baseline_cache = mapper.map(
                target_full.cache,
                draft_rotary=draft_rotary,
                include_residual=False,
            )
        mapped_cache = mapper.map(target_full.cache, draft_rotary=draft_rotary)

        draft_prefix = mapped_cache.slice(0, mapped_cache.seq_len - 1).clone()
        draft_boundary = forward_incremental(
            draft, ids[:, -1:], draft_prefix, inference=False
        )
        q = probs(draft_boundary.logits)
        q_rows, tokens, draft_cache = [], [], mapped_cache.clone()
        for _ in range(args.gamma):
            q_rows.append(q[0])
            token = int(torch.multinomial(q.detach()[0], 1).item())
            tokens.append(token)
            step = forward_incremental(
                draft,
                torch.tensor([[token]], device=device),
                draft_cache,
                inference=False,
            )
            draft_cache.append(step.cache)
            q = probs(step.logits)

        proposal_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            target_step = forward_incremental(
                target, proposal_ids, target_full.cache, inference=False
            )
            p_rows = [target_next[0]]
            if args.gamma > 1:
                p_rows.extend(
                    torch.softmax(
                        target_step.logits[0, :-1].float(), dim=-1
                    )
                )
            p = torch.stack(p_rows).unsqueeze(0)
        q_tensor = torch.stack(q_rows).unsqueeze(0)

        block_loss, block_metrics = block_acceptance_loss(p, q_tensor)
        if args.objective == "block":
            objective_loss = block_loss
        else:
            objective_loss, _ = one_step_acceptance_loss(p, q_tensor)

        delta = torch.zeros((), device=device, dtype=torch.float32)
        denom = torch.zeros((), device=device, dtype=torch.float32)
        for current, base in zip(
            mapped_cache.layers, baseline_cache.layers, strict=True
        ):
            delta = delta + (current.key.float() - base.key.float()).pow(2).sum()
            delta = delta + (current.value.float() - base.value.float()).pow(2).sum()
            denom = (
                denom
                + base.key.float().pow(2).sum()
                + base.value.float().pow(2).sum()
            )
        reg = delta / (denom + 1e-6)
        loss = objective_loss + args.lambda_reg * reg
        return (
            loss,
            block_metrics["expected_length"].mean(),
            block_metrics["alpha"][:, 0].mean(),
            reg.detach(),
        )

    history = []
    optimizer.zero_grad(set_to_none=True)
    micro_index = 0
    current_step = 0
    sums = {
        "loss": 0.0,
        "expected_length": 0.0,
        "first_acceptance": 0.0,
        "regularizer": 0.0,
    }
    for optimizer_step, microbatch in optimizer_microbatch_schedule(
        args.steps, args.grad_accum
    ):
        if optimizer_step != current_step:
            current_step = optimizer_step
            sums = {key: 0.0 for key in sums}
        length = lengths[micro_index % len(lengths)]
        micro_index += 1
        loss, expected_length, first_acceptance, reg = train_one(
            next_prefix(length)
        )
        (loss / args.grad_accum).backward()
        sums["loss"] += float(loss.detach())
        sums["expected_length"] += float(expected_length)
        sums["first_acceptance"] += float(first_acceptance)
        sums["regularizer"] += float(reg)

        if microbatch != args.grad_accum:
            continue
        clip_grad_norm_(mapper.residual_parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "optimizer_step": optimizer_step,
            "microbatches_seen": micro_index,
            **{key: value / args.grad_accum for key, value in sums.items()},
        }
        history.append(row)
        if optimizer_step == 1 or optimizer_step % 10 == 0:
            print(json.dumps(row))
        if optimizer_step % args.save_every == 0 or optimizer_step == args.steps:
            mapper.save(args.output)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "optimizer_step": optimizer_step,
                    "history": history,
                },
                args.output + ".optim.pt",
            )

    Path(args.output + ".json").write_text(
        json.dumps({"args": vars(args), "history": history}, indent=2)
    )
    if args.merge_output:
        merged = RidgeKVMapper.from_state_dict(mapper.state_dict()).merge_residual()
        merged.save(args.merge_output)
        print(f"saved merged inference mapper to {args.merge_output}")
    print(
        f"trained {args.steps} optimizer steps ({micro_index} microbatches) "
        f"with objective={args.objective} and saved {args.output}"
    )


if __name__ == "__main__":
    main()
