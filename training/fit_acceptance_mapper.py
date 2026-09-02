#!/usr/bin/env python3
"""Train the phase-2 low-rank mapper against the block-acceptance surrogate.

The default command trains 500 steps with batch size 1 and gradient accumulation.
Use ``--initialize-only`` to create the zero-gated adapter without training.
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

from verifier_anchored_sd.spec_decode.hf_runtime import forward_incremental
from verifier_anchored_sd.spec_decode.target_to_draft_mapper import RidgeKVMapper
from verifier_anchored_sd.training.block_acceptance_loss import block_acceptance_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--text-file", help="disjoint JSONL/raw text prefixes; defaults to streaming FineWeb-Edu")
    ap.add_argument("--steps", type=int, default=500)
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
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--merge-output", help="optional final checkpoint with W0+gUV^T merged")
    ap.add_argument("--initialize-only", action="store_true")
    args = ap.parse_args()
    mapper = RidgeKVMapper.load(args.mapper, map_location=args.device)
    if mapper.u is None:
        mapper.add_low_rank(args.rank)
    mapper.save(args.output)
    if args.initialize_only:
        print(f"initialized zero-gated rank-{args.rank} residual at {args.output}; no LLM training was run")
        return

    torch.manual_seed(args.seed)
    tokenizer, target, draft = load_hf_pair(args.target, args.draft, args.device, args.dtype)
    for model in (target, draft):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    device = next(target.parameters()).device
    baseline_mapper = RidgeKVMapper.load(args.mapper, map_location=device)
    optimizer = torch.optim.AdamW(mapper.residual_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lengths = [int(x) for x in args.context_lengths.split(",")]
    text_stream = iter_texts(args.text_file, limit=max(args.max_steps * 8, 4000))
    token_buffer = torch.empty(0, dtype=torch.long)

    def next_prefix(length: int) -> torch.Tensor:
        nonlocal token_buffer
        while token_buffer.numel() < length:
            try:
                text = next(text_stream)
            except StopIteration as exc:
                raise RuntimeError("not enough training text; provide a larger disjoint --text-file") from exc
            token_buffer = torch.cat((token_buffer, tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]))
        result, token_buffer = token_buffer[:length], token_buffer[length:]
        return result

    def probs(logits):
        return torch.softmax(logits[:, -1, :].float(), dim=-1)

    def train_one(ids: torch.Tensor):
        ids = ids.unsqueeze(0).to(device)
        target_full = forward_incremental(target, ids)
        target_next = probs(target_full.logits)
        baseline_cache = baseline_mapper.map(target_full.cache)
        mapped_cache = mapper.map(target_full.cache)
        draft_prefix = mapped_cache.slice(0, mapped_cache.seq_len - 1).clone()
        draft_boundary = forward_incremental(draft, ids[:, -1:], draft_prefix, inference=False)
        q = probs(draft_boundary.logits)
        q_rows, tokens, draft_cache = [], [], mapped_cache.clone()
        for _ in range(args.gamma):
            q_rows.append(q[0])
            token = int(torch.multinomial(q.detach()[0], 1).item())
            tokens.append(token)
            step = forward_incremental(
                draft, torch.tensor([[token]], device=device), draft_cache, inference=False
            )
            draft_cache.append(step.cache)
            q = probs(step.logits)
        proposal_ids = torch.tensor([tokens], device=device)
        target_step = forward_incremental(target, proposal_ids, target_full.cache)
        p_rows = [target_next[0]]
        if args.gamma > 1:
            p_rows.extend(torch.softmax(target_step.logits[0, :-1].float(), dim=-1))
        p = torch.stack(p_rows).unsqueeze(0)
        q_tensor = torch.stack(q_rows).unsqueeze(0)
        block_loss, metrics = block_acceptance_loss(p, q_tensor)
        delta = torch.zeros((), device=device, dtype=torch.float32)
        denom = baseline_cache.layers[0].key.float().pow(2).sum()
        for current, base in zip(mapped_cache.layers, baseline_cache.layers):
            delta = delta + (current.key.float() - base.key.float()).pow(2).sum()
            delta = delta + (current.value.float() - base.value.float()).pow(2).sum()
            denom = denom + base.value.float().pow(2).sum()
        reg = delta / (denom + 1e-6)
        return block_loss + args.lambda_reg * reg, float(metrics["expected_length"].mean()), float(reg.detach())

    history = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        length = lengths[(step - 1) % len(lengths)]
        loss, expected_length, reg = train_one(next_prefix(length))
        (loss / args.grad_accum).backward()
        if step % args.grad_accum == 0 or step == args.steps:
            clip_grad_norm_(mapper.residual_parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        row = {"step": step, "loss": float(loss.detach()), "expected_length": expected_length, "regularizer": reg, "context_length": length}
        history.append(row)
        if step == 1 or step % 10 == 0:
            print(json.dumps(row))
        if step % args.save_every == 0 or step == args.steps:
            mapper.save(args.output)
            torch.save({"optimizer": optimizer.state_dict(), "step": step, "history": history}, args.output + ".optim.pt")
    Path(args.output + ".json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2))
    if args.merge_output:
        merged = RidgeKVMapper.from_state_dict(mapper.state_dict()).merge_residual()
        merged.save(args.merge_output)
        print(f"saved merged inference mapper to {args.merge_output}")
    print(f"trained {args.steps} steps and saved {args.output}")


if __name__ == "__main__":
    main()
