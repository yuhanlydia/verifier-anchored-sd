# Verifier-Anchored Speculative Decoding

Experimental implementation of **Verifier-Anchored Draft Cache Refresh** for the
Qwen3-4B → Qwen3-1.7B matched-KV pair.

The first milestone is deliberately limited to the three kill tests:

1. **E0** — fit a per-layer/head affine target→draft ridge mapper.
2. **E1 / KT-A** — measure target prefill + mapping against target prefill + native draft prefill.
3. **E2 / KT-B** — compare Native SD, Ridge Init-only, and Ridge + Verifier Refresh on 200 prompts.

The block-acceptance residual is only initialized by the provided phase-2 command;
it is not trained until E0–E2 pass.

## What is implemented

`src/verifier_anchored_sd/spec_decode/` contains the model-agnostic cache state,
closed-form ridge mapper, exact rejection sampler, and refresh state machine.
`spec_decode/hf_runtime.py` is the optional incremental Transformers adapter used by
the pilot scripts. It keeps target and draft caches separate, commits only the
accepted target prefix, and permits at most one unmaterialized correction frontier.

The runtime uses Qwen3's matched KV-head/head-dimension regime. Both models must
expose the same number of KV heads and head dimension. The calibration and runtime
currently assume compatible Qwen3 rotary geometry; a pair with different rotary
scaling must add explicit inverse/forward RoPE transforms before using the mapper.

## Setup

```bash
cd verifier-anchored-sd
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,dev]'
```

Hugging Face access to both Qwen checkpoints is required. Use a local JSONL file for
calibration text (`{"text": "..."}` or one document per line); otherwise E0 streams
the FineWeb-Edu sample through `datasets`.

## Run the pilot

```bash
# E0: 500 × 1024, observations sampled every 4 tokens, k=8
python bench/fit_ridge_calibration.py \
  --output checkpoints/qwen3_4b_to_1p7b_ridge.pt

# E1: 20 warmups + 100 CUDA-event measurements per length
python bench/benchmark_prefill_bridge.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt

# E2: provide 200 held-out prompt documents as JSONL/raw lines
python bench/eval_acceptance_pilot.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --text-file data/heldout_prompts.jsonl
```

For a long-generation curve after E2:

```bash
python bench/eval_generation_drift.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --text-file data/heldout_prompts.jsonl
```

Outputs are JSON/JSONL under `results/`, which is intentionally git-ignored. E1
reports median, P95, mapper-only time, end-to-end bridge time, native total time,
speedup, and peak allocated VRAM. E2 reports MAL and acceptance rate per method.

## Correctness contract

The target distribution is never altered. For each proposal position, the exact
rejection sampler accepts with `min(1, p/q)` and samples a rejection correction from
`normalize(max(p-q, 0))`. On rejection, the correction token is computed by the
target only on the next round; until then it is the single permitted draft-native
pending frontier and is replaced in place after target materialization.

The first draft boundary query recomputes the final prompt token with the mapped
cache truncated by one position, then discards that query's K/V. This supplies the
draft next-token logits without duplicating the final prompt token in the persistent
cache. The subsequent proposal K/V is genuinely incremental; no round reruns the
whole prefix.

## Tests

```bash
pytest -q
```

Tests are CPU-only and cover exact acceptance, cache positions/lengths, mapper
shapes, pending-frontier replacement, rejection rollback guards, and the block
acceptance loss. They do not download models.

## Handoff / go-no-go

Do not train the residual yet. First inspect E1 and E2:

- G0: bridge total is below native total at 4K/8K, ideally ≥1.5×.
- G1: Ridge mapped MAL does not catastrophically collapse.
- G2: Refresh is flatter/better than Init-only over 512 generated tokens.

If G0 or G2 fails, stop and record the failure rather than spending GPU time on the
block objective. If they pass, initialize the rank-8 zero-gated adapter:

```bash
python training/fit_acceptance_mapper.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --output checkpoints/qwen3_4b_to_1p7b_block_init.pt
```

`training/block_acceptance_loss.py` implements the intended surrogate
`-mean(sum_j prod_{i<=j}(1-TV(p_i,q_i))/gamma)` for that later phase.

