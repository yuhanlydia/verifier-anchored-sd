# Phase-2 acceptance mapper training plan

Phase 2 starts **only after E0-E2 pass**. It does not fine-tune either LLM. It trains
a small residual on top of the paper-faithful target-to-draft ridge mapper so the
draft proposal distribution better matches the verifier over a speculative block.

## Preconditions

Primary pair:

```text
target = Qwen/Qwen3-4B
draft  = Qwen/Qwen3-1.7B
```

Required gates:

- **E0:** refit the audited mapper using the pinned KVBridge backend, 500 x 1024,
  stride 4, `k=8`, `lambda=0.01`, content-space K mapping, all-source-head features.
- **G0 / E1:** full target-prefill+bridge beats target+native-draft initialization at
  4K/8K (preferred threshold >=1.5x).
- **G1 / E2:** mapped draft MAL is not catastrophically below Native SD.
- **G2 / E2:** Ridge Refresh is better/flatter than Ridge Init-only over long
  generation.

Do not use ridge checkpoints from the pre-audit simplified mapper.

Keep three disjoint data pools:

| Pool | Use | Default |
|---|---|---:|
| calibration | fit `W0` | 500 x 1024 FineWeb-Edu, stride 4 |
| acceptance train | fit `U,V,g` | 2,000 FineWeb-Edu prefixes |
| held-out eval | checkpoint selection/reporting | disjoint from both |

Do not train on SPEED-Bench or E2 held-out prompts.

## Parameterization

For every draft layer/head and separately for K and V:

```text
W = W0 + g U V^T
U: [d_out, r]
V: [r, d_in]
r = 8
g = 0 at initialization
```

`U,V ~ N(0,1e-3)`. `g=0` makes the first forward exactly Ridge while still allowing
the gate to receive a gradient because `U,V` are nonzero. `W0` and the ridge bias
are frozen. After training, merge `gUV^T` into `W0`; inference keeps the same single
affine mapper graph.

## Gradient path

One training example:

1. Target prefills the prefix with `torch.no_grad()` and
   `forward_incremental(..., inference=False)`. This is deliberate: target tensors
   are ordinary no-grad tensors, not inference tensors, because the mapper must save
   them for gradients with respect to `U,V,g`.
2. Verifier K is inverse-RoPE transformed to content space. The trainable mapper
   produces draft-content K/V and the draft model's exact RoPE is applied.
3. Recompute the final prompt token against the mapped draft prefix to obtain the
   first draft distribution; discard the query token's new K/V.
4. Sample `gamma=4` proposal tokens on-policy from the draft. Token IDs are detached;
   probability rows stay attached to the mapper/draft graph.
5. The frozen draft processes the four proposal tokens incrementally. Draft
   parameters have `requires_grad=False`, but autograd remains enabled through the
   draft computations so gradients reach the mapped cache.
6. Target verifies the sampled block under ordinary `no_grad`. Align rows as
   `p(y1|prefix)` followed by verification logits for later positions.
7. Backpropagate only through the draft computation and mapper residual.

Target weights: frozen. Draft weights: frozen. No SFT, RLHF, GRPO, or target-logit
backpropagation.

## Objective

Per-position exact acceptance mass:

```text
alpha_i = 1 - TV(p_i, q_i)
        = 1 - 0.5 * sum_v |p_i(v) - q_i(v)|
```

Block surrogate:

```text
A_gamma = sum_{j=1..gamma} prod_{i=1..j} alpha_i
L_block = -mean(A_gamma / gamma)
```

Ridge-preservation regularizer:

```text
L_reg = ||C_D(current) - C_D(ridge)||_F^2
        / (||C_D(ridge)||_F^2 + 1e-6)
```

Final loss:

```text
L = L_block + 1e-3 * L_reg
```

The pure ridge cache is produced with the **same mapper object** using
`include_residual=False`; do not load a second 1-2GB base mapper on the 24GB GPU.

## Default optimization

| Setting | Value |
|---|---:|
| optimizer | AdamW |
| learning rate | `2e-4` |
| weight decay | `1e-4` |
| gradient clip | `1.0` |
| physical batch | `1` |
| gradient accumulation | `16` |
| precision | BF16 LLMs |
| gamma | `4` |
| rank | `8` |
| first run | `500 optimizer updates` |
| extension ceiling | `1000 optimizer updates` |
| save interval | every `100 optimizer updates` |
| regularizer | `1e-3` |

With accumulation 16, 500 optimizer updates means **8,000 microbatches**. The code
uses an explicit optimizer/microbatch schedule so `--steps` cannot silently mean
microbatches.

Start with context lengths 512/1024/2048. On a 24GB card, if memory is too high,
reduce context length before changing the model pair or gamma, and record the
change.

### 16GB A4000 feasibility profile

The trainer's physical batch is already 1. `grad_accum` changes the number of
microbatches per optimizer update, but it does not materially lower the peak model
memory. To run a feasibility pass on a 16GB card, use `--low-vram`; it enforces
physical batch 1, context lengths 256/384/512, gradient accumulation 4, and
controlled CPU offload while keeping BF16 weights and `gamma=4`:

```bash
python training/fit_acceptance_mapper.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --output checkpoints/qwen3_4b_to_1p7b_block_16gb.pt \
  --text-file data/fineweb_acceptance_disjoint.jsonl \
  --low-vram --steps 500 --grad-accum 4 \
  --context-lengths 256,384,512 --device cuda --dtype bfloat16
```

This profile is for OOM/API feasibility only. Do not compare its latency against
the 24GB paper gate. Re-run E1/E2 without `--low-vram` on the target 24GB machine
for scientific numbers.

## Command

```bash
python training/fit_acceptance_mapper.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --output checkpoints/qwen3_4b_to_1p7b_block_step500.pt \
  --text-file data/fineweb_acceptance_disjoint.jsonl \
  --target Qwen/Qwen3-4B --draft Qwen/Qwen3-1.7B \
  --steps 500 --max-steps 1000 --gamma 4 \
  --context-lengths 512,1024,2048 --rank 8 \
  --lr 2e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --grad-accum 16 --lambda-reg 1e-3 --save-every 100 \
  --merge-output checkpoints/qwen3_4b_to_1p7b_block_merged.pt \
  --device cuda --dtype bfloat16
```

Artifacts:

- main `*.pt`: mapper + current residual;
- `*.pt.optim.pt`: optimizer state, optimizer-step index, history;
- `*.pt.json`: arguments and per-optimizer-step metrics;
- optional merged checkpoint: single affine inference mapper.

## Checkpoint selection

Evaluate steps 0/100/200/300/400/500 on fixed held-out prompts with identical seeds.
Compare:

1. Ridge Init-only;
2. Ridge Refresh;
3. one-step-TV Refresh (ablation);
4. block-acceptance Refresh (ours candidate).

Primary selection metric: held-out **Mean Accepted Length (MAL)**. Secondary:
wall-clock output tokens/s. Also report acceptance rate, bonus rate, one-step TV,
block surrogate, KV similarity to native/Ridge, TTFT, TPOT, mapper/refresh time,
target verifier calls/output token, and peak VRAM.

Never select by training loss alone.

## Required ablations after the first successful checkpoint

- objective: KV-MSE vs one-step TV vs block acceptance;
- refresh: init-only vs init+refresh;
- gamma: 2 / 4 / 6 / 8;
- context: 1K / 4K / 8K / 16K;
- rank: 4 / 8 / 16 (supplementary).

Do not change dataset split, seeds, revisions, and prompts across matched ablations.

## Stop conditions

Stop and keep the last good checkpoint if:

- loss/regularizer/cache contains NaN or Inf;
- cache length/position changes unexpectedly;
- held-out MAL drops >5 percentage points below Ridge Refresh;
- surrogate improves while measured throughput decreases;
- residual norm grows without acceptance improvement.

Debug in this order:

1. target/draft probability-row alignment;
2. proposal IDs are detached;
3. target/draft parameters are frozen;
4. target tensors entering the mapper are not inference tensors;
5. gate and residual gradients are nonzero;
6. `g=0` output matches Ridge within BF16 tolerance;
7. receiver RoPE positions match the target-token positions being refreshed.
