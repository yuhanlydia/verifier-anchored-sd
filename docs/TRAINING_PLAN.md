# Phase-2 acceptance mapper training plan

This is the executable handoff for the acceptance-optimized residual. It starts
only after E0–E2 have established that the pair, mapper, and refresh phenomenon are
viable. The goal is not to fine-tune either language model. The goal is to move a
small translated KV mapper toward higher multi-token speculative utility while
staying close to the ridge cache.

## 1. Preconditions and data separation

Use the same primary pair and mapper as the pilot:

```text
target = Qwen/Qwen3-4B
draft  = Qwen/Qwen3-1.7B
```

Before training, verify:

- E0 produced `checkpoints/qwen3_4b_to_1p7b_ridge.pt` with `k=8`, `lambda=0.01`.
- G0 bridge total beats native initialization at 4K/8K.
- G1 Ridge translated-cache MAL is not catastrophically below native SD.
- G2 Ridge Refresh is better/flatter than Ridge Init-only over 512 generated tokens.
- target/draft weights are frozen and both models fit with the training context on the 24GB GPU.

Keep three disjoint data pools:

| Pool | Use | Default |
|---|---|---:|
| calibration | fit `W0` ridge | 500 × 1024 FineWeb-Edu sequences |
| acceptance train | optimize `U,V,g` | 2,000 FineWeb-Edu prefixes |
| held-out eval | choose checkpoint and report results | separate from both above |

Do not train on SPEED-Bench or the 200 E2 prompts. A JSONL line may be
`{"text":"..."}` or raw text. The command supports streaming FineWeb-Edu when no
`--text-file` is supplied, but a frozen local file is preferred for reproducibility.

## 2. Parameterization and initialization

Start from the fitted ridge mapper `W0`. For every draft layer, KV head, and K/V
kind, train:

```text
W = W0 + g U V^T
U: [d_out, r]
V: [r, d_in]
r = 8
g = 0 at initialization
```

`U` and `V` are initialized with small `N(0, 1e-3)` values. The zero gate makes the
first forward pass exactly Ridge, while nonzero `U,V` allow the gate to receive a
gradient. K and V have separate residuals. The base ridge weights and biases are
not optimizer parameters. After training, merge the residual into `W*` and use the
same inference graph as Ridge.

The initialization command is:

```bash
python training/fit_acceptance_mapper.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --output checkpoints/qwen3_4b_to_1p7b_block_init.pt \
  --rank 8 --initialize-only
```

## 3. Training example and gradient path

Each optimization example is one prefix at a randomly cycled context length
`L ∈ {512, 1024, 2048}`. The implementation follows this order:

1. Target prefills the prefix once under inference/no-grad and produces exact
   target cache `C_T` and the target boundary distribution.
2. The trainable mapper produces `C_D = M_(W0+gUV^T)(C_T)`.
3. The draft boundary query recomputes the final prompt token with the last mapped
   K/V omitted; its K/V is discarded. This obtains the first draft distribution
   without duplicating the prompt token in persistent cache.
4. For `gamma=4`, sample each proposal token from the current draft distribution.
   Token IDs are detached; the probability rows remain attached to the graph.
5. The frozen draft processes each sampled token incrementally. Its temporary K/V
   stays in the graph for the four-step block but is not committed as persistent
   verifier-refresh state.
6. Target verifies the sampled block under no-grad. The target rows are aligned as
   `p(y1 | prefix)` from the boundary logits, then `p(yi | prefix,y< i)` from the
   verification logits.
7. Compute block acceptance loss, cache residual penalty, backpropagate, clip, and
   update only `U,V,g`.

The target never receives gradients. The draft model never receives gradients.
There is no SFT, RLHF, GRPO, or target-logit backward pass.

## 4. Objective

For each position, use exact acceptance mass:

```text
alpha_i = 1 - TV(p_i, q_i)
        = 1 - 0.5 * sum_v |p_i(v) - q_i(v)|
```

Later positions count only if earlier positions would have been accepted:

```text
A_gamma = sum_{j=1..gamma} prod_{i=1..j} alpha_i
L_block = -mean(A_gamma / gamma)
```

Add the normalized ridge-preservation penalty on the mapped prompt cache:

```text
L_reg = ||C_D(current) - C_D(ridge)||_F^2
        / (||C_D(ridge)||_F^2 + 1e-6)
```

Use:

```text
L = L_block + 1e-3 * L_reg
```

`training/block_acceptance_loss.py` implements `L_block`. The trainer computes
`L_reg` over K and V for the current prefix. Do not replace this with perplexity or
plain KV-MSE as the primary objective.

## 5. Optimization and memory budget

| Setting | Value |
|---|---:|
| optimizer | AdamW |
| learning rate | `2e-4` |
| weight decay | `1e-4` |
| gradient clip | `1.0` |
| physical batch | `1` |
| gradient accumulation | `16` |
| precision | BF16 |
| gamma | `4` |
| rank | `8` |
| first run | `500` optimizer steps |
| extension ceiling | `1000` steps |
| save interval | every `100` steps |
| regularizer | `lambda_reg=1e-3` |

Only the mapper output and four draft token forwards retain autograd state. Target
prefill/verification run in inference mode. Start with 512/1024/2048 contexts; add
4096 only if peak VRAM remains below the host's safety margin. If memory exceeds
24GB, reduce context length first, then accumulation, while recording the change;
do not silently change the model pair or gamma.

## 6. Run command

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

The trainer writes:

- `*.pt`: mapper weights and residual;
- `*.pt.optim.pt`: optimizer state, step, and training history;
- `*.pt.json`: arguments and per-step loss/expected-length history.

When `--merge-output` is supplied, the additional checkpoint contains only the
merged affine mapper (`W0+gUV^T`) and is the one to use for inference benchmarks.

The current trainer starts from the ridge checkpoint and creates a zero-gated
residual automatically if needed. `--initialize-only` exits after that checkpoint.

## 7. Checkpoint selection and evaluation

At steps 0, 100, 200, 300, 400, and 500, evaluate on held-out prompts with:

1. Ridge Init-only;
2. Ridge Refresh;
3. one-step-TV Refresh, if implemented for the ablation;
4. block-acceptance Refresh (the candidate).

Report, with identical prompts/seeds/gamma:

- MAL and acceptance rate;
- one-step TV and block surrogate;
- KV cosine similarity to native draft cache and to Ridge cache;
- output tokens/sec, end-to-end latency, TTFT, TPOT;
- target verification calls/output token;
- mapper and refresh time;
- peak VRAM.

Select the checkpoint by held-out MAL first, then wall-clock throughput. Never select
by training loss alone. After selecting, merge `W0+gUV^T` and rerun the E1 timing
test; the merged checkpoint must have the same inference graph and no residual
module call.

## 8. Required ablations

Run these after the first successful 500-step checkpoint:

- objective: KV-MSE vs one-step TV vs block acceptance;
- refresh: init-only vs init+refresh;
- gamma: 2, 4, 6, 8;
- context: 1K, 4K, 8K, 16K;
- rank: 4, 8, 16 in supplementary results.

Keep the dataset split, seed list, model revision, and evaluation prompts fixed
across ablations. The main comparison must not change both objective and refresh
policy at once.

## 9. Stop conditions and debugging

Stop the run and keep the last good checkpoint if any of these occurs:

- loss or regularizer becomes NaN;
- mapped cache contains NaN/Inf;
- cache sequence length changes unexpectedly;
- MAL on held-out data drops more than 5 percentage points below Ridge Refresh;
- training improves surrogate but decreases wall-clock throughput;
- mapper residual norm grows without acceptance improvement.

First debug in this order: verify target/draft probability row alignment, verify
that sampled IDs are detached, verify that target/draft parameters have
`requires_grad=False`, verify gate and residual gradients are nonzero, then compare
the trainer's Ridge-at-gate-zero output against the Ridge baseline bitwise up to
BF16 tolerance.
