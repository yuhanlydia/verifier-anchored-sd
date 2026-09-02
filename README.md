# Verifier-Anchored Speculative Decoding

Experimental implementation of **Verifier-Anchored Draft Cache Refresh** for the
matched-KV pair:

```text
verifier / target: Qwen/Qwen3-4B
draft:             Qwen/Qwen3-1.7B
```

The repository is intentionally staged around falsifiable kill tests. Do **not**
train the acceptance residual until E0-E2 pass.

## Research contract

The target distribution is never approximated. The target keeps its native KV cache
and performs exact speculative verification. Cross-model translation affects only
the draft proposal cache.

At initialization:

```text
prompt -> target prefill -> exact target KV -> target-to-draft mapper -> draft KV
```

During generation, accepted verifier K/V is translated back into the persistent
draft cache. A rejection correction, or the canonical target **bonus token** after a
fully accepted block, is temporarily represented by one draft-native pending
frontier. On the next verifier forward its exact target KV is materialized and the
pending draft KV is replaced in place.

Thus the persistent draft history is verifier-anchored while target sampling remains
exact.

## Important audit fixes

The original pilot implementation was useful for plumbing but was **not suitable for
scientific E0 results**. The audited implementation now enforces:

1. **Paper-faithful cross-head features.** Each draft KV head may read all verifier
   KV heads from every selected verifier layer. For Qwen3 with `k=8`, the active
   input width is `8 layers x 8 KV heads x 128 = 8192`, not `8 x 128`.
2. **Calibration-R2 source-layer selection.** Production E0 delegates fitting to the
   pinned KVBridge backend instead of choosing source layers by depth proximity.
3. **Explicit RoPE strip/reapply.** Verifier keys are inverse-rotated to content
   space, mapped there, then rotated with the draft model's own RoPE factors.
4. **Transformers DynamicCache runtime.** Incremental generation no longer relies on
   legacy tuple-cache support.
5. **Canonical speculative bonus token.** A fully accepted draft block emits one
   target-sampled bonus token, using the same pending-frontier mechanism as a
   rejection correction.
6. **Correct optimizer-step semantics.** `--steps 500 --grad-accum 16` means 500
   optimizer updates and 8,000 microbatches, not ~32 updates.
7. **Autograd-safe target tensors.** Phase-2 target forwards use ordinary `no_grad`
   tensors rather than inference tensors inside the trainable mapper path.
8. **Frozen calibration contract.** E0 records model revisions/tokenizer/config and
   consumes exactly the requested calibration shard set; stale or mismatched shard
   directories fail fast.

**Any ridge checkpoint produced before these audit fixes must be discarded and E0
must be refit.** The feature layout is different by construction.

## 24GB setup

The primary target/draft pair is chosen so both BF16 LLMs plus a BF16 live mapper and
moderate KV caches fit a 24GB GPU for the pilot. E0 fitting is out-of-core: calibration
KV is persisted to SafeTensors and both LLMs are unloaded before the large ridge
normal equations are solved. The full 500 x 1024 / stride-4 calibration cache can use
tens of GB of disk, so place `artifacts/` on a filesystem with adequate space.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,kvbridge,dev]'
```

`kvbridge` is pinned to the audited research commit used by this project.

### 16GB feasibility profile

The physical batch is already `1`; lowering gradient accumulation alone does not
reduce model-weight residency. For an RTX A4000/16GB feasibility run, add
`--low-vram` to E0/E1/E2/training. This keeps exact BF16 weights, enables controlled
CPU offload, uses training contexts `256,384,512`, sets gradient accumulation to 4,
and shortens pilot generation to 128 tokens. It is intended to prove that the
runtime works and to catch OOM/API bugs, not to claim the 24GB paper speed gate.

```bash
# training smoke/profile: physical batch=1, 500 optimizer updates, 2,000 microbatches
python training/fit_acceptance_mapper.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --output checkpoints/qwen3_4b_to_1p7b_block_16gb.pt \
  --text-file data/fineweb_acceptance_disjoint.jsonl \
  --low-vram --steps 500 --gamma 4 --grad-accum 4 \
  --context-lengths 256,384,512 --device cuda --dtype bfloat16

# E0 can keep the scientific 1024-token calibration and offload model weights.
python bench/fit_ridge_calibration.py \
  --target Qwen/Qwen3-4B --draft Qwen/Qwen3-1.7B \
  --sequences 500 --seq-len 1024 --stride 4 --k 8 --lambda 0.01 \
  --low-vram --accumulation-device cpu \
  --calibration-dir artifacts/e0_qwen3_4b_to_1p7b_calibration_16gb \
  --kvbridge-artifact artifacts/e0_qwen3_4b_to_1p7b_kvbridge_16gb \
  --output checkpoints/qwen3_4b_to_1p7b_ridge_16gb.pt

# E1/E2 feasibility smoke (not the paper gate):
python bench/benchmark_prefill_bridge.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge_16gb.pt \
  --lengths 256,512,1024,2048,4096 --warmup 2 --repetitions 10 --low-vram
python bench/eval_acceptance_pilot.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge_16gb.pt \
  --text-file data/heldout_prompts.jsonl --prompts 20 --low-vram
```

`--low-vram` does not change the scientific defaults; omit it on the intended 24GB
machine. If the A4000 still OOMs while loading both models, the remaining safe
fallback is `--device auto` with CPU offload, but its wall-clock numbers must not be
used for G0/G4.

## E0 - fit the target-to-draft ridge mapper

Default scientific configuration:

```text
FineWeb-Edu
500 sequences x 1024 tokens
observation stride = 4  (~128k token observations)
k = 8 source layers per draft layer
ridge lambda = 0.01
content-space K mapping = on
all-source-KV-head features = on
```

Run:

```bash
python bench/fit_ridge_calibration.py \
  --target Qwen/Qwen3-4B \
  --draft Qwen/Qwen3-1.7B \
  --sequences 500 --seq-len 1024 --stride 4 \
  --k 8 --lambda 0.01 \
  --calibration-dir artifacts/e0_qwen3_4b_to_1p7b_calibration \
  --kvbridge-artifact artifacts/e0_qwen3_4b_to_1p7b_kvbridge \
  --output checkpoints/qwen3_4b_to_1p7b_ridge.pt
```

For reproducibility, prefer a frozen local FineWeb-Edu JSONL via `--text-file`.
Changing model revision, tokenizer, sequence count/length, or stride requires a fresh
`--calibration-dir`.

## E1 / KT-A - does mapping actually beat draft prefill?

```bash
python bench/benchmark_prefill_bridge.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --lengths 512,1024,2048,4096,8192,16384 \
  --warmup 20 --repetitions 100 \
  --mapper-dtype bfloat16 \
  --output results/e1_prefill_bridge.json
```

E1 reports target prefill, native draft prefill, mapper-only time, complete bridge
time, P95, speedup, and method-specific peak VRAM. Native timing is not charged for
RoPE instrumentation that only the bridge needs.

**G0:** the full bridge must beat target+draft native initialization at 4K/8K;
`>=1.5x` is the preferred continuation threshold.

## E2 / KT-B - is the translated cache useful for real SD?

Use held-out prompts disjoint from E0 calibration:

```bash
python bench/eval_acceptance_pilot.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --text-file data/heldout_prompts.jsonl \
  --prompts 200 --prompt-tokens 512 --new-tokens 512 --gamma 4 \
  --mapper-dtype bfloat16 \
  --output results/e2_acceptance_pilot.json
```

Matched methods:

```text
Native SD          native draft prefill + native persistent draft KV
Ridge Init-only    mapped prompt KV + draft-native generated KV
Ridge Refresh      mapped prompt KV + verifier-refreshed accepted KV
```

Report MAL, acceptance rate, target-bonus rate, wall-clock output tokens/s, and peak
VRAM. All three use the same exact target verifier and canonical target bonus rule.

**G1:** Ridge Init-only must not catastrophically collapse relative to Native SD.

**G2:** Ridge Refresh must improve or flatten long-generation acceptance compared
with Ridge Init-only.

For the position-resolved check:

```bash
python bench/eval_generation_drift.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --text-file data/heldout_prompts.jsonl \
  --prompts 50 --prompt-tokens 512 --new-tokens 512 --gamma 4 \
  --mapper-dtype bfloat16 \
  --output results/generation_drift.json
```

The output buckets are inclusive `1-64`, `65-128`, `129-256`, and `257-512` and are
weighted by the actual output positions covered by each speculative block.

## Phase 2 - block-acceptance residual

Only after G0-G2 pass, add a rank-8 residual to the frozen ridge mapper:

```text
W = W0 + g U V^T,   g=0 initially
```

Target and draft weights remain frozen. The training objective is based on exact
per-position speculative acceptance mass

```text
alpha_i = 1 - TV(p_i, q_i)
A_gamma = sum_j product_{i<=j} alpha_i
L = -mean(A_gamma / gamma) + 1e-3 * L_cache_reg
```

Run:

```bash
python training/fit_acceptance_mapper.py \
  --mapper checkpoints/qwen3_4b_to_1p7b_ridge.pt \
  --output checkpoints/qwen3_4b_to_1p7b_block_step500.pt \
  --text-file data/fineweb_acceptance_disjoint.jsonl \
  --steps 500 --max-steps 1000 --gamma 4 \
  --context-lengths 512,1024,2048 --rank 8 \
  --lr 2e-4 --weight-decay 1e-4 --grad-clip 1.0 \
  --grad-accum 16 --lambda-reg 1e-3 --save-every 100 \
  --merge-output checkpoints/qwen3_4b_to_1p7b_block_merged.pt
```

`--steps` is explicitly **optimizer updates**. With the default accumulation, this
500-step run consumes 8,000 microbatches. Checkpoint selection is by held-out MAL
first and wall-clock throughput second, never by training loss alone. The final
residual is merged into the affine mapper, leaving the inference graph unchanged.

See `docs/TRAINING_PLAN.md` for the full phase-2 protocol.

## Tests

CPU regression suite:

```bash
pytest -q
python -m compileall -q src bench training
```

The tests cover rejection sampling, canonical bonus frontier, cache positions and
lengths, pending-frontier replacement, cross-head mapping, explicit RoPE
strip/reapply, optimizer-step accounting, output-position bucketing, and the block
acceptance loss. Model/GPU integration still has to be validated by running E0-E2
on the intended 24GB CUDA machine.

## Stop rules

Do not convert a negative kill test into a large hyperparameter search.

- G0 fails: stop the prefill-saving story for this pair.
- G1 fails badly: the translated cache is not a viable draft state; change pair or
  mapper before training anything.
- G2 fails: verifier refresh is not a contribution; do not train the block adapter
  just to rescue it.
- Only if G0-G2 pass: train the acceptance-aligned residual, then run objective,
  refresh, gamma, context-length, and rank ablations.
