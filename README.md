# Verifier-Anchored Speculative Decoding

Research implementation for **Verifier-Anchored Draft Cache Refresh** with the
matched-KV Qwen3 pair:

```text
verifier / target: Qwen/Qwen3-4B
draft:             Qwen/Qwen3-1.7B
```

The repository is organized around falsifiable kill tests.  Do **not** train the
acceptance residual until E0-E2 establish that translation is useful and continual
verifier refresh is a real phenomenon.

## Core idea

The target distribution is never approximated.  The target keeps native KV and
performs exact speculative verification.  Cross-model translation affects only the
draft proposal cache.

At initialization:

```text
prompt -> target prefill -> exact target KV -> target-to-draft mapper -> draft KV
```

During generation, the target has already computed exact K/V for accepted proposal
tokens.  Instead of permanently retaining the draft-generated K/V for those tokens,
**Ridge Refresh** discards it and maps the verifier's exact new K/V into the
persistent draft cache.  A rejection correction, or the canonical target bonus
token after a fully accepted block, is the only one-token draft-native pending
frontier; on the next target forward that slot is replaced in place.

The scientific question is therefore not whether cross-model KV transfer is
possible.  It is:

> Once target-to-draft KV translation is accurate and cheap, does repeatedly
> anchoring persistent draft history to newly materialized verifier state improve
> speculative acceptance / long-generation stability enough to justify refresh?

## Important baseline update: CacheBridge

CacheBridge (arXiv:2609.00891, Sep 2026) shows that architecture-indexed matched-head
cross-model KV mapping can sharply reduce mapper size, construction cost, and
application latency.  **Matched-head mapping is not a novelty claim of this repo.**

We now expose two translator supports:

```text
--head-mode full
    Original Full-Head support: every draft KV head reads all verifier KV heads
    from each selected verifier layer.

--head-mode matched
    Head-local centered ridge: draft head h reads verifier head h from each selected
    layer.  This is an efficiency baseline inspired by the new matched-head result;
    it does NOT implement CacheBridge's attention-sensitivity weighting or fused
    sufficient-statistics kernel.
```

For Qwen3-4B -> Qwen3-1.7B with `k=8`, the affine input width is 8192 for Full-Head
and 1024 for matched-head.  The weight count is about 469.8M versus 58.7M, an 8x
reduction before biases.

Our paper can only stand if verifier refresh adds value **after** using a strong,
cheap translator baseline.

## Correctness audit

The current branch fixes several issues that made earlier smoke checkpoints
unsuitable as scientific evidence:

1. Full-Head uses all source KV heads, not a same-head shortcut.
2. R² source-layer selection is used for scientific E0; depth selection is smoke-only.
3. Keys are inverse-RoPE transformed to content space, mapped, then re-rotated with
   draft-model factors.
4. Transformers 5.x uses `DynamicCache` for incremental forwards.
5. Exact speculative decoding emits a target bonus token after a fully accepted
   block.
6. Correction/bonus frontiers share the one-token pending-frontier state machine.
7. Phase-2 `--steps` means optimizer updates, not microbatches.
8. Target tensors entering a trainable mapper are ordinary `no_grad` tensors, not
   inference-mode tensors.
9. E0 calibration directories have a frozen manifest and exact shard set.
10. E0 16GB/24GB fitting reuses CUDA **after both LLMs are unloaded**, instead of
    wasting the freed GPU while fitting on CPU.
11. E1 directly times complete native initialization instead of summing separate
    medians.
12. E1 sweeps batches and records OOM as a capacity boundary.
13. E2 reports both realized MAL and deterministic conditional acceptance mass with
    paired-bootstrap confidence intervals.

**Discard pre-audit mapper checkpoints and rerun E0.**

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,kvbridge,dev]'
```

For reproducible science, use frozen local files for calibration/evaluation rather
than allowing E0 and E2 to stream overlapping FineWeb-Edu prefixes.

## Recommended next run: 16GB

Use an RTX A4000 / 16GB-class card as the first **kill test**, not as a reduced copy
of the full 24GB paper experiment.

```bash
export HF_HUB_CACHE=/path/with/free/space
export EVAL_TEXT=/path/to/heldout_prompts.jsonl
bash scripts/run_16gb_next.sh
```

The script runs:

### E0-16 matched-head

```text
Qwen3-4B -> Qwen3-1.7B
128 x 1024-token calibration sequences
stride = 4                -> 32,768 fit observations
R² layer selection on 32 sequences
k = 8
ridge lambda = 0.01
capture: controlled CPU offload
fit: CUDA after both models are deleted
matched-head fit block: 8 draft layers
```

Why 128 rather than immediately forcing 500?  The matched-head input is 1024-D, so
32,768 observations already give a 32x observation/feature ratio.  The point of the
16GB run is to cheaply establish G0/G1/G2 before spending time on confirmatory
calibration scaling.

### E1-16

```text
lengths: 512, 1K, 2K, 4K, 8K
batch:   1, 2, 4
warmup:  5
repeats: 20
```

Batch 1 is the latency result.  Batches 2/4 are the utilization/throughput curve.
`--continue-on-oom` records the maximum feasible batch per context length instead of
throwing away previous rows.

### E2-16

```text
64 held-out prompts
512 prompt tokens
64 generated tokens
gamma = 4
Native SD vs Ridge Init-only vs Ridge Refresh
5000 paired bootstrap samples
```

This replaces the previous one-prompt smoke.  More independent prompts are more
valuable for deciding whether refresh is real than one 512-token continuation on a
slow/offloaded 16GB setup.

Only if E2 passes G2 should the 16GB machine run a 16-prompt x 256-token drift curve.

## Confirmatory run: 24GB

```bash
export HF_HUB_CACHE=/path/with/free/space
export EVAL_TEXT=/path/to/heldout_prompts.jsonl
bash scripts/run_24gb_next.sh
```

The 24GB script fits and evaluates both translator supports.

### Full-Head baseline

```text
500 x 1024 calibration
stride 4 (~128K observations)
k = 8
R² selection on all 500
post-capture CUDA fitting
```

### Matched-head baseline

Uses the same 500 calibration shards and all 500 final-fit sequences, with R² layer
selection on 64 sequences.  This isolates mapper support while keeping the model
pair, benchmark prompts, and content-space treatment fixed.

### E1-24

```text
lengths: 512, 1K, 2K, 4K, 8K, 16K
batch:   1, 2, 4, 8
warmup:  20
repeats: 100
```

### E2-24

```text
200 held-out prompts
512 prompt tokens
512 generated tokens
gamma = 4
10,000 paired bootstrap samples
```

## Go / no-go gates

The exact preregistration is in `docs/NEXT_EXPERIMENTS.md`.

### G0 - systems opportunity

Use directly timed native initialization:

```text
native = target prefill + native draft prefill in one timed function
bridge = target prefill + target->draft mapping
```

Hard minimum: batch-1 bridge speedup >1.0 at 4K and 8K.  Preferred continuation
signal: >=1.5x at either 4K or 8K.

### G1 - translated draft remains useful

Using conditional expected accepted length:

```text
E[MAL](Ridge Init-only) / E[MAL](Native SD) >= 0.80
```

If G1 fails, translator error dominates; do not train an acceptance residual.

### G2 - verifier refresh is real

Primary statistic:

```text
Delta = E[MAL](Ridge Refresh) - E[MAL](Ridge Init-only)
```

Require the paired-bootstrap 95% CI lower bound for `Delta` to be >0.  A positive
one-seed realized MAL is not enough.

This gate is now the central paper decision.  If a good matched-head translator
makes the refresh benefit disappear, the earlier gain was likely repair of a poor
mapper rather than a general verifier-anchoring principle.

### G3 - long-generation stabilization

Only after G2: measure MAL/acceptance by output-position buckets.  Refresh should
reduce late-generation degradation relative to Init-only.

### G4 - block-acceptance residual

Only after G0-G2 pass, train the rank-8 residual and compare:

```text
Ridge Refresh
one-step TV Refresh
block-acceptance Refresh
```

The block objective must beat both held-out baselines without destroying the E1
systems gain after merging `W0 + gUV^T`.

## E0 CLI

Full-Head example:

```bash
python bench/fit_ridge_calibration.py \
  --head-mode full --fit-profile 24gb \
  --sequences 500 --selection-sequences 500 \
  --seq-len 1024 --stride 4 --k 8 --lambda 0.01 \
  --layer-selection r2 \
  --calibration-dir artifacts/e0_full/calibration \
  --kvbridge-artifact artifacts/e0_full/kvbridge \
  --output checkpoints/full.pt
```

Matched-head example:

```bash
python bench/fit_ridge_calibration.py \
  --head-mode matched --fit-profile 16gb --low-vram \
  --sequences 128 --selection-sequences 32 \
  --seq-len 1024 --stride 4 --k 8 --lambda 0.01 \
  --layer-selection r2 \
  --calibration-dir artifacts/e0_matched/calibration \
  --output checkpoints/matched.pt
```

## E1 CLI

```bash
python bench/benchmark_prefill_bridge.py \
  --mapper checkpoints/matched.pt \
  --memory-profile 16gb --batch-sizes 1,2,4 \
  --lengths 512,1024,2048,4096,8192 \
  --continue-on-oom --mapper-dtype bfloat16
```

For the old offload-only feasibility path, add `--low-vram`; do not mix its
wall-clock numbers with resident 24GB G0 results.

## E2 CLI

```bash
python bench/eval_acceptance_pilot.py \
  --mapper checkpoints/matched.pt \
  --memory-profile 16gb \
  --text-file data/heldout_prompts.jsonl \
  --bootstrap-samples 5000 --mapper-dtype bfloat16
```

The output JSON contains per-prompt rows, summaries, paired-bootstrap differences,
and explicit G1/G2 booleans.

## Phase 2

Phase 2 remains disabled until the kill tests pass.  The frozen LLMs and base ridge
mapper receive a small mergeable residual:

```text
W = W0 + g U V^T,  g=0 at initialization
```

Available objectives are `one_step_tv` and `block`.  The block objective uses
prefix-weighted speculative acceptance mass rather than KV reconstruction error.
See `docs/TRAINING_PLAN.md` for the training protocol.

## Tests

```bash
python -m compileall -q src bench training
pytest -q
```

CPU tests cover exact rejection sampling, bonus/correction frontiers, cache
positions, RoPE strip/reapply, Full-Head and matched-head support, streaming
matched-head centered ridge, optimizer-step accounting, acceptance-mass statistics,
paired bootstrap, resource profiles, and output-position bucketing.

GPU/model integration is deliberately a separate claim: rerun E0 -> E1 -> E2 on the
intended 16GB/24GB CUDA hosts before drawing scientific conclusions.
