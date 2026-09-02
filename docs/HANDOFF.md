# Handoff checklist

This repository is intentionally independent from `/root/vla-gap-lab`; that worktree
had unrelated uncommitted changes.

## Before using a GPU

- Confirm both Qwen3 checkpoints are accessible and that `transformers>=4.51` is installed.
- Confirm the model configs report `num_key_value_heads=8` and the same `head_dim=128`.
- Prepare disjoint calibration and held-out prompt files.
- Run `pytest -q` on the execution host.

## Required order

1. E0 calibration: do not change the default `k=8`, `lambda=0.01`, `500 × 1024`, stride 4 on the first run.
2. E1 prefill bridge: inspect 4K and 8K end-to-end bridge speedup and peak VRAM.
3. E2 acceptance pilot: compare all three policies with `gamma=4`, not just one-step TV.
4. Only if G0/G1/G2 pass, initialize and train the rank-8 residual on a disjoint corpus.

## Known implementation boundaries

- Qwen3 rotary geometry is assumed compatible between the two models. For a pair with different RoPE scaling, add explicit inverse target RoPE plus draft RoPE before fitting/mapping keys.
- The first mapped-cache boundary requires one draft query of the final prompt token with its K/V discarded. This is necessary to obtain the draft next-token distribution without duplicating the last prompt token.
- The current pilot does not append a separate bonus token after an all-accepted block; this keeps MAL accounting simple and identical across methods. Add the bonus path only after the kill tests pass.
- `eval_generation_drift.py` reports per-block MAL assigned to the output-position bucket touched by that block. For a publication-quality curve, retain per-token output positions and bootstrap confidence intervals.
- The HF adapter currently uses tuple cache compatibility. If a future Transformers release removes tuple support, replace `forward_incremental`'s cache argument with a `DynamicCache` conversion in one place.

