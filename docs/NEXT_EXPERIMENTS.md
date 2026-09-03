# Next experiments: verifier-anchored speculative decoding

## Current evidence boundary

The 16GB A4000 smoke is encouraging but not yet scientific evidence.  The only
completed E0 checkpoint used 8 sequences, `k=4`, depth-based source-layer selection,
and CPU fitting.  Under that non-paper smoke mapper:

- the 4K bridge initialization was measured at roughly 1.53x versus the old
  component-sum native baseline;
- one E2 prompt gave realized MAL 0.800 for Ridge Init-only and 1.222 for Ridge
  Refresh;
- the paper-faithful `k=8`, R²-selected E0 fit did not finish.

Therefore the current sign is **promising integration signal, not a method result**.
The dominant diagnosed failure was the E0 execution path: after calibration capture
both LLMs were already unloaded, yet the expensive selector / normal equations were
run on CPU.

## External update that changes the baseline

CacheBridge (arXiv:2609.00891, 1 Sep 2026) shows that architecture-indexed
matched-head affine support can dramatically reduce cross-model KV mapper storage,
application latency, calibration need, and construction time.  Matched-head mapping
is consequently **not a novelty claim of this project**.  We include a simple
matched-head centered-ridge backend as a strong translator baseline; it does not
implement CacheBridge's attention-sensitivity weighting or fused construction
kernel.

Our remaining scientific question is narrower and cleaner:

> Once target-to-draft KV translation is sufficiently accurate and cheap, does
> continually replacing persistent draft history with newly materialized verifier
> KV improve speculative acceptance / stability enough to justify the refresh?

## Resource-aware protocol

### 16GB kill test

Run `scripts/run_16gb_next.sh`.

Primary translator:

- Qwen3-4B verifier -> Qwen3-1.7B draft;
- matched KV topology;
- `k=8`, lambda=0.01, content-space K mapping;
- 128 x 1024-token calibration sequences, stride 4 = 32,768 fit observations;
- 32 calibration sequences for R² source-layer selection;
- controlled model offload only during capture;
- after capture, delete both LLMs and use CUDA for R² / centered normal equations;
- matched-head final fitter uses 8 draft layers per statistics block.

The matched-head affine support has 1,024 inputs per output head at `k=8`, versus
8,192 inputs in Full-Head.  For this 28-layer / 8-KV-head receiver that is about
58.7M versus 469.8M affine weights (8x smaller before biases).

E1:

- context lengths 512 / 1K / 2K / 4K / 8K;
- batch sweep 1 / 2 / 4;
- batch=1 is the latency result;
- batch>1 is the utilization / throughput curve;
- OOM is recorded as the capacity frontier rather than aborting prior rows;
- first pilot uses 5 warmups + 20 repeats.

E2:

- 64 held-out prompts, disjoint from E0;
- 512 prompt tokens;
- 64 generated tokens;
- gamma=4;
- Native SD, Ridge Init-only, Ridge Refresh;
- report realized MAL and conditional expected accepted length;
- paired bootstrap Refresh minus Init on the same prompts.

Only if E2 passes should the 16GB machine run a 16-prompt x 256-token drift curve.

### 24GB confirmatory run

Run `scripts/run_24gb_next.sh`.

1. Full-Head baseline: 500 x 1024, stride 4, `k=8`, all 500 sequences for R²
   selection, post-capture CUDA fitting.
2. Matched-head baseline: reuse the same 500 calibration shards, use 64 sequences
   for R² selection and all 500 for final centered ridge.
3. E1: 512 / 1K / 2K / 4K / 8K / 16K and batch 1 / 2 / 4 / 8, 20 warmups + 100
   repeats.
4. E2: 200 held-out prompts x 512 generated tokens, gamma=4, 10k paired bootstrap.

## Pre-registered gates

### G0: bridge has an actual systems opportunity

Use **directly timed** native initialization, not the sum of separately measured
medians.

- hard minimum: bridge speedup > 1.0 at 4K and 8K, batch=1;
- preferred: >=1.5x at either 4K or 8K;
- report mapper-only time and peak VRAM;
- report batch throughput separately from latency.

Failure of G0 means verifier anchoring may still be diagnostically interesting but
there is no compelling end-to-end speculative-decoding systems story.

### G1: translated draft is not catastrophically damaged

Let `E[MAL]` be the conditional acceptance-mass metric.  Require

`E[MAL](Ridge Init-only) / E[MAL](Native SD) >= 0.80`.

If a stronger translator cannot reach this gate, do not train the acceptance
adapter; translator error dominates the experiment.

### G2: verifier refresh is a real phenomenon

Primary test:

`Delta = E[MAL](Ridge Refresh) - E[MAL](Ridge Init-only)`.

Require the paired-bootstrap 95% CI lower bound for Delta to be > 0 on the 16GB
64-prompt pilot.  Confirm on the 24GB 200-prompt run.

A realized-MAL gain without a positive paired expected-MAL CI is not enough.

### G3: long-generation stabilization

Only after G2 passes, compare acceptance in output-position buckets.  Refresh should
reduce late-generation degradation relative to Init-only.  Do not use this curve to
rescue a failed G2.

### G4: acceptance-optimized residual is necessary

Only after G0-G2 pass, train the small residual mapper.  Compare:

- Ridge Refresh;
- one-step TV Refresh;
- block-acceptance Refresh.

The block objective must beat both baselines on held-out MAL and not lose the E1
systems gain after merging `W0 + gUV^T`.

## Decision tree

- **G0 fail:** stop the systems paper direction.
- **G0 pass, G1 fail:** translator/pair is the bottleneck; do not attribute failure
  to verifier anchoring.
- **G0/G1 pass, G2 fail:** stop the continual verifier-refresh contribution.  A
  target-to-draft prefill bridge alone is too close to existing cross-model transfer
  work to carry this paper.
- **G0/G1/G2 pass:** the core phenomenon is established; proceed to long-generation
  analysis and the block-acceptance residual.
- **G2 only passes for a poor Full-Head mapper but disappears for the matched-head
  baseline:** interpret refresh as an error-repair mechanism, not a general
  verifier-anchoring principle, and reconsider the paper framing.
