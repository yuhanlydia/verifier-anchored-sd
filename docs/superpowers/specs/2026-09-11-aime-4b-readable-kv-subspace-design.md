# AIME 4B-Readable Teacher-KV Subspace Design

## Goal

Test whether AIME correctness gradients from Qwen3-4B identify a low-rank
subspace of Qwen3-8B prompt KV that is more useful to the 4B decoder than the
unfiltered teacher KV. Run the experiment for both shape-compatible direct
8B KV transfer and the existing Full-Head ridge mapping.

The central computation is:

```text
AIME problem -> Qwen3-8B prompt KV
             -> direct transfer OR Full-Head mapping into 4B KV geometry
             -> 4B answer-span loss gradient with respect to transferred KV
             -> per-layer/head K/V SVD
             -> projected transferred KV -> 4B evaluation
```

This is an adaptation of Self-Policy Distillation (SPD). SPD extracts a
capability subspace from correctness-aligned KV gradients within one model.
Here, the loss and gradients come from the receiving 4B decoder, while the KV
state originated in the 8B model.

## Fixed model and data contracts

Use the already frozen model pair and exact BF16 weights:

```text
teacher: Qwen/Qwen3-8B
revision: b968826d9c46dd6066d109eabc6255188de91218

receiver: Qwen/Qwen3-4B
revision: 1cfa9a7208912126459214e8b04321603b3df60c

dtype: bfloat16
```

Both models have 36 layers, 8 KV heads, and head dimension 128. Direct KV
transfer is therefore shape-compatible, but remains an experimental control:
matching tensor dimensions do not imply matching learned coordinates.

Use `allenai/aime-2022-2025` at revision
`73e1eba765ad5847cdb5d1e2e7aaf7b22b585798`. The source has one `train`
split with fields `id`, `problem`, `solution`, `answer`, `url`, and `year`.
Every materialized row records the source revision, row ID, year, URL, and a
SHA256 digest of the normalized content.

## Evaluation protocols

Two protocols answer different questions and must be reported separately.

### AIME2025 five-fold cross-fit

Use all 30 AIME2025 I/II problems. Construct five deterministic folds with
three AIME I and three AIME II problems in each held-out fold. For fold `j`,
fit branch-specific SVD bases on the other 24 problems and evaluate on the six
held-out problems. Concatenate only held-out predictions when computing final
metrics, so no evaluated problem contributes to its own basis.

This protocol directly tests whether AIME2025 contains stable, shared
4B-readable KV directions. It does not produce one globally selected rank from
AIME2025. All pre-registered ranks are reported, and the paper-faithful
half-rank setting `r=64` is the primary row.

### Cross-year generalization

Fit bases on AIME2022-2024 and reserve every AIME2025 problem for evaluation.
Use a deterministic split inside 2022-2024 for selecting optional rank and
soft-projection strength, then freeze that choice before touching AIME2025.
This protocol tests whether the readable directions generalize to new
competition problems.

The two protocols share model revisions, prompt formatting, answer-span loss,
transfer branches, metrics, and controls.

## AIME normalization and correctness span

Normalize each dataset row as:

```json
{
  "example_id": "aime-<year>-<source-id>",
  "problem": "...",
  "gold_solution": "...",
  "gold_answer": "000" ,
  "year": 2025,
  "source_url": "..."
}
```

`gold_answer` is the integer answer formatted as exactly three digits, matching
the AIME answer convention. Extract the first complete solution from the
dataset's potentially multi-author `solution` field and append a canonical
suffix:

```text
Answer: <gold_answer>
```

Tokenize the full prompt and teacher-forced solution, and mask every label
except the token positions belonging to `<gold_answer>` in the canonical
suffix. Reject a row if the answer is outside `[000, 999]`, the suffix cannot
be located exactly, or the resulting correctness span is empty.

The aligned loss is:

\[
\mathcal L_{\mathrm{answer}}
=-\frac{1}{|S|}\sum_{t\in S}
\log p_{4B}(y_t\mid y_{<t}, KV_{\mathrm{prefix}}).
\]

## Transfer branches

Fit and evaluate separate subspaces for two branches. Never reuse a basis
fitted on one branch for the other branch's primary result.

### Direct branch

1. Run the AIME problem prompt through Qwen3-8B.
2. Convert 8B keys from position space to content space using recorded 8B RoPE
   factors; values remain unchanged.
3. Treat the 36-layer, 8-head, 128-dimensional tensors as 4B-space candidate
   KV without a learned mapper.
4. Apply 4B RoPE when constructing the receiver cache.

The runtime must verify layer count, KV-head count, head dimension, tokenizer
contract, positions, and rotary provenance before accepting the cache.

### Full-Head mapped branch

1. Run the same prompt through Qwen3-8B.
2. Apply the frozen strongest Full-Head ridge mapper selected by the completed
   paper-vs-subspace suite.
3. Keep keys in content space while fitting and applying the subspace.
4. Apply 4B RoPE only when constructing the receiver cache.

The mapper checkpoint and metadata SHA256 values are part of every downstream
artifact contract.

## Gradient collection in 4B KV space

For each branch, make the transferred content-space K tensors and V tensors
leaf tensors with gradients enabled. Use them as the prompt cache for the 4B
decoder, teacher-force the gold solution, and compute only the answer-span
loss. Generated solution-token KV remains native 4B KV; the gradients of
interest are those with respect to the transferred prompt cache.

For each receiver layer `l`, KV kind `a`, and head `h`, flatten the prompt-token
gradients across calibration examples into:

\[
G_{b,l,a,h}\in\mathbb R^{M\times128},
\]

where `b` is `direct` or `full_head`. Accumulate the Gram matrix in FP64 rather
than storing all token gradients:

\[
C_{b,l,a,h}=G^\top G.
\]

Eigendecomposition of `C` yields the same right-singular basis as SVD of `G`.
Store eigenvalues, explained-gradient-energy curves, sample counts, gradient
norms, and the top 64 vectors. Abort the fit if any layer/head receives no
finite nonzero gradient.

## Projection methods

The primary method follows SPD and uses hard half-rank projection:

\[
z'=P_{64}z,\qquad P_r=V_rV_r^\top.
\]

Evaluate the following matrix independently for direct and Full-Head branches:

```text
unprojected transfer
gradient hard: ranks 8, 16, 32, 64
gradient soft: ranks 8, 16, 32, 64; beta 0.25, 0.50, 0.75
gradient orthogonal complement: rank 64
PCA activation control: ranks 16, 32, 64
seeded random orthonormal control: ranks 16, 32, 64
```

Soft projection is:

\[
z'=(1-\beta)z+\beta P_rz.
\]

Apply K bases to content-space keys and V bases to values. Reapply receiver
RoPE after K projection. The unprojected direct and Full-Head branches are the
mandatory baselines for their corresponding projected methods.

## Metrics and interpretation

Report metrics on identical held-out examples and decoding seeds:

```text
answer-span NLL under 4B
probability assigned to the gold AIME answer tokens
1 - TV between 8B and intervened 4B next-token distributions
top-1 agreement with 8B
greedy exact-match AIME accuracy
block speculative mean accepted length and acceptance rate
```

The primary mechanistic comparison for each branch is paired answer-span NLL:

```text
gradient projected transfer - unprojected transfer
```

The primary deployment comparison is paired exact-match accuracy and block
acceptance against native 4B and unprojected Full-Head transfer. Because
AIME2025 has only 30 examples, publish every per-example result and paired
bootstrap interval; do not turn a small positive mean into a success claim
when the interval crosses zero.

Performance thresholds are diagnostic. Negative results do not stop artifact
generation or prevent the other branch from running.

## Components and interfaces

Add these focused components:

```text
src/verifier_anchored_sd/aime_data.py
    pinned dataset-row normalization, answer-span construction, deterministic folds

src/verifier_anchored_sd/kv_gradient_subspace.py
    branch-neutral FP64 Gram accumulation, eigendecomposition, basis artifact

src/verifier_anchored_sd/spec_decode/direct_kv_adapter.py
    checked direct 8B-to-4B cache conversion and rotary provenance

src/verifier_anchored_sd/spec_decode/gradient_projector.py
    hard/soft K/V projection in receiver content space

bench/prepare_aime_kv_subspace.py
    materialize frozen AIME rows and fold manifests

bench/fit_aime_kv_gradient_subspace.py
    collect 4B answer-loss gradients for one branch/fold and fit bases

bench/eval_aime_kv_subspace.py
    evaluate the full paired method matrix and emit per-example results

scripts/run_aime_kv_subspace.sh
    reproducible cross-fit and cross-year orchestration
```

Reuse the existing `CacheState`, Full-Head mapper, probability metrics,
bootstrap implementation, atomic JSON writer, SHA utilities, and low-VRAM
loading path. Do not alter or overwrite the completed FineWeb A-E artifacts.

## Artifact layout

Write only small manifests/results to Git:

```text
results/aime_kv_subspace_2026-09-11/
  data_manifest.json
  crossfit/fold_<0-4>/{direct,full_head}.json
  crossfit/final.json
  cross_year/{direct,full_head}.json
  cross_year/final.json

artifacts/aime_kv_subspace_2026-09-11/
  fold_<0-4>/{direct,full_head}_gradient_basis.pt
  cross_year/{direct,full_head}_gradient_basis.pt
```

Large captured KV shards remain local. Basis artifacts may remain local when
they exceed repository limits; their metadata JSON and SHA256 must still be
published.

Every result records dataset revision and row digests, model/tokenizer
contracts, branch, fold membership, mapper SHA where applicable, basis SHA,
rank, beta, seed, dtype, code commit, and hardware.

## Failure handling

Before GPU work, validate all dataset rows, folds, tokenizer contracts, model
geometry, mapper provenance, and disjointness. Each long phase writes atomic
progress JSON and a terminal status file. On CUDA OOM, preserve the failure
artifact and change only model residency/offload settings; do not silently
change model, dtype, rank matrix, examples, or decoding lengths.

## Tests

Unit tests cover:

```text
AIME answer normalization and exact answer-token masking
five-fold balance and train/eval disjointness
direct adapter geometry and rotary checks
FP64 Gram accumulation equivalence to explicit SVD on synthetic gradients
basis orthonormality and deterministic sign canonicalization
hard, soft, orthogonal, PCA, and seeded-random projection behavior
branch-specific artifact rejection
mapper/dataset/model SHA contract enforcement
paired result completeness and method/example uniqueness
```

An integration test uses tiny synthetic cache tensors and a toy differentiable
decoder to prove that answer loss produces nonzero cache gradients and that
the fitted basis changes only the intended K/V coordinates. GPU smoke tests
run one AIME example per branch before launching the full experiment.

## Success criteria

The implementation is complete when both direct and Full-Head branches:

1. produce branch-specific 4B-gradient bases for all five AIME2025 folds;
2. evaluate every held-out problem exactly once per pre-registered method;
3. finish the cross-year AIME2022-2024 to AIME2025 experiment;
4. publish complete finite paired metrics and provenance;
5. preserve negative as well as positive outcomes without performance-gated
   early termination.
