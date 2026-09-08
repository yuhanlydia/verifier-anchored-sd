# Next Stage — Student-Readable KV Subspace

## What this experiment asks

The current Qwen3-8B -> Qwen3-4B result is mixed:

- mapped 4B agrees with 8B top-1 more often than pure 4B;
- but full-distribution overlap/KL/NLL are worse.

The experiment tests whether 8B KV contains useful information that 4B can only
consume in a low-dimensional decoder-readable subspace.

The scientific object is historical draft KV only. The newest causal frontier token
is always processed by 4B natively.

## Frozen pair

```text
verifier: Qwen/Qwen3-8B
draft:    Qwen/Qwen3-4B
mapper:   existing audited matched-head content-space BF16 8B->4B mapper
```

Do not refit the mapper in this experiment.

## Three basis families

### 1. SPD-style student sensitivity

Use frozen 4B gradients:

```text
L = KL(p_8B || q_4B)
g = dL / d C_4B
S_grad = E[g g^T]
```

Top eigenvectors define the 4B-sensitive KV subspace.

### 2. Signed beneficial-transfer basis — primary method

At native 4B history:

```text
g = d KL(p_8B || q_native) / d C_native
Delta = C_mapped - C_native
S_benefit = -0.5 E[g Delta^T + Delta g^T]
```

For an orthogonal projector P:

```text
-E[g^T P Delta] = trace(P S_benefit)
```

Positive eigenvectors are predicted-helpful mapped directions. Negative eigenvectors
are a signed harmful control.

### 3. Activation PCA

No target gradients: top variance directions of mapped KV. This tests whether any
gain is generic low-rank regularization rather than student-readable transfer.

K/V are fitted independently for every 4B layer and KV head. Key bases are defined
in inverse-RoPE content space.

## Intervention families

For ranks:

```text
4, 8, 16, 32, 64
```

run:

### Mapped-only / deployment-valid

```text
hard: C = P C_mapped
soft: C = P C_mapped + beta (I-P) C_mapped
beta = 0, .25, .5, .75
```

for gradient and positive-benefit bases.

Controls:

```text
PCA hard
random hard
gradient orthogonal I-P
negative-benefit hard
```

### Native-base delta / mechanism-only upper bounds

```text
projected delta:
C = C_native + alpha P(C_mapped-C_native)
alpha = .25, .5, 1.0, 1.5

orthogonal-delta shrink:
C = C_native + P Delta + beta(I-P)Delta
beta = 0, .25, .5, .75
```

These require native 4B history, so they are **not** deployment/prefill-skip methods.
Never select them as the final systems method.

## Data split — mandatory

Use four logically disjoint sources:

```text
A mapper calibration       (already frozen)
B SUBSPACE_TEXT             basis fit only
C EVAL_TEXT                 intervention matrix only
D E2_TEXT                   speculative acceptance only after winner is frozen
```

Exact token-row overlap is rejected by code.

## 32GB run

```bash
export SUBSPACE_TEXT=/path/to/frozen_subspace_fit.jsonl
export EVAL_TEXT=/path/to/disjoint_intervention_eval.jsonl
export GPU_MEMORY_GIB=28
export METHOD_BATCH_SIZE=8
bash scripts/run_student_readable_subspace.sh
```

## 48GB run

```bash
export SUBSPACE_TEXT=/path/to/frozen_subspace_fit.jsonl
export EVAL_TEXT=/path/to/disjoint_intervention_eval.jsonl
export GPU_MEMORY_GIB=44
export METHOD_BATCH_SIZE=16
bash scripts/run_student_readable_subspace.sh
```

Optional only after a deployment winner exists:

```bash
export E2_TEXT=/path/to/third_disjoint_e2.jsonl
```

## Default scientific settings

### Basis fit B

```text
64 prefixes x 512 tokens
2 backward passes per prefix:
  native history -> benefit matrix
  mapped history -> gradient-sensitivity matrix
max rank = 64
objective = KL(p_8B || q_4B)
statistics = CPU FP64 sufficient statistics
no per-token gradients saved
```

### Evaluation C

```text
128 prefixes x 1024 tokens
ranks = 4,8,16,32,64
10,000 document-cluster paired bootstrap samples
method batch = 8 (32GB) / 16 (48GB)
```

The same 4B model and mapper are used across every method. Only historical KV
intervention changes.

## Primary metrics

For each method:

```text
A_target = 1 - TV(p_8B, q_method)
KL(p_8B || q_method)
target top-1 agreement
next-token NLL
```

The primary success gate is deliberately strict. A deployment-valid method must beat
both baselines:

```text
CI95_low[A_target(method) - A_target(native)] > 0
AND
CI95_low[A_target(method) - A_target(full_mapped)] > 0
```

Only then does the result JSON contain:

```text
decision.status = go_e2
winner != null
```

## How to interpret outcomes

### A. benefit+ / grad beats native + full mapped; random/PCA do not

Strong support for student-readable transfer:

```text
8B state contains useful information, but 4B can only consume selected directions.
```

### B. positive-benefit improves and negative-benefit degrades

Strongest signed-mechanism evidence. The local first-order geometry predicts causal
helpful vs harmful teacher directions.

### C. random/PCA match gradient/benefit

This is a compression/regularization effect, not a decoder-readable-subspace result.
Do not claim the gradient mechanism.

### D. delta upper bound wins but mapped-only methods do not

Interesting representational phenomenon, but it does not solve draft-prefill skip.
It means useful teacher information exists only when anchored on native 4B state.
Do not call it a systems win.

### E. no method beats native 4B target overlap

Stop the subspace-transfer direction. The previous top-1 improvement was too local to
support useful distribution-level speculative transfer.

## Debug / STOP contract

Stop instead of patching around any of these:

1. mapper checkpoint SHA differs from mapper metadata;
2. 8B/4B revisions or tokenizer hash differ from mapper metadata;
3. old cache-only screen artifact is used instead of schema-v3 KV+probability artifact;
4. cache/probability/token row sequence ID, token digest, next-token ID or revision mismatch;
5. B and C exact token-row digests overlap;
6. target probability has NaN/Inf, negative mass, wrong vocab length, or sum != 1 +/- 1e-5;
7. any draft layer/KV-kind/head has zero or non-finite gradient during the 4-prefix smoke;
8. basis is not orthonormal within 1e-4;
9. positive-benefit family includes a non-positive eigenvalue as a usable direction;
10. beta=1 endpoint differs from full mapped or alpha=0 differs from native;
11. mapped path fails to materialize the final frontier token with native 4B forward;
12. OOM causes an automatic model, dtype, rank, method-batch, or precision change;
13. a delta/native-base method is selected as deployment winner;
14. E2 data is read before the intervention winner is frozen.

OOM must be recorded as incomplete. Change `METHOD_BATCH_SIZE` only by an explicit new
run/configuration, not silently within a scientific result.

## Expected result files

```text
artifacts/student_readable_subspace_2026-09-08/student_readable_basis.pt
artifacts/student_readable_subspace_2026-09-08/student_readable_basis.pt.json
results/student_readable_subspace_2026-09-08/qwen3_8b_to_4b_subspace_interventions.json
results/student_readable_subspace_2026-09-08/artifact_inventory.json
```

The intervention result contains every per-prefix method row, summaries, paired
bootstrap intervals, candidate eligibility, and the frozen deployment winner if one
exists.

## GPU agent reporting requirement

Commit only small result/provenance files, not large KV shards. Report:

```text
GPU + VRAM
host RAM
branch + commit
mapper SHA256
basis SHA256
B/C input SHA256
fit/eval completed rows
min/max gradient norm
positive/negative benefit effective-rank range
native A_target / KL
full_mapped A_target / KL
best grad, benefit+, PCA, random, benefit-negative results
winner + rank/beta if any
all paired CIs vs native and full_mapped for winner
OOM/fallback: must say none, or mark run incomplete
```
