# Student-Readable KV Subspace Design

## Goal

Test the hypothesis that mapped Qwen3-8B KV contains useful verifier information that Qwen3-4B can only consume in a low-dimensional, decoder-readable subspace. The experiment must separate useful transferred directions from incompatible directions without changing the existing 8B->4B mapper.

## Evidence motivating the experiment

For the current audited Qwen3-8B -> Qwen3-4B mapper, mapped history improved verifier top-1 agreement while degrading full-distribution overlap/KL/NLL. This is consistent with a mixed transfer signal:

```text
mapped state = student-readable verifier signal + student-incompatible interference
```

The new experiment must test that interpretation causally.

## Frozen model / mapper contract

Primary pair:

```text
verifier: Qwen/Qwen3-8B
draft:    Qwen/Qwen3-4B
mapper:   existing audited matched-head, content-space, BF16 8B->4B mapper
```

Do not refit or change the mapper for the first subspace experiment.

The verifier remains exact. The 4B decoder is frozen. No LoRA, SFT, RL, quantization, or acceptance residual is allowed in this stage.

## Data split contract

Use four logically disjoint sources:

```text
A: mapper calibration (already frozen by mapper metadata)
B: SUBSPACE_TEXT -- fit subspace statistics only
C: EVAL_TEXT     -- evaluate intervention matrix only
D: E2_TEXT       -- speculative decoding only after a positive intervention result
```

The subspace artifact records token-row digests from B. Evaluation must reject any exact row overlap with mapper calibration A or subspace fit B.

## Causal state

For one prefix x_1:t, define:

```text
p_T      = verifier next-token distribution after x_1:t
C_N      = native 4B KV for x_1:t-1
C_M      = mapped 8B->4B KV for x_1:t-1
q_N      = 4B distribution after native frontier x_t on C_N
q_M      = 4B distribution after native frontier x_t on C_M
Delta_C  = C_M - C_N
```

The newest frontier x_t is always materialized by a native 4B forward for every intervention. Subspace operations apply only to historical KV x_1:t-1.

Keys are analyzed/intervened in RoPE-free content space. Values remain in their native value space. Any output cache is re-rotated with the same draft rotary factors before the frontier forward.

## Subspace family 1: gradient-sensitivity basis (SPD-style baseline)

Use 4B decoder gradients, not verifier gradients.

Primary fitting loss:

```text
L_target = KL(p_T || q_4)
```

For each draft layer l, KV head h, kind z in {K,V}, collect gradients with respect to the 4B historical cache:

```text
g_{l,h,z} = d L_target / d C_{l,h,z}
```

For keys, inverse-rotate gradient rows into content space before accumulation.

Accumulate the per-head covariance:

```text
S_grad = E[g g^T]   in R^{128 x 128}
```

Take eigenvectors in descending eigenvalue order. The rank-r projector is:

```text
P_grad(r) = U_r U_r^T
```

This is the closest analogue to SPD: it identifies directions to which the frozen 4B decoder's verifier-alignment loss is most sensitive.

## Subspace family 2: beneficial-transfer basis (primary proposed method)

Sensitivity alone does not tell whether the teacher delta moves the student in a helpful or harmful direction. Use native 4B gradients and the actual mapped-minus-native teacher delta.

For each layer/head/KV kind:

```text
g = d KL(p_T || q_N) / d C_N
Delta = C_M - C_N
```

Construct the symmetric first-order benefit matrix:

```text
S_benefit = -0.5 * E[g Delta^T + Delta g^T]
```

For a rank-r orthogonal projector P, the first-order predicted improvement from injecting P Delta is:

```text
-E[g^T P Delta] = trace(P S_benefit)
```

Therefore the optimal rank-r projector under this approximation is the eigenspace of the largest positive eigenvalues of S_benefit.

Store all eigenvectors/eigenvalues. At evaluation, never include eigenvectors whose eigenvalue is <= 0. If fewer than r positive directions exist, use the available positive rank and report effective_rank.

This is the main new method: the teacher state is filtered by directions that the student decoder predicts will be beneficial.

## Subspace family 3: activation-PCA control

Use mapped-history activations only, without any gradient or target distribution information.

For each layer/head/KV kind accumulate centered covariance of C_M and take the top-r PCA basis:

```text
P_pca(r)
```

This controls for the possibility that generic low-rank compression alone improves calibration.

## Intervention matrix

All interventions use exactly the same frozen mapper and prefix.

### Baselines

1. `native`:

```text
C = C_N
```

2. `full_mapped`:

```text
C = C_M
```

### A. Hard mapped projection

For basis P in {grad, benefit, pca}:

```text
C = P C_M
```

Implemented per layer/head/KV kind in content/value space.

### B. Soft mapped projection

For beta in {0.0, 0.25, 0.5, 0.75, 1.0}:

```text
C = P C_M + beta (I-P) C_M
```

beta=0 is hard projection; beta=1 is full mapped.

### C. Projected teacher delta

Preserve the native 4B state and inject only the projected mapped-minus-native delta:

```text
C = C_N + alpha P (C_M - C_N)
```

with:

```text
alpha in {0.25, 0.5, 1.0, 1.5}
```

alpha=0 would be native and is not rerun.

This is a mechanistic upper-bound diagnostic because it requires native draft history and therefore does not preserve the prefill-skip systems benefit.

### D. Delta shrinkage toward full mapped

Use the readable delta fully and retain only beta of the orthogonal delta:

```text
C = C_N + P Delta + beta (I-P) Delta
```

with beta in {0.0, 0.25, 0.5, 0.75, 1.0}.

beta=0 is readable-delta only; beta=1 exactly recovers full mapped.

### E. Causal controls

1. `random_subspace`: deterministic rank-matched random orthonormal basis, same intervention formulas as hard projection and projected delta.
2. `grad_orthogonal`: use I-P_grad at the same nominal rank complement and report the actual retained dimension.
3. `benefit_negative`: use eigenvectors of S_benefit with the most negative eigenvalues. This is a signed causal control: if the first-order theory is meaningful, it should be worse than the positive-benefit subspace.

## Rank sweep

Primary ranks:

```text
r in {4, 8, 16, 32, 64}
```

Head dimension is 128. The same nominal ranks are used for K and V, but K/V bases are fitted independently.

For `benefit`, effective rank may be lower if there are fewer positive eigenvalues.

## Statistics accumulation

Do not store per-token gradients.

For each layer/head/kind accumulate on CPU FP64 or FP32 sufficient statistics:

```text
Gradient covariance:      sum(g g^T), count
Benefit cross-statistic:  sum(g Delta^T + Delta g^T), count
Activation PCA:           sum(x), sum(x x^T), count
```

The maximum matrix is 128x128, so all layer/head statistics are small enough to keep in host memory.

Gradient fitting is sequential by prefix. Only the frozen 4B model and mapper must be live; verifier KV/probabilities are loaded from the sequential screen artifact.

## Gradient correctness

For gradient fitting:

1. mapped/native historical cache tensors are detached from model construction;
2. cache tensors are cloned with `requires_grad_(True)`;
3. model parameters remain `requires_grad=False`;
4. after one native-frontier forward, compute target KL and call backward;
5. require nonzero finite gradients for every fitted layer/head/KV kind;
6. clear graph/tensors every prefix.

A 4-prefix gradient smoke must report nonzero finite norms before the full fit.

## Primary evaluation metrics

For every intervention report:

```text
A_target = 1 - TV(p_T, q_method)
Delta_A_target_vs_native
Delta_A_target_vs_full_mapped
KL(p_T || q_method)
target top-1 agreement
next-token NLL under q_method
```

Use document-cluster paired bootstrap for differences on the same evaluation rows.

Primary success criterion:

A subspace method must satisfy both:

```text
95% CI low[ A_target(method) - A_target(full_mapped) ] > 0
95% CI low[ A_target(method) - A_target(native) ] > 0
```

This is intentionally stronger than the old top-1 signal. It requires the intervention to beat both the current mapped state and pure 4B on distributional speculative overlap.

Secondary desirable condition:

```text
KL(p_T || q_method) < KL(p_T || q_native)
```

with paired bootstrap support.

## Method selection for E2

Do not choose the best method on the same prefixes used for final scientific E2.

On C, select one configuration by:

1. highest mean A_target among configurations whose CI beats full_mapped;
2. tie-break by lower target KL;
3. tie-break by lower rank;
4. freeze method/basis/rank/beta/alpha before D.

Only that frozen configuration enters E2 alongside:

```text
native_sd
full_mapped/native-frontier init
subspace-mapped/native-frontier init
subspace accepted-only refresh (only if the intervention supports refresh semantics)
```

Delta methods that require C_N are mechanistic diagnostics and cannot be claimed as prefill-skip systems methods.

## Direct-run hardware profile

Primary hardware is 32GB or 48GB single GPU.

```text
32GB: GPU_MEMORY_GIB=28
48GB: GPU_MEMORY_GIB=44
```

8B and 4B need not be loaded together during basis fitting. Verifier artifacts are captured sequentially first. No automatic precision change or quantization is allowed.

## Debug / STOP conditions

The GPU agent must stop rather than patch around:

1. mapper SHA/revision/tokenizer mismatch;
2. overlap between subspace-fit rows and evaluation rows;
3. missing verifier probability shard;
4. cache/probability/token-row binding mismatch;
5. K/V basis shape not [draft_layers, kv_heads, head_dim, max_rank];
6. non-orthonormal basis beyond tolerance 1e-4;
7. NaN/Inf gradient, covariance, eigenvalue, intervention cache, or probability;
8. zero gradient for a layer/head/kind during the smoke;
9. benefit basis includes non-positive eigenvalues as positive-readable directions;
10. hard/soft endpoint identities fail (`beta=1` != full_mapped or delta-shrink beta=1 != full_mapped);
11. random control does not use exactly the same rank/effective shape;
12. frontier token is not materialized natively after intervention;
13. OOM triggers an automatic model/dtype/rank change;
14. method selection reads E2 data.

## Falsification logic

### Strong support

A gradient or benefit subspace beats both native and full mapped in target overlap, with paired CI lower bounds > 0, and matched-rank random/PCA controls do not.

Interpretation:

```text
8B state contains useful information, but transfer must be filtered through student-readable directions.
```

### Compression-only result

PCA/random low-rank controls match the gradient methods.

Interpretation:

```text
the gain is generic regularization/compression, not decoder-readable transfer.
```

### No support

No intervention beats native 4B target overlap.

Interpretation:

```text
the top-1 improvement was too local to support a useful subspace-transfer mechanism.
```

### Signed-mechanism support

Positive-benefit eigenspace improves alignment while negative-benefit eigenspace degrades it.

Interpretation:

```text
the first-order student-gradient/teacher-delta geometry identifies causal beneficial versus harmful transfer directions.
```
