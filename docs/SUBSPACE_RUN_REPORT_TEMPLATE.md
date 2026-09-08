# Student-Readable KV Subspace Run Report

## Reproducibility

```text
branch:
commit:
GPU:
VRAM:
host RAM:
CUDA:
PyTorch:
Transformers:
```

```text
mapper path:
mapper SHA256:
mapper metadata SHA256:
subspace basis path:
subspace basis SHA256:
SUBSPACE_TEXT SHA256:
EVAL_TEXT SHA256:
E2_TEXT SHA256:          # only if block acceptance was run
```

## Basis-fit B

```text
requested prefixes:
completed prefixes:
prefix tokens:
objective: KL(p_8B || q_4B)
max rank:
native gradient norm min/max:
mapped gradient norm min/max:
positive-benefit effective rank min/max:
negative-benefit effective rank min/max:
mean native target KL:
mean mapped target KL:
OOM/fallback: none | incomplete (describe; never silently change config)
```

## Intervention screen C

### Baselines

| Method | A_target | KL(target||method) | target top-1 | next-token NLL |
|---|---:|---:|---:|---:|
| native 4B | | | | |
| full mapped 8B->4B | | | | |

### Best methods by family

| Family | mode | rank | beta/alpha | A_target | Δ vs native CI95 | Δ vs full mapped CI95 | KL | deployment-valid? |
|---|---|---:|---:|---:|---|---|---:|---|
| gradient sensitivity | | | | | | | | |
| positive benefit | | | | | | | | |
| negative benefit control | | | | | | | | no/control |
| activation PCA | | | | | | | | |
| random | | | | | | | | control |
| gradient orthogonal | | | | | | | | control |
| projected-delta upper bound | | | | | | | | no |

### Frozen deployment winner

```text
decision.status:
winner method:
family:
mode:
rank:
beta:
A_target:
KL:
CI95 vs native:
CI95 vs full mapped:
```

Interpretation must be one of:

```text
student-readable mechanism support
signed benefit mechanism support
compression-only / PCA-random match
native-anchor-only upper bound
no support
```

## Block-level speculative acceptance D

Run only if C produced `decision.status=go_e2`.

| Method | realized MAL | acceptance rate | verifier blocks | bonus rate | tok/s | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| native_sd | | | | | | |
| full_mapped_init_only | | | | | | |
| full_mapped_accepted_only | | | | | | |
| subspace_mapped_init_only | | | | | | |
| subspace_mapped_accepted_only | | | | | | |

### Paired block-level tests

```text
subspace_init - full_mapped_init realized MAL:
  mean difference:
  CI95:
  status: support | stop | inconclusive

subspace_init - native realized MAL:
  mean difference:
  CI95:

subspace_init - full_mapped_init acceptance rate:
  mean difference:
  CI95:

subspace_refresh - subspace_init realized MAL:
  mean difference:
  CI95:
  refresh status: support | stop | inconclusive

subspace_refresh - subspace_init throughput:
  mean difference:
  CI95:
```

## Final decision

Choose exactly one:

```text
A. GO — readable subspace improves both one-step target overlap and block acceptance.
B. MECHANISM ONLY — subspace works only with native-base delta; no prefill-skip win.
C. COMPRESSION ONLY — PCA/random matches gradient/benefit methods.
D. ONE-STEP ONLY — one-step winner does not improve block acceptance.
E. STOP — no useful subspace transfer.
F. INCOMPLETE — OOM/artifact/debug contract prevented a scientific result.
```

Do not report GPU integration as successful if any STOP/debug contract was violated.
