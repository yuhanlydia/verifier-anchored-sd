# Next experiments: verifier-target alignment before refresh

## Scientific correction

The 2026-09-07 pair screen established that Qwen3-8B -> Qwen3-4B reconstructs the
native draft distribution much better than 4B -> 1.7B, but neither is near-lossless:

```text
8B -> 4B   A_transfer = 0.800360
4B -> 1.7B A_transfer = 0.600986
```

That remains valid **native-fidelity evidence**. It is not the correct speculative-
decoding gate.

Speculative decoding depends on verifier/proposal overlap. The next primary statistic
is therefore:

```text
p_T      = verifier next-token distribution
q_native = native draft distribution
q_mapped = mapped-verifier-history + native-draft-frontier distribution

A_target_native = 1 - TV(p_T, q_native)
A_target_mapped = 1 - TV(p_T, q_mapped)
Delta_target = A_target_mapped - A_target_native
```

Use a document-cluster paired bootstrap for `Delta_target`:

```text
95% CI low  > 0 -> support      -> go_sd
95% CI high < 0 -> harm         -> stop_pair
CI crosses 0    -> inconclusive -> expand held-out prefixes
```

`A_transfer > 0.95` remains a diagnostic for near-lossless native reconstruction. It
cannot veto speculative decoding when verifier-target alignment is significantly
improved.

Full design:

```text
docs/superpowers/specs/2026-09-07-target-alignment-gate-design.md
```

Direct testing handoff:

```text
docs/NEXT_STAGE_TARGET_ALIGNMENT.md
```

## Stage A — existing Qwen3-8B -> Qwen3-4B mapper

Do **not** refit the mapper. Reuse:

```text
artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
```

Re-capture the held-out verifier prefixes into a **fresh artifact root** because the
old screen did not store verifier next-token distributions.

Run on 32GB:

```bash
export EVAL_TEXT=/path/to/frozen_pair_selection.jsonl
export E2_TEXT=/path/to/third_disjoint_e2.jsonl   # optional until go_sd
export GPU_MEMORY_GIB=28
bash scripts/run_target_alignment_next.sh
```

Run on 48GB:

```bash
export EVAL_TEXT=/path/to/frozen_pair_selection.jsonl
export E2_TEXT=/path/to/third_disjoint_e2.jsonl
export GPU_MEMORY_GIB=44
bash scripts/run_target_alignment_next.sh
```

The script runs a non-scientific 4-prefix smoke, then 128 x 1,024 scientific
prefixes with 10,000 document-cluster bootstrap samples. If the gate is inconclusive,
it can expand to 512 prefixes without changing the mapper.

## Stage A decision

### `go_sd`

Run native-frontier E2 on a third held-out prompt source:

```text
native_sd
mapped_init_only
mapped_accepted_only
```

The causal state is:

```text
C_draft(t) = [M(C_verifier(1:t-1)); KV_draft_native(t)]
```

The newest frontier stays native. Legacy full-refresh remains diagnostic only.

### `stop_pair`

Do not tune refresh, HellaSwag, k/lambda, or an acceptance residual to rescue 8B->4B.
Proceed to Stage B.

### `expand`

Increase only the held-out evaluation clusters/prefixes. Do not change the mapper.

## Stage B — Qwen3-32B -> Qwen3-14B

The prior literature's strong Qwen3 result is not enough for this project because SD
needs the reverse large-verifier -> small-draft direction. Screen 32B -> 14B directly.

Fixed configuration:

```text
exact BF16 weights
sequential model loading
128 x 1,024 calibration windows
stride = 4
selection sequences = 32
matched-head content-space ridge
k = 8
lambda = 0.01
selection ridge = 1e-6
128 x 1,024 held-out prefixes
FP32 verifier probability artifacts
10,000 document-cluster bootstrap samples
```

48GB GPU:

```bash
export CALIBRATION_TEXT=/path/to/frozen_calibration.jsonl
export EVAL_TEXT=/path/to/disjoint_pair_selection.jsonl
export GPU_MEMORY_GIB=44
bash scripts/run_32b_to_14b_pair_screen.sh
```

32GB GPU:

```bash
export CALIBRATION_TEXT=/path/to/frozen_calibration.jsonl
export EVAL_TEXT=/path/to/disjoint_pair_selection.jsonl
export GPU_MEMORY_GIB=28
bash scripts/run_32b_to_14b_pair_screen.sh
```

Exact 32B BF16 will require CPU offload on these single-GPU profiles. Recommend at
least 96 GiB host RAM; 128 GiB is preferred. OOM must remain an incomplete result;
do not silently quantize.

## E2 and task confirmation eligibility

Both `scripts/run_native_frontier_e2.sh` and HellaSwag confirmation now require:

```text
target_alignment.gate.status == support
decision.status == go_sd
```

Native-fidelity failure alone does not block them.

The current runtime's conditional sampled-proposal expected-accepted-length estimator
is different from the **withdrawn** overlap-product expected-MAL values stored in the
2026-09-03 E2 artifact. Never reuse or compare the withdrawn old formula/numbers as
if they were the current estimator.

## Stop rules

- Target alignment `harm`: stop that pair.
- Target alignment `support`: allow native-frontier E2.
- Target alignment `inconclusive`: expand held-out clusters only.
- Incomplete/OOM: debug execution; do not interpret partial metrics.
- No acceptance residual training until a pair receives `go_sd` and the native-
  frontier E2 establishes a positive refresh phenomenon.

The exact debug STOP contract, artifact requirements, and remote-agent reporting
format are in `docs/NEXT_STAGE_TARGET_ALIGNMENT.md`.
