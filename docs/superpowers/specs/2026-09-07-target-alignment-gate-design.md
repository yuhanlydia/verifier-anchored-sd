# Target-Alignment Gate Design

## Problem

The current pair screen treats native-draft reconstruction as the scientific gate:

```text
A_transfer = 1 - TV(q_native_draft, q_mapped_draft)
```

That quantity answers whether translated verifier KV reproduces the draft model's original state. It does **not** answer the speculative-decoding question. Exact speculative decoding accepts proposals according to overlap between the verifier distribution and the proposal distribution, so the relevant comparison is whether mapped draft state moves the proposal distribution toward or away from the verifier.

A pair may therefore fail native reconstruction while still being useful for speculative decoding:

```text
q_mapped != q_native
but
TV(p_target, q_mapped) < TV(p_target, q_native)
```

The next experiment must separate **native fidelity** from **target alignment**.

## Scientific question

For the same held-out prefix and the same native causal frontier token, compare:

```text
p_T       = verifier next-token distribution
q_native  = draft distribution from native draft history
q_mapped  = draft distribution from mapped verifier history + native draft frontier
```

Define:

```text
A_native_target = 1 - TV(p_T, q_native)
A_mapped_target = 1 - TV(p_T, q_mapped)
Delta_target    = A_mapped_target - A_native_target
```

`Delta_target` is the primary pair-selection statistic.

The old native-fidelity statistic remains diagnostic:

```text
A_transfer = 1 - TV(q_native, q_mapped)
```

It must no longer block speculative experiments by itself.

## Causal-state contract

Every distribution comparison uses the same token prefix `x_1:t` and predicts `x_{t+1}`.

Native path:

```text
draft native prefill x_1:t -> q_native
```

Mapped path:

```text
verifier KV(x_1:t-1) -> mapper -> mapped draft history
draft native forward x_t on mapped history -> q_mapped
```

Verifier path:

```text
verifier native prefill x_1:t -> p_T
```

The latest token `x_t` is native in the draft for both native and mapped paths. This avoids the previous frontier-provenance mismatch.

## Sequential artifact design

The experiment must work for pairs that cannot fit simultaneously, including Qwen3-32B -> Qwen3-14B. Models are therefore loaded sequentially.

During verifier screen capture, save two artifacts per held-out row:

1. verifier KV cache for `x_1:t`;
2. verifier next-token probability vector `p_T(. | x_1:t)` in FP32.

The probability artifact is stored separately from the cache shard and carries the same sequence ID, token digest, next-token ID, model revision, vocabulary size, and dtype contract.

During draft evaluation, load only the draft model, mapper, verifier cache shard, and verifier probability shard. Compute `q_native`, `q_mapped`, native-fidelity metrics, and target-alignment metrics in one pass.

## Metrics

Per row record:

```text
A_transfer
KL(q_native || q_mapped)
native/mapped top-1 agreement
A_native_target
A_mapped_target
Delta_target
KL(p_T || q_native)
KL(p_T || q_mapped)
target/native top-1 agreement
target/mapped top-1 agreement
next-token NLL under target/native/mapped
attention-output cosine(native draft, mapped draft)
```

Aggregate by source document, not by window, using document-cluster bootstrap.

## Target-alignment gate

Use 10,000 document-cluster bootstrap samples for the mean `Delta_target`.

```text
support:      95% CI lower bound > 0
harm:         95% CI upper bound < 0
inconclusive: CI crosses 0
```

Decision logic:

```text
support:
    mapped draft is significantly closer to verifier;
    allow native-frontier speculative E2 even if A_transfer < 0.95.

harm:
    mapped draft is significantly farther from verifier;
    reject the pair for verifier-anchored SD with this translator.

inconclusive:
    expand held-out evaluation from 128 to 512 prefixes before any SD experiment.
```

`A_transfer > 0.95` is retained only as a near-lossless reconstruction diagnostic and as evidence for a prefill-reuse-only story. It is not the speculative-decoding gate.

## Experiment order

### Stage A — Re-evaluate existing Qwen3-8B -> Qwen3-4B mapper

Use the existing audited mapper checkpoint and frozen held-out data. Re-capture the verifier screen artifacts because old screen shards did not store verifier probabilities.

Required settings:

```text
mapper: artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
held-out prefixes: 128 x 1,024 tokens
bootstrap: 10,000 document clusters
exact weights: BF16
verifier probability storage: FP32
native frontier: enabled
```

Do not refit the mapper before this test. The purpose is to isolate the decision-metric error from translator changes.

### Stage B — Qwen3-32B -> Qwen3-14B directional screen

Run only if Stage A is `harm`, or after Stage A finishes if additional evidence is desired.

Use sequential exact-weight loading. No quantization is allowed in the scientific screen.

Initial scientific configuration:

```text
calibration: 128 x 1,024 FineWeb-Edu windows
stride: 4 -> 32,768 observations
selection sequences: 32
matched-head content-space ridge
k = 8
lambda = 0.01
held-out screen: 128 x 1,024 prefixes
bootstrap: 10,000 document clusters
```

Hardware profiles:

```text
48GB GPU: GPU_MEMORY_GIB=44
32GB GPU: GPU_MEMORY_GIB=28
```

Qwen3-32B will require CPU offload even on 48GB in BF16. The run should therefore have at least 96GB system RAM; 128GB is preferred. If exact-weight loading cannot complete, record the run as incomplete. Do not silently switch to quantization.

## Native-frontier E2 eligibility

Only a target-alignment `support` decision permits confirmatory native-frontier SD.

The E2 method matrix is:

```text
native_sd
mapped_init_only          # mapped history + native frontier
mapped_accepted_only      # mapped history + native frontier; refresh accepted historical KV only
```

Legacy full-refresh methods remain diagnostic only.

Primary E2 metrics are realized accepted length, acceptance rate, verifier calls per emitted token, end-to-end tokens/s, and paired bootstrap differences on the same prompts. The previously withdrawn overlap-product expected-MAL estimator must not be used for scientific gates.

## Debug contract

The direct-testing agent must stop rather than patch around any of the following failures:

1. **Token mismatch:** cache shard, target-probability shard, and frozen token row do not share the same sequence ID, token digest, and next-token ID.
2. **Revision mismatch:** loaded verifier/draft revision differs from mapper or artifact metadata.
3. **Tokenizer mismatch:** verifier and draft tokenizer hashes differ.
4. **Vocabulary mismatch:** verifier probability vector length differs from draft output vocabulary.
5. **Probability invalidity:** NaN/Inf, negative values, or row sum outside `1 +/- 1e-5`.
6. **Frontier mismatch:** mapped path predicts after a different final token than native/target paths.
7. **Artifact contamination:** old screen shards without target-probability artifacts are mixed into the new run.
8. **Data leakage:** calibration and evaluation token-row digests overlap.
9. **OOM:** write an incomplete result with phase and completed-row count; do not change model precision or quantize automatically.
10. **Metric misuse:** do not use the withdrawn expected-MAL gate.

Before a full GPU run, execute the complete CPU test suite and a 4-prefix integration smoke. A smoke is not scientific evidence and must be stored under a separate `smoke/` artifact root.
