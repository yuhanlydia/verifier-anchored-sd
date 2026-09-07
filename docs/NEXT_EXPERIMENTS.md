# Next experiments: pair quality before verifier refresh

## Evidence boundary

The completed Qwen3-4B verifier -> Qwen3-1.7B run is a negative pair-quality
result. Its new distribution screen reached only `A_transfer=0.601`; the old
reported expected-MAL retention of 0.717 used an invalid autoregressive estimator
and is withdrawn. The old realized refresh delta was negative, but it cannot
distinguish a bad model pair from a bad refresh principle, so 4B -> 1.7B is now a
stress control.

The next primary candidate is Qwen3-8B verifier -> Qwen3-4B draft. Both have 36
layers, 8 KV heads, head dimension 128, the same tokenizer vocabulary, and the same
RoPE base. These structural matches justify screening; they do not guarantee
transfer quality.

## Frozen experiment order

1. **Pair selection.** Fit the same matched-head content-space ridge mapper for
   8B -> 4B and 4B -> 1.7B from disjoint calibration data.
2. **Near-lossless transfer.** Compare native draft state against mapped verifier
   history followed by one native draft frontier token.
3. **Task confirmation.** Run mapped/native HellaSwag only for a distribution-screen
   pass. It cannot rescue a failed distribution screen.
4. **Speculative compatibility.** Require mapped/native expected-MAL retention of
   at least 0.90 for confirmatory refresh inference. Retention from 0.85 through
   0.90 is exploratory; below 0.85 rejects the pair for refresh work.
5. **Verifier refresh.** Compare accepted-only historical refresh with mapped
   init-only while keeping the newest causal frontier native in both methods.

The full design and implementation contract are in
`docs/superpowers/specs/2026-09-07-pair-screen-native-frontier-design.md`.

## E0: sequential exact-weight mapper calibration

Each directional pair uses:

- 128 frozen FineWeb-Edu calibration windows of 1,024 tokens;
- stride 4, giving 32,768 token observations;
- matched KV heads, content-space keys, `k=8`, ridge lambda 0.01;
- source-layer selection by head-averaged K/V R² on 32 sequences;
- exact BF16 model weights, with verifier and draft loaded in separate processes;
- immutable manifests containing exact Hub revisions, tokenizer hash, token-row
  digests, geometry, and capture parameters.

Calibration and evaluation token windows must have no exact or partial row overlap.
Existing shards are accepted only when their metadata and exact shard set match the
manifest.

## E1: distribution transfer screen

The primary screen uses 128 disjoint 1,024-token windows. It records the source
document for every window and resamples whole documents in the bootstrap. For each prefix it
computes:

```text
native:  draft native prefill of tokens 1..t
mapped:  verifier KV for 1..t-1 -> mapper -> draft KV,
         then native draft forward of token t
```

The newest token is native in both paths, so logits and the cache state that will
be used next have the same causal provenance. Report mean, median, fifth percentile,
and minimum `A_transfer`, native-to-mapped KL, top-1 agreement, next-token NLL
delta, attention-output cosine diagnostics, model revisions, hashes, hardware,
elapsed time, and requested/completed rows.

The gate is preregistered:

```text
A_transfer = 1 - TV(q_native, q_mapped)
pass:         document-cluster bootstrap 95% CI lower bound > 0.95
fail:         document-cluster bootstrap 95% CI upper bound <= 0.95
inconclusive: interval crosses 0.95; expand to 512 prefixes
```

If a pair passes, repeat a context-stability diagnostic on 32 prefixes at 2,048 and
8,192 tokens. Record OOM per length without discarding completed lengths.

## E2: native-frontier speculative ablation

The method matrix is:

| Method | Initialization | Refresh |
|---|---|---|
| `native_sd` | native draft prefill | none |
| `legacy_mapped_init_only` | old full-prompt map/logit mismatch | none |
| `mapped_init_only` | mapped history + native frontier | none |
| `legacy_full_refresh` | old full-prompt map/logit mismatch | all verified/frontier KV |
| `mapped_accepted_only` | mapped history + native frontier | accepted history only |

The primary state is

```text
C_draft(t) = [M(C_verifier(1:t-1)); KV_draft_native(t)]
```

The mapper gate precedes the refresh gate:

```text
mapped/native expected-MAL retention >= 0.90: confirmatory
0.85 <= retention < 0.90: exploratory
retention < 0.85: reject
```

For confirmatory pairs, compute the paired bootstrap interval for

```text
Delta = E[MAL](mapped_accepted_only) - E[MAL](mapped_init_only)
```

- CI lower bound > 0 supports historical verifier anchoring with a native frontier.
- CI upper bound < 0 stops the verifier-anchored refresh paper direction.
- An interval crossing zero is inconclusive and requires more held-out prompts.

Legacy full refresh remains only to explain the prior negative result. It is not
the primary method.

## Commands

```bash
export CALIBRATION_TEXT=/path/to/frozen_calibration.jsonl
export EVAL_TEXT=/path/to/disjoint_frozen_evaluation.jsonl
bash scripts/run_pair_screen.sh
```

On the 100GB reference host, the runner writes hashes to
`artifact_inventory.json` and prunes completed cache shards after each candidate.
Set `PRUNE_COMPLETED_SHARDS=0` only when the host has enough disk to retain them.

For a passing result:

```bash
export SCREEN_RESULT=results/pair_screen_2026-09-07/qwen3_8b_to_4b.json
export MAPPER=artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
export EVAL_TEXT=/path/to/e2_prompts.jsonl
bash scripts/run_native_frontier_e2.sh
```

HellaSwag confirmation uses `bench/eval_mapped_hellaswag.py`. It requires a passing
screen JSON unless `--allow-failed-screen` is supplied for an explicitly diagnostic
run. Floor-normalized mapped/native retention must be at least 0.95 before making a
task-quality claim.

Do not train an acceptance residual, run refresh as a confirmatory experiment, or
advance the paper claim until the preceding gates pass.
