# Pair Screening and Native-Frontier Refresh Design

## Purpose

The current Qwen3-4B verifier to Qwen3-1.7B draft result cannot isolate the
scientific value of verifier refresh because its new distribution screen gives
only `A_transfer=0.601`. The earlier 0.7166 expected-acceptance retention estimate
used an invalid autoregressive overlap product and is withdrawn. The next experiment
must first find a verifier-to-draft pair whose mapped cache preserves the draft's own behavior,
then test refresh without replacing the causal frontier that produced the next
draft logits.

This design replaces the old Qwen3-4B to Qwen3-1.7B 24GB confirmatory run as the
next experiment. It keeps that pair as a measured negative-control candidate and
adds Qwen3-8B to Qwen3-4B as the primary candidate.

## Scientific questions

The experiment answers three questions in order:

1. **Pair transfer:** after translating verifier history into draft KV, does the
   draft retain its native next-token distribution?
2. **Speculative compatibility:** for a pair that passes transfer screening, does
   mapped initialization preserve at least 90% of Native SD conditional expected
   accepted length?
3. **Refresh structure:** with a sufficiently faithful mapper, does mapping only
   accepted historical tokens while preserving a native causal frontier improve
   acceptance over mapped initialization alone?

The three quantities must remain separate. Draft-to-draft transfer fidelity does
not use the verifier probability distribution. Speculative acceptance retention
compares mapped and native draft proposals against the verifier. Refresh gain
compares two mapped-cache policies.

## Candidate pairs

The first screen uses:

| Name | Verifier | Draft | Role |
| --- | --- | --- | --- |
| `qwen3_8b_to_4b` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-4B` | primary candidate |
| `qwen3_4b_to_1p7b` | `Qwen/Qwen3-4B` | `Qwen/Qwen3-1.7B` | measured comparison |

Every model revision and tokenizer hash is frozen in artifact manifests. Both
models in a pair must have an identical tokenizer contract, the same KV-head count,
and the same head dimension. The mapper is directional; reverse directions require
separate calibration and are outside this run.

## Data isolation

Calibration and evaluation use disjoint frozen JSONL files. The initial protocol
uses 128 calibration windows and 128 held-out evaluation windows, each 1,024 tokens.
All pairs consume the same token IDs because the candidate Qwen3 models share a
tokenizer contract. Manifests record the input file digest, ordered token-window
digest, model revision, tokenizer digest, sequence length, stride, and count.

The existing `data/fineweb_edu_heldout_offset4096.jsonl` may be used for screening
only if it is not used for mapper fitting. Calibration may stream FineWeb-Edu, but
the selected token windows must be frozen before either model is loaded.

## Resource-aware sequential capture

Qwen3-8B and Qwen3-4B are not required to coexist on GPU or CPU. Sequential capture
has four resumable phases:

1. Tokenize and freeze the ordered calibration windows.
2. Load only the verifier, capture sampled KV plus exact RoPE factors, save one
   source shard per sequence, then unload it and clear CUDA memory.
3. Load only the draft, capture matching native KV shards for the same token IDs,
   save them, then unload it and clear CUDA memory.
4. Fit the matched-head centered-ridge mapper from the paired on-disk shards after
   both LLMs are unloaded.

Each phase validates its manifest before reusing existing shards. Partial runs
resume only missing shards. A mismatched manifest is a hard error and requires a
new artifact directory. On a 16GB GPU, an individual model may use exact-BF16 CPU
offload with an explicit GPU memory cap; quantization is not allowed in screening.

The initial mapper uses matched-head support, content-space key mapping, `k=8`,
ridge lambda `0.01`, 128 x 1,024-token calibration sequences, stride 4, and R² layer
selection on 32 sequences. Full-head and learned residual mappers are excluded from
the first screen so mapper support is held constant across pairs.

## Pair-screen evaluation

Evaluation is also sequential. First load the verifier alone and capture held-out
prefix KV shards. Then unload it, load the mapper and draft, and evaluate every
held-out prefix in two modes:

- **Native:** run the complete prefix through the draft and retain the probability
  distribution produced at the final prefix token.
- **Mapped native-frontier:** map verifier KV for all prefix tokens except the last,
  then run the last prefix token natively through the draft. The resulting logits
  and final native KV describe the same state.

The primary per-prefix transfer score is

`A_transfer = 1 - 0.5 * sum_v |q_native(v) - q_mapped(v)|`.

The screen records:

- mean `A_transfer` with a source-document-cluster bootstrap 95% confidence interval;
- median, fifth percentile, and minimum `A_transfer`;
- `KL(q_native || q_mapped)` with probabilities clamped before logarithms;
- native-versus-mapped top-1 agreement;
- change in negative log-likelihood for the real next token when available;
- per-layer cosine similarity between the draft attention outputs at the native
  frontier, as a diagnostic rather than a gate;
- peak GPU memory, elapsed time, model revisions, and all artifact hashes.

The primary screen uses 128 disjoint 1,024-token windows and records their source
document IDs. Bootstrap resampling treats the document as the independent unit. If it passes, the
same implementation runs a context-stability diagnostic on 32 prefixes at 2,048
and 8,192 tokens. OOM is recorded per length without discarding completed rows.

### Pair-screen decision

- **Pass:** document-cluster bootstrap 95% CI lower bound for mean `A_transfer` is greater than 0.95.
- **Fail:** bootstrap 95% CI upper bound is at most 0.95.
- **Inconclusive:** the interval crosses 0.95; expand the held-out screen to 512
  prefixes before making a pair decision.

HellaSwag mapped-versus-native retention is a confirmatory quality test only for a
pair that passes the distribution screen. It reports native accuracy, mapped
accuracy, and mapped/native normalized retention. A retention of at least 0.95 is
required before making a task-quality claim, but HellaSwag cannot override a failed
`A_transfer` screen.

## Initialization and refresh policies

The runtime separates initialization from refresh. Initialization becomes an
explicit mode:

- `native`: ordinary draft prefill.
- `legacy_mapped`: reproduce the current implementation by mapping the complete
  prompt while obtaining logits from a native forward of the last prompt token;
  this mode is retained only to connect new results to the recorded 16GB result.
- `mapped_native_frontier`: map prompt history `1..n-1`, run prompt token `n`
  natively, and keep both the returned logits and native KV.

The Boolean `refresh` switch becomes an explicit policy:

- `none`: native or mapped initialization, then keep accepted draft KV natively.
- `full`: reproduce the current behavior by mapping accepted tokens and replacing a
  correction or target-bonus frontier with mapped verifier KV when it materializes.
- `accepted_only`: map accepted proposal tokens after verification, but never
  replace the newest correction or bonus frontier generated by the draft.

Primary new methods use `mapped_native_frontier`. The legacy modes remain explicit
state-inconsistency ablations and must be labeled `legacy` in outputs; they cannot
support the structural claim even if their metric improves.

At every proposal boundary, the cached KV for the last token and the logits used to
sample the next token must come from the same draft forward. Debug validation may
recompute the frontier and compare KV, but production evaluation reuses the stored
native frontier and avoids an unnecessary second draft forward.

## Speculative-decoding experiment

Only a pair that passes pair screening enters this experiment. The first pilot uses
64 held-out prompts, 512 prompt tokens, 64 generated tokens, gamma 4, and 5,000
paired-bootstrap samples. A confirmatory run uses 200 prompts, 512 generated tokens,
and 10,000 bootstrap samples.

All methods use identical prompts and per-prompt random seeds:

1. `native_sd`: `native` initialization, policy `none`.
2. `legacy_mapped_init_only`: `legacy_mapped` initialization, policy `none`.
3. `mapped_init_only`: `mapped_native_frontier` initialization, policy `none`.
4. `legacy_full_refresh`: `legacy_mapped` initialization, policy `full`.
5. `mapped_accepted_only`: `mapped_native_frontier` initialization, policy
   `accepted_only`.

The output retains realized MAL, conditional expected accepted length, acceptance
rate, bonus rate, throughput, and peak VRAM. It adds paired confidence intervals
for Accepted-Only minus the native-frontier Init-only baseline and legacy Full
minus legacy Init-only. The latter contrast must reproduce the direction of the
recorded result before it is used as historical context.

### Speculative gates

Let `R = E[MAL](mapped_init_only) / E[MAL](native_sd)`.

- `R >= 0.90`: confirmatory-quality mapper; evaluate refresh as the primary test.
- `0.85 <= R < 0.90`: exploratory mapper; report refresh but do not make the
  structural claim.
- `R < 0.85`: reject the pair for the refresh question.

For a confirmatory-quality mapper, let
`Delta = E[MAL](mapped_accepted_only) - E[MAL](mapped_init_only)`.

- CI lower bound greater than zero supports Historical Verifier Anchoring plus a
  Native Frontier.
- CI upper bound below zero stops the verifier-anchored refresh paper direction.
- A CI crossing zero is inconclusive and triggers the predeclared larger sample,
  not a mapper or policy redesign.

Full Refresh remains an ablation. Accepted-Only must beat Init-only; merely beating
Full Refresh is insufficient.

## Command surface and artifacts

The implementation provides:

- `bench/capture_sequential_calibration.py`: frozen windows and resumable per-model
  calibration capture;
- `bench/fit_sequential_mapper.py`: R² selection and matched-head fitting from
  sequential shards;
- `bench/capture_screen_prefixes.py`: verifier-only held-out capture;
- `bench/eval_pair_transfer.py`: draft-only native versus mapped evaluation;
- `bench/eval_mapped_hellaswag.py`: optional mapped/native task retention;
- `scripts/run_pair_screen.sh`: runs both candidate pairs through the primary
  screen, resuming completed phases;
- `scripts/run_native_frontier_e2.sh`: runs the five-method E2 experiment for a
  selected mapper.

Machine-readable JSON outputs live under ignored `results/`; model caches,
calibration shards, and mapper weights live under ignored `artifacts/` and
`checkpoints/`. Small manifests and completed result summaries may be explicitly
force-added for scientific record after inspection.

Every JSON result includes a schema version, full CLI configuration, Git commit,
hardware description, model revisions, tokenizer hash, input digests, per-example
rows, aggregates, confidence intervals, and gate decisions. Writes use a temporary
file followed by an atomic rename so interruption cannot create a valid-looking
partial result.

## Failure handling

- Tokenizer, KV geometry, model revision, token-window, or manifest mismatch is a
  hard failure before fitting or evaluation.
- Missing shards resume; corrupt shards fail with their exact path.
- CUDA OOM clears cached allocations, records the failed phase/length, and exits
  without deleting completed artifacts.
- NaN or non-normalized probability rows fail the run rather than entering metrics.
- Evaluation refuses overlap between calibration and held-out token-window digests.
- A result with fewer than the requested valid prefixes is incomplete and cannot
  pass a gate.

## Tests

CPU tests cover artifact manifests and disjointness, atomic JSON writes, transfer
metrics and bootstrap decisions, explicit refresh-policy parsing, native-frontier
cache transitions, mapped initialization with a native last token, and five-method
E2 aggregation. Synthetic cache tests verify that Accepted-Only maps accepted
history while preserving the frontier exactly.

GPU integration checks use a tiny compatible Hugging Face model pair when
available. Scientific runs record success only after the requested real-model row
count, finite metrics, artifact hashes, and gate fields are present. Existing exact
speculative-decoding and mapper tests remain required.

## Non-goals

This experiment does not optimize simultaneous 8B plus 4B serving, introduce model
quantization, train a nonlinear translator, train the acceptance residual, or claim
novelty for matched-head KV mapping. Those steps are considered only after a pair
passes transfer screening and the Accepted-Only refresh gate.
