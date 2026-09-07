# Target-Alignment Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the native-reconstruction-only pair gate with a verifier-target alignment gate that works with sequential model loading and directly determines whether a pair may enter native-frontier speculative decoding.

**Architecture:** Extend held-out verifier capture to persist exact next-token probability vectors beside verifier KV shards. Extend pair evaluation to compare target/native/mapped distributions on the same causal frontier, aggregate `Delta_target = A_mapped_target - A_native_target` with document-cluster bootstrap, and drive the experiment runner from that gate while retaining `A_transfer` as diagnostic metadata.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, existing KVBridge-backed mapper, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-07-target-alignment-gate-design.md`

## Global Constraints

- Exact BF16 LLM weights for scientific capture; no automatic quantization.
- Verifier next-token probabilities are computed in FP32 and stored as FP32.
- Native and mapped draft paths must share the same native causal frontier token.
- Calibration and held-out token rows must remain disjoint by exact row digest.
- `A_transfer` remains diagnostic; it is not the speculative-decoding gate.
- Target-alignment gate uses 10,000 document-cluster bootstrap samples.
- The withdrawn overlap-product expected-MAL estimator must not be used for scientific gating.
- GPU OOM produces an explicit incomplete artifact; code must not silently change precision or model pair.

---

### Task 1: Target-alignment metric primitives

**Files:**
- Modify: `src/verifier_anchored_sd/transfer_metrics.py`
- Test: `tests/test_target_alignment_metrics.py`

**Interfaces:**
- Consumes: normalized probability tensors shaped `[rows, vocab]`.
- Produces: `target_alignment_rows(target_probs, native_probs, mapped_probs, next_ids=None) -> list[dict]`, `summarize_target_alignment(rows, requested, samples=10000, seed=0, cluster_ids=None) -> dict`.

- [ ] **Step 1: Write failing tests**

Test a hand-computed binary case where mapped is closer to target than native and assert positive `delta_target_alignment`. Test support/harm/inconclusive CI classification. Test incomplete runs withhold a scientific decision.

- [ ] **Step 2: Run RED test**

Run:

```bash
pytest -q tests/test_target_alignment_metrics.py
```

Expected: import failure because `target_alignment_rows` and `summarize_target_alignment` do not exist.

- [ ] **Step 3: Implement minimal metric code**

Per row compute:

```python
a_target_native = 1.0 - 0.5 * (target - native).abs().sum()
a_target_mapped = 1.0 - 0.5 * (target - mapped).abs().sum()
delta = a_target_mapped - a_target_native
```

Also record target-to-native/mapped KL and target top-1 matches. Bootstrap the row-level delta by document cluster. Decision is `support` when CI low > 0, `harm` when CI high < 0, otherwise `inconclusive`.

- [ ] **Step 4: Run GREEN tests**

```bash
pytest -q tests/test_target_alignment_metrics.py tests/test_transfer_metrics.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/verifier_anchored_sd/transfer_metrics.py tests/test_target_alignment_metrics.py
git commit -m "feat: add verifier-target alignment metrics"
```

### Task 2: Sequential verifier-probability artifacts

**Files:**
- Create: `src/verifier_anchored_sd/distribution_artifacts.py`
- Modify: `bench/capture_screen_prefixes.py`
- Test: `tests/test_distribution_artifacts.py`

**Interfaces:**
- Produces: `save_probability_shard(path, probabilities, metadata)`, `load_probability_shard(path)`, `exact_probability_paths(directory, count)`.
- Capture writes `screen_dir/target_probs/00000.pt ...` aligned one-to-one with `screen_dir/shards/00000.pt ...`.

- [ ] **Step 1: Write failing artifact tests**

Assert FP32 round-trip, probability normalization validation, exact shard-set validation, and rejection of malformed schema.

- [ ] **Step 2: Run RED test**

```bash
pytest -q tests/test_distribution_artifacts.py
```

Expected: module import failure.

- [ ] **Step 3: Implement probability artifacts**

Store a plain tensor payload:

```python
{
    "schema_version": 1,
    "metadata": {...},
    "probabilities": probabilities.float().cpu(),
}
```

Require one-dimensional finite non-negative probabilities summing to one within `1e-5`.

- [ ] **Step 4: Extend verifier screen capture**

Change the verifier forward from cache-only to:

```python
out = forward_incremental(target, ids)
probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)[0]
```

Save `out.cache` and `probs` under matching sequence IDs and token digests. Existing cache shards without a matching probability shard must not count as a complete new-format row.

- [ ] **Step 5: Run GREEN tests**

```bash
pytest -q tests/test_distribution_artifacts.py tests/test_cache_artifacts.py tests/test_pair_screen_contract.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_sd/distribution_artifacts.py bench/capture_screen_prefixes.py tests/test_distribution_artifacts.py
git commit -m "feat: capture verifier probability artifacts"
```

### Task 3: Pair evaluator uses target alignment

**Files:**
- Modify: `bench/eval_pair_transfer.py`
- Test: `tests/test_target_alignment_contract.py`

**Interfaces:**
- Consumes: verifier KV shard, verifier probability shard, frozen token row, draft model, mapper.
- Produces result sections: `native_fidelity`, `target_alignment`, and top-level `decision`.

- [ ] **Step 1: Write failing contract tests**

Assert evaluator-result decision rules from synthetic aggregates:

```text
target support -> go_sd
target harm -> stop_pair
target inconclusive -> expand
```

Assert `A_transfer` failure alone cannot force `stop_pair` when target alignment is support.

- [ ] **Step 2: Run RED test**

```bash
pytest -q tests/test_target_alignment_contract.py
```

Expected: failure because target-alignment decision helper is absent.

- [ ] **Step 3: Implement decision helper**

Add a pure helper in `transfer_metrics.py`:

```python
def target_alignment_decision(target_status: str, fidelity_status: str) -> str:
    if target_status == "support":
        return "go_sd"
    if target_status == "harm":
        return "stop_pair"
    if target_status == "inconclusive":
        return "expand"
    return "incomplete"
```

The fidelity status is retained in the result but cannot override target `support`.

- [ ] **Step 4: Update evaluator**

Load the matching target probability shard for each row, validate sequence ID/token digest/next-token ID/vocabulary size, then compute both metric families. Preserve old `summary`/`gate` aliases only for backward-readable artifacts and label them `native_fidelity_diagnostic` in new results.

- [ ] **Step 5: Run GREEN tests**

```bash
pytest -q tests/test_target_alignment_contract.py tests/test_target_alignment_metrics.py tests/test_transfer_metrics.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add bench/eval_pair_transfer.py src/verifier_anchored_sd/transfer_metrics.py tests/test_target_alignment_contract.py
git commit -m "feat: gate pair screening on verifier alignment"
```

### Task 4: Direct-run experiment scripts

**Files:**
- Create: `scripts/run_target_alignment_next.sh`
- Modify: `scripts/run_pair_screen.sh`
- Modify: `configs/pair_screen.yaml`

**Interfaces:**
- `run_target_alignment_next.sh` first re-evaluates the existing Qwen3-8B -> 4B mapper without refitting.
- `run_pair_screen.sh` uses top-level `decision` for expansion/stop behavior on future pairs.

- [ ] **Step 1: Add Stage A runner**

Require:

```bash
CALIBRATION_TEXT
EVAL_TEXT
```

Reuse:

```text
artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
```

Use a fresh screen directory `artifacts/target_alignment_2026-09-07/qwen3_8b_to_4b/screen` so old cache-only screen shards cannot contaminate the run.

Run 4-prefix smoke first under `artifacts/target_alignment_2026-09-07/smoke/`, then 128 x 1,024 scientific prefixes and 10,000 bootstrap samples.

- [ ] **Step 2: Add 32B -> 14B candidate configuration**

Add model IDs to config and document two exact-weight memory profiles:

```text
48GB: GPU_MEMORY_GIB=44
32GB: GPU_MEMORY_GIB=28
```

Use sequential model loading. Keep 128 calibration windows, stride 4, k=8, lambda=0.01, 32 selection sequences, 128 held-out prefixes.

- [ ] **Step 3: Update pair-screen orchestration**

Read `result["decision"]["status"]`:

```text
go_sd -> permit native-frontier E2
stop_pair -> stop this pair
expand -> recapture/evaluate 512 held-out prefixes
```

Do not use legacy `A_transfer` gate to block SD.

- [ ] **Step 4: Shell syntax validation**

```bash
bash -n scripts/run_target_alignment_next.sh
bash -n scripts/run_pair_screen.sh
```

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_target_alignment_next.sh scripts/run_pair_screen.sh configs/pair_screen.yaml
git commit -m "exp: add target-alignment run protocol"
```

### Task 5: Agent-facing debug and execution handoff

**Files:**
- Create: `docs/NEXT_STAGE_TARGET_ALIGNMENT.md`
- Modify: `docs/NEXT_EXPERIMENTS.md`
- Modify: `README.md`

**Interfaces:**
- Provides exact commands, expected artifacts, STOP/GO decisions, and debug requirements for the remote testing agent.

- [ ] **Step 1: Write direct execution order**

The document must state:

```text
0. git checkout feat/target-alignment-gate
1. install exact project extras
2. pytest -q
3. 4-prefix target-alignment smoke
4. 128-prefix 8B -> 4B target-alignment run
5. inspect decision.status
6. only go_sd -> run native-frontier E2
7. stop_pair -> run 32B -> 14B sequential screen
```

- [ ] **Step 2: Write debug STOP conditions**

Include the ten conditions in the design spec: token/revision/tokenizer/vocab/probability/frontier/artifact/data/OOM/metric misuse.

- [ ] **Step 3: Document result reporting**

The testing agent must commit result JSON plus a concise `RUN_REPORT.md` with exact GPU, peak VRAM, completed rows, artifact hashes, gate status, and any incomplete phase. It must not commit reconstructible large KV shards.

- [ ] **Step 4: Commit**

```bash
git add docs/NEXT_STAGE_TARGET_ALIGNMENT.md docs/NEXT_EXPERIMENTS.md README.md
git commit -m "docs: hand off target-alignment experiments"
```

### Task 6: Full verification

**Files:** none beyond prior tasks.

- [ ] **Step 1: Run complete CPU verification**

```bash
python -m compileall -q src bench training
pytest -q
bash -n scripts/run_target_alignment_next.sh
bash -n scripts/run_pair_screen.sh
```

Expected: zero failures.

- [ ] **Step 2: Inspect diff**

Verify no scientific result files were fabricated, no old result was overwritten, no quantization path was added, and new docs consistently use target alignment rather than native fidelity as the SD gate.

- [ ] **Step 3: Open PR**

Open a PR from `feat/target-alignment-gate` to `main` summarizing the metric correction, sequential probability artifacts, execution protocol, and verification evidence.
