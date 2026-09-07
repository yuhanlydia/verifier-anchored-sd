# Pair Screening and Native-Frontier Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable sequential Qwen3 pair screen, then provide a causally consistent native-frontier refresh ablation for pairs that pass.

**Architecture:** Frozen token windows and model-specific cache shards allow verifier and draft models to run sequentially on a 16GB GPU. Pure artifact and metric modules support thin benchmark CLIs. The speculative runtime separates initialization mode from refresh policy so legacy results remain reproducible while Accepted-Only preserves the native causal frontier.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers/Datasets/Accelerate, KVBridge, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-07-pair-screen-native-frontier-design.md`

## Global Constraints

- Primary pairs are `Qwen/Qwen3-8B -> Qwen/Qwen3-4B` and `Qwen/Qwen3-4B -> Qwen/Qwen3-1.7B`.
- Calibration and held-out token-window digests must be disjoint.
- Primary mapper is matched-head, content-space, `k=8`, lambda `0.01`, 128 x 1,024 tokens, stride 4, R² selection on 32 sequences.
- Screening uses exact BF16 weights with sequential model loading; quantization is forbidden.
- Pair pass requires the 95% bootstrap CI lower bound for mean `A_transfer` to exceed `0.95`.
- Confirmatory speculative eligibility requires mapped/native expected-MAL retention of at least `0.90`.
- Scientific artifacts must record frozen model revisions, tokenizer hash, input digests, Git commit, hardware, requested row count, and gate status.
- Production code follows failing-test-first TDD; all existing tests remain required.

---

### Task 1: Artifact contracts and transfer statistics

**Files:**
- Create: `src/verifier_anchored_sd/experiment_artifacts.py`
- Create: `src/verifier_anchored_sd/transfer_metrics.py`
- Test: `tests/test_experiment_artifacts.py`
- Test: `tests/test_transfer_metrics.py`

**Interfaces:**
- Produces: `sha256_file(path) -> str`, `token_rows_digest(rows) -> str`, `atomic_write_json(path, value) -> None`, `validate_disjoint(calibration_digest, evaluation_digest) -> None`.
- Produces: `distribution_transfer_rows(native_probs, mapped_probs, next_ids=None) -> list[dict]`, `summarize_transfer(rows, samples, seed, threshold) -> dict`.

- [x] **Step 1: Write failing artifact tests**

```python
def test_atomic_json_is_complete_and_digest_is_order_sensitive(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert not (tmp_path / "result.json.tmp").exists()
    assert token_rows_digest([[1, 2], [3]]) != token_rows_digest([[3], [1, 2]])

def test_calibration_and_evaluation_must_be_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint("same", "same")
```

- [x] **Step 2: Run artifact tests and verify failure from missing module**

Run: `.venv/bin/pytest tests/test_experiment_artifacts.py -q`
Expected: collection error for `verifier_anchored_sd.experiment_artifacts`.

- [x] **Step 3: Implement deterministic hashing, atomic JSON, and disjointness**

```python
def token_rows_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<Q", len(row)))
        for token in row:
            digest.update(struct.pack("<q", int(token)))
    return digest.hexdigest()

def atomic_write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
```

- [x] **Step 4: Write failing metric tests**

```python
def test_identical_distributions_have_perfect_transfer():
    probs = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    rows = distribution_transfer_rows(probs, probs)
    assert [row["a_transfer"] for row in rows] == [1.0, 1.0]
    assert all(row["kl_native_mapped"] == 0.0 for row in rows)

def test_gate_distinguishes_pass_fail_and_inconclusive():
    assert classify_transfer_gate(0.951, 0.970, 0.95) == "pass"
    assert classify_transfer_gate(0.920, 0.950, 0.95) == "fail"
    assert classify_transfer_gate(0.940, 0.960, 0.95) == "inconclusive"
```

- [x] **Step 5: Run metric tests and verify missing behavior**

Run: `.venv/bin/pytest tests/test_transfer_metrics.py -q`
Expected: import or assertion failure for the new transfer functions.

- [x] **Step 6: Implement normalized TV, KL, Top-1, NLL delta, bootstrap, and gate**

```python
def classify_transfer_gate(ci_low, ci_high, threshold=0.95):
    if ci_low > threshold:
        return "pass"
    if ci_high <= threshold:
        return "fail"
    return "inconclusive"

def _one_row(native, mapped, next_id=None):
    p = native.float() / native.float().sum().clamp_min(1e-12)
    q = mapped.float() / mapped.float().sum().clamp_min(1e-12)
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("probabilities must be finite")
    row = {
        "a_transfer": float((1.0 - 0.5 * (p - q).abs().sum()).clamp(0, 1)),
        "kl_native_mapped": float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum()),
        "top1_agreement": int(p.argmax() == q.argmax()),
    }
    if next_id is not None:
        row["next_token_nll_delta"] = float(-q[next_id].clamp_min(1e-12).log() + p[next_id].clamp_min(1e-12).log())
    return row
```

- [x] **Step 7: Run focused and existing evaluation tests**

Run: `.venv/bin/pytest tests/test_experiment_artifacts.py tests/test_transfer_metrics.py tests/test_evaluation_stats.py -q`
Expected: all pass.

- [x] **Step 8: Commit**

```bash
git add src/verifier_anchored_sd/experiment_artifacts.py src/verifier_anchored_sd/transfer_metrics.py tests/test_experiment_artifacts.py tests/test_transfer_metrics.py
git commit -m "feat: add pair-screen artifact and metric contracts"
```

### Task 2: Sequential model loading and cache shards

**Files:**
- Modify: `bench/common.py`
- Create: `src/verifier_anchored_sd/cache_artifacts.py`
- Create: `bench/capture_sequential_calibration.py`
- Test: `tests/test_cache_artifacts.py`
- Test: `tests/test_sequential_capture_contract.py`

**Interfaces:**
- Produces: `load_hf_model(model_id, device, dtype, *, gpu_memory_gib=None, offload_folder=None) -> (tokenizer, model)`.
- Produces: `save_cache_shard(path, cache, metadata)`, `load_cache_shard(path) -> (CacheState, dict)`.
- Consumes: Task 1 hashing and atomic manifest functions.

- [x] **Step 1: Write failing cache round-trip and manifest tests**

```python
def test_cache_shard_round_trip_preserves_rotary_and_content_flag(tmp_path):
    original = cache_with_rotary(tokens=3)
    save_cache_shard(tmp_path / "00000.pt", original, {"sequence_id": "00000"})
    restored, metadata = load_cache_shard(tmp_path / "00000.pt")
    assert metadata["sequence_id"] == "00000"
    assert restored.keys_are_content == original.keys_are_content
    assert torch.equal(restored.layers[0].key, original.layers[0].key)
    assert torch.equal(restored.rotary.cos, original.rotary.cos)

def test_existing_manifest_rejects_changed_model_revision(tmp_path):
    write_or_validate_manifest(tmp_path, manifest(revision="a"))
    with pytest.raises(RuntimeError, match="different contract"):
        write_or_validate_manifest(tmp_path, manifest(revision="b"))
```

- [x] **Step 2: Run tests and verify failure from absent shard API**

Run: `.venv/bin/pytest tests/test_cache_artifacts.py tests/test_sequential_capture_contract.py -q`
Expected: collection failure for missing modules.

- [x] **Step 3: Implement plain-tensor CacheState serialization**

```python
def cache_state_payload(cache, metadata):
    return {
        "schema_version": 1,
        "metadata": dict(metadata),
        "keys": [layer.key.cpu() for layer in cache.layers],
        "values": [layer.value.cpu() for layer in cache.layers],
        "keys_are_content": cache.keys_are_content,
        "rotary_cos": None if cache.rotary is None else cache.rotary.cos.cpu(),
        "rotary_sin": None if cache.rotary is None else cache.rotary.sin.cpu(),
        "rotary_interleaved": None if cache.rotary is None else cache.rotary.interleaved,
    }
```

- [x] **Step 4: Extract one-model loading from `load_hf_pair`**

```python
def load_hf_model(model_id, device, dtype="bfloat16", *, gpu_memory_gib=None, offload_folder=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = {"torch_dtype": resolve_dtype(dtype), "trust_remote_code": True, "low_cpu_mem_usage": True}
    if gpu_memory_gib is not None:
        kwargs.update(device_map="auto", max_memory={0: f"{gpu_memory_gib}GiB", "cpu": "80GiB"}, offload_state_dict=True, offload_folder=offload_folder)
    else:
        kwargs["device_map"] = {"": device}
    return tokenizer, AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
```

- [x] **Step 5: Implement the resumable calibration capture CLI**

The CLI accepts `--role source|draft`, freezes token IDs under
`PAIR_DIR/tokens`, validates the pair tokenizer contract, captures sampled cache
states under `PAIR_DIR/source` or `PAIR_DIR/draft`, and writes a role manifest only
after all requested shards validate.

- [x] **Step 6: Run focused tests and CLI help**

Run: `.venv/bin/pytest tests/test_cache_artifacts.py tests/test_sequential_capture_contract.py -q && .venv/bin/python bench/capture_sequential_calibration.py --help >/dev/null`
Expected: all tests and help command pass.

- [x] **Step 7: Commit**

```bash
git add bench/common.py bench/capture_sequential_calibration.py src/verifier_anchored_sd/cache_artifacts.py tests/test_cache_artifacts.py tests/test_sequential_capture_contract.py
git commit -m "feat: capture calibration caches sequentially"
```

### Task 3: Local R² layer selection and mapper fitting

**Files:**
- Modify: `src/verifier_anchored_sd/spec_decode/head_local_fit.py`
- Create: `bench/fit_sequential_mapper.py`
- Test: `tests/test_head_local_selection.py`

**Interfaces:**
- Produces: `select_source_layers_by_r2(pairs, *, target_layers, draft_layers, kv_heads, head_dim, top_k, device, layer_block_size, content_space) -> (list[list[int]], list[list[float]])`.
- Consumes: Task 2 `load_cache_shard`; existing `fit_matched_head_mapper_from_cache_pairs`.

- [x] **Step 1: Write a failing synthetic selection test**

```python
def test_r2_selection_finds_the_constructed_source_layer():
    pairs = [synthetic_pair(draft_from_source_layer=1, seed=i) for i in range(6)]
    selected, scores = select_source_layers_by_r2(lambda: iter(pairs), target_layers=3, draft_layers=1, kv_heads=2, head_dim=2, top_k=1, device="cpu", layer_block_size=1, content_space=False)
    assert selected == [[1]]
    assert scores[0][1] > scores[0][0]
```

- [x] **Step 2: Run the test and verify missing selector failure**

Run: `.venv/bin/pytest tests/test_head_local_selection.py -q`
Expected: import failure for `select_source_layers_by_r2`.

- [x] **Step 3: Implement streaming centered univariate-layer R² selection**

For each draft layer and source layer, accumulate per-head centered sufficient
statistics in FP32, solve ridge with alpha `1e-6`, compute held-in explained
variance across K and V, average across heads, and select the stable descending
top-k with source-layer index as tie-breaker.

- [x] **Step 4: Implement mapper fit CLI with exact shard-set validation**

```python
def pair_factory(paths):
    for source_path, draft_path in paths:
        source, source_meta = load_cache_shard(source_path)
        draft, draft_meta = load_cache_shard(draft_path)
        if source_meta["token_digest"] != draft_meta["token_digest"]:
            raise RuntimeError(f"calibration token mismatch: {source_path}")
        yield source, draft
```

The CLI selects on the first `--selection-sequences` ordered pairs, fits on every
requested pair, saves the mapper and JSON metadata atomically, and rejects missing,
extra, or differently ordered shards.

- [x] **Step 5: Run selector, existing fitter, and CLI checks**

Run: `.venv/bin/pytest tests/test_head_local_selection.py tests/test_head_local_fit.py -q && .venv/bin/python bench/fit_sequential_mapper.py --help >/dev/null`
Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add src/verifier_anchored_sd/spec_decode/head_local_fit.py bench/fit_sequential_mapper.py tests/test_head_local_selection.py
git commit -m "feat: fit sequential matched-head mappers"
```

### Task 4: Sequential held-out pair screen

**Files:**
- Create: `bench/capture_screen_prefixes.py`
- Create: `bench/eval_pair_transfer.py`
- Create: `tests/test_pair_screen_contract.py`
- Modify: `src/verifier_anchored_sd/transfer_metrics.py`

**Interfaces:**
- `capture_screen_prefixes.py` produces verifier cache shards and held-out manifest.
- `eval_pair_transfer.py` consumes those shards plus a mapper, loads only the draft,
  and writes a complete screen JSON.

- [x] **Step 1: Write failing contract tests for valid row counts and disjointness**

```python
def test_screen_cannot_pass_with_missing_rows():
    result = finalize_screen([transfer_row(0.99)], requested=2, samples=100, seed=0)
    assert result["gate"]["status"] == "incomplete"

def test_screen_rejects_calibration_digest_reuse():
    with pytest.raises(ValueError, match="overlap"):
        validate_screen_inputs({"token_rows_digest": "x"}, {"token_rows_digest": "x"})
```

- [x] **Step 2: Run the tests and observe the intended missing-function failures**

Run: `.venv/bin/pytest tests/test_pair_screen_contract.py -q`
Expected: import failures for new screen functions.

- [x] **Step 3: Implement verifier-only held-out capture**

Use the Task 2 shard format without stride. Store the real next token ID separately
from each prefix and include token-window digests in every shard.

- [x] **Step 4: Implement native versus mapped-native-frontier draft evaluation**

```python
native = forward_incremental(draft, prefix_ids)
mapped_history = mapper.map(
    target_cache.slice(0, target_cache.seq_len - 1),
    draft_rotary=draft_rotary,
)
mapped = forward_incremental(draft, prefix_ids[:, -1:], mapped_history)
rows.extend(distribution_transfer_rows(probs(native.logits), probs(mapped.logits), [next_id]))
```

Register hooks on each draft `self_attn` module only when
`--attention-cosine` is supplied. Reduce the final-token output immediately and
remove every hook in `finally`.

- [x] **Step 5: Run focused tests and both CLI help commands**

Run: `.venv/bin/pytest tests/test_pair_screen_contract.py tests/test_transfer_metrics.py -q && .venv/bin/python bench/capture_screen_prefixes.py --help >/dev/null && .venv/bin/python bench/eval_pair_transfer.py --help >/dev/null`
Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add bench/capture_screen_prefixes.py bench/eval_pair_transfer.py src/verifier_anchored_sd/transfer_metrics.py tests/test_pair_screen_contract.py
git commit -m "feat: evaluate sequential pair transfer"
```

### Task 5: Native-frontier runtime policies

**Files:**
- Modify: `src/verifier_anchored_sd/spec_decode/hf_runtime.py`
- Modify: `src/verifier_anchored_sd/spec_decode/verifier_cache_refresh.py`
- Create: `tests/test_native_frontier_policy.py`
- Modify: `tests/test_pending_frontier.py`
- Modify: `tests/test_refresh_cache_length.py`

**Interfaces:**
- Produces: `InitMode = Literal["native", "legacy_mapped", "mapped_native_frontier"]`.
- Produces: `RefreshPolicy = Literal["none", "full", "accepted_only"]`.
- `QwenPairRuntime(..., init_mode, refresh_policy)` replaces the Boolean API while
  accepting legacy `refresh=` temporarily only if needed by existing callers.

- [x] **Step 1: Write failing cache transition tests**

```python
def test_accepted_only_resolves_pending_without_replacing_native_kv():
    state = VerifierAnchoredCache(kv(2, 1), mapper())
    state.append_pending(7, kv(1, 3), next_probs=torch.tensor([[0.2, 0.8]]))
    before = state.draft_cache.layers[0].key[..., -1:, :].clone()
    probs = state.resolve_pending_native(7)
    assert torch.equal(state.draft_cache.layers[0].key[..., -1:, :], before)
    assert torch.equal(probs, torch.tensor([[0.2, 0.8]]))
    assert state.pending is None

def test_full_refresh_replaces_pending_for_legacy_reproduction():
    state = pending_state(native_value=3)
    state.materialize_pending(7, kv(1, 9))
    assert not torch.equal(state.draft_cache.layers[0].key[..., -1:, :], kv(1, 3).layers[0].key)
```

- [x] **Step 2: Run transition tests and verify expected failures**

Run: `.venv/bin/pytest tests/test_native_frontier_policy.py -q`
Expected: missing `next_probs`/`resolve_pending_native` behavior.

- [x] **Step 3: Store frontier logits with the native pending KV**

Extend `PendingFrontier` with `next_probs`. `append_pending` receives the native
draft token KV plus its next-token probabilities, stores both, and
`resolve_pending_native` clears only the marker.

- [x] **Step 4: Write failing runtime-policy tests with deterministic fake forwards**

Tests patch `forward_incremental` at the model boundary and assert:

```python
assert runtime.anchored.draft_cache.layers[0].key[..., -1, :].equal(native_last)
assert runtime.draft_next_probs.equal(probs_from_that_same_forward)
assert mapper.map_calls == expected_historical_ranges
```

- [x] **Step 5: Implement explicit init and refresh policies**

For `mapped_native_frontier`, map target prompt slice `0:n-1`, run token `n`
natively, and append its returned KV. For `accepted_only`, append mapped accepted
proposal KV but resolve correction/bonus pending state without replacement. Reuse
stored draft probabilities at the next proposal boundary.

- [x] **Step 6: Run all state-machine tests**

Run: `.venv/bin/pytest tests/test_native_frontier_policy.py tests/test_pending_frontier.py tests/test_refresh_cache_length.py tests/test_rejection_rollback.py tests/test_bonus_frontier.py -q`
Expected: all pass.

- [x] **Step 7: Commit**

```bash
git add src/verifier_anchored_sd/spec_decode/hf_runtime.py src/verifier_anchored_sd/spec_decode/verifier_cache_refresh.py tests/test_native_frontier_policy.py tests/test_pending_frontier.py tests/test_refresh_cache_length.py
git commit -m "feat: preserve native causal frontiers"
```

### Task 6: Five-method E2 and decision gates

**Files:**
- Modify: `bench/eval_acceptance_pilot.py`
- Create: `tests/test_acceptance_method_matrix.py`
- Modify: `tests/test_evaluation_stats.py`

**Interfaces:**
- Produces: `acceptance_methods() -> dict[str, dict[str, str]]`.
- Produces: `classify_mapper_retention(value) -> Literal["confirmatory", "exploratory", "reject"]` and `classify_refresh_delta(ci_low, ci_high) -> Literal["support", "stop", "inconclusive"]`.

- [x] **Step 1: Write failing method-matrix and gate tests**

```python
def test_e2_method_matrix_keeps_legacy_and_primary_contrasts_separate():
    methods = acceptance_methods()
    assert methods["legacy_full_refresh"] == {"init_mode": "legacy_mapped", "refresh_policy": "full"}
    assert methods["mapped_accepted_only"] == {"init_mode": "mapped_native_frontier", "refresh_policy": "accepted_only"}

def test_refresh_decision_requires_confidence_interval_sign():
    assert classify_refresh_delta(0.01, 0.10) == "support"
    assert classify_refresh_delta(-0.10, -0.01) == "stop"
    assert classify_refresh_delta(-0.01, 0.02) == "inconclusive"
```

- [x] **Step 2: Run tests and verify failures**

Run: `.venv/bin/pytest tests/test_acceptance_method_matrix.py tests/test_evaluation_stats.py -q`
Expected: missing matrix/gate functions.

- [x] **Step 3: Implement the five methods and paired contrasts**

Output keys are `accepted_only_minus_init_expected_mal`,
`legacy_full_minus_legacy_init_expected_mal`, and the corresponding realized-MAL
contrasts. Preserve per-prompt seeds across all methods.

- [x] **Step 4: Implement exact decision fields**

```python
retention = mapped_init / max(native, 1e-12)
gates = {
    "mapper_retention": retention,
    "mapper_status": classify_mapper_retention(retention),
    "accepted_only_delta": primary_ci["mean_difference"],
    "accepted_only_status": classify_refresh_delta(primary_ci["ci_low"], primary_ci["ci_high"]),
}
```

- [x] **Step 5: Run focused E2 and state-machine tests**

Run: `.venv/bin/pytest tests/test_acceptance_method_matrix.py tests/test_evaluation_stats.py tests/test_native_frontier_policy.py -q`
Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add bench/eval_acceptance_pilot.py tests/test_acceptance_method_matrix.py tests/test_evaluation_stats.py
git commit -m "feat: evaluate accepted-only refresh gates"
```

### Task 7: Optional HellaSwag confirmation

**Files:**
- Create: `bench/eval_mapped_hellaswag.py`
- Create: `src/verifier_anchored_sd/multiple_choice.py`
- Test: `tests/test_multiple_choice.py`

**Interfaces:**
- Produces: `choice_nll(logits, labels, context_length) -> float` and `normalized_retention(native_accuracy, mapped_accuracy, random_floor=0.25) -> float`.
- Consumes: the Task 4 mapped-native-frontier initialization path.

- [x] **Step 1: Write failing scoring and normalization tests**

```python
def test_choice_nll_scores_only_continuation_tokens():
    logits = perfect_logits_for([4, 5])
    assert choice_nll(logits, torch.tensor([9, 4, 5]), context_length=1) < 1e-3

def test_floor_normalized_retention():
    assert normalized_retention(0.75, 0.70, random_floor=0.25) == pytest.approx(0.9)
```

- [x] **Step 2: Run tests and verify missing scoring behavior**

Run: `.venv/bin/pytest tests/test_multiple_choice.py -q`
Expected: missing module failure.

- [x] **Step 3: Implement teacher-forced continuation scoring**

Score each HellaSwag ending twice with identical tokens: native draft context and
mapped-history/native-frontier context. Average token NLL selects the answer. Record
raw and floor-normalized mapped/native retention.

- [x] **Step 4: Implement CLI validation and atomic result output**

The CLI refuses to run unless the pair-screen JSON gate is `pass`, unless
`--allow-failed-screen` is explicitly supplied for diagnosis.

- [x] **Step 5: Run tests and CLI help**

Run: `.venv/bin/pytest tests/test_multiple_choice.py -q && .venv/bin/python bench/eval_mapped_hellaswag.py --help >/dev/null`
Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add bench/eval_mapped_hellaswag.py src/verifier_anchored_sd/multiple_choice.py tests/test_multiple_choice.py
git commit -m "feat: confirm mapped-cache task retention"
```

### Task 8: Orchestration, documentation, and real execution

**Files:**
- Create: `configs/pair_screen.yaml`
- Create: `scripts/run_pair_screen.sh`
- Create: `scripts/run_native_frontier_e2.sh`
- Modify: `README.md`
- Modify: `docs/NEXT_EXPERIMENTS.md`
- Create after run: `docs/PAIR_SCREEN_2026-09-07.md`

**Interfaces:**
- Consumes all previous CLIs and records their exact commands.
- Produces resumable candidate result JSON and a human-readable decision record.

- [ ] **Step 1: Add the frozen candidate matrix and orchestration scripts**

`run_pair_screen.sh` runs source capture, draft capture, fit, held-out verifier
capture, and draft evaluation for both candidates. It accepts `HF_HUB_CACHE`,
`CALIBRATION_TEXT`, and `EVAL_TEXT`; it never uses the evaluation file for fitting.

- [ ] **Step 2: Add E2 orchestration with a screen gate check**

`run_native_frontier_e2.sh` reads the screen JSON and exits unless status is `pass`.
It then runs the five-method pilot and prints only the mapper and refresh decisions.

- [ ] **Step 3: Update project docs to supersede the old next experiment**

Document Pair Selection -> Near-lossless Transfer -> Speculative Compatibility ->
Native-Frontier Refresh, the exact stop rules, 16GB sequential limitations, and
the distinction between legacy and structurally consistent methods.

- [ ] **Step 4: Run complete static and CPU verification**

Run: `.venv/bin/python -m compileall -q src bench training && .venv/bin/ruff check . && .venv/bin/pytest -q`
Expected: zero compile errors, zero Ruff errors, all tests pass without warnings.

- [ ] **Step 5: Install real-run extras and verify pinned dependencies**

Run: `.venv/bin/python -m pip install -e '.[hf,kvbridge,dev]'`
Expected: Transformers, Datasets, Accelerate, and pinned KVBridge import successfully.

- [ ] **Step 6: Run a one-sequence real-model integration probe**

Run the 4B to 1.7B pipeline with one 64-token calibration window, depth selection
restricted to the integration probe, one held-out 64-token prefix, and CPU fitting.
Expected: source and draft load sequentially, artifacts resume, one finite transfer
row is written, and the result is labeled `integration_probe`, never scientific.

- [ ] **Step 7: Run the preregistered pair screen on the 16GB A4000**

Run: `bash scripts/run_pair_screen.sh`
Expected: completed result JSON for both pairs or a precise durable phase/OOM record
that can resume on the next host without repeating completed capture.

- [ ] **Step 8: Write the dated result record from machine-readable outputs**

Record exact commands, revisions, environment, metrics, confidence intervals,
decisions, incomplete phases, and the next permitted experiment. Do not infer a
gate result from partial rows.

- [ ] **Step 9: Re-run final verification and inspect repository state**

Run: `.venv/bin/python -m compileall -q src bench training && .venv/bin/ruff check . && .venv/bin/pytest -q && git diff --check && git status --short`
Expected: all checks pass; only intentional result/doc changes remain.

- [ ] **Step 10: Commit orchestration and recorded results**

```bash
git add configs/pair_screen.yaml scripts/run_pair_screen.sh scripts/run_native_frontier_e2.sh README.md docs/NEXT_EXPERIMENTS.md docs/PAIR_SCREEN_2026-09-07.md
git commit -m "exp: add pair-screen and native-frontier protocol"
```
