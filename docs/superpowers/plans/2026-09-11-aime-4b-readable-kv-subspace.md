# AIME 4B-Readable Teacher-KV Subspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible AIME experiment that transfers Qwen3-8B prompt KV directly or through Full-Head ridge, derives branch-specific low-rank bases from Qwen3-4B answer-span gradients, and evaluates projected KV on leakage-free AIME2025 examples.

**Architecture:** A pinned dataset adapter materializes normalized AIME rows and deterministic folds. A branch-neutral gradient accumulator consumes 4B-space cache gradients, while separate direct and Full-Head adapters create those caches. GPU fit and evaluation entry points reuse the repository's existing cache, mapper, metric, provenance, and low-VRAM infrastructure.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Hugging Face Dataset Viewer/Hub files, pytest, Bash, existing `verifier_anchored_sd` modules.

**Spec:** `docs/superpowers/specs/2026-09-11-aime-4b-readable-kv-subspace-design.md`

## Global Constraints

- Teacher is `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`.
- Receiver is `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`.
- Model weights and inference dtype remain exact BF16.
- Dataset is `allenai/aime-2022-2025@73e1eba765ad5847cdb5d1e2e7aaf7b22b585798`.
- Both `direct` and `full_head` branches run on identical folds and examples.
- SVD/eigendecomposition statistics are computed in FP64 in 4B KV coordinates.
- AIME2025 evaluation examples never contribute to their own cross-fit basis.
- Negative performance does not terminate the other branch or suppress artifacts.
- Existing `results/paper_vs_subspace_2026-09-09` outputs are immutable.

---

### Task 1: AIME data and fold contract

**Files:**
- Create: `src/verifier_anchored_sd/aime_data.py`
- Create: `tests/test_aime_data.py`
- Create: `bench/prepare_aime_kv_subspace.py`

**Interfaces:**
- Produces: `AIMERow`, `normalize_aime_row(raw)`, `answer_span_ids(tokenizer, row)`, `make_aime2025_folds(rows, folds=5)`, and a frozen JSONL/manifest writer.
- Consumes: existing `atomic_write_json`, `sha256_file`, and canonical JSON hashing conventions.

- [ ] **Step 1: Write failing normalization and fold tests**

```python
def test_normalize_aime_row_zero_pads_answer():
    row = normalize_aime_row({
        "id": 7, "problem": "Find n.", "solution": "Thus n=7.\n~author",
        "answer": "7", "url": "https://example/2025_AIME_I_Problems/Problem_1",
        "year": 2025,
    })
    assert row.gold_answer == "007"
    assert row.gold_solution == "Thus n=7."

def test_five_folds_hold_out_each_problem_once():
    rows = make_rows(15, exam="I") + make_rows(15, exam="II")
    folds = make_aime2025_folds(rows)
    assert [len(f.eval_ids) for f in folds] == [6] * 5
    assert all(sum(r.exam == "I" for r in f.eval_rows) == 3 for f in folds)
    assert sorted(x for f in folds for x in f.eval_ids) == sorted(r.example_id for r in rows)
    assert all(set(f.fit_ids).isdisjoint(f.eval_ids) for f in folds)
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_aime_data.py -q`

Expected: collection fails because `verifier_anchored_sd.aime_data` does not exist.

- [ ] **Step 3: Implement immutable row types and validation**

```python
@dataclass(frozen=True)
class AIMERow:
    example_id: str
    problem: str
    gold_solution: str
    gold_answer: str
    year: int
    exam: Literal["I", "II"]
    source_url: str
    digest: str

def normalize_aime_row(raw: Mapping[str, object]) -> AIMERow:
    answer_int = int(str(raw["answer"]).strip())
    if not 0 <= answer_int <= 999:
        raise ValueError("AIME answer must lie in [0, 999]")
    solution = str(raw["solution"]).split("\n~", 1)[0].strip()
    if not solution:
        raise ValueError("AIME solution must be non-empty")
    exam = parse_exam_from_url(str(raw["url"]))
    answer = f"{answer_int:03d}"
    # Build example_id and digest from canonical serialized fields.
```

- [ ] **Step 4: Implement exact answer-token masking tests and code**

```python
def answer_span_ids(tokenizer, row: AIMERow) -> AnswerSpan:
    prefix = format_problem_prompt(row.problem)
    completion_prefix = row.gold_solution + "\nAnswer: "
    full_ids = tokenizer(prefix + completion_prefix + row.gold_answer, add_special_tokens=False)["input_ids"]
    before_ids = tokenizer(prefix + completion_prefix, add_special_tokens=False)["input_ids"]
    answer_ids = full_ids[len(before_ids):]
    if not answer_ids:
        raise ValueError("AIME answer token span is empty")
    return AnswerSpan(full_ids=tuple(full_ids), answer_positions=tuple(range(len(before_ids), len(full_ids))))
```

The test uses a deterministic tokenizer stub whose answer occupies two tokens and asserts only those positions survive label masking.

- [ ] **Step 5: Implement dataset preparation CLI**

`bench/prepare_aime_kv_subspace.py` accepts `--dataset`, `--revision`, and `--output-dir`, downloads the pinned source, normalizes every row, writes `aime_2022_2024.jsonl`, `aime_2025.jsonl`, five fold manifests, and `data_manifest.json`, then asserts row-digest disjointness.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_aime_data.py -q`

Expected: all AIME data tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: existing suite plus new tests passes.

- [ ] **Step 7: Commit**

```bash
git add src/verifier_anchored_sd/aime_data.py tests/test_aime_data.py bench/prepare_aime_kv_subspace.py
git commit -m "feat: add pinned AIME subspace data contract"
```

### Task 2: FP64 gradient-subspace artifact

**Files:**
- Create: `src/verifier_anchored_sd/kv_gradient_subspace.py`
- Create: `tests/test_kv_gradient_subspace.py`

**Interfaces:**
- Consumes: gradients shaped `[layers, 2, batch, heads, tokens, head_dim]` plus example provenance.
- Produces: `KVGradientAccumulator.update(gradients)`, `KVGradientBasis.fit(accumulator, max_rank)`, `save(path)`, `load(path)`, and `project(cache, rank, beta)` metadata contracts.

- [ ] **Step 1: Write a failing explicit-SVD equivalence test**

```python
def test_gram_eigenvectors_span_explicit_right_singular_vectors():
    rows = torch.tensor([[3., 0.], [0., 2.], [1., 0.]], dtype=torch.float64)
    acc = KVGradientAccumulator(layers=1, heads=1, head_dim=2)
    acc.update_kind(layer=0, kind=0, head=0, rows=rows)
    basis = acc.fit(max_rank=2)
    _, _, vh = torch.linalg.svd(rows, full_matrices=False)
    assert torch.allclose(basis.projector(0, 0, 0, 2), vh.T @ vh, atol=1e-10)
```

- [ ] **Step 2: Run the test and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_kv_gradient_subspace.py -q`

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement FP64 accumulation and deterministic basis fitting**

```python
class KVGradientAccumulator:
    def update_kind(self, *, layer: int, kind: int, head: int, rows: Tensor) -> None:
        x = rows.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != self.head_dim or not torch.isfinite(x).all():
            raise ValueError("gradient rows must be finite [tokens, head_dim]")
        self.gram[layer, kind, head].add_(x.T @ x)
        self.row_counts[layer, kind, head] += x.shape[0]
        self.norm_sums[layer, kind, head] += torch.linalg.vector_norm(x).item()

def canonicalize_basis_sign(vectors: Tensor) -> Tensor:
    pivot = vectors.abs().argmax(dim=-2, keepdim=True)
    signs = torch.gather(vectors, -2, pivot).sign().clamp_min(0).mul(2).sub(1)
    return vectors * signs
```

- [ ] **Step 4: Add artifact validation tests**

Tests assert orthonormal columns, descending non-negative eigenvalues, nonzero coverage for all layer/kind/head cells, exact branch/model/dataset metadata, SHA checking on load, and rejection when a direct basis is supplied to the Full-Head branch.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_kv_gradient_subspace.py -q && .venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/verifier_anchored_sd/kv_gradient_subspace.py tests/test_kv_gradient_subspace.py
git commit -m "feat: add FP64 KV gradient subspace artifacts"
```

### Task 3: Direct adapter and receiver-space projector

**Files:**
- Create: `src/verifier_anchored_sd/spec_decode/direct_kv_adapter.py`
- Create: `src/verifier_anchored_sd/spec_decode/gradient_projector.py`
- Create: `tests/test_direct_kv_adapter.py`
- Create: `tests/test_gradient_projector.py`

**Interfaces:**
- Consumes: teacher `CacheState`, teacher/receiver model contracts, rotary factors, and a branch-matched `KVGradientBasis`.
- Produces: `direct_teacher_cache_to_receiver_content(...) -> CacheState` and `GradientProjectedKVMapper` implementing the existing mapper `map_cache(...)` protocol.

- [ ] **Step 1: Write failing direct-adapter geometry and rotary tests**

```python
def test_direct_adapter_preserves_values_and_rebases_keys():
    source = synthetic_position_cache(layers=2, heads=2, tokens=3, dim=4)
    converted = direct_teacher_cache_to_receiver_content(source, matching_contracts())
    assert converted.keys_are_content
    assert torch.equal(converted.layers[0].value, source.layers[0].value)
    assert converted.rotary is None

def test_direct_adapter_rejects_head_dimension_mismatch():
    with pytest.raises(ValueError, match="head dimension"):
        direct_teacher_cache_to_receiver_content(source, mismatched_contracts())
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_direct_kv_adapter.py tests/test_gradient_projector.py -q`

Expected: collection fails because the modules do not exist.

- [ ] **Step 3: Implement checked direct conversion**

Use existing `CacheState.to_content_space()`/rotary helpers. Compare exact layer, KV-head, head-dimension, tokenizer-hash, prompt positions, and model revision metadata before returning content-space keys and unchanged values.

- [ ] **Step 4: Implement and test hard/soft projection**

```python
def project_rows(x: Tensor, basis: Tensor, beta: float) -> Tensor:
    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must lie in (0, 1]")
    projected = (x @ basis) @ basis.transpose(-1, -2)
    return projected if beta == 1.0 else x + beta * (projected - x)
```

Tests cover identity at full rank, rank-restricted output, soft interpolation,
orthogonal complement, deterministic random controls, and content-space-only K projection.

- [ ] **Step 5: Implement mapper-compatible wrapper**

`GradientProjectedKVMapper(base_mapper, basis, branch, rank, beta, family)` calls the Full-Head mapper for `full_head`, calls the direct adapter for `direct`, validates artifact provenance, projects every K/V head, and returns receiver position-space cache through existing RoPE utilities.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_direct_kv_adapter.py tests/test_gradient_projector.py -q && .venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/verifier_anchored_sd/spec_decode/direct_kv_adapter.py src/verifier_anchored_sd/spec_decode/gradient_projector.py tests/test_direct_kv_adapter.py tests/test_gradient_projector.py
git commit -m "feat: project direct and mapped teacher KV in receiver space"
```

### Task 4: AIME answer-gradient fitting CLI

**Files:**
- Create: `bench/fit_aime_kv_gradient_subspace.py`
- Create: `tests/test_aime_gradient_fit_contract.py`
- Modify: `bench/common.py`

**Interfaces:**
- Consumes: one frozen fit JSONL, branch name, model revisions, optional Full-Head mapper, output artifact path, and low-VRAM residency limits.
- Produces: one branch/fold `.pt` basis, metadata JSON, atomic progress JSON, and terminal failure/status JSON.

- [ ] **Step 1: Write failing CLI/contract tests**

```python
def test_full_head_branch_requires_mapper(tmp_path):
    with pytest.raises(ValueError, match="mapper"):
        validate_fit_args(branch="full_head", mapper=None)

def test_direct_branch_rejects_mapper(tmp_path):
    with pytest.raises(ValueError, match="direct"):
        validate_fit_args(branch="direct", mapper=tmp_path / "mapper.pt")
```

Also test that fit/eval row digests overlap is rejected and that BF16/model revisions appear in artifact metadata.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_aime_gradient_fit_contract.py -q`

Expected: collection or imports fail because the fit module is absent.

- [ ] **Step 3: Add reusable 4B cached teacher-forcing helper**

Extend `bench/common.py` with a narrowly scoped function:

```python
def receiver_answer_loss_with_cache(
    model, *, prompt_cache: CacheState, full_ids: Sequence[int],
    answer_positions: Sequence[int], device: str,
) -> torch.Tensor:
    """Return mean NLL only at answer_positions while retaining cache gradients."""
```

The function constructs `past_key_values` using existing cache adapters,
feeds the teacher-forced continuation, gathers shifted logits at exact answer
positions, and returns their mean cross-entropy.

- [ ] **Step 4: Implement branch-specific gradient capture loop**

For every fit example: capture 8B prompt KV, construct direct or Full-Head
content-space 4B cache leaves, run the 4B answer loss, call `backward()`, flatten
each prompt-cache gradient to `[tokens, 128]`, update the FP64 accumulator,
clear gradients, and atomically record completed example IDs and norms.

- [ ] **Step 5: Fit PCA control statistics in the same pass**

Accumulate centered activation Gram matrices separately from gradient Gram
matrices. Store gradient and PCA bases under distinct artifact families with
the same branch/fold provenance.

- [ ] **Step 6: Run CPU contract tests and one-example GPU smoke**

Run: `.venv/bin/python -m pytest tests/test_aime_gradient_fit_contract.py -q && .venv/bin/python -m pytest -q`

Then run one AIME2025 fit example for `direct` and one for `full_head` with
exact BF16. Assert nonzero finite gradients for all 36×2×8 cells and save smoke metadata.

- [ ] **Step 7: Commit**

```bash
git add bench/common.py bench/fit_aime_kv_gradient_subspace.py tests/test_aime_gradient_fit_contract.py
git commit -m "feat: fit AIME answer-gradient KV bases"
```

### Task 5: Paired AIME transfer evaluation

**Files:**
- Create: `bench/eval_aime_kv_subspace.py`
- Create: `src/verifier_anchored_sd/aime_subspace_metrics.py`
- Create: `tests/test_aime_subspace_metrics.py`
- Create: `tests/test_aime_eval_contract.py`

**Interfaces:**
- Consumes: held-out rows, both branch artifacts, Full-Head mapper, rank/beta matrix, model contracts, and decoding seed.
- Produces: per-example metrics, paired summaries/intervals, method completeness validation, and final diagnostic classification.

- [ ] **Step 1: Write failing completeness and paired-metric tests**

```python
def test_every_method_has_every_example_once():
    rows = synthetic_rows(methods=("native", "direct", "mapped"), examples=6)
    assert_complete_matrix(rows, expected_methods=3, expected_examples=6)
    with pytest.raises(ValueError, match="duplicate"):
        assert_complete_matrix(rows + [rows[0]], expected_methods=3, expected_examples=6)

def test_lower_answer_nll_is_positive_improvement():
    paired = paired_answer_nll(projected=[1.0, 2.0], baseline=[2.0, 3.0], samples=1000, seed=0)
    assert paired.mean_improvement == 1.0
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_aime_subspace_metrics.py tests/test_aime_eval_contract.py -q`

Expected: collection fails because evaluation modules do not exist.

- [ ] **Step 3: Implement method matrix and provenance checks**

Generate the exact methods from the spec for both branches. Require branch,
fold, dataset revision, fit/eval row digests, model revisions, mapper SHA, and
basis SHA to match before model loading. Seed random controls from
`sha256(branch + fold + layer + kind + head)`.

- [ ] **Step 4: Implement per-example evaluation**

For each held-out problem and method, compute answer-span NLL, gold-token
probability, 8B-vs-4B `1-TV`, top-1 agreement, greedy three-digit exact match,
and optional block acceptance. Write progress after every method/example row.

- [ ] **Step 5: Implement paired aggregation**

Aggregate cross-fit rows only after verifying each AIME2025 example occurs in
exactly one held-out fold. Report paired bootstrap intervals by example for
projected-minus-unprojected metrics and exact-match paired counts. Preserve all
rows regardless of sign.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_aime_subspace_metrics.py tests/test_aime_eval_contract.py -q && .venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add bench/eval_aime_kv_subspace.py src/verifier_anchored_sd/aime_subspace_metrics.py tests/test_aime_subspace_metrics.py tests/test_aime_eval_contract.py
git commit -m "feat: evaluate AIME receiver-gradient KV subspaces"
```

### Task 6: Reproducible suite orchestration

**Files:**
- Create: `scripts/run_aime_kv_subspace.sh`
- Create: `tests/test_aime_suite_shell.py`
- Create: `docs/AIME_KV_SUBSPACE_RUNBOOK.md`

**Interfaces:**
- Consumes: pinned dataset/model revisions, Full-Head mapper paths, GPU residency, and profile.
- Produces: five-fold direct/full-head artifacts, cross-fit results, cross-year artifacts/results, logs, statuses, and a final manifest.

- [ ] **Step 1: Write failing shell-contract test**

The test invokes `bash -n`, asserts `set -euo pipefail`, verifies all pinned
revisions are passed explicitly, checks both branches and all five folds are
enumerated, and confirms each long command has progress/failure/status paths.

- [ ] **Step 2: Run test and verify missing-script failure**

Run: `.venv/bin/python -m pytest tests/test_aime_suite_shell.py -q`

Expected: fails because `scripts/run_aime_kv_subspace.sh` does not exist.

- [ ] **Step 3: Implement pilot and full profiles**

`pilot` runs one smoke example per branch, then the complete AIME2025 cross-fit
primary rows at rank 64 before the secondary matrix. `full` additionally runs
cross-year generalization and block acceptance. Both profiles continue after
negative metrics and stop only on invalid provenance, nonfinite outputs,
missing rows, or unrecoverable runtime failure.

- [ ] **Step 4: Write runbook**

Document exact environment variables, commands, expected artifacts, GPU
residency-only recovery, resume semantics, and interpretation rules. Include
commands that verify JSON completeness and SHA256 values before publishing.

- [ ] **Step 5: Run shell, focused, and full tests**

Run: `bash -n scripts/run_aime_kv_subspace.sh`

Run: `.venv/bin/python -m pytest tests/test_aime_suite_shell.py -q && .venv/bin/python -m pytest -q`

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_aime_kv_subspace.sh tests/test_aime_suite_shell.py docs/AIME_KV_SUBSPACE_RUNBOOK.md
git commit -m "feat: orchestrate AIME KV subspace experiments"
```

### Task 7: Run, validate, and publish the experiment

**Files:**
- Create: `results/aime_kv_subspace_2026-09-11/**`
- Create locally: `artifacts/aime_kv_subspace_2026-09-11/**`

**Interfaces:**
- Consumes: completed suite code, frozen dataset files, model cache, Full-Head mapper.
- Produces: complete small JSON results/provenance on GitHub; large KV shards and basis tensors remain local with published hashes.

- [ ] **Step 1: Materialize and validate frozen data**

Run `bench/prepare_aime_kv_subspace.py` at the pinned revision. Verify 30
AIME2025 rows, five balanced folds, no fit/eval digest overlap, and expected
2022-2024 rows.

- [ ] **Step 2: Run one-example direct and Full-Head smoke fits/evals**

Verify every layer/head has finite nonzero gradient, basis columns are
orthonormal, direct and Full-Head artifacts have distinct branch contracts,
and both evaluation paths emit finite metrics.

- [ ] **Step 3: Run AIME2025 five-fold cross-fit**

Run `SUITE_PROFILE=pilot bash scripts/run_aime_kv_subspace.sh`. Monitor GPU,
RAM, disk, atomic progress, and terminal status. Preserve any OOM failure and
adjust only target/draft residency before resuming.

- [ ] **Step 4: Run cross-year generalization and secondary controls**

After cross-fit integrity passes, run `SUITE_PROFILE=full`. Do not select a
method on AIME2025. Complete direct and Full-Head branches even when one is
negative.

- [ ] **Step 5: Validate final artifacts**

Check every JSON parses, all numeric metrics are finite, every expected
method/example pair appears once, fold membership is disjoint, source/model/
mapper/basis hashes match, and no progress file is mistaken for a final result.

- [ ] **Step 6: Run final regression suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all tests pass after the exact code used for GPU results.

- [ ] **Step 7: Publish and verify remote bytes**

Commit/push code, tests, runbook, small results, status, and provenance to
`feat/student-readable-kv-subspace`. Fetch the remote branch and compare
SHA256 for every published result. Do not push large KV shards or model/basis
binaries.
