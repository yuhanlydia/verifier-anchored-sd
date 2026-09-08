# Student-Readable KV Subspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable gradient/benefit/PCA KV subspace fitting, intervention sweeps, causal controls, and a direct GPU runner for Qwen3-8B -> Qwen3-4B without changing the existing mapper.

**Architecture:** A standalone `kv_subspace.py` owns basis artifacts and cache interventions. `subspace_stats.py` accumulates 128-D per-head sufficient statistics. A sequential GPU fitter uses verifier cache/probability artifacts plus the frozen 4B decoder to build bases. A separate evaluator applies a frozen method matrix on disjoint prefixes and produces paired-bootstrap comparisons to both native and full-mapped baselines. Deployment-valid mapped-only interventions can optionally be wrapped around the mapper for a later speculative acceptance matrix.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, existing `CacheState` / `RidgeKVMapper` / target-alignment artifact contracts, pytest, bash.

**Spec:** `docs/superpowers/specs/2026-09-08-student-readable-kv-subspace-design.md`

## Global Constraints

- Keep Qwen3-8B, Qwen3-4B, tokenizer revisions, mapper SHA, BF16 model precision frozen.
- Do not refit the mapper in the primary experiment.
- Keys are fitted/intervened in RoPE-free content space; values in value space.
- The newest causal frontier token is always a native 4B forward.
- Basis-fit rows, intervention-eval rows, and E2 rows are disjoint.
- Rank sweep is `4,8,16,32,64` with head dimension 128.
- Primary gradient objective is `KL(p_target || q_4B)`.
- No automatic quantization, model substitution, rank reduction, or dtype fallback on OOM.

---

### Task 1: Basis artifact and cache intervention primitives

**Files:**
- Create: `src/verifier_anchored_sd/kv_subspace.py`
- Create: `tests/test_kv_subspace.py`

**Interfaces:**
- Produces `SubspaceBasisArtifact`, `BasisFamily`, `InterventionSpec`.
- Produces `project_cache(...)`, `intervene_mapped_cache(...)`, `intervene_delta_cache(...)`.
- Later tasks consume the exact basis tensor layout `[layers, 2, heads, head_dim, max_rank]`.

- [ ] **Step 1: Write failing tests** for orthonormal basis validation, hard projection, soft endpoint identity, delta endpoint identity, random rank matching, key content-space round trip, and benefit effective-rank clipping.
- [ ] **Step 2: Run** `pytest -q tests/test_kv_subspace.py` and confirm import/behavior failures.
- [ ] **Step 3: Implement minimal primitives.** Store FP32 bases/eigenvalues and immutable provenance metadata. For a cache `X`, compute projection by `coeff = X @ U_r; proj = coeff @ U_r.T` per layer/head/kind. Convert keys to content space before intervention and reapply the original rotary factors afterward.
- [ ] **Step 4: Run** `pytest -q tests/test_kv_subspace.py` and require pass.
- [ ] **Step 5: Commit** `feat: add KV subspace basis and intervention primitives`.

### Task 2: Streaming sufficient statistics and eigensolvers

**Files:**
- Create: `src/verifier_anchored_sd/subspace_stats.py`
- Create: `tests/test_subspace_stats.py`

**Interfaces:**
- Produces `GradientCovarianceStats`, `BenefitCrossStats`, `ActivationCovarianceStats`.
- Produces `sensitivity_basis(...)`, `benefit_basis(...)`, `pca_basis(...)` returning orthonormal eigenvectors and eigenvalues.

- [ ] **Step 1: Write failing analytic tests.** Use 2-D synthetic examples where the top gradient direction is known, the benefit matrix has one positive and one negative eigenvector, centered PCA has a known principal axis, and benefit rank clipping excludes non-positive eigenvalues.
- [ ] **Step 2: Run** `pytest -q tests/test_subspace_stats.py` and confirm failures.
- [ ] **Step 3: Implement streaming accumulators** without storing per-token rows. Use FP64 CPU accumulators for sums/cross-products and `torch.linalg.eigh` on symmetric 128x128 matrices.
- [ ] **Step 4: Run** the focused tests and then `pytest -q`.
- [ ] **Step 5: Commit** `feat: add streaming subspace sufficient statistics`.

### Task 3: Gradient / benefit / PCA basis fitting CLI

**Files:**
- Create: `bench/fit_student_readable_subspace.py`
- Create: `tests/test_subspace_fit_contract.py`
- Modify: `src/verifier_anchored_sd/experiment_artifacts.py` only if an additional exact-row overlap helper is required.

**Interfaces:**
- Consumes screen artifact directory produced by `capture_screen_prefixes.py`, frozen mapper + metadata, and the 4B draft model.
- Produces `student_readable_subspace.pt` plus `.json` metadata containing mapper SHA, screen manifest SHA, basis-fit input SHA, token-row digests, gradient objective, counts, nonzero-gradient diagnostics, and basis artifact SHA.

- [ ] **Step 1: Write failing contract tests** for mapper/revision mismatch, old cache-only screen rejection, missing target probabilities, non-disjoint rows, zero/NaN gradient smoke, and artifact metadata round-trip.
- [ ] **Step 2: Run focused tests** and confirm RED.
- [ ] **Step 3: Implement the fitter.** For each prefix: compute native 4B history and mapped 4B history; create leaf historical K/V tensors with gradients; compute native target-KL gradient for benefit stats and mapped target-KL gradient for sensitivity stats; inverse-RoPE key gradients/deltas; update gradient, benefit, delta/activation sufficient statistics; clear graph per prefix.
- [ ] **Step 4: Add 4-prefix smoke mode** that requires finite nonzero gradient norms for every layer/head/KV kind before full fitting.
- [ ] **Step 5: Solve and save** `grad`, `benefit_positive`, `benefit_negative`, and `pca` bases through max rank 64.
- [ ] **Step 6: Run focused tests and full `pytest -q`.**
- [ ] **Step 7: Commit** `feat: fit student-readable KV subspaces from 4B gradients`.

### Task 4: Intervention matrix evaluator

**Files:**
- Create: `bench/eval_subspace_interventions.py`
- Create: `src/verifier_anchored_sd/subspace_metrics.py`
- Create: `tests/test_subspace_metrics.py`

**Interfaces:**
- Consumes disjoint target cache/probability screen artifacts, mapper, subspace artifact.
- Produces one JSON containing every intervention row, aggregated method table, paired bootstrap differences vs native and full-mapped, and a frozen deployment-valid winner candidate.

- [ ] **Step 1: Write failing metric tests** for paired method-vs-baseline bootstrap, winner eligibility, tie-break order, and incomplete-row withholding.
- [ ] **Step 2: Run focused tests** and confirm RED.
- [ ] **Step 3: Implement method generator** for ranks `4,8,16,32,64` and:
  - `full_mapped`
  - `grad_hard`
  - `grad_soft(beta=0.25,0.5,0.75)` plus hard/full endpoints without duplicate forwards
  - `benefit_hard`
  - `benefit_soft(beta=0.25,0.5,0.75)`
  - `pca_hard`
  - `projected_delta(alpha=0.25,0.5,1.0,1.5)` for grad and benefit bases
  - `delta_shrink(beta=0,0.25,0.5,0.75)` for grad and benefit bases, with beta=1 reused as full-mapped
  - `random_hard` and `random_delta`
  - `grad_orthogonal`
  - `benefit_negative`
- [ ] **Step 4: Implement evaluation loop.** Compute native 4B cache once per prefix, mapped cache once, then one native-frontier incremental forward per intervention. No verifier model is loaded during evaluation.
- [ ] **Step 5: Report per-method** target overlap, target KL, target top-1 agreement, next-token NLL, effective rank, beta/alpha, and latency. Use document-cluster paired bootstrap against both native and full-mapped.
- [ ] **Step 6: Winner selection.** Deployment-valid winner must use mapped-only intervention (`hard` or `soft` projection), beat full-mapped in target overlap with CI low >0, and beat native with CI low >0. Delta methods are reported as mechanistic upper bounds but cannot be selected for prefill-skip E2.
- [ ] **Step 7: Run focused and full tests.**
- [ ] **Step 8: Commit** `feat: evaluate student-readable KV intervention matrix`.

### Task 5: Speculative acceptance integration for mapped-only winners

**Files:**
- Create: `src/verifier_anchored_sd/spec_decode/subspace_mapper.py`
- Create: `tests/test_subspace_mapper.py`
- Modify: `bench/eval_acceptance_pilot.py` to accept an optional subspace artifact + intervention spec, without changing existing behavior when absent.

**Interfaces:**
- Produces `SubspaceMappedKVMapper`, a wrapper exposing `.map(...)` compatible with `RidgeKVMapper` and applying a frozen mapped-only subspace intervention after mapping.
- E2 adds `subspace_mapped_init_only` and `subspace_mapped_accepted_only` when a subspace spec is supplied.

- [ ] **Step 1: Write failing wrapper tests** proving `beta=1` is identical to base mapper, hard projection changes only historical mapped KV, and accepted-only refresh also passes through the same frozen intervention.
- [ ] **Step 2: Run focused tests** and confirm RED.
- [ ] **Step 3: Implement wrapper** and optional E2 CLI flags `--subspace-artifact`, `--subspace-family`, `--subspace-rank`, `--subspace-beta`.
- [ ] **Step 4: Add E2 provenance binding** so the selected method exactly matches the intervention-evaluation winner JSON if `--winner-result` is supplied.
- [ ] **Step 5: Run focused and full tests.**
- [ ] **Step 6: Commit** `feat: add subspace-filtered speculative acceptance methods`.

### Task 6: Direct GPU runner, experiment handoff, and CI checks

**Files:**
- Create: `scripts/run_student_readable_subspace.sh`
- Create: `docs/NEXT_STAGE_STUDENT_READABLE_SUBSPACE.md`
- Create: `docs/SUBSPACE_RUN_REPORT_TEMPLATE.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Runner accepts `SUBSPACE_TEXT`, `EVAL_TEXT`, optional `E2_TEXT`, existing mapper path, and 32GB/48GB memory profile.

- [ ] **Step 1: Write the direct workflow**:
  1. CPU tests;
  2. capture 4-prefix subspace smoke artifacts;
  3. gradient smoke;
  4. capture B at 64 prefixes x 512 by default, then fit bases;
  5. capture C at 128 prefixes x 1024;
  6. run full intervention matrix;
  7. if a deployment-valid winner exists and `E2_TEXT` is provided, run the acceptance matrix on D;
  8. otherwise stop with the preregistered reason.
- [ ] **Step 2: Default fit settings**: `SUBSPACE_PROMPTS=64`, `SUBSPACE_PREFIX_TOKENS=512`, `MAX_RANK=64`, objective `target_kl`. Allow scaling to 128 x 1024 only as a confirmatory rerun.
- [ ] **Step 3: Add fail-fast debug contract** from the design doc, result inventory hashes, and no automatic quantization/dtype/rank fallback.
- [ ] **Step 4: Add CI shell syntax check** for the new runner.
- [ ] **Step 5: Run full verification**: `python -m compileall -q src bench training`, `pytest -q`, `bash -n scripts/run_student_readable_subspace.sh`.
- [ ] **Step 6: Commit** `docs: add direct student-readable subspace experiment protocol`.

## Final verification checklist

- [ ] Full CPU suite green.
- [ ] New runner shell syntax green.
- [ ] RED/GREEN evidence exists for every production module.
- [ ] Existing target-alignment workflow remains behaviorally unchanged without subspace flags.
- [ ] No old target-alignment or E2 result is rewritten as new evidence.
- [ ] GPU/model integration is explicitly left to the 32GB/48GB test host and is not claimed by CI.
