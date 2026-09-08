# Six-Hour Method Debug Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use one fixed Qwen3-8B → Qwen3-4B mapper to debug every speculative-decoding cache policy for six GPU-hours without turning exploratory measurements into a scientific claim.

**Architecture:** The batch keeps the verifier, draft, mapper, tokenizer, dtype, and prompt source fixed. It exercises native SD, mapped initialization with a native frontier, accepted-only historical refresh, and legacy full refresh through small reproducible pilots, then runs longer-generation and resource diagnostics. Formal E2 remains gated by the target-alignment result and is not run while that result is `expand`.

**Tech Stack:** Python 3.10, PyTorch 2.14 CUDA, Transformers 5.16, KVBridge 0.2.0, existing `QwenPairRuntime`, JSON progress artifacts, and the existing 8B→4B BF16 mapper.

**Spec:** `docs/superpowers/specs/2026-09-07-target-alignment-gate-design.md`

## Global Constraints

- Reuse mapper `artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt`; never refit it in this batch.
- Keep Qwen3-8B as verifier and Qwen3-4B as draft; use exact BF16 weights and the pinned model revisions.
- Use `data/fineweb_edu_heldout_offset4096.jsonl` only as a debug prompt source; it is not an independent E2 source.
- Label every output `scientific=false` or `diagnostic=true`; do not report it as formal E2 evidence.
- Do not bypass `validate_e2_artifacts` for a confirmatory result. The formal 128-row screen is `expand`.
- Any cache/tokenizer/vocabulary/probability mismatch, NaN/Inf, OOM, or provenance failure stops that arm and writes its failure artifact.
- Keep each arm in a separate output directory and record GPU, cap, commit, mapper SHA, input SHA, and elapsed time.

---

### Task 1: Freeze the debug contract and align generation-drift coverage

**Files:**
- Modify: `bench/eval_generation_drift.py`
- Create: `tests/test_generation_drift_contract.py`
- Modify: `docs/RUN_REPORT_TEMPLATE.md`

**Interfaces:**
- `eval_generation_drift.py` must construct `QwenPairRuntime` with `init_mode` in `{native, legacy_mapped, mapped_native_frontier}` and `refresh_policy` in `{none, full, accepted_only}`.
- The debug default must be the audited Qwen3-8B → Qwen3-4B pair; the existing 4B → 1.7B default remains a separate stress control.
- The debug runner must emit `config`, `scientific=false`, per-method rows, per-position MAL buckets, and a failure sidecar on OOM.

- [ ] Add a unit test that checks the 8B→4B defaults and the explicit four-policy matrix; retain compatibility aliases but do not use them in new runs.
- [ ] Run `pytest tests/test_generation_drift_contract.py -q` and confirm it fails before the coverage change.
- [ ] Map public labels to `native_sd`, `mapped_init_only`, `mapped_accepted_only`, and `legacy_full_refresh`; retain the old 4B→1.7B invocation only as a named stress control.
- [ ] Run the focused test again and confirm it passes.
- [ ] Run `ruff check bench/eval_generation_drift.py tests/test_generation_drift_contract.py`.
- [ ] Commit with `fix: align generation drift debug runner with runtime API`.

### Task 2: Run the target-alignment expansion diagnostic

**Files:**
- Use: `scripts/run_target_alignment_next.sh`
- Use: `data/fineweb_edu_heldout_offset4096.jsonl`
- Use: existing mapper and metadata from the prior pair-screen worktree
- Create: `results/debug_batch_2026-09-07/target_alignment_256.json`

**Interfaces:**
- Input contract: same mapper SHA `c1e388f652ceb071bd0087f6f1e7cf057530a43004c293e456bdfc3e88f6cd80`, same evaluation input, BF16, `GPU_MEMORY_GIB=12`.
- Output contract: 256 completed windows if available, document-cluster bootstrap, `decision.status`, and no mapper changes.

- [ ] Run the existing 4-prefix smoke with the fixed mapper and record it separately.
- [ ] Run a 256-window target-alignment diagnostic using the same frozen source and a fresh artifact root.
- [ ] If fewer than 512 windows exist, record `expand_unavailable` with the exact count; do not reuse calibration rows.
- [ ] Compare `mean_delta_target_alignment`, CI, target NLL, target KL, and native-fidelity diagnostics against the existing 128-row result.

### Task 3: Exercise the full five-method acceptance matrix on a debug split

**Files:**
- Use: `bench/eval_acceptance_pilot.py`
- Use: the completed 4-row smoke result as the diagnostic screen contract
- Create: `results/debug_batch_2026-09-07/e2_gamma{1,4,8}.json`

**Interfaces:**
- Methods must remain the existing matrix: `native_sd`, `legacy_mapped_init_only`, `mapped_init_only`, `legacy_full_refresh`, and `mapped_accepted_only`.
- Use `PROMPTS=4`, `PROMPT_TOKENS=256`, `NEW_TOKENS=32`, and one gamma per run (`1`, `4`, `8`).

- [ ] Run each gamma with `LOW_VRAM=1` and separate output/progress paths.
- [ ] Verify every method has four complete prompts before reading summaries.
- [ ] Check expected accepted length, realized accepted length, bonus rate, throughput, and peak VRAM.
- [ ] Compare `mapped_accepted_only` against `mapped_init_only` and `legacy_full_refresh` against `legacy_mapped_init_only`; label all confidence intervals exploratory because the source is not an independent E2 set.
- [ ] Stop immediately on a contract error rather than changing model, dtype, gamma, or mapper inside a run.

### Task 4: Run long-generation causal-frontier drift diagnostics

**Files:**
- Use: repaired `bench/eval_generation_drift.py`
- Create: `results/debug_batch_2026-09-07/generation_drift.json`

**Interfaces:**
- Use four prompts, 256-token prompt prefix, 128 generated tokens, gamma 4, BF16, and `--low-vram`.
- Compare position buckets `1–64`, `65–128`, `129–256`, and `257–512`; buckets beyond generated length remain null.

- [ ] Run native, mapped-init/native-frontier, accepted-only, and legacy full-refresh arms with identical seeds.
- [ ] Verify `frontier_kinds` and cache sequence lengths remain causal after every block.
- [ ] Record where accepted length or MAL begins to diverge; this is a debugging trace, not a paper metric.

### Task 5: Run resource and determinism checks

**Files:**
- Use: `bench/eval_speedbench.py`
- Use: `tests/` and existing artifact validators
- Create: `results/debug_batch_2026-09-07/resource_profile.json`

**Interfaces:**
- Use the same four prompt rows, new tokens 64, gamma 4, and caps 10 and 12 GiB in separate runs.

- [ ] Run each cache policy twice with the same seed and compare token outputs and accepted-length traces byte-for-byte.
- [ ] Record peak VRAM, CPU offload usage, elapsed time, output tokens per second, and OOM status.
- [ ] Confirm changing only the GPU cap never changes tokenization, mapper SHA, model revision, or output contract.
- [ ] Stop the cap arm if OOM occurs and preserve the failure artifact.

### Task 6: Assemble the six-hour debug report

**Files:**
- Create: `results/debug_batch_2026-09-07/report.json`
- Create: `docs/DEBUG_BATCH_2026-09-07.md`

**Interfaces:**
- The report must list each arm, start/end time, elapsed seconds, exact command/config, artifact paths, completion counts, mapper/input/model hashes, and failure status.
- The report must separate `runtime_bug`, `cache_provenance_issue`, `resource_issue`, and `behavioral_observation`.

- [ ] Recompute summaries from raw rows rather than copying console output.
- [ ] Mark the formal Stage A result as `expand` and the debug matrix as `diagnostic`.
- [ ] State the only permitted scientific next step: obtain additional independent held-out clusters and rerun target alignment; do not run confirmatory E2 from this batch.
- [ ] Run `pytest -q`, `python -m compileall -q src bench training`, all relevant `bash -n` checks, and `git diff --check`.
- [ ] Commit the report and runner repair separately from any future scientific result.

## Six-hour allocation

| Window | Arm | Purpose | Scientific status |
|---|---|---|---|
| 0:00–0:20 | contracts + smoke | catch API/cache regressions | diagnostic |
| 0:20–1:10 | target alignment, 256 windows | reduce current `expand` uncertainty | exploratory |
| 1:10–3:10 | five methods × gamma 1/4/8 | debug acceptance and refresh transitions | diagnostic |
| 3:10–4:20 | long-generation drift | find causal-frontier failures | diagnostic |
| 4:20–5:20 | caps 10/12 GiB + repeats | isolate resource/determinism issues | diagnostic |
| 5:20–6:00 | report + replay checks | seal evidence and failures | diagnostic |

The batch must not silently proceed from an inconclusive target-alignment screen to confirmatory E2. A future E2 requires a third disjoint prompt source and a `go_sd` screen result.
