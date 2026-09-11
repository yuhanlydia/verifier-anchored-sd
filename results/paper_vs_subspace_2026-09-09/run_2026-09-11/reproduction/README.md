# 2026-09-11 pilot reproduction notes

This run is still in progress. Progress snapshots and smoke diagnostics are not final D or E results.

## Frozen inputs

Use the repository root as the working directory. The archived `prepare.py` records how the five JSONL files were generated. It originally resolved the dataset revision at runtime. For an exact reproduction, replace that lookup with the revision recorded in `../data_manifest.json`:

```python
rev = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
```

Copy the preparation script into `data/paper_vs_subspace_2026-09-11/` before running it; it writes beside itself and refuses to overwrite existing split files. Compare all five generated file SHA256 values with `../data_manifest.json`. Do not substitute fresh dataset HEAD, reuse B as D, or pool selection and evaluation rows.

## Suite invocation

Use code containing the exploratory selection change (`335de06`) and the bounded-memory MLP validation fix (`f5f01a2`). Model revisions and BF16 remain those specified in `docs/NEXT_STAGE_PAPER_VS_SUBSPACE.md`.

```bash
export CALIBRATION_TEXT="$PWD/data/paper_vs_subspace_2026-09-11/A.jsonl"
export MAPPER_EVAL_TEXT="$PWD/data/paper_vs_subspace_2026-09-11/B.jsonl"
export SUBSPACE_TEXT="$PWD/data/paper_vs_subspace_2026-09-11/C.jsonl"
export EVAL_TEXT="$PWD/data/paper_vs_subspace_2026-09-11/D.jsonl"
export E2_TEXT="$PWD/data/paper_vs_subspace_2026-09-11/E.jsonl"
export PYTHON="$PWD/.venv/bin/python"
export SUITE_PROFILE=pilot
export SELECTION_POLICY=exploratory
export GPU_MEMORY_GIB=20
export METHOD_BATCH_SIZE=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
bash scripts/run_paper_vs_subspace_strict.sh
```

The actual test GPU is an RTX 3090 with 24 GiB. The 20 GiB runtime cap is recorded explicitly; the original suggested 44 GiB cap assumed a 48 GiB GPU. This is a pilot with 32 MLP training sequences and two epochs, not a full paper-profile reproduction. Compiler tooling was installed after the first capture attempt failed. The subsequent MLP failure and memory fix are recorded under `../attempt_02_mlp_validation_oom/`.

Performance thresholds are advisory for this run. The suite retains metrics, confidence intervals, strict-gate diagnostics and negative results, and freezes an eligible mapped-only candidate for exploratory E evaluation. Data isolation, artifact hashes, model identity and numerical validity remain required.

## Resume and storage conditions

The current process resumed at MLP training after the eight ridge fits and B evaluations had completed. After successful MLP training and its B evaluation, the large A and B KV shards were removed to recover disk space. Their file inventories are archived in `../calibration_shard_inventory.json` and `../B_shard_inventory.json`; raw split files, manifests, tokens, learned mapper checkpoints and result JSON remain on the test machine.

Do not blindly restart an earlier training or B-evaluation stage against these pruned directories. Reconstruct the relevant shards from the frozen split first, or resume only at a stage whose required artifacts are present. Keep D shards until every final D baseline has finished. Large KV shards and model checkpoints are not uploaded to GitHub; their small metadata and hashes are retained.
