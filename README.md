# Verifier-Anchored Speculative Decoding

## Current next run: paper-faithful mapper baselines vs student-readable subspace

The current 8B -> 4B evidence is mixed: mapped verifier KV improves verifier top-1 agreement but degrades full-distribution overlap/KL/NLL. Before attributing that failure to the student decoder, the repository now compares the current matched-head mapper against the original Cross-Model KV paper's **Full-Head ridge** and its **independent nonlinear MLP** baseline, then tests whether an explicit 4B-readable KV subspace still adds value.

The complete five-split protocol is in:

```text
docs/NEXT_STAGE_PAPER_VS_SUBSPACE.md
```

Direct 32GB/48GB runner:

```bash
git fetch origin
git checkout feat/student-readable-kv-subspace
git pull --ff-only origin feat/student-readable-kv-subspace

python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,kvbridge,dev]'

export CALIBRATION_TEXT=/data/A_calibration.jsonl
export MAPPER_EVAL_TEXT=/data/B_mapper_selection.jsonl
export SUBSPACE_TEXT=/data/C_subspace_fit.jsonl
export EVAL_TEXT=/data/D_final_eval.jsonl
# export E2_TEXT=/data/E_block_acceptance.jsonl  # only needed after a frozen winner
export SUITE_PROFILE=pilot
export GPU_MEMORY_GIB=28       # 32GB; use 44 on a 48GB card
export METHOD_BATCH_SIZE=8     # use 16 on 48GB if desired
bash scripts/run_paper_vs_subspace_suite.sh
```

The runner evaluates:

```text
Native 4B
Matched-head ridge: k = 8,12,16,20
Paper Full-Head ridge: k = 8,12,16,20
Paper nonlinear MLP on the best Full-Head k
Student-readable grad / signed-benefit subspaces
PCA / random / orthogonal / negative-benefit controls
Speculative block acceptance only after a frozen deployment-valid winner
```

All final mapper/subspace comparisons are repeated on the same held-out split D. The primary metric is verifier/proposal overlap:

```text
A_target = 1 - TV(p_8B, q_4B_method)
```

A subspace claim requires paired document-cluster bootstrap evidence against native 4B, the strongest linear mapper, and—when present—the paper MLP. Pilot MLP results are capacity diagnostics only; `SUITE_PROFILE=paper` enforces the paper's 500 x 1024 calibration, stride 4, 1024-1024 MLP, Adam 1e-3, 20 epochs, batch 4096, and MSE.

---

This repository studies a narrow question in dual-model speculative decoding:

> Can verifier state be translated into the draft model so that we both avoid
> redundant draft prefill and improve the draft proposal distribution by anchoring
> its historical cache to verifier-computed state?

The target/verifier is always exact. Cross-model translation changes only the draft
proposal state.

## Core system

Standard speculative decoding redundantly prefills the prompt twice:

```text
prompt -> verifier prefill -> verifier KV
prompt -> draft prefill    -> draft KV
```

The bridge removes the second prefill:

```text
prompt -> verifier prefill -> verifier KV -> M_target_to_draft -> draft KV
```

During generation the verifier also materializes exact KV for accepted proposal
tokens. The verifier-anchored idea asks whether those accepted historical states can
replace draft-generated historical KV.

The causal frontier must remain native to the draft:

```text
C_draft(t) = [ M(C_verifier(1:t-1)) ; KV_draft_native(t) ]
```

This avoids the old state inconsistency where logits were produced from a native
frontier but the next block consumed a mapped frontier.

## Current evidence

The audited 2026-09-07 native-fidelity screen found:

| verifier -> draft | mean A_transfer | bootstrap 95% CI | top-1 | KL(native||mapped) |
|---|---:|---:|---:|---:|
| Qwen3-8B -> Qwen3-4B | 0.800360 | [0.765924, 0.833559] | 0.8125 | 0.398420 |
| Qwen3-4B -> Qwen3-1.7B | 0.600986 | [0.523788, 0.669038] | 0.640625 | 1.238085 |

where

```text
A_transfer = 1 - TV(q_native_draft, q_mapped_draft)
```

This proves that 8B -> 4B is materially more transferable than 4B -> 1.7B under the
same matched-head ridge protocol, but neither is near-lossless native-state
reconstruction.

**Important correction:** native reconstruction is not the primary speculative-
decoding gate. SD cares about verifier/proposal overlap.

## Current scientific gate: verifier-target alignment

For one fixed prefix `x_1:t`, compute:

```text
p_T      = verifier next-token distribution
q_native = native draft next-token distribution
q_mapped = mapped verifier history + native draft frontier distribution
```

Then:

```text
A_target_native = 1 - TV(p_T, q_native)
A_target_mapped = 1 - TV(p_T, q_mapped)
Delta_target = A_target_mapped - A_target_native
```

Use 10,000 document-cluster paired-bootstrap samples:

```text
95% CI low  > 0 -> support      -> go_sd
95% CI high < 0 -> harm         -> stop_pair
CI crosses 0    -> inconclusive -> expand held-out prefixes
```

`A_transfer > 0.95` remains a useful native-fidelity diagnostic. It cannot veto SD
when target alignment significantly improves.

The design is frozen in:

```text
docs/superpowers/specs/2026-09-07-target-alignment-gate-design.md
```

The direct testing handoff is:

```text
docs/NEXT_STAGE_TARGET_ALIGNMENT.md
```

## Stage A: reuse the existing Qwen3-8B -> Qwen3-4B mapper

Do **not** refit the mapper. This isolates the metric correction from translator
changes.

Expected local mapper artifact:

```text
artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
```

32GB GPU:

```bash
export EVAL_TEXT=/path/to/frozen_pair_selection.jsonl
export E2_TEXT=/path/to/third_disjoint_e2.jsonl   # optional until go_sd
export GPU_MEMORY_GIB=28
bash scripts/run_target_alignment_next.sh
```

48GB GPU:

```bash
export EVAL_TEXT=/path/to/frozen_pair_selection.jsonl
export E2_TEXT=/path/to/third_disjoint_e2.jsonl
export GPU_MEMORY_GIB=44
bash scripts/run_target_alignment_next.sh
```

The script runs a non-scientific 4-prefix smoke, then a scientific 128 x 1,024
screen. It captures verifier KV and FP32 verifier next-token probabilities in a
fresh artifact root, so old cache-only screen shards cannot contaminate the run.

## Stage B: Qwen3-32B -> Qwen3-14B

If Stage A returns `stop_pair`, test the reverse large-to-small direction directly.
Do not infer 32B -> 14B quality from prior 14B -> 32B results.

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

Fixed scientific settings:

```text
exact BF16 weights, no automatic quantization
sequential model loading
128 x 1,024 calibration windows
stride = 4 -> 32,768 observations
32 layer-selection sequences
matched-head content-space ridge
k = 8
ridge lambda = 0.01
selection ridge = 1e-6
128 x 1,024 held-out prefixes
FP32 verifier probability artifacts
10,000 document-cluster bootstrap samples
```

Qwen3-32B exact BF16 requires CPU offload on 32/48GB single-GPU hosts. Recommend at
least 96 GiB host RAM; 128 GiB is preferred. OOM is an incomplete result, not a
reason to silently quantize or change the pair.

## Native-frontier E2

Only `decision.status=go_sd` may enter E2. `run_native_frontier_e2.sh` compares
native SD, mapped/native-frontier initialization, accepted-only historical refresh,
and legacy controls. The newest causal frontier remains native in the primary
methods.

## Student-readable subspace experiment

The dedicated subspace design and earlier direct runner remain available at:

```text
docs/NEXT_STAGE_STUDENT_READABLE_SUBSPACE.md
scripts/run_student_readable_subspace.sh
```

Use the new paper-vs-subspace suite above for reviewer-safe baseline comparison.
