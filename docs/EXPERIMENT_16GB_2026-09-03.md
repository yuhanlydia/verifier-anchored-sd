# 16GB matched-head kill test — 2026-09-03

## Outcome

This run is a **NO-GO** for Phase 2 under the preregistered gates.

- G0 could not be evaluated at the required 4K/8K resident points because batch 1
  was already OOM. At 512/1K/2K, the directly timed bridge speedup was about 1.26x.
- G1 failed: mapped Init-only expected-MAL retention was 0.716625, below 0.80.
- G2 failed in the opposite direction: Refresh minus Init-only expected MAL was
  -0.199519, paired-bootstrap 95% CI [-0.249085, -0.149204].
- Phase 2 was not run.

## Code and environment

- Branch: `fix/paper-faithful-mapper-runtime`
- Starting commit: `5b4b1b6b6417b5dda5b78ed2ead7041949c76195`
- GPU: NVIDIA RTX A4000, 16,376 MiB
- NVIDIA driver: 580.82.09
- Python: 3.10.12
- PyTorch: 2.14.0+cu130
- CUDA runtime used by PyTorch: 13.0
- Transformers: 5.16.1
- Datasets: 5.0.1
- Accelerate: 1.14.0
- KVBridge: 0.2.0, repository-pinned commit
  `0d75f31dcde6eeceaa609d3affed6ca1401deb77`

The machine required `build-essential` and `python3.10-dev` for the first Triton
CUDA driver JIT. Before the full run, a real Qwen3 CUDA forward was used to verify
the toolchain.

## Invocation

```bash
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export EVAL_TEXT=/root/verifier-anchored-sd/data/fineweb_edu_heldout_offset4096.jsonl
bash scripts/run_16gb_next.sh
```

## Data isolation

E0 used the default streaming `HuggingFaceFW/fineweb-edu` `sample-10BT` source.
The held-out file was generated from dataset revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, skipping the first 4,096 source
rows and then retaining 128 documents of at least 4,000 characters. E0 asks the
loader for at most the first 1,024 source rows, so the E2 source indices are
disjoint by construction.

## E0 — matched-head ridge mapper

- Target: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`
- Draft: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Tokenizer hash:
  `59ecfb3dfee770c00883902baad7168e85734be6db5ae2441239bbe75c4e29ae`
- Head mode/backend: matched / matched-head-centered-ridge
- Content-space key mapping: enabled
- Calibration: 128 sequences x 1,024 tokens
- Stride: 4
- Final fitting observations: 32,768
- R2 layer-selection sequences: 32
- Selection observations: 8,192
- Selected source layers per draft layer: 8 (`k=8`)
- Feature width: 1,024
- Ridge lambda: 0.01
- Accumulation: CUDA float32
- Fit layer block: 8
- Model capture: controlled low-VRAM offload, BF16
- Mapper file size: 235,113,639 bytes

The exact selected layers and selection scores are stored in
`checkpoints/qwen3_4b_to_1p7b_matched_16gb.pt.json`.

## E1 — resident BF16 initialization benchmark

- Lengths: 512, 1,024, 2,048, 4,096, 8,192
- Batch sizes: 1, 2, 4
- Warmups: 5
- Repetitions: 20
- Seed: 0
- Direct native timing: target prefill + draft prefill in one timed operation
- Bridge timing: target prefill + mapper
- Mapper dtype: BF16
- Continue on OOM: enabled

Batch-1 results:

| Context | Native median | Bridge median | Speedup | Result |
|---:|---:|---:|---:|---|
| 512 | 0.226224 s | 0.180196 s | 1.255432x | completed |
| 1,024 | 0.428861 s | 0.339538 s | 1.263074x | completed |
| 2,048 | 0.878973 s | 0.697942 s | 1.259378x | completed |
| 4,096 | — | — | — | OOM |
| 8,192 | — | — | — | OOM |

Capacity boundary:

- 512: batches 1/2/4 completed.
- 1,024: batches 1/2 completed; batch 4 OOM.
- 2,048: batch 1 completed; batches 2/4 OOM.
- 4,096 and 8,192: all requested batches OOM.

## E2 — 64-prompt resident BF16 acceptance test

- Prompts: 64 independent held-out documents
- Prompt tokens: 512
- New tokens: 64
- Gamma: 4
- Methods: Native SD, Ridge Init-only, Ridge Refresh
- Mapper dtype: BF16
- Per-prompt runtime seed: prompt index, shared across methods
- Paired bootstrap samples: 5,000
- Confidence level: 95%

Aggregate results:

| Method | Expected MAL | Realized MAL | Acceptance rate | Tokens/s |
|---|---:|---:|---:|---:|
| Native SD | 2.082077 | 2.106303 | 0.526576 | 5.176404 |
| Ridge Init-only | 1.492069 | 1.499546 | 0.374887 | 4.054573 |
| Ridge Refresh | 1.292550 | 1.310022 | 0.327506 | 3.571801 |

Paired results:

| Contrast | Mean difference | 95% CI |
|---|---:|---:|
| Refresh - Init, expected MAL | -0.199519 | [-0.249085, -0.149204] |
| Refresh - Init, realized MAL | -0.189524 | [-0.275639, -0.102418] |
| Refresh - Native, tokens/s | -1.604604 | [-1.826336, -1.385082] |

## Gate decision

The mapper misses G1, and G2 is not merely inconclusive: both deterministic and
realized paired intervals are entirely below zero. Under the preregistered protocol,
do not run Phase 2 or the optional long-generation drift curve from this checkpoint.

Raw records:

- `results/e1_matched_16gb_resident.json`
- `results/e2_matched_16gb_resident.json`
- `checkpoints/qwen3_4b_to_1p7b_matched_16gb.pt.json`
- `data/fineweb_edu_heldout_offset4096.jsonl`
