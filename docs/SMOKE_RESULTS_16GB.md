# 16GB A4000 smoke results

These are feasibility results from an NVIDIA RTX A4000 (16,376 MiB), **not paper
results**. The run used Qwen3-4B -> Qwen3-1.7B in BF16 with the explicit
`--low-vram` profile. That profile placed up to 6 GiB of target weights and 4 GiB
of draft weights on GPU and offloaded the remaining target layers to CPU.

## E0 mapper smoke

The intended Full-Head configuration was `500 x 1024`, stride 4, `k=8`, R² layer
selection, and `lambda=0.01`. An 8-sequence `k=8` smoke and an 8-sequence `k=4` R²
smoke were started, but the CPU R² selector/final normal-equation solves were too
slow and were stopped before producing a checkpoint.

The only completed checkpoint used:

- 8 sequences x 1024 tokens, stride 4;
- `k=4`;
- depth-based source-layer selection;
- low-VRAM model placement;
- CPU fitting;
- about 470 MB KVBridge artifact / 940 MB runtime mapper.

That checkpoint is suitable for runtime/API smoke only. Never use it for paper
mapper, acceptance, or systems claims.

### Post-smoke diagnosis

This was primarily an **execution-path problem**, not proof that 16GB is too small
for formal fitting. After cache capture, both LLMs were already deleted and CUDA
memory was released, yet the expensive R² selection and ridge statistics were sent
to CPU. The current code therefore separates capture residency from fit residency:
16GB capture may offload model weights, but after capture the selector and normal
equations reuse the freed CUDA device.

A matched-head centered-ridge baseline was also added after CacheBridge
(arXiv:2609.00891) made clear that Full-Head mapper size/application cost is itself a
strong modern baseline issue. For Qwen3-4B -> Qwen3-1.7B at `k=8`, matched-head uses
1024 input features per output head versus 8192 for Full-Head, reducing affine
weights from about 469.8M to 58.7M.

## E1 prefill smoke

The 256-2048 measurements used 1 warmup and 3 repetitions. The 4096 measurement
used 0 warmups and 1 repetition because CPU offload made this configuration slow.

| prompt length | reported bridge speedup | bridge peak VRAM |
| ---: | ---: | ---: |
| 256 | 1.115x | 12.27 GB |
| 512 | 1.207x | 12.57 GB |
| 1024 | 1.099x | 13.22 GB |
| 2048 | 1.107x | 14.36 GB |
| 4096 | 1.529x | 14.94 GB |

The 4K sign is encouraging, and an earlier placement OOM at 15.55 GiB was avoided
by the 6/4 GiB split. However, the old script formed its native baseline from the
**sum of separately measured medians**. That is not a clean G0 statistic. The
current E1 directly times the complete native initialization and complete bridge
initialization, synchronizes the actual CUDA shard under offload, and sweeps batch
1/2/4 on 16GB. Therefore all speedup numbers above must be rerun before publication.

## E2 correctness smoke

One FineWeb-Edu prompt, 256 prompt tokens, 16 generated tokens, `gamma=4`:

| method | MAL | acceptance rate | elapsed | output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native SD | 1.000 | 0.250 | 21.61 s | 0.740 |
| Ridge Init-only | 0.800 | 0.200 | 25.07 s | 0.638 |
| Ridge + Refresh | 1.222 | 0.306 | 22.69 s | 0.705 |

Refresh > Init-only has the **right sign**, and the pending-frontier path completed
without an exception. It is still only an integration signal: `N=1`, the mapper is
non-paper, and realized MAL is stochastic.

The current E2 therefore adds conditional expected accepted length
`sum_j prod_{i<=j}(1-TV(p_i,q_i))` and a paired bootstrap on identical held-out
prompts. The 16GB kill test uses 64 independent prompts x 64 generated tokens; the
24GB confirmatory run uses 200 prompts x 512 generated tokens.

## Current decision

The direction is **not disproven**, but it is not established. The one-prompt
Refresh gain and 4K bridge sign justify one stronger kill test, not phase-2 training.
The key question is now whether Refresh still improves paired expected acceptance
when the translator is a much cheaper matched-head mapper.

Use:

```bash
export HF_HUB_CACHE=/path/with/free/space
export EVAL_TEXT=/path/to/disjoint_heldout_prompts.jsonl
bash scripts/run_16gb_next.sh
```

The script fits a 128-sequence `k=8` R²-selected matched-head mapper, performs CUDA
post-capture fitting, tries fully resident 16GB E1/E2 first, sweeps E1 batch 1/2/4,
and falls back to controlled offload only if resident inference cannot fit.

Do not train the acceptance residual until G0, G1, and G2 in
`docs/NEXT_EXPERIMENTS.md` pass.