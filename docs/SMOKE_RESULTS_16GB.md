# 16GB A4000 smoke results

These are feasibility results from an NVIDIA RTX A4000 (16,376 MiB), not paper
results. The run used Qwen3-4B → Qwen3-1.7B in BF16 with the explicit `--low-vram`
profile. That profile places up to 6 GiB of target weights and 4 GiB of draft
weights on the GPU and offloads the remaining target layers to CPU.

## E0 mapper smoke

The paper configuration remains `500 × 1024`, stride 4, `k=8`, R² layer selection,
and `lambda=0.01`. An 8-sequence `k=8` smoke and an 8-sequence `k=4` R² smoke were
started, but the CPU R² selector/final normal-equation solves were too slow for this
machine and were stopped before producing a checkpoint.

The explicit non-paper smoke mode completed:

- 8 sequences × 1024 tokens, stride 4
- `k=4`
- `--layer-selection depth --low-vram --accumulation-device cpu`
- KVBridge artifact: about 470 MB
- runtime mapper: about 940 MB

The resulting checkpoint is suitable for runtime/API smoke only. Never use it for
the paper mapper or acceptance claims.

## E1 prefill smoke

The 256–2048 measurements used 1 warmup and 3 repetitions. The 4096 measurement
used 0 warmups and 1 repetition because CPU offload makes this configuration slow.

| prompt length | bridge speedup | bridge peak VRAM |
| ---: | ---: | ---: |
| 256 | 1.115× | 12.27 GB |
| 512 | 1.207× | 12.57 GB |
| 1024 | 1.099× | 13.22 GB |
| 2048 | 1.107× | 14.36 GB |
| 4096 | 1.529× | 14.94 GB |

The 4096 result is a feasibility signal for G0, not the required 20-warmup/100-
repetition paper measurement. An earlier 4096 run with the old single-budget
placement OOMed at 15.55 GiB; the split 6/4 GiB placement completed it.

## E2 correctness smoke

One FineWeb-Edu prompt, 256 prompt tokens, 16 generated tokens, `gamma=4`:

| method | MAL | acceptance rate | elapsed | output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native SD | 1.000 | 0.250 | 21.61 s | 0.740 |
| Ridge Init-only | 0.800 | 0.200 | 25.07 s | 0.638 |
| Ridge + Refresh | 1.222 | 0.306 | 22.69 s | 0.705 |

Refresh is higher than Init-only in this smoke, and the pending-frontier path
completed without an exception. This is only an integration signal: E2/G1/G2 need
the disjoint held-out pilot (at least 200 prompts, 512 new tokens) on a 24GB
machine before drawing a scientific conclusion.

## Next run

Use `HF_HUB_CACHE` consistently so model files are not duplicated:

```bash
HF_HUB_CACHE=/path/with/at-least-40GB-free \
python bench/fit_ridge_calibration.py \
  --target Qwen/Qwen3-4B --draft Qwen/Qwen3-1.7B \
  --sequences 500 --seq-len 1024 --stride 4 --k 8 --lambda 0.01 \
  --calibration-dir artifacts/e0_qwen3_4b_to_1p7b_calibration \
  --kvbridge-artifact artifacts/e0_qwen3_4b_to_1p7b_kvbridge \
  --output checkpoints/qwen3_4b_to_1p7b_ridge.pt
```

Do not train the acceptance residual until this formal E0, the full E1 wall-clock
measurement, and the 200-prompt E2 pilot are available.
