# Qwen3 verifier-to-draft pair screen — 2026-09-07

## Decision

Neither candidate is sufficiently close to native draft state for verifier-refresh
experiments. Both completed all 128 preregistered held-out rows, and both confidence
intervals are wholly below the `A_transfer > 0.95` boundary.

| Verifier -> draft | Mean `A_transfer` | Bootstrap 95% CI | P05 | Top-1 agreement | Decision |
|---|---:|---:|---:|---:|---|
| Qwen3-8B -> Qwen3-4B | 0.800360 | [0.766157, 0.832042] | 0.494013 | 0.8125 | fail |
| Qwen3-4B -> Qwen3-1.7B | 0.600986 | [0.549008, 0.649657] | 0.034002 | 0.640625 | fail |

Qwen3-8B -> Qwen3-4B is materially better than 4B -> 1.7B on the same prefixes:
the paired mean difference is 0.199374 with a bootstrap 95% interval of
[0.148097, 0.250628]. Architecture alignment helped, but it did not produce
near-lossless transfer.

The preregistered consequence is:

- do not run HellaSwag confirmation for either pair;
- do not interpret speculative expected-MAL or refresh deltas for either mapper;
- do not run Accepted-Only Refresh as a confirmatory method experiment;
- retain the native-frontier implementation and use it only after a future pair
  passes the distribution screen.

## Metrics

| Metric | 8B -> 4B | 4B -> 1.7B |
|---|---:|---:|
| Median `A_transfer` | 0.832785 | 0.645708 |
| Minimum `A_transfer` | 0.010841 | 0.0000029 |
| Mean KL(native || mapped) | 0.398420 | 1.238085 |
| Mean next-token NLL delta | +0.295181 | +1.426215 |
| Mean per-row attention-output cosine | 0.816571 | 0.742118 |
| Minimum per-row attention-output cosine | 0.558357 | 0.515143 |
| Evaluation elapsed time | 249.67 s | 225.89 s |

The very low tail values matter: a mean near 0.8 hides prefixes on which mapped
state almost completely changes the draft distribution. The HellaSwag gate cannot
repair this distribution-level failure.

## Frozen protocol

Both directional mappers used exact BF16 weights and the same calibration tokens:

```text
128 x 1,024-token FineWeb-Edu windows
stride 4 -> 32,768 observations
matched head, content-space key mapping
k = 8
ridge lambda = 0.01
R² selection on 32 sequences
```

The held-out screen used 128 independent 1,024-token prefixes. Each comparison was
between native draft prefill and mapped verifier history through token `t-1`
followed by a native draft forward at frontier token `t`. The gate used 10,000
percentile-bootstrap samples with seed 0.

Calibration input:

```text
artifacts/pair_screen_2026-09-07/inputs/fineweb_edu_calibration.jsonl
SHA-256 685ece4471f351018308ec5b712223eb185de4e0264fdca33f7631d35263d885
```

Evaluation input:

```text
data/fineweb_edu_heldout_offset4096.jsonl
SHA-256 5a62cda38d879827cc07c015b87179878faa80c70e1f482be497dc9c879bc16c
```

The files are different FineWeb-Edu record ranges. The artifact layer also checked
all frozen token-row digests and found no row overlap.

## Model and code revisions

```text
Qwen/Qwen3-8B   b968826d9c46dd6066d109eabc6255188de91218
Qwen/Qwen3-4B   1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
repository      3ee991d2786da4bf0222a5906e6f5ea402ac170a
```

Environment:

```text
GPU          NVIDIA RTX A4000, 16,376 MiB
OS           Ubuntu 22.04 / Linux 5.15 x86_64
Python       3.10
PyTorch      2.14.0+cu130
CUDA runtime 13.0
Transformers 5.16.1
Datasets     5.0.1
Accelerate   1.14.0
KVBridge     0.2.0 at 0d75f31dcde6eeceaa609d3affed6ca1401deb77
```

The 8B capture used Accelerate exact-weight CPU offload with a 14GiB GPU cap. Peak
observed allocation from `nvidia-smi` was approximately 13.6GB. No quantization was
used.

## Execution

The reproducible entry point is:

```bash
export CALIBRATION_TEXT=artifacts/pair_screen_2026-09-07/inputs/fineweb_edu_calibration.jsonl
export EVAL_TEXT=data/fineweb_edu_heldout_offset4096.jsonl
export GPU_MEMORY_GIB=14
bash scripts/run_pair_screen.sh
```

The run executed these phase interfaces for each directional pair:

```bash
python bench/capture_sequential_calibration.py --role source \
  --sequences 128 --seq-len 1024 --stride 4 --dtype bfloat16 --gpu-memory-gib 14 ...
python bench/capture_sequential_calibration.py --role draft \
  --sequences 128 --seq-len 1024 --stride 4 --dtype bfloat16 --gpu-memory-gib 14 ...
python bench/fit_sequential_mapper.py --k 8 --lambda 0.01 \
  --selection-ridge 0.000001 --selection-sequences 32 \
  --accumulation-device cuda --selection-layer-block 4 --fit-layer-block 4 ...
python bench/capture_screen_prefixes.py \
  --prompts 128 --prefix-tokens 1024 --dtype bfloat16 --gpu-memory-gib 14 ...
python bench/eval_pair_transfer.py \
  --prompts 128 --bootstrap-samples 10000 --threshold 0.95 \
  --dtype bfloat16 --gpu-memory-gib 14 --attention-cosine ...
```

The native Qwen3-4B calibration tensors are identical for primary-draft and
stress-control-source roles. To reduce GPU work and disk use, those 128 tensors were
moved between pair artifact directories, their role metadata was atomically changed
from `draft` to `source`, and the source capture CLI then revalidated every tensor,
token digest, model revision, sequence ID, and shape before writing completion.

Before the formal run, a one-row integration probe used 4B -> 1.7B, one 256-token
calibration row, `k=1`, and one 64-token held-out prefix. It completed the entire
capture-fit-map-evaluate path and was explicitly labeled `scientific=false`. Its
`A_transfer=0.129988` is not scientific evidence.

The first probe attempt exposed missing local Triton build dependencies. Installing
`build-essential` and `python3.10-dev` resolved the compiler and `Python.h` errors;
the exact command then completed on CUDA.

## Artifact hashes and retention

The primary mapper checkpoint is:

```text
artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
SHA-256 c1e388f652ceb071bd0087f6f1e7cf057530a43004c293e456bdfc3e88f6cd80
```

The stress-control mapper checkpoint is:

```text
artifacts/pair_screen_2026-09-07/qwen3_4b_to_1p7b/mapper.pt
SHA-256 def56f57c36499cff051440a2c8444e20c921c28aa57fd0f7345ecd21b64dea4
```

Machine-readable results:

```text
results/pair_screen_2026-09-07/qwen3_8b_to_4b.json
SHA-256 8e94dac188013b1eae35182af7675750c8026890715046c314758ad70eb6468c

results/pair_screen_2026-09-07/qwen3_4b_to_1p7b.json
SHA-256 8be2a3e4af138d8007b24f6bf3e59950560ad5fb5850d7570bf8ebcf3af57901
```

`artifact_inventory.json` in each pair directory records hashes for mapper,
manifests, frozen tokens, completion markers, and result JSON. Large cache shards
were removed after complete result hashes were recorded because the 100GB host
cannot retain both pairs' approximately 27GB of reconstructible caches. Frozen
tokens, contracts, mapper checkpoints, inventories, and row-level result JSON were
retained.

## Next permitted experiment

Search for another target-to-draft pair or a substantially stronger translator and
rerun the same distribution gate. Qwen3-14B -> Qwen3-32B results from prior work do
not establish the required reverse 32B -> 14B quality, and that pair is not feasible
on this 16GB host without more aggressive sequential/offload resources.

Accepted-Only Refresh remains implemented and unit tested. It becomes scientifically
eligible only after a mapper yields mapped/native expected-MAL retention of at least
0.90, and its distribution screen should first pass the stricter `A_transfer` gate.
