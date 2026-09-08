# 下一阶段直接测试：Target-Alignment Gate

> 给远端测试 agent 的执行合同。不要自行改变模型、精度、数据、mapper 或 gate；出现下面的 STOP 条件时先提交 failure artifact，再停止。

## 0. 这次到底测试什么

旧 pair screen 问的是：

```text
mapped draft 是否仍然像 native draft？
A_transfer = 1 - TV(q_native, q_mapped)
```

这只能衡量 KV reconstruction fidelity。

Speculative decoding 真正关心的是：

```text
mapped draft 是否比 native draft 更接近 verifier target？
```

同一 prefix `x_1:t` 上定义：

```text
p_T      = verifier distribution
q_native = native draft distribution
q_mapped = mapped verifier history + native draft frontier distribution

A_target_native = 1 - TV(p_T, q_native)
A_target_mapped = 1 - TV(p_T, q_mapped)
Delta_target = A_target_mapped - A_target_native
```

### Primary scientific gate

对 `Delta_target` 做 10,000 次 **document-cluster paired bootstrap**：

```text
95% CI low  > 0   -> support -> go_sd
95% CI high < 0   -> harm    -> stop_pair
CI crosses 0      -> inconclusive -> expand to 512 prefixes
```

`A_transfer > 0.95` 只保留为 native reconstruction diagnostic，**不能单独阻止 speculative decoding**。

---

## 1. 拉代码并验证 CPU 逻辑

```bash
git fetch origin
git checkout feat/target-alignment-gate
git pull --ff-only origin feat/target-alignment-gate

python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,kvbridge,dev]'

python -m compileall -q src bench training
pytest -q
bash -n scripts/run_target_alignment_next.sh
bash -n scripts/run_32b_to_14b_pair_screen.sh
bash -n scripts/run_native_frontier_e2.sh
```

任何一个失败：**不要上 GPU**。先修测试/环境并提交日志。

---

# Stage A — 先重判已经跑过的 Qwen3-8B -> Qwen3-4B

## 2. 原则

**不要重新 fit mapper。**

必须继续使用已经跑出的 audited mapper：

```text
artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
```

这样本次和 2026-09-07 pair screen 的唯一科学变化是：

```text
old: q_native vs q_mapped
new: p_target vs q_native vs q_mapped
```

如果 mapper binary 不在当前测试机，脚本必须停止。不要自动重训替代。

## 3. 数据

`EVAL_TEXT` 必须是之前冻结的 held-out pair-selection 数据，与 mapper calibration 数据 disjoint。

推荐继续使用：

```text
data/fineweb_edu_heldout_offset4096.jsonl
```

如果要在 `go_sd` 后自动跑 E2，则 `E2_TEXT` 必须是**第三份独立数据**，不能复用 calibration，也不能复用本次 pair-selection `EVAL_TEXT`。

## 4. 32GB GPU

```bash
export EVAL_TEXT=/path/to/frozen_pair_selection.jsonl
export E2_TEXT=/path/to/third_disjoint_e2.jsonl   # 可先不设
export GPU_MEMORY_GIB=28
bash scripts/run_target_alignment_next.sh
```

## 5. 48GB GPU

```bash
export EVAL_TEXT=/path/to/frozen_pair_selection.jsonl
export E2_TEXT=/path/to/third_disjoint_e2.jsonl   # 可先不设
export GPU_MEMORY_GIB=44
bash scripts/run_target_alignment_next.sh
```

### Stage A 自动执行内容

1. 4-prefix × 256-token **non-scientific smoke**；
2. 新 verifier capture：128 × 1,024 prefixes；
3. 每行保存 verifier KV + FP32 `p_T`；
4. draft 侧计算 `q_native`；
5. mapped history + native frontier 计算 `q_mapped`；
6. 计算 native fidelity + target alignment；
7. 10,000 document-cluster bootstrap；
8. 输出 `decision.status`；
9. 如果 `expand` 且数据足够，自动扩到 512 prefixes；
10. 只有 `go_sd` 且设置了 `E2_TEXT` 才进入 native-frontier E2。

### Stage A 主要结果

```text
results/target_alignment_2026-09-07/qwen3_8b_to_4b_target_alignment.json
```

必须包含：

```text
native_fidelity.summary
target_alignment.summary
target_alignment.gate
decision.status
rows
hardware
mapper_checkpoint_sha256
mapper_metadata_sha256
```

---

# Stage A 决策

## A1. `decision.status = go_sd`

这说明即使 `A_transfer < 0.95`，mapped KV 仍显著把 draft proposal 拉近 verifier。

允许跑：

```text
native_sd
mapped_init_only          # mapped history + native frontier
mapped_accepted_only      # only accepted historical KV refreshed; frontier stays native
```

不再用旧 legacy full refresh 作为主方法。

若脚本没有自动跑 E2：

```bash
export SCREEN_RESULT=results/target_alignment_2026-09-07/qwen3_8b_to_4b_target_alignment.json
export MAPPER=artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
export EVAL_TEXT=/path/to/third_disjoint_e2.jsonl
export LOW_VRAM=0
bash scripts/run_native_frontier_e2.sh
```

32/48GB 对 8B+4B 应优先 resident；只有真实 OOM 才设：

```bash
export LOW_VRAM=1
```

但 offload wall-clock 不能和 resident paper speed number 混用。

## A2. `decision.status = stop_pair`

立即停止 8B->4B：

```text
不要调 refresh
不要跑 HellaSwag 试图救结果
不要训练 one-step-TV / block-acceptance residual
不要改 k/lambda 后把它冒充同一次 hypothesis test
```

下一步跑 Stage B：32B->14B。

## A3. `decision.status = expand`

只增加 held-out clusters / prefixes，不改 mapper，不改模型，不改 gate。

## A4. `decision.status = incomplete`

只 debug execution；不能解释 partial metrics。

---

# Stage B — Qwen3-32B -> Qwen3-14B directional screen

这是原 cross-model KV 文献中最有实证支持的 family pairing 的**反方向**：我们需要 large verifier -> small draft，所以必须自己测 `32B -> 14B`，不能把已有 `14B -> 32B` 结果当作证明。

## 6. 数据与硬件

需要：

```text
CALIBRATION_TEXT : frozen calibration data
EVAL_TEXT        : disjoint pair-selection data
```

推荐 host RAM：

```text
>= 96 GiB
128 GiB preferred
```

因为 Qwen3-32B exact BF16 在 32/48GB GPU 上需要 CPU offload。科学 screen **禁止自动量化**。

### 48GB GPU

```bash
export CALIBRATION_TEXT=/path/to/frozen_calibration.jsonl
export EVAL_TEXT=/path/to/disjoint_pair_selection.jsonl
export GPU_MEMORY_GIB=44
bash scripts/run_32b_to_14b_pair_screen.sh
```

### 32GB GPU

```bash
export CALIBRATION_TEXT=/path/to/frozen_calibration.jsonl
export EVAL_TEXT=/path/to/disjoint_pair_selection.jsonl
export GPU_MEMORY_GIB=28
bash scripts/run_32b_to_14b_pair_screen.sh
```

### Stage B fixed settings

```text
target: Qwen/Qwen3-32B
draft:  Qwen/Qwen3-14B
Hub revisions: resolved to immutable SHAs before run
weights: exact BF16
loading: sequential

calibration:
  128 x 1,024 tokens
  stride = 4
  32,768 observations
  selection sequences = 32
  matched-head, content-space K
  k = 8
  ridge lambda = 0.01
  selection ridge = 1e-6

screen:
  128 x 1,024 held-out prefixes
  verifier probabilities stored FP32
  10,000 document-cluster bootstrap
  primary gate = Delta_target CI
```

如果 Stage B `go_sd`，先保存 pair-quality 证据；不要把 32/48GB 单卡大量 CPU-offload 的 E2 wall-clock 当 paper systems number。机制 E2 可以后续在合适的 simultaneous exact-weight 配置上补。

---

# 7. Debug STOP Contract

下面任何一条出现时，测试 agent 必须**停止，不要“先调到能跑”为止”**。

### D1 Token mismatch

KV shard、target-probability shard、frozen token row 的：

```text
sequence_id
token_digest
next_token_id
```

必须一致。

### D2 Model revision mismatch

实际加载的 target/draft Hub SHA 必须与 mapper / artifact metadata 完全一致。

### D3 Tokenizer mismatch

Target/draft tokenizer hash 必须一致，且与 mapper calibration contract 一致。

### D4 Vocabulary mismatch

```text
len(p_T) == len(q_native) == len(q_mapped)
```

否则直接停止。

### D5 Invalid verifier probabilities

`p_T` 必须：

```text
FP32 storage
finite
non-negative
sum = 1 +/- 1e-5
temperature = 1.0
```

### D6 Frontier mismatch

三个路径必须预测同一个 `x_{t+1}`，且 mapped draft 的最新 `x_t` 必须通过 draft native forward materialize。

禁止重新引入：

```text
logit from native frontier
but persistent cache uses mapped frontier
```

### D7 Artifact contamination

旧 `pair_screen_2026-09-07/.../screen` 是 cache-only schema，不能混进新 target-alignment artifact root。

### D8 Data leakage

Calibration token-row digest 与 evaluation token-row digest 不能有 exact/partial overlap。

### D9 OOM

必须写：

```text
status=incomplete
phase
requested_rows
completed_rows
error
```

然后停止。

**禁止自动：**

```text
BF16 -> INT8/INT4
换模型
缩短 prefix
换 batch
改 GPU cap 后继续把结果当同一实验
```

### D10 Metric misuse

2026-09-03 旧 E2 JSON 里的 overlap-product expected-MAL 已经 withdrawn。

当前 runtime 中新的 conditional-on-sampled-proposal estimator 是另一套实现，可以继续作为 E2 secondary metric；**旧 JSON 数字和旧公式不能与新结果混用**。

---

# 8. 测试 agent 提交要求

每个科学 GPU run 结束后必须提交：

```text
result JSON
failure/progress JSON（如有）
artifact_inventory.json
RUN_REPORT.md
```

`RUN_REPORT.md` 至少写：

```text
repo commit SHA
GPU 型号 / VRAM
host RAM
CUDA / PyTorch / Transformers
exact target/draft revision SHA
mapper SHA256
mapper metadata SHA256
calibration input SHA256
evaluation input SHA256
requested/completed rows
peak GPU memory
decision.status
CI low/high
是否有 OOM / fallback
```

不要把可重建的大型 KV shards 提交到 GitHub；在写完 hash/inventory 以后可以清理。target probability shards 较小，建议保留用于复核。

---

# 9. 当前禁止事项

直到出现 `target_alignment= support`：

```text
NO acceptance residual training
NO GRPO/RL
NO block-acceptance optimization
NO confirmatory refresh claim
NO HellaSwag rescue
```

下一阶段只回答一个问题：

```text
Does translated verifier state move the draft toward the verifier distribution?
```

先把这个问题回答干净，再决定 verifier-anchored SD 是否继续。

## Multi-GPU math pilot

`bench/eval_math_accuracy.py` supports placing the verifier and draft on separate
GPUs. On a host with at least two devices, use for example:

```bash
python bench/eval_math_accuracy.py \
  --target-device cuda:0 --draft-device cuda:1 \
  --mapper artifacts/pair_screen_2026-09-07/qwen3_8b_to_4b/mapper.pt
```

The runtime transfers mapped verifier KV to the draft device at the cache
boundary. With one GPU, keep both devices as `cuda`; the low-VRAM CPU-offload
profile remains the fallback.
