# Next Stage — Paper-Faithful Mapper vs Student-Readable Subspace

## 1. 为什么必须补这组实验

当前 8B -> 4B 结果是混合的：mapped KV 让 4B 的 verifier top-1 agreement 从
0.7500 提升到 0.7969，但 target overlap / KL / NLL 变差。这证明 teacher-specific
signal 存在，但不能证明当前 matched-head ridge 已经是强 translator。

原 Cross-Model KV 方法的最终 ridge 不是 matched-head：每个 receiver KV head 会读取
selected source layers 的 **全部 source KV heads**。论文还在 ridge 失败 pair 上使用独立
nonlinear MLP：

```text
Linear(k * H * D, 1024)
ReLU
Linear(1024, 1024)
ReLU
Linear(1024, D)
```

每个 `(receiver layer, K/V, receiver head)` 参数独立，Adam, lr=1e-3, MSE,
20 epochs, batch=4096。

因此 reviewer-safe 的问题必须是：

```text
matched-head ridge 失败
    ↓
paper Full-Head ridge 能否修复？
    ↓
paper nonlinear MLP 能否修复？
    ↓
student-readable subspace 是否仍然带来额外收益？
```

如果不补前两项，subspace 的任何 gain 都可能只是 weak-baseline artifact。

---

## 2. 五份数据，禁止复用

必须准备五个冻结文件：

```text
A CALIBRATION_TEXT   mapper fit only
B MAPPER_EVAL_TEXT   k/support selection only
C SUBSPACE_TEXT       subspace basis fit only
D EVAL_TEXT           final subspace + Full-Head + MLP comparison
E E2_TEXT             speculative block acceptance, optional until winner exists
```

runner 首先检查五个文件 SHA256 互不相同；各 artifact 还继续检查 exact token-row overlap。

不要把 B 当最终结果：B 只用于选 k / strongest linear mapper。论文表中的主要比较应来自 D。

---

## 3. 固定模型

```text
verifier: Qwen/Qwen3-8B
revision: b968826d9c46dd6066d109eabc6255188de91218

draft: Qwen/Qwen3-4B
revision: 1cfa9a7208912126459214e8b04321603b3df60c

dtype: exact BF16
```

禁止自动 INT8/INT4、换模型、改 dtype 或在 OOM 后偷偷减 k。

---

## 4. Baseline matrix

### B0 — Native 4B

```text
q_native = 4B native prefix + native frontier
```

### B1 — Matched-head ridge

同一 A、同一 k、同一 lambda：

```text
receiver head h <- source head h only
```

扫：

```text
k = 8, 12, 16, 20
lambda = .01
content-space K
```

### B2 — Paper Full-Head ridge

最终 multi-layer fit 使用：

```text
receiver head h <- all source heads from selected source layers
input width = k * H * D
```

扫同样：

```text
k = 8, 12, 16, 20
lambda = .01
selection ridge = 1e-6
```

B 上按：

```text
max mean A_target = 1-TV(p_8B, q_mapped)
tie -> lower target KL
tie -> lower k
```

分别冻结：

```text
best_linear
best_full_head
```

### B3 — Paper nonlinear MLP

只在 `best_full_head` 的 selected layers / k 上训练，不对所有 k 重复训练。

`paper` profile 完全按论文：

```text
500 x 1024 calibration
stride 4
hidden 1024 -> 1024
Adam lr=1e-3
20 epochs
batch 4096
MSE
independent receiver heads
```

`pilot` profile 只用于先判断 capacity：

```text
128 x 1024 calibration capture
MLP uses 32 sequences
same hidden 1024 -> 1024
2 epochs
batch 1024
Adam lr=1e-3
MSE
```

**pilot 结果不能写成 paper reproduction。**

### B4 — Student-readable subspace

Subspace 基于 B 上冻结的 strongest **linear** mapper，而不是根据 D 改 mapper。

包含：

```text
grad sensitivity: E[g g^T]
benefit positive: -0.5 E[g Delta^T + Delta g^T]
PCA
random
benefit negative
grad orthogonal
```

rank：

```text
4,8,16,32,64
```

mapped-only hard/soft projection 是 deployment-valid；native-base delta 方法只作为机制 upper bound。

---

## 5. Primary metric

所有 mapper / subspace 方法统一使用：

```text
A_target = 1 - TV(p_8B, q_4B_method)
```

同时报告：

```text
KL(p_8B || q_method)
target top-1 agreement
next-token NLL
```

不能只以 top-1 判断成功。

---

## 6. 最关键的 D-split 判读

D 上 Full-Head、MLP、subspace winner 使用**完全相同的 verifier KV / target-prob rows**。

### Case A — Full-Head >> matched

结论：

```text
之前 8B->4B 的主要问题是 matched-head support 太限制，不能用旧结果否定 cross-model KV。
```

### Case B — MLP >> Full-Head，subspace 没额外收益

结论：

```text
主要问题是 nonlinear mapping / residual error placement，不足以支持 student-readable-subspace novelty。
```

### Case C — subspace > Full-Head 且 > MLP

若 paired document-cluster bootstrap：

```text
CI95_low[A_subspace - A_full_head] > 0
CI95_low[A_subspace - A_MLP] > 0
```

这是最强结果：

```text
nonlinear mapper 被动修复 error placement 仍不够；
显式使用 4B decoder geometry 选择 teacher directions 更有效。
```

### Case D — benefit+ 提升、benefit- 恶化

这是 signed mechanism evidence：

```text
4B gradient × teacher delta 的一阶几何可以区分 helpful / harmful transfer directions。
```

### Case E — PCA/random ~= grad/benefit

结论：

```text
只是低秩 regularization/compression，不是 student-specific readable subspace。
```

### Case F — 所有方法都打不过 native 4B

STOP：

```text
top-1 improvement 太局部，不足以形成有用的 full-distribution / speculative-decoding gain。
```

---

## 7. 先跑 pilot

32GB：

```bash
git fetch origin
git checkout feat/student-readable-kv-subspace
git pull --ff-only origin feat/student-readable-kv-subspace

export CALIBRATION_TEXT=/data/A_calibration.jsonl
export MAPPER_EVAL_TEXT=/data/B_mapper_selection.jsonl
export SUBSPACE_TEXT=/data/C_subspace_fit.jsonl
export EVAL_TEXT=/data/D_final_eval.jsonl
# E2_TEXT 可先不设
export SUITE_PROFILE=pilot
export GPU_MEMORY_GIB=28
export METHOD_BATCH_SIZE=8
bash scripts/run_paper_vs_subspace_suite.sh
```

48GB：

```bash
export CALIBRATION_TEXT=/data/A_calibration.jsonl
export MAPPER_EVAL_TEXT=/data/B_mapper_selection.jsonl
export SUBSPACE_TEXT=/data/C_subspace_fit.jsonl
export EVAL_TEXT=/data/D_final_eval.jsonl
export SUITE_PROFILE=pilot
export GPU_MEMORY_GIB=44
export METHOD_BATCH_SIZE=16
bash scripts/run_paper_vs_subspace_suite.sh
```

Pilot 默认：

```text
A: 128 x 1024, stride 4
matched/full-head k sweep: 8,12,16,20
B: 128 x 1024
MLP: selected k, 32 calibration sequences, 2 epochs
C: 64 x 512 subspace fit
D: 128 x 1024 final evaluation
bootstrap: 10,000
```

---

## 8. 只有 pilot 显示值得时，再跑 paper profile

```bash
export SUITE_PROFILE=paper
export GPU_MEMORY_GIB=44
bash scripts/run_paper_vs_subspace_suite.sh
```

Paper profile：

```text
A: 500 x 1024, stride 4
matched/full-head k sweep: 8,12,16,20
Full-Head uses all A for R² selection + final fit
MLP: exactly 500 sequences, 20 epochs, batch 4096
C/D settings unchanged unless preregistered before run
```

MLP artifact 很大。若 selected k 很高，建议 host RAM >= 64GB，最好 96GB+，并预留几十 GB 磁盘。

---

## 9. 输出

关键结果：

```text
results/paper_vs_subspace_2026-09-09/linear_mapper_selection.json
results/paper_vs_subspace_2026-09-09/mapper_selection_B/*.json
results/paper_vs_subspace_2026-09-09/subspace/qwen3_8b_to_4b_subspace_interventions.json
results/paper_vs_subspace_2026-09-09/subspace/final_D/paper_full_head.json
results/paper_vs_subspace_2026-09-09/subspace/final_D/paper_mlp.json
results/paper_vs_subspace_2026-09-09/subspace/paper_vs_subspace_final.json
```

由于 master runner 调用 subspace runner 时会把 RESULT_ROOT 切到其子目录，最终 D summary 位于 `subspace/` 下面，这是预期路径。

---

## 10. Debug / STOP

出现以下任何情况必须停止，不得自动改实验：

1. A/B/C/D/E 文件 SHA 相同；
2. exact token-row overlap；
3. source/draft revision / tokenizer / geometry mismatch；
4. Full-Head 不是 cross-head feature；
5. MLP 没有复用 best Full-Head 的 selected source layers；
6. MLP receiver heads 共享参数；
7. `paper` profile 不是 500×1024/stride4/20 epochs/batch4096/lr1e-3；
8. key 没在 content space fit/map；
9. causal frontier 不是 native 4B；
10. OOM 后自动减 k、减 hidden、改 dtype、量化；
11. 用 B 作为最终 paper comparison；
12. 用 E2 数据选择 subspace method。

GPU agent 应提交 result JSON、failure/progress JSON（如有）、GPU/RAM/版本信息和所有 mapper/basis SHA256。
