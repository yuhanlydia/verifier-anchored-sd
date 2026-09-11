# 直接跑：Paper Mapper vs Student-Readable KV Subspace

> 2026-09-11：性能关卡已改为参考。subspace runner 默认
> `SELECTION_POLICY=exploratory`，会保留严格判定和全部指标，并把 D 上冻结的
> 最佳可部署候选送入独立 E 验证，不要求通过 CI 或提升阈值。
> 探索性结果不会标成严格通过；详见 `NEXT_STAGE_PAPER_VS_SUBSPACE.md` 顶部更新。

> 测试 agent 请按这个文件执行。不要自行改模型、k、dtype、MLP hidden size、数据 split 或 gate。

## 1. 拉代码

```bash
git fetch origin
git checkout feat/student-readable-kv-subspace
git pull --ff-only origin feat/student-readable-kv-subspace

python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,kvbridge,dev]'

python -m compileall -q src bench training
pytest -q
bash -n scripts/run_paper_vs_subspace_strict.sh
```

任何 CPU/shell check 失败：先停止，不上 GPU。

## 2. 准备 4+1 份数据

必须是不同的冻结文件：

```text
A CALIBRATION_TEXT   mapper calibration
B MAPPER_EVAL_TEXT   choose k / strongest linear mapper
C SUBSPACE_TEXT       fit 4B-readable basis
D EVAL_TEXT           final paper mapper vs subspace comparison
E E2_TEXT             optional; block acceptance only after a frozen winner
```

推荐使用 FineWeb-Edu 的互不重叠 document/offset 区间生成 A/B/C/D/E。不要只把同一个文件复制成不同名字；runner 会先检查文件 SHA，artifact contract 还会检查 exact token-row overlap。

## 3. 32GB — 先跑 pilot

```bash
export CALIBRATION_TEXT=/data/A_calibration.jsonl
export MAPPER_EVAL_TEXT=/data/B_mapper_selection.jsonl
export SUBSPACE_TEXT=/data/C_subspace_fit.jsonl
export EVAL_TEXT=/data/D_final_eval.jsonl
# export E2_TEXT=/data/E_block_acceptance.jsonl   # 可以先不设

export SUITE_PROFILE=pilot
export GPU_MEMORY_GIB=28
export METHOD_BATCH_SIZE=8

bash scripts/run_paper_vs_subspace_strict.sh
```

strict wrapper 自动固定：

```text
A = 128 x 1024, stride 4
matched layer selection = all 128 A rows
Full-Head layer selection = all 128 A rows
k = 8,12,16,20
B = 128 x 1024
paper-architecture MLP pilot = selected k, 32 A rows, 2 epochs
C = 64 x 512
D = 128 x 1024
bootstrap = 10,000
```

## 4. 48GB — pilot

```bash
export CALIBRATION_TEXT=/data/A_calibration.jsonl
export MAPPER_EVAL_TEXT=/data/B_mapper_selection.jsonl
export SUBSPACE_TEXT=/data/C_subspace_fit.jsonl
export EVAL_TEXT=/data/D_final_eval.jsonl

export SUITE_PROFILE=pilot
export GPU_MEMORY_GIB=44
export METHOD_BATCH_SIZE=16

bash scripts/run_paper_vs_subspace_strict.sh
```

如果 selected-k MLP checkpoint 加 4B receiver 在 GPU 上真实 OOM，**不要自动缩 hidden/k/dtype**。提交 failure artifact；可以在下一次预注册 run 中显式把 MLP mapper evaluation 放 CPU，但不要把两种 wall-clock 混在 systems claim 里。

## 5. 只有 pilot 值得时才跑 paper profile

建议优先在 48GB + >=96GB host RAM 上跑：

```bash
export SUITE_PROFILE=paper
export GPU_MEMORY_GIB=44
export METHOD_BATCH_SIZE=16
bash scripts/run_paper_vs_subspace_strict.sh
```

paper profile 固定：

```text
A = 500 x 1024, stride 4
matched selection = all 500
Full-Head selection = all 500
k sweep = 8,12,16,20
paper MLP = best Full-Head k
MLP = Linear(kHD,1024)-ReLU-Linear(1024,1024)-ReLU-Linear(1024,128)
Adam lr=1e-3
20 epochs
batch=4096
MSE
receiver layer/head/KV parameters independent
```

## 6. 实验矩阵

同一个 8B verifier / 4B receiver：

```text
Native 4B
Matched-head ridge k=8,12,16,20
Paper Full-Head ridge k=8,12,16,20
Paper nonlinear MLP at best Full-Head k
Grad sensitivity subspace
Signed benefit-positive subspace
PCA control
Random rank-matched control
Gradient orthogonal control
Benefit-negative control
Hard / soft mapped projection
Projected-delta / delta-shrink mechanism upper bounds
```

主指标：

```text
A_target = 1 - TV(p_8B, q_method)
```

同时必须看：

```text
KL(p_8B || q_method)
target top-1 agreement
next-token NLL
```

## 7. 最终如何判断

### Full-Head > matched

在 selection data 完全匹配后仍成立：说明 cross-head support 本身重要；旧 matched-head 8B->4B 不是足够强的 paper baseline。

### MLP > Full-Head，但 subspace <= MLP

说明主要是 nonlinear mapper capacity / error placement；subspace novelty 不够强。

### Subspace > native + Full-Head + MLP

如果 D split 上：

```text
CI95_low[subspace - native] > 0
CI95_low[subspace - Full-Head] > 0
CI95_low[subspace - MLP] > 0
```

这是最强证据：显式 student-readable geometry 比单纯增加 mapper nonlinear capacity 更有效。

### Benefit-positive > 0 且 benefit-negative < 0

最强 signed mechanism：4B gradient × teacher delta 几何能区分 helpful/harmful teacher directions。

### PCA/random ~= grad/benefit

只有 generic compression/regularization，不能 claim decoder-readable subspace。

### 所有方法 <= native 4B

STOP：之前 top-1 gain 太局部，不能转成 full-distribution / speculative gain。

## 8. 关键输出

```text
results/paper_vs_subspace_2026-09-09/linear_mapper_selection.json
results/paper_vs_subspace_2026-09-09/mapper_selection_B/
results/paper_vs_subspace_2026-09-09/subspace/qwen3_8b_to_4b_subspace_interventions.json
results/paper_vs_subspace_2026-09-09/subspace/final_D/paper_full_head.json
results/paper_vs_subspace_2026-09-09/subspace/final_D/paper_mlp.json
results/paper_vs_subspace_2026-09-09/subspace/paper_vs_subspace_final.json
```

如果你的 run 因为环境变量导致 final_D 位于 suite root 而不是 `subspace/`，以脚本最终打印的 `Final decision artifact:` 为准并完整提交路径；不要手工移动结果后丢失 provenance。

## 9. STOP / Debug

立即停止并提交 failure/progress artifact，如果出现：

- model revision/tokenizer mismatch；
- A/B/C/D/E SHA 或 token-row overlap；
- Full-Head 最终 fit 没有使用 all source heads；
- MLP 没复用 best Full-Head selected layers；
- MLP receiver heads 发生参数共享；
- key 不在 content space fit/map；
- causal frontier 不是 native 4B；
- NaN/Inf；
- OOM；
- 自动改 k/hidden/dtype/model；
- 用 B 作为 final result；
- 用 E 数据重新选择方法。
