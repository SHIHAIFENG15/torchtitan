# GPU 还需要跑什么

给 `node-5-15`（或同等 8×A100）上的执行者。NPU 侧第 1 行性能/内存对照已跑完，数在 `docs/gpu_baseline_perf_mem.md`。**现在缺的是 GPU 用同一套 CONFIG 跑完**，才能判定性能、内存是否达标。

分支：`npu-muse-spark-debug-r04`（至少包含 `58f5904a` 及 perf/mem 配方）。

## 已经齐的，不要重跑

| 项 | 状态 |
| --- | --- |
| 行 1 / 2①②③ / 4 / 5 的 10 步×2 | 已记在 `docs/gpu_debugmodel_baseline.md` |
| 六份 1000 步导出 | `docs/gpu_baseline/*_1000.txt` |
| 多份 step 0 权重 | 已进仓库 `outputs/checkpoint-*`（含 ultradata 分片） |
| Dist-MoE / mxfp8 | A100 SM80 **不要跑** |

## 必须新跑的三组（第 1 行）

与 NPU 完全相同的 CONFIG。先 `git pull` 到含 `muse_spark_debugmodel_perf_*` / `mem_1device` 的提交。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

### 1. 单卡内存

```bash
NGPU=1 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_mem_1device \
  ./run_train.sh | tee logs/mem_1device_gpu.log
```

记：丢掉 step 1 后的稳定 `memory`（GiB）。NPU 对照是 **0.26GiB**。

### 2. 性能 4 卡（分母）

```bash
NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_perf_dp2 \
  ./run_train.sh | tee logs/perf_dp2_gpu.log
```

`steps=200`。丢掉 step≤100，对剩余 `tps` 算术平均。NPU 对照是 **322.89**。

### 3. 性能 8 卡（分子，只加倍 DP）

```bash
NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_perf_dp4 \
  ./run_train.sh | tee logs/perf_dp4_gpu.log
```

网格必须是 `dp_shard=4, tp=2, ep=2`。NPU 对照稳定 tps **307.98**，线性度 **0.9538**。

配方已设 `init_timeout_seconds=train_timeout_seconds=1800`。若 GPU 首步编译仍超时，可再加大，但不要改网格。

## 跑完怎么交数

1. 把三份日志路径、HEAD、日期记下来。
2. 填 `docs/gpu_baseline_perf_mem.md` 里 GPU 行：

| 端 | 4 卡稳定 tps | 8 卡稳定 tps | 线性度 | 与对端差（百分点） | 结论 |
| --- | --- | --- | --- | --- | --- |
| NPU | 322.89 | 307.98 | 0.9538 | （填完 GPU 后算） | 待 GPU |
| GPU | （填） | （填） | （填） | `|gpu_ratio - 0.9538| * 100`，≤2 达标 | |

| 端 | 单卡稳定 memory (GiB) | 4 卡稳定 memory (GiB) | 结论 |
| --- | --- | --- | --- |
| NPU | 0.26 | 0.11 | 待 GPU |
| GPU | （填） | （填，可用 perf_dp2 稳定步） | |

3. 同步改 `docs/npu_debugmodel_verification.md` 第 1 行性能/内存结论，以及 `docs/npu_gpu_acceptance_report.md`。
4. 提交并推到 `npu-muse-spark-debug-r04`（不要 force-push）。

## 不算本次必跑

- 再导出 1000 步 / 再建 step 0（已有）。
- 第 2① / 4 / 5 的 perf·mem 变体（等第 1 行 GPU 对照闭环后再复制）。
- NPU 加载 step 0 算 MAE（在 NPU 做，不是 GPU 再跑训练）。

## 参考

- 跑法细节：`docs/gpu_baseline_perf_mem.md`
- NPU 实测与失败踩坑：同文件「NPU 实测结果」一节
- 环境总则：`docs/gpu_baseline_setup.md`

## 状态（2026-10-31）

三组均已完成。GPU 数与判定分析见 `docs/npu_gpu_row1_perf_mem_analysis.md`；填数见 `docs/gpu_baseline_perf_mem.md` GPU 实测节。验收总表由 `docs/npu_gpu_acceptance_report.md` 单独维护。
