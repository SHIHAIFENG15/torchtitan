# 第 1 行性能线性度与单卡内存对照

> GPU 必跑清单见 [`docs/gpu_todo_runs.md`](gpu_todo_runs.md)。下面保留完整协议与 NPU 实测表。


对齐 `docs/gpu_baseline_setup.md` 里的性能、内存两项。配方加在
`torchtitan_recipes/tests/models/muse_spark.py`，只服务清单第 1 行
`muse_spark_debugmodel`。NPU 与 GPU 必须用**同一对 CONFIG**。

完整验收结论仍写在各行的精度/性能/内存/确定性里，总表见
`docs/npu_gpu_acceptance_report.md`。本文件只记怎么跑、记什么数。

## 新增配方

| CONFIG | NGPU | 网格 | steps | 用途 |
| --- | --- | --- | --- | --- |
| `muse_spark_debugmodel_perf_dp2` | 4 | `dp_shard=2, tp=2, ep=2` | 200 | 性能分母 |
| `muse_spark_debugmodel_perf_dp4` | 8 | `dp_shard=4, tp=2, ep=2` | 200 | 性能分子（只加倍 DP） |
| `muse_spark_debugmodel_mem_1device` | 1 | 全部并行度 1 | 20 | 单卡内存 |

不要把验收用的 `muse_spark_debugmodel`（10 步）改成 8 卡。不要拿第 2 行③的 8 卡去除第 1 行。

## 性能：怎么跑

环境与确定性验收一致即可；性能可以不开 `MUSE_DEBUG_DETERMINISTIC`，但 NPU/GPU 两侧要一致。

NPU（容器里，先 `source /workspace/env.sh`，flex 补丁与 ascend 入口按当前机器约定）：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export HCCL_DETERMINISTIC=true
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHNPU_PRECOMPILE_THREADS=1

NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_perf_dp2 \
  ./run_train.sh | tee logs/perf_dp2_npu.log

NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_perf_dp4 \
  ./run_train.sh | tee logs/perf_dp4_npu.log
```

GPU（`node-5-15` 同类环境）：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1

NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_perf_dp2 \
  ./run_train.sh | tee logs/perf_dp2_gpu.log

NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_perf_dp4 \
  ./run_train.sh | tee logs/perf_dp4_gpu.log
```

## 性能：怎么记

1. 去掉 ANSI 后，取 `step > 100` 的 `tps`，算算术平均（记为稳定段平均 tps）。
2. `线性度 = mean_tps_8card / mean_tps_4card`。
3. NPU 与 GPU 各算一个线性度，差值（百分点）= `|npu_ratio - gpu_ratio| * 100`。
4. 达标：差值 ≤ 2。

写入验收文档时用下面这张表（跑完填数，不要编造）：

| 端 | 4 卡稳定 tps | 8 卡稳定 tps | 线性度 | 与对端差（百分点） | 结论 |
| --- | --- | --- | --- | --- | --- |
| NPU | | | | | |
| GPU | | | | | |

日志字段含义不变：`tps` 是单卡 token/s，已按 CP×TP×PP 折算，直接用比值。

## 内存：怎么跑

```bash
# NPU
NGPU=1 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_mem_1device \
  ./run_train.sh | tee logs/mem_1device_npu.log

# GPU
NGPU=1 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_mem_1device \
  ./run_train.sh | tee logs/mem_1device_gpu.log
```

再各跑一次现有 4 卡验收配方（或 `perf_dp2`），取稳定步 `memory`。

## 内存：怎么记

1. 只比日志 `memory`（该步窗口最大预留）。不要用 `nvidia-smi` / `npu-smi` 整卡占用判这项。
2. 丢掉第 1 步（常含编译）。单卡取 step 2–20 的典型稳定值；多卡取稳定步。
3. 先填单卡，再填 4 卡 FSDP2。

| 端 | 单卡稳定 memory (GiB) | 4 卡稳定 memory (GiB) | 结论 |
| --- | --- | --- | --- |
| NPU | | | |
| GPU | | | |

## 记录要求

- 命令、CONFIG、NGPU、HEAD、日期写进 `docs/npu_debugmodel_verification.md` 第 1 行「本行验收指标」和 `docs/npu_gpu_acceptance_report.md`。
- 原始日志路径写进文档，便于复查。
- 改配方或补丁记 `docs/npu_pta_fixes.md`。
- 没跑完之前，性能/内存结论保持「不达标（对照未跑）」；填表后才能改判定。

## 尚未覆盖

第 2① `with_cp` 的 4↔8 与单卡（单卡须非 CP 内核）、第 4/5 行的 4 卡减半变体，等第 1 行这套跑通后再复制。


## NPU 实测结果（2026-10-31，`e8bd720c`+超时补丁，容器 `npu-y00839695-torchtitan-20261028`）

环境：`ENTRYPOINT=torchtitan_ascend.train`，`HCCL_DETERMINISTIC=true`，`MUSE_DEBUG_DETERMINISTIC=1`，`TORCHINDUCTOR_COMPILE_THREADS=1`，`TORCHNPU_PRECOMPILE_THREADS=1`。`perf_*` 配方另设 `init_timeout_seconds=train_timeout_seconds=1800`（8 卡首步 flex 编译会超过默认 300s watchdog）。

| 端 | 4 卡稳定 tps (step>100) | 8 卡稳定 tps (step>100) | 线性度 | 与对端差（百分点） | 结论 |
| --- | --- | --- | --- | --- | --- |
| NPU | 322.89 | 307.98 | 0.9538 | 3.20 | 已对齐，判定见 GPU 行 |
| GPU | 1003.27 | 924.81 | 0.9218 | 3.20 | **不达标**（差 3.20 个百分点 > 2 阈值） |

| 端 | 单卡稳定 memory (GiB) | 4 卡稳定 memory (GiB) | 结论 |
| --- | --- | --- | --- |
| NPU | 0.26（step2–20；step1 为 0.33 含编译） | 0.11 | 已对齐，判定见 GPU 行 |
| GPU | 0.24（step2–20） | 0.11（perf_dp2 step>100） | **达标**（单卡差 0.02GiB，4 卡持平） |

日志：
- `logs/mem_1device_npu.log`（成功）
- `logs/perf_dp2_npu_r2.log`（成功；首轮未开 `MUSE_DEBUG_DETERMINISTIC` 时首步非有限梯度失败）
- `logs/perf_dp4_npu_r3.log`（成功；此前两轮 8 卡在首步 HCCL/watchdog 失败，加长 init/train timeout 后通过）

GPU 请用同一三个 CONFIG 按上文命令跑完，填表后再算百分点差。

## GPU 实测结果（2026-10-31，HEAD `7ed23db6` 基线 + perf/mem 配方，容器 `gpu-z00824525-tt`）

详表与判定分析见 [`docs/npu_gpu_row1_perf_mem_analysis.md`](npu_gpu_row1_perf_mem_analysis.md)。

环境：`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`TORCH_DEVICE_BACKEND_AUTOLOAD=1`，`MUSE_DEBUG_DETERMINISTIC=1`（与 NPU 侧一致）。配方未改网格，未改超时（GPU 无 HCCL watchdog 问题）。

| 项 | 结果 |
| --- | --- |
| 单卡内存 | 稳定 0.24GiB（step2–20 全部 0.24；NPU 0.26） |
| 4 卡稳定 tps | 1003.27（step 101–200 共 100 步算术平均；前后半段 993.8 / 1012.8，平稳） |
| 8 卡稳定 tps | 924.81（step 101–200 共 100 步算术平均；趋势平稳 898–975） |
| 线性度 | 924.81 / 1003.27 = **0.9218** |
| 与 NPU 差 | \|0.9538 − 0.9218\| × 100 = **3.20 个百分点** |
| 判定 | 性能线性度**不达标**（> 2）；单卡/4 卡内存**达标** |

日志：
- `logs/mem_1device_gpu.log`（成功，20 步）
- `logs/perf_dp2_gpu.log`（成功，200 步）
- `logs/perf_dp4_gpu.log`（成功但**作废**：tps 随时间从 ~420 退化到 ~300，疑似与外部任务争卡，已用重跑取代）
- `logs/perf_dp4_gpu_r2.log`（成功，采用其数据；运行期间监控确认无外部任务）

说明：A100 (NVLink) 上 8 卡相对 4 卡的吞吐比 0.92，低于 NPU 的 0.95；差值 3.2 个百分点超出 2 个百分点阈值，按协议如实记为不达标。若需复测，建议在独占时段重跑 `perf_dp4`。
