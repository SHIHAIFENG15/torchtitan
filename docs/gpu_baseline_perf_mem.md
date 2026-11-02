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

## 第 2① `with_cp` 配方（NPU 已跑，GPU 待跑）

| CONFIG | NGPU | 网格 | steps | 用途 |
| --- | --- | --- | --- | --- |
| `muse_spark_debugmodel_with_cp_perf_dp1` | 4 | `dp_shard=1, cp=2, tp=2, ep=2` | 200 | 性能分母 |
| `muse_spark_debugmodel_with_cp_perf_dp2` | 8 | `dp_shard=2, cp=2, tp=2, ep=2` | 200 | 性能分子（只加倍 DP） |
| `muse_spark_debugmodel_with_cp_mem_1device` | 1 | 全部并行度 1（无 CP） | 20 | 单卡内存 |

```bash
NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_with_cp_perf_dp1 \
  ./run_train.sh | tee logs/with_cp_perf_dp1_gpu.log

NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_with_cp_perf_dp2 \
  ./run_train.sh | tee logs/with_cp_perf_dp2_gpu.log

NGPU=1 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_with_cp_mem_1device \
  ./run_train.sh | tee logs/with_cp_mem_1device_gpu.log
```

### NPU 实测（2026-11-02，容器 `npu-y00839695-torchtitan-20261028`）

| 端 | 4 卡稳定 tps (step>100) | 8 卡稳定 tps (step>100) | 线性度 | 与对端差（百分点） | 结论 |
| --- | --- | --- | --- | --- | --- |
| NPU | 148.53 | 144.21 | 0.9709 | 1.35 | 已对齐，判定见 GPU 行 |
| GPU | 485.99 | 478.41 | 0.9844 | 1.35 | **达标**（1.35 ≤ 2 阈值） |

| 端 | 单卡稳定 memory (GiB) | 4 卡稳定 memory (GiB) | 结论 |
| --- | --- | --- | --- |
| NPU | 0.25 | 0.10 | 已对齐，判定见 GPU 行 |
| GPU | 1.47–3.85（随步爬升，无稳定值，末步 3.85） | 0.65（step>100 末值） | **不达标**（GPU 日志预留显著高于 NPU：单卡约 6–15×、4 卡约 6.5×；GPU 端预留随步爬升，疑似 allocator 预留增长或内核实现差异，建议 allocator 行为核查后复测） |

日志：`logs/with_cp_perf_dp1_npu.log`、`logs/with_cp_perf_dp2_npu.log`、`logs/with_cp_mem_1device_npu.log`；GPU：`logs/with_cp_perf_dp1_gpu.log`、`logs/with_cp_perf_dp2_gpu.log`、`logs/with_cp_mem_1device_gpu.log`。

## 尚未覆盖 / GPU 待跑

配方已扩展到第 2①②③、4、5 行（及 Dist-MoE 仅 NPU）。**GPU 侧完整必跑清单与命令**见 [`docs/gpu_agent_perf_mem_todo.md`](gpu_agent_perf_mem_todo.md)。第 1 行及第 2①②③、4、5 行 GPU/NPU perf/mem 已于 2026-11-02 闭环（见下表）。

## GPU 实测结果（2026-11-02，HEAD `75c46f03` + 本分支新配方，容器 `gpu-z00824525-tt`，独占 8 卡）

环境与第 1 行相同（`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`TORCH_DEVICE_BACKEND_AUTOLOAD=1`、`MUSE_DEBUG_DETERMINISTIC=1`）。15 次运行全部 `Training completed`，无 `ChildFailedError`。逐次顺序执行，一次只跑一个 torchrun。

| 行 | CONFIG（4 卡 / 8 卡） | 4 卡稳定 tps | 8 卡稳定 tps | GPU 线性度 | NPU 线性度 | 差（百分点） | 性能判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2① | `with_cp_perf_dp1` / `_perf_dp2` | 485.99 | 478.41 | 0.9844 | 0.9709 | 1.35 | **达标** |
| 2② | `fsdp2_cp2_ep4_perf_dp2` / `_perf_dp4` | 335.02 | 324.08 | 0.9673 | 0.9556 | 1.17 | **达标** |
| 2③ | `mm_fsdp2_…_perf_dp1` / `_perf_dp2` | 134.18 | 109.40 | 0.8153 | 0.8106 | 0.47 | **达标** |
| 4 | `debugmodel_mm_perf_dp1` / `_perf_dp2` | 374.36 | 358.08 | 0.9565 | 0.9696 | 1.31 | **达标** |
| 5 | `ultradata_…_perf_dp1` / `_perf_dp2`（120 步） | 10230.50 | 9874.95 | 0.9652 | 0.8990 | 6.62 | **不达标** |

| 行 | GPU 单卡 memory (step2–20) | GPU 4 卡 memory | NPU 单卡 / 4 卡 | 内存判定 |
| --- | --- | --- | --- | --- |
| 2① | 1.47–3.85（末 3.85，爬升） | 0.65 | 0.25 / 0.10 | **不达标** |
| 2② | 1.46–3.85（末 3.85，爬升） | 0.49 | 0.25 / 0.12 | **不达标** |
| 2③ | 2.39–14.15（末 14.15，爬升） | 2.24 | 0.28 / 0.15 | **不达标** |
| 4 | 2.34–14.15（末 14.15，爬升） | 3.19 | 0.28 / 0.15 | **不达标** |
| 5 | 0.24–0.54（末 0.54，爬升） | 0.44 | 13.25 / 4.32 | **不达标**（本行是 NPU 预留更高） |

日志（`logs/`）：`with_cp_mem_1device_gpu.log`、`with_cp_perf_dp1_gpu.log`、`with_cp_perf_dp2_gpu.log`、`fsdp2_mem_1device_gpu.log`、`fsdp2_perf_dp2_gpu.log`、`fsdp2_perf_dp4_gpu.log`、`mm_fsdp2_mem_1device_gpu.log`、`mm_fsdp2_perf_dp1_gpu.log`、`mm_fsdp2_perf_dp2_gpu.log`、`mm_mem_1device_gpu.log`、`mm_perf_dp1_gpu.log`、`mm_perf_dp2_gpu.log`、`ultradata_mem_1device_gpu.log`、`ultradata_perf_dp1_gpu.log`、`ultradata_perf_dp2_gpu.log`。

配方修正：`muse_spark_debugmodel_ultradata_lora_sft_perf_dp1` 原样在 4 卡上触发 `ValueError: expert_parallel_degree (8) must divide dp_shard * cp * tp (4)`——基函数 `expert_parallel_degree=8` 硬编码且未随卡数折减。已改为基函数新增 `expert_parallel_degree: int = 8` 参数、`perf_dp1` 传 4（4 整除 dp_shard×cp×tp=4，与 8 卡"全卡 EP"语义一致）；8 卡路径数值不变。NPU 侧队列脚本同配置同参数，拉取本修复后其 4 卡档才能通过。

内存记数说明：GPU 端各 mem_1device 日志的 `memory` 普遍随步单调爬升（缓存预留增长），与 NPU 侧平稳值不同口径，按协议如实记录区间与末值；达标判定待两侧口径确认后复核。

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

NPU 第 5 行 8 卡定稿日志：`logs/ultradata_perf_dp2_npu.ep8.log`（后续同名 `ultradata_perf_dp2_npu.log` 若被覆盖重跑，验收以 ep8 定稿为准）。
