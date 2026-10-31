# 第 1 行性能 / 内存 NPU↔GPU 对照分析

> 本文件只做第 1 行 `muse_spark_debugmodel` 的 **性能线性度** 与 **内存** 对照分析。
> 四项验收总表由 `docs/npu_gpu_acceptance_report.md` 维护（可能有其它同学在改），此处不覆盖总表。
> 原始协议与填数表见 `docs/gpu_baseline_perf_mem.md`；必跑清单见 `docs/gpu_todo_runs.md`（三组均已完成）。

日期：2026-10-31  
配方提交：含 `muse_spark_debugmodel_perf_dp2` / `perf_dp4` / `mem_1device`（起自 `e8bd720c`，超时补丁后 NPU 实测记在 `58f5904a`）  
GPU 填数提交：`75c46f03`  
NPU 容器：`npu-y00839695-torchtitan-20261028`（910B4）  
GPU 容器：`gpu-z00824525-tt`（`node-5-15`，8×A100-80GB）

## 1. 结论摘要

| 指标 | NPU | GPU | 差 | 判定 |
| --- | --- | --- | --- | --- |
| 4 卡稳定 tps（step>100） | 322.89 | 1003.27 | 绝对吞吐不比 | — |
| 8 卡稳定 tps（step>100） | 307.98 | 924.81 | 绝对吞吐不比 | — |
| 线性度（8卡/4卡） | 0.9538 | 0.9218 | **3.20 个百分点** | **不达标**（阈值 ≤2） |
| 单卡稳定 memory (GiB) | 0.26 | 0.24 | 0.02 | **达标** |
| 4 卡稳定 memory (GiB) | 0.11 | 0.11 | 0 | **达标** |

要点：

1. **只比线性度，不比绝对 tps。** A100 绝对吞吐约为 NPU 的 3 倍，属硬件差异，验收协议不把绝对 tps 当门槛。
2. **性能项不达标的唯一依据**是线性度差 3.20pp > 2pp。NPU 扩到 8 卡几乎不掉（0.95），GPU 掉到 0.92。
3. **内存项达标。** 单卡同量级，4 卡日志预留两边都是 0.11GiB。
4. GPU 首轮 `perf_dp4` 因疑似外部争卡 tps 退化已作废；采用 `perf_dp4_gpu_r2.log`。分析以重跑为准。

## 2. 对齐条件（两边一致才算数）

| 项 | 取值 |
| --- | --- |
| CONFIG | `muse_spark_debugmodel_perf_dp2`（4 卡）/ `perf_dp4`（8 卡）/ `mem_1device`（1 卡） |
| 4 卡网格 | `dp_shard=2, tp=2, ep=2` |
| 8 卡网格 | `dp_shard=4, tp=2, ep=2`（只加倍 DP） |
| 单卡网格 | 全部并行度 1 |
| 稳定段 | 丢掉 step≤100；对剩余 tps 算术平均 |
| 确定性开关 | 两边均 `MUSE_DEBUG_DETERMINISTIC=1`（NPU 另开 `HCCL_DETERMINISTIC`） |
| 内存字段 | 日志 `memory`（该步窗口最大预留），不用 `nvidia-smi` / `npu-smi` |

未满足上述条件的旧 10 步 / 异配方 8 卡数，不参与本项判定。

## 3. 性能：数怎么来的

### 3.1 NPU

- 日志：`logs/perf_dp2_npu_r2.log`、`logs/perf_dp4_npu_r3.log`
- 环境：`ENTRYPOINT=torchtitan_ascend.train`；`perf_*` 设 `init_timeout_seconds=train_timeout_seconds=1800`（8 卡首步 flex 编译会踩默认 300s watchdog）
- 稳定段：step 101–200 均值 → 322.89 / 307.98
- 线性度：307.98 / 322.89 = **0.9538**

踩坑（已解决，不影响定稿数）：

- 未开 `MUSE_DEBUG_DETERMINISTIC` 时 `perf_dp2` 首步非有限梯度
- 8 卡两轮因 HCCL/watchdog 失败，加长 timeout 后通过

### 3.2 GPU

- 日志：`logs/perf_dp2_gpu.log`、`logs/perf_dp4_gpu_r2.log`（`perf_dp4_gpu.log` **作废**）
- 环境：`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`TORCH_DEVICE_BACKEND_AUTOLOAD=1`，`MUSE_DEBUG_DETERMINISTIC=1`
- 稳定段：step 101–200 各 100 步均值 → 1003.27 / 924.81
  - 4 卡前后半段 993.8 / 1012.8，平稳
  - 8 卡重跑趋势约 898–975，平稳；首轮从 ~420 掉到 ~300，疑似争卡
- 线性度：924.81 / 1003.27 = **0.9218**

### 3.3 差值与判定

```
|0.9538 - 0.9218| * 100 = 3.20 个百分点  >  2  →  不达标
```

解读：

- NPU 在只加倍 DP 时扩展效率更高（掉吞吐更少）。
- GPU 在 NVLink A100 上 8 卡相对 4 卡仍掉约 7.8% 吞吐（1−0.9218），相对 NPU 的 4.6% 掉幅多出约 3.2 个百分点。
- 协议不允许用「硬件不同所以线性度可比宽」自行放宽；按字面 ≤2pp 记不达标。
- 若需申诉复测：建议 GPU **独占时段**再跑一遍 `perf_dp4`，确认不是残余争卡；NPU 数已稳定，无强制重跑。

## 4. 内存：数怎么来的

| 端 | 单卡（`mem_1device`，step2–20） | 4 卡（`perf_dp2` 稳定步） |
| --- | --- | --- |
| NPU | 0.26GiB（step1=0.33 含编译已丢） | 0.11GiB |
| GPU | 0.24GiB（step2–20 全 0.24） | 0.11GiB |

判定：**达标**。单卡差 0.02GiB 同量级；4 卡持平。字段均为训练日志预留，不是整卡占用。

日志：`logs/mem_1device_npu.log`、`logs/mem_1device_gpu.log`、`logs/perf_dp2_gpu.log`（4 卡预留）。

## 5. 和总验收表的关系

本分析覆盖清单第 1 行的 **性能**、**内存** 两项闭环。

不在本文件范围、仍由验收报告 / 验证文档跟进：

| 项 | 状态（截至本分析成稿） |
| --- | --- |
| 精度 MAE | step 0 已入库 `outputs/checkpoint-*`；NPU 加载后 1000 步 + 对 `docs/gpu_baseline/*.txt` 算 MAE **未做** |
| 确定性跨设备 | 各自两次达标；跨设备第 1 步起不同 → 不达标 |
| 第 2① / 4 / 5 行 perf·mem 变体 | 按协议等第 1 行闭环后再复制；**尚未开跑** |
| Dist-MoE / mxfp8 GPU | A100 SM80 **不可做** |

请维护 `docs/npu_gpu_acceptance_report.md` 的同学以本文件与 `docs/gpu_baseline_perf_mem.md`「GPU 实测结果」为准同步第 1 行性能/内存单元格；本文件不直接改总表，以免并行编辑冲突。

## 6. 原始引用

- 协议与双端填表：`docs/gpu_baseline_perf_mem.md`
- GPU 必跑清单（已完成）：`docs/gpu_todo_runs.md`
- 行级验收叙述：`docs/npu_debugmodel_verification.md` 第 1 行「本行验收指标」
- 配方：`torchtitan_recipes/tests/models/muse_spark.py` 中 `muse_spark_debugmodel_perf_*` / `mem_1device`
