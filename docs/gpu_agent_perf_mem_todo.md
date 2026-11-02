# GPU agent 任务：Muse Spark 性能 / 内存对照

给 `node-5-15`（或同等 **8×A100-80GB**）上的另一个 agent / 执行者。

NPU 侧已在 `179.173.2.3` 容器 `npu-y00839695-torchtitan-20261028` 上按协议补跑（或正在跑）各行 perf/mem。本文件只列 **GPU 还必须跑的同 CONFIG**，跑完才能和 NPU 比线性度与内存并改验收判定。

权威协议：`docs/gpu_baseline_setup.md`、`docs/gpu_baseline_perf_mem.md`。总表：`docs/npu_gpu_acceptance_report.md`。

## 目标

对下表每一组 CONFIG：

1. 用与 NPU **完全相同**的 `MODULE` / `CONFIG` / `NGPU` 跑完。
2. 把日志落到 `logs/<name>_gpu.log`（命名与下表一致）。
3. 按「怎么记」算出稳定段 tps / memory，填回 `docs/gpu_baseline_perf_mem.md` 对应表的 GPU 行，并更新 `docs/npu_gpu_acceptance_report.md` 与 `docs/npu_debugmodel_verification.md` 该行的性能/内存判定。

## 环境

- 分支：含下列配方的 `npu-muse-spark-debug-r04`（或已 cherry-pick 这些 CONFIG 的等价提交）。先 `git pull`。
- 机器：独占 8 卡。争卡会导致 8 卡 tps 随时间掉下去（第 1 行 `perf_dp4_gpu.log` 已作废先例），必须重跑。
- 入口：GPU 用默认 `torchtitan.train`（不要设 `torchtitan_ascend.train`）。
- 建议环境变量：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
# 与 NPU 记数方式对齐时可开；两侧要一致
export MUSE_DEBUG_DETERMINISTIC=1
cd /path/to/torchtitan
mkdir -p logs
```

## 怎么记（所有行共用）

**性能**

1. 去掉 ANSI 后，取 `step > 100` 的 `tps`，算术平均 = 稳定段平均 tps。
2. `线性度 = mean_tps_大档 / mean_tps_小档`（大档 = 8 卡，小档 = 4 卡）。
3. 与 NPU 同一对 CONFIG 的线性度比：差值（百分点）= `|npu_ratio - gpu_ratio| * 100`。
4. 达标：差值 ≤ 2。不比绝对 tps。

**内存**

1. 只比日志 `memory`（该步窗口最大预留）。不要用 `nvidia-smi` 整卡占用。
2. 单卡：丢掉第 1 步，取 step 2–20 的稳定值。
3. 多卡：取对应 4 卡 perf 配方稳定步（step>100）的 `memory`。
4. 先比单卡，再比多卡。

## 已完成（不要重跑，除非要复测）

| 行 | CONFIG | GPU 日志 | 结论 |
| --- | --- | --- | --- |
| 1 | `muse_spark_debugmodel_perf_dp2` / `_perf_dp4` / `_mem_1device` | `logs/perf_dp2_gpu.log`、`perf_dp4_gpu_r2.log`、`mem_1device_gpu.log` | 性能不达标（差 3.20pp）；内存达标 |

## 必跑清单（按推荐顺序）

模块缩写：

- `SPARK` = `torchtitan_recipes.tests.models.muse_spark`
- `IT` = `tests.integration_tests.models`

### A. 第 2 行① `with_cp`（NPU 已有数）

NPU 对照：4 卡 tps **148.53**，8 卡 **144.21**，线性度 **0.9709**；单卡 mem **0.25GiB**，4 卡 **0.10GiB**。

```bash
NGPU=1 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_with_cp_mem_1device \
  ./run_train.sh | tee logs/with_cp_mem_1device_gpu.log

NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_with_cp_perf_dp1 \
  ./run_train.sh | tee logs/with_cp_perf_dp1_gpu.log

NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_with_cp_perf_dp2 \
  ./run_train.sh | tee logs/with_cp_perf_dp2_gpu.log
```

网格核对：4 卡 `dp_shard=1, cp=2, tp=2, ep=2`；8 卡只把 `dp_shard` 加到 2。

### B. 第 2 行② `fsdp2_cp2_ep4`

```bash
NGPU=1 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_fsdp2_cp2_ep4_mem_1device \
  ./run_train.sh | tee logs/fsdp2_mem_1device_gpu.log

NGPU=4 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_fsdp2_cp2_ep4_perf_dp2 \
  ./run_train.sh | tee logs/fsdp2_perf_dp2_gpu.log

NGPU=8 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_fsdp2_cp2_ep4_perf_dp4 \
  ./run_train.sh | tee logs/fsdp2_perf_dp4_gpu.log
```

网格：4 卡 `dp_shard=2, cp=2, tp=1, ep=4`；8 卡 `dp_shard=4, cp=2, tp=1, ep=4`。

### C. 第 2 行③ `mm_fsdp2_tp2_cp2_ep4`

```bash
NGPU=1 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_mem_1device \
  ./run_train.sh | tee logs/mm_fsdp2_mem_1device_gpu.log

NGPU=4 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_perf_dp1 \
  ./run_train.sh | tee logs/mm_fsdp2_perf_dp1_gpu.log

NGPU=8 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_perf_dp2 \
  ./run_train.sh | tee logs/mm_fsdp2_perf_dp2_gpu.log
```

网格：4 卡 `dp_shard=1, cp=2, tp=2, ep=4`；8 卡 `dp_shard=2, ...`。视觉长度变化会触发 flex 重编译，部分步 tps 会掉到个位数——**丢掉这些编译步后再对 step>100 取平均**，或至少保证稳定段样本够；不要把编译掉速当成线性度失败。

### D. 第 4 行 `debugmodel_mm`（Ulysses EP2）

```bash
NGPU=1 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_mm_mem_1device \
  ./run_train.sh | tee logs/mm_mem_1device_gpu.log

NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_mm_perf_dp1 \
  ./run_train.sh | tee logs/mm_perf_dp1_gpu.log

NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_mm_perf_dp2 \
  ./run_train.sh | tee logs/mm_perf_dp2_gpu.log
```

### E. 第 5 行 `ultradata_lora_sft`（32K，贵）

需要本机已有：

- `assets/hf/Muse-Spark-Open-816B-A42B/tokenizer.json`（公开副本即可）
- `assets/datasets/muse_spark_ultradata/General_Agent_part-*.jsonl`（至少 part-01）

或设 `MUSE_SPARK_ULTRADATA_JSONL_PATTERN`。

```bash
NGPU=1 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_ultradata_lora_sft_mem_1device \
  ./run_train.sh | tee logs/ultradata_mem_1device_gpu.log

NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_ultradata_lora_sft_perf_dp1 \
  ./run_train.sh | tee logs/ultradata_perf_dp1_gpu.log

NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_ultradata_lora_sft_perf_dp2 \
  ./run_train.sh | tee logs/ultradata_perf_dp2_gpu.log
```

perf 配方是 **120 步**（仍满足 step>100）。首步 preprocess 可能很长，属正常。

## 不要跑 / 不可对照

| 行 | 原因 |
| --- | --- |
| 第 3 行 Dist-MoE bf16 / mxfp8 | A100 是 SM80；官方 Dist-MoE / mxfp8 要 SM100。精度/性能/内存交叉对照标 **不可做**。不要编造 GPU 基线。 |

## 填表模板（每行复制一张）

```text
行: <1/2①/2②/2③/4/5>
4卡稳定 tps (step>100):
8卡稳定 tps (step>100):
GPU 线性度:
NPU 线性度:（从 docs/gpu_baseline_perf_mem.md 或 acceptance report 抄）
差（百分点）:
性能判定: 达标 / 不达标

单卡 memory (step2–20):
4卡 memory (稳定步):
NPU 单卡 / 4卡:（抄 NPU）
内存判定: 达标 / 不达标

日志路径:
HEAD / 日期 / 是否独占 8 卡:
```

## 完成后改哪些文件

1. `docs/gpu_baseline_perf_mem.md` — 补各行 GPU 实测表  
2. `docs/npu_gpu_acceptance_report.md` — 总表与该行「性能 / 内存」小节  
3. `docs/npu_debugmodel_verification.md` — 对应行「本行验收指标」里的性能/内存句  
4. 本文件顶部可加一行「GPU 已于 YYYY-MM-DD 完成」

## NPU 侧状态查询（GPU agent 不必等）

NPU 队列脚本：`scripts/run_npu_perf_mem_queue.sh`  
状态：`logs/npu_perf_mem_queue_status.txt`  
NPU 日志命名与上表 `_gpu` 对称，后缀 `_npu.log`。

GPU agent **可以立刻开跑**；判定时用当时文档里已写入的 NPU 数。若某行 NPU 尚未写入，先看 `logs/*_npu.log` 是否已有 `Training completed`，用下面脚本算均值后再比：

```bash
python3 - <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
tps, mem = [], []
for m in re.finditer(r"step:\s*(\d+).*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)", text):
    step, memory, t = int(m.group(1)), float(m.group(2)), int(m.group(3).replace(",", ""))
    if step > 100:
        tps.append(t); mem.append(memory)
print("n", len(tps), "mean_tps", sum(tps)/len(tps) if tps else None, "last_mem", mem[-1] if mem else None)
print("completed", "Training completed" in text)
PY
logs/with_cp_perf_dp1_npu.log
```
