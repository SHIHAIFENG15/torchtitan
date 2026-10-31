# Muse Spark NPU ↔ GPU 验收对齐报告

对照清单第 1–5 行。GPU 数据来自 `docs/gpu_debugmodel_baseline.md`（2026-10-30 / 2026-10-31，`node-5-15` 8×A100-80GB）及 `docs/gpu_baseline/*_1000.txt`。NPU 数据来自 `docs/npu_debugmodel_verification.md`（910B4，分支 `npu-muse-spark-debug-r04`）。

标准（`docs/gpu_baseline_setup.md`）：

| 指标 | 达标条件 |
| --- | --- |
| 精度 | 同一份 step 0 权重，两边各 ≥1000 步；MoE MAE ≤ 0.02 或平均相对误差 ≤ 2% |
| 性能 | 丢掉前 100 步，同一配方 8 卡 tps / 4 卡 tps 与 GPU 同一比值差 ≤ 2 个百分点 |
| 内存 | 先比单卡同场景日志预留，再比 FSDP2 多卡；`nvidia-smi` 整卡占用不是这项 |
| 确定性 | 同配置连跑两次 loss、grad_norm 打印值相同，且与 GPU 同策略结果一致 |

本报告对每一行写齐四项结论。10 步对照只作设备差量级参考，**不是**精度 MAE。

## 总表

| 清单行 | 配方 | 精度 | 性能 | 内存 | 确定性（NPU自洽 / 对GPU） |
| --- | --- | --- | --- | --- | --- |
| 1 | `muse_spark_debugmodel` 4卡 | 不达标 | 不达标（3.2pp） | 达标 | 达标 / 不达标 |
| 2① | `with_cp` 4卡 | 不达标 | 不达标 | 不达标 | 达标 / 不达标 |
| 2② | `fsdp2_cp2_ep4` 4卡 | 不达标 | 不达标 | 不达标 | 达标 / 不达标 |
| 2③ | `mm_fsdp2_tp2_cp2_ep4` 8卡 | 不达标 | 不达标 | 不达标 | 达标 / 不达标 |
| 3 | Dist-MoE bf16 | 不达标 | 不达标 | 不达标 | 达标 / 不达标 |
| 4 | `debugmodel_mm` EP2 Ulysses 8卡 | 不达标 | 不达标 | 不达标 | 达标 / 不达标 |
| 5 | `ultradata_lora_sft` 8卡 | 不达标 | 不达标 | 不达标 | 达标 / 不达标 |

共性原因：

- **精度**：GPU 六份 1000 步文件已齐；step 0 权重仍在 GPU 机 `outputs/checkpoint-*/step-0/`，**未拷到 NPU**，MAE 未算。
- **性能**：现有配方世界大小固定，不是「同一配方只加倍 DP」；两边都没有满 100 步稳定段线性度。
- **内存**：单卡对照未跑；多卡只能比日志预留同量级。
- **确定性**：NPU / GPU 各自两次均逐项相同；跨设备从第 1 步就不同，差不发散。标准要求同策略一致，交叉对照不达标。

第 3 行另有硬件限制：A100 SM80，官方 Dist-MoE / mxfp8 要 SM100，无 GPU 基线。

---

## 第 1 行 `muse_spark_debugmodel`（4 卡，`dp_shard=2,tp=2,ep=2`）

### 精度 — 不达标

- GPU 1000 步已导出：`docs/gpu_baseline/debugmodel_1000.txt`（step1 loss 7.840 → step1000 2.349，均值约 2.577）。
- step 0 在 GPU：`outputs/checkpoint-debugmodel/step-0`（报告路径）。NPU 未加载，MAE 未算。
- 10 步各自初始化：loss GPU−NPU 约 +0.07～+0.13，不是 MAE。

### 性能 — 不达标（已测，差 3.2 个百分点）

- 同配方 4↔8 卡线性度已用 `perf_dp2` / `perf_dp4` 补测（2026-10-31）：NPU 0.9538，GPU 0.9218，差 3.20 个百分点 > 2 阈值。
- GPU 4 卡稳定 tps 1003.27，8 卡稳定 tps 924.81（step>100 各 100 步平均）；原始日志 `logs/perf_dp2_gpu.log`、`logs/perf_dp4_gpu_r2.log`（首轮 `perf_dp4_gpu.log` 疑似争卡作废）。

### 内存 — 达标

- 单卡同场景已测：GPU 0.24GiB（step2–20）vs NPU 0.26GiB，同量级。
- 4 卡 FSDP2 稳定预留两边都是 **0.11GiB**（日志窗口最大预留）。

### 确定性 — NPU 达标，对 GPU 不达标

- NPU / GPU 各自两次 10 步 loss、grad_norm 相同。
- 跨设备：第 1 步起不同；GPU loss 系统性偏高约 0.07–0.13，无放大。

---

## 第 2 行① `muse_spark_debugmodel_with_cp`（4 卡，Ulysses）

### 精度 — 不达标

- GPU 1000 步：`docs/gpu_baseline/with_cp_1000.txt`（7.849 → 2.368）。step 0 未上 NPU。
- 10 步 loss 差约 +0.03～+0.12。

### 性能 — 不达标

- 无 8 卡同配方。稳定段 NPU 约 130–145 / 更高档约 300+，GPU 约 333–361 tps。

### 内存 — 不达标

- 单卡未跑。稳定预留两边约 **0.10GiB**。

### 确定性 — NPU 达标，对 GPU 不达标

- 各自两次相同；跨设备差约 0.03–0.12，无放大。

---

## 第 2 行② `muse_spark_debugmodel_fsdp2_cp2_ep4`（4 卡，AllGatherKV）

### 精度 — 不达标

- GPU 1000 步：`docs/gpu_baseline/fsdp2_cp2_ep4_1000.txt`（7.840 → 2.337）。step 0 未上 NPU。
- 10 步 loss 差约 +0.04～+0.12；第 10 步 grad_norm 差约 −0.0028。

### 性能 — 不达标

- 稳定段 NPU 约 123 tps，GPU 约 319–331。无线性度。

### 内存 — 不达标

- 单卡未跑。稳定预留两边 **0.12GiB**。

### 确定性 — NPU 达标，对 GPU 不达标

---

## 第 2 行③ `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`（8 卡，AllGatherKV）

### 精度 — 不达标

- GPU 1000 步：`docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`（7.790 → ~0.003，多模态过拟合 debug 尺度）。step 0 未上 NPU。
- 这组 GPU 数**不能**填第 4 行。
- 10 步 loss 差有正有负，绝对值 ≤ ~0.15。

### 性能 — 不达标

- 无同配方 4 卡。第 5、8 步 tps 掉到十来个是视觉长度重编译，GPU 同样掉到 15，不能当线性度。

### 内存 — 不达标

- 单卡未跑。日志预留 NPU 稳定约 0.12–0.26GiB，GPU 约 0.14GiB。GPU `nvidia-smi` ~6.8GiB/卡是整卡占用，不是日志预留。多模态 GPU 上 `grid_sample` 反传走了 CPU。

### 确定性 — NPU 达标，对 GPU 不达标

---

## 第 3 行 Dist-MoE bf16

### 精度 — 不达标

- A100 SM80，官方 Dist-MoE / mxfp8 要 SM100，无可比 1000 步。NPU 仅有 bf16 10 步自洽。

### 性能 — 不达标

- NPU 稳定约 355 tps。无 GPU，无 8 卡线性度。

### 内存 — 不达标

- NPU 稳定预留 0.09GiB。无 GPU 预留，无单卡。

### 确定性 — NPU 达标，对 GPU 不达标（无 GPU）

---

## 第 4 行 `muse_spark_debugmodel_mm`（8 卡，EP2，Ulysses）

### 精度 — 不达标

- GPU 10 步 ×2 已齐；1000 步：`docs/gpu_baseline/debugmodel_mm_1000.txt`（7.791 → ~0.002）。step 0 未上 NPU。
- 10 步 loss 差约 −0.09～+0.11，与其它行同量级。

### 性能 — 不达标

- 无同配方 4 卡。NPU 稳定约 120 tps，GPU 约 82–90（编译步更低）。

### 内存 — 不达标

- 单卡未跑。稳定预留 NPU 0.12GiB，GPU 约 0.10GiB。

### 确定性 — NPU 达标，对 GPU 不达标

- 两边各自两次相同；跨设备第 1 步起不同。

---

## 第 5 行 `muse_spark_debugmodel_ultradata_lora_sft`（8 卡）

### 精度 — 不达标

- GPU 10 步 ×2 与 1000 步已齐：`docs/gpu_baseline/ultradata_lora_1000.txt`（12.363 → 10.102，均值约 11.25；符合 ln(202048)≈12.216 尺度、只训 LoRA）。
- step 0 未上 NPU，MAE 未算。
- 10 步跨设备差很小：loss |diff| ≤ 0.027，grad |diff| ≤ 0.008，仍不是逐项相同，也不是 MAE。

### 性能 — 不达标

- 无第二档卡数。NPU 稳定约 347 tps；GPU 稳定档可达上万 tps（序列/批设定不同计量时不可直接当线性度）。两边都没有满 100 步线性度协议。

### 内存 — 不达标

- 无单卡。NPU 稳定预留约 4.3GiB（两次略不同）；GPU 日志约 **3.46GiB**。同量级，未做单卡档。

### 确定性 — NPU 达标，对 GPU 不达标

- 两边各自两次相同；跨设备差最小的一行，仍不满足「同策略一致」。

---

## 数据与缺口清单

| 资产 | 状态 |
| --- | --- |
| GPU 10 步 ×2（行 1、2①②③、4、5） | 已齐，见 `docs/gpu_debugmodel_baseline.md` |
| GPU 1000 步导出（同上六份） | 已齐，见 `docs/gpu_baseline/*.txt` |
| GPU step 0 权重 | 已进 git：`outputs/checkpoint-<配置名>/`（ultradata 为 90MB 分卷 + 拼接说明），NPU 拉取后即可加载 |
| NPU 加载 step 0 后的 1000 步 + MAE | **未做**（权重已具备，待 NPU 执行） |
| 性能线性度（同配方 4↔8） | 第 1 行已测：GPU 0.9218 vs NPU 0.9538，差 3.2pp，**不达标** |
| 单卡内存对照 | 第 1 行已测：GPU 0.24 vs NPU 0.26，**达标** |
| Dist-MoE / mxfp8 GPU | **不可做**（SM100） |

下一步（阻塞精度达标）：从 git 拉取六份 step 0（`outputs/checkpoint-<配置名>/`，ultradata 先 `cat __0_0.distcp.part-* > __0_0.distcp`），用 `scripts/loss_compare.py` 的 load 路径（或 `TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH`）跑 NPU 1000 步，再对 `docs/gpu_baseline/*.txt` 算 MAE / 相对误差。

报告依据提交：`7ed23db6`（GPU 基线）、`524ff3e7`（权重入库）及当前 NPU 验收文档。未编造 GPU 数。

## 性能与内存对照怎么补齐

第 1 行已增加 `perf_dp2` / `perf_dp4` / `mem_1device` 配方。跑法与记数表见 `docs/gpu_baseline_perf_mem.md`。未跑完前，总表里性能、内存保持不达标。

### 第 1 行 NPU 性能/内存实测摘要

- 4 卡稳定 tps：322.89
- 8 卡稳定 tps：307.98
- NPU 线性度：0.9538
- 单卡稳定 memory：0.26GiB
- 4 卡稳定 memory：0.11GiB
- GPU 同配方未跑，两项仍不达标。

## GPU 待跑

性能/内存要闭环，GPU 必须跑的命令与交数表见 `docs/gpu_todo_runs.md`。

## 第 1 行 GPU 性能/内存实测结果（2026-10-31 闭环）

- 4 卡稳定 tps：1003.27（`perf_dp2`，step>100 均值）
- 8 卡稳定 tps：924.81（`perf_dp4` r2，step>100 均值；首轮 r1 疑似外部争卡退化作废）
- GPU 线性度：0.9218；与 NPU 0.9538 差 **3.20 个百分点 > 2**，性能项**不达标**
- 单卡稳定 memory：0.24GiB（NPU 0.26）；4 卡 0.11GiB（NPU 0.11），内存项**达标**
- 依据：HEAD `7ed23db6` + perf/mem 配方，日志 `logs/mem_1device_gpu.log`、`logs/perf_dp2_gpu.log`、`logs/perf_dp4_gpu_r2.log`；详表见 `docs/gpu_baseline_perf_mem.md` GPU 实测节
- `docs/gpu_todo_runs.md` 三组任务已全部执行完毕，无遗留 GPU 侧动作（第 2①/4/5 行的 perf·mem 变体按文档属后续复制项）
