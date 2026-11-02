# Muse Spark NPU ↔ GPU 验收对齐报告

对照清单第 1–5 行。本文把四项验收按「有没有按协议测过、数怎么来的、判什么」写完。10 步对照只说明设备差的量级，**不是**精度 MAE。

数据来源：

- GPU 10 步与 1000 步：`docs/gpu_debugmodel_baseline.md`、`docs/gpu_baseline/*_1000.txt`（`node-5-15`，8×A100-80GB）
- GPU 第 1 行性能 / 内存必跑三组：`docs/gpu_baseline_perf_mem.md`、`docs/gpu_todo_runs.md`、`docs/npu_gpu_row1_perf_mem_analysis.md`
- NPU 10 步与确定性：`docs/npu_debugmodel_verification.md`（910B4）
- NPU 共享 step 0 的 1000 步：`docs/npu_baseline/`（从 `outputs/checkpoint-*/step-0/` 加载，入口 `torchtitan_ascend.train`）。六行队列已于 2026-10-31 18:26 UTC 结束，过程在 `/workspace/npu_load_1000/monitor.log`

标准（`docs/gpu_baseline_setup.md`）：

| 指标 | 达标条件 |
| --- | --- |
| 精度 | 同一份 step 0 权重，两边各 ≥1000 步。MoE loss 的平均绝对误差 ≤ 0.02，**或**平均相对误差 ≤ 2%。两条满足一条即可 |
| 性能 | 丢掉前 100 步，同一配方只加倍 DP 的 8 卡 tps / 4 卡 tps，与 GPU 的同一比值相差 ≤ 2 个百分点。不比绝对 tps |
| 内存 | 先比单卡同场景日志 `memory`（窗口最大预留），再比同场景多卡。`nvidia-smi` / `npu-smi` 整卡占用不是这项 |
| 确定性 | 同配置连跑两次，loss 和 grad_norm 的打印值逐项相同，并且和 GPU 同策略的打印值一致 |

`grad_norm` 不进入精度的 MAE 门槛。下面仍给出 grad_norm 的误差，只作旁证。

## 总表

| 清单行 | 配方 | 精度 | 性能 | 内存 | 确定性（NPU 自洽 / 对 GPU） |
| --- | --- | --- | --- | --- | --- |
| 1 | `muse_spark_debugmodel` 4 卡 | **达标**（相对误差 1.28%） | **不达标**（已测，差 3.20pp） | **达标** | 达标 / 不达标 |
| 2① | `with_cp` 4 卡 | **达标**（MAE 0.019，相对 0.73%） | **达标**（已测，差 1.35pp） | **不达标**（GPU 预留显著更高且随步爬升） | 达标 / 不达标 |
| 2② | `fsdp2_cp2_ep4` 4 卡 | **达标**（MAE 0.014，相对 0.58%） | **达标**（已测，差 1.17pp） | **不达标**（GPU 单卡爬升 1.46–3.85 vs NPU 0.25） | 达标 / 不达标 |
| 2③ | `mm_fsdp2_tp2_cp2_ep4` 8 卡 | **不达标**（MAE 0.031，相对 16.8%） | **达标**（已测，差 0.47pp） | **不达标**（GPU 单卡爬升 2.39–14.15 vs NPU 0.28） | 达标 / 不达标 |
| 3 | Dist-MoE bf16 | 不可做（无 GPU 基线） | 不可做 | 不可做 | 达标 / 不可做 |
| 4 | `debugmodel_mm` 8 卡 | **不达标**（MAE 0.031，相对 19.8%） | **达标**（已测，差 1.31pp） | **不达标**（GPU 单卡爬升 2.34–14.15 vs NPU 0.28） | 达标 / 不达标 |
| 5 | `ultradata_lora_sft` 8 卡 | **达标**（MAE 0.000835，相对 0.0075%） | **不达标**（已测，差 6.62pp） | **不达标**（NPU 单卡 13.25 vs GPU 0.24–0.54） | 达标 / 不达标 |

「未测」是协议要求的对照还没跑，不是测完失败。「不达标」是按字面已经可以下结论。1000 步队列已结束。**GPU 侧五行的 perf/mem 已于 2026-11-02 全部跑完**（HEAD `75c46f03` + `ultradata perf_dp1` EP 修复，独占 8 卡，日志 `logs/*_gpu.log`；详表见 `docs/gpu_baseline_perf_mem.md`「GPU 实测结果（2026-11-02）」节）。第 2①②③/4/5 性能与内存均已用现成 NPU 日志闭环（详表见 `docs/gpu_baseline_perf_mem.md`「闭环对照」）；2①②③/4 性能达标、5 性能不达标，五行内存均不达标。


## 不达标项总览与读法

总表红格多，但不是「整网算错」。按指标归类：

| 类型 | 行 | 一句话 |
| --- | --- | --- |
| 精度 | 2③、4 | debug 多模态过拟合后 loss ~1e-3，相对误差被小分母放大；第 1 步已对齐，不是未加载权重 |
| 性能 | 1 | 线性度差 3.20pp；**GPU** 加倍 DP 掉得更多（NPU 0.95 vs GPU 0.92），不是 NPU 绝对吞吐更差 |
| 性能 | 5 | 线性度差 6.62pp；**NPU** 掉得更多（0.899 vs GPU 0.965），32K LoRA + EP 加倍路径，与第 1 行方向相反 |
| 内存 A | 2①②③、4 | GPU 日志 `memory` 随步爬升（预留），NPU 平稳 ~0.25；对照第 1 行无 CP 单卡两边都 ~0.25 且不爬 → 跟 **CP / 多形状 flex / CUDA caching allocator 预留** 相关，不是整机口径坏了 |
| 内存 B | 5 | **方向反过来**：NPU 单卡 13.25 平稳，GPU 0.24–0.54；不能套用「GPU 预留虚高」，32K 路径要单独查 |
| 确定性对 GPU | 几乎全行 | NPU 自洽达标；跨设备打印值不一致是协议字面要求，bf16 路径本就不保证 bit-identical |

协议比的内存是日志窗口**最大预留**，不是 `npu-smi` / `nvidia-smi`。性能只比「只加倍 DP」的 8/4 卡线性度，不比绝对 tps。下面各行「原因分析」只解释本行不达标项。

---

## 第 1 行 `muse_spark_debugmodel`（4 卡，`dp_shard=2, tp=2, ep=2`）

### 精度 — 达标

两边加载同一份 step 0：`outputs/checkpoint-debugmodel/step-0`。GPU 曲线 `docs/gpu_baseline/debugmodel_1000.txt`，NPU 曲线 `docs/npu_baseline/debugmodel_1000.txt`（TensorBoard 全精度，1000 步对齐）。

| 窗口 | loss MAE | loss 平均相对误差 | 判定 |
| --- | --- | --- | --- |
| 1–1000 | 0.03105 | 1.28% | 相对误差达标 |
| 1–100 | 0.00583 | 0.19% | 两条都过 |
| 101–500 | 0.01995 | 0.77% | 两条都过 |
| 501–1000 | 0.04497 | 1.90% | 只过相对误差 |

第 1 步 loss：NPU 7.84031，GPU 7.84006，绝对差 2.5e-4。权重确实加载上了，不是各自初始化。第 1000 步：NPU 2.29888，GPU 2.34908。最大绝对差 0.086，在第 925 步。全程 NPU loss 略低于 GPU，差随步数变大，不是来回抖动。后 500 步的相对误差 1.90% 仍低于 2%，所以全段平均能过线，不是靠前 100 步把平均数拉下来的。

grad_norm 全段 MAE 0.045、相对误差 7.56%，两条都不过。最大绝对差 0.533，在第 360 步。这不改精度判定。

10 步各自初始化的 loss 差大约 +0.07～+0.13，和这次共享权重的第 1 步差 2.5e-4 不是同一件事。

### 性能 — 不达标（已测）

必跑三组用的是第 1 行同一模型、只改 DP，不是拿别的行来除。

| 端 | CONFIG | NGPU | 稳定 tps（step 101–200 均值） |
| --- | --- | --- | --- |
| NPU | `muse_spark_debugmodel_perf_dp2` | 4 | 322.89 |
| NPU | `muse_spark_debugmodel_perf_dp4` | 8 | 307.98 |
| GPU | `muse_spark_debugmodel_perf_dp2` | 4 | 1003.27 |
| GPU | `muse_spark_debugmodel_perf_dp4` | 8 | 924.81 |

线性度 = 8 卡 tps / 4 卡 tps。NPU 307.98 / 322.89 = **0.9538**。GPU 924.81 / 1003.27 = **0.9218**。差 |0.9538 − 0.9218| × 100 = **3.20 个百分点**，大于 2。

绝对 tps 不进门槛。A100 大约是 910B4 的 3 倍，协议不比这个。NPU 从 4 卡加到 8 卡掉 4.6% 吞吐，GPU 掉 7.8%。GPU 4 卡前后半段是 993.8 / 1012.8，平稳。8 卡采用 `logs/perf_dp4_gpu_r2.log`；首轮 `perf_dp4_gpu.log` 从约 420 掉到约 300，疑似争卡，作废。NPU 日志是 `logs/perf_dp2_npu_r2.log`、`logs/perf_dp4_npu_r3.log`。单卡内存日志是 `logs/mem_1device_npu.log`、`logs/mem_1device_gpu.log`。

拆开看，不达标只来自线性度这一条，不是 4 卡或 8 卡没跑完。

#### 原因分析（性能）

1. 门槛只看 |NPU 线性度 − GPU 线性度| ≤ 2pp。本行差 **3.20pp**，只超这一条；不是 4/8 卡没跑完。
2. 谁掉更多：NPU 4→8 掉 **4.6%**（线性度 0.9538），GPU 掉 **7.8%**（0.9218）。**GPU 扩展更差**，NPU 相对更好，但差值仍超阈值。
3. GPU 8 卡采用平稳的 `perf_dp4_gpu_r2.log`（首轮争卡已作废），不是脏数。
4. 与第 5 行方向相反（第 5 是 NPU 掉更多），不要用同一句话概括两行。
5. 若要翻盘：独占复测，或评估是否放宽到 3–4pp（改口径 ≠ 重跑自动过线）。


### 内存 — 达标

比的是日志 `memory`，丢掉第 1 步。

| 端 | 单卡 `mem_1device`（step 2–20） | 4 卡 `perf_dp2` 稳定段 |
| --- | --- | --- |
| NPU | 0.26GiB（step 1 为 0.33，含编译，已丢） | 0.11GiB |
| GPU | 0.24GiB（step 2–20 全是 0.24） | 0.11GiB |

单卡差 0.02GiB，同一量级。4 卡持平。

### 确定性 — NPU 达标，对 GPU 不达标

NPU、GPU 各自连跑两次，10 步 loss 和 grad_norm 的打印值相同。两边互比从第 1 步就不同，loss 差大约 0.07～0.13，10 步内没有放大。标准要求打印值一致，所以交叉对照不达标。tps 两次可以差一截，不拿来判确定性。

这次共享权重的 1000 步第 1 步已经对齐到 2.5e-4，说明 10 步那条差主要来自各卡自己的初始化，不是加载路径坏了。它仍然不能把 10 步确定性改成达标。

#### 原因分析（确定性对 GPU）

NPU 自洽（连跑两次打印值相同）已达标。跨设备要求打印值一致；bf16 + 不同后端本就不保证 bit-identical，第 1 步起即可有差、10 步内不发散。共享 step-0 的 1000 步精度项另算。本项按协议字面对 GPU 不达标，**不表示训练发散**。

**复验结果（2026-11-02）**：同一 `outputs/checkpoint-debugmodel/step-0` 重跑 1000 步完成——`docs/gpu_baseline/debugmodel_1000_r2.txt` 与 r1 **逐位相同**（MAE = 0），r2 对 NPU MAE 0.031 / 相对 1.28%，达标结论不变。日志 `logs/debugmodel_precision_r2_gpu.log`。

---

## 第 2 行① `muse_spark_debugmodel_with_cp`（4 卡，Ulysses）

### 精度 — 达标

step 0：`outputs/checkpoint-with_cp/step-0`。GPU `docs/gpu_baseline/with_cp_1000.txt`，NPU `docs/npu_baseline/with_cp_1000.txt`。

| 窗口 | loss MAE | loss 平均相对误差 |
| --- | --- | --- |
| 1–1000 | 0.01900 | 0.73% |
| 1–100 | 0.00978 | 0.32% |
| 101–500 | 0.01945 | 0.71% |
| 501–1000 | 0.02049 | 0.82% |

全段 MAE 和相对误差都过线。后 500 步 MAE 0.0205，刚过 0.02，但门槛用的是 1000 步平均。第 1 步 7.84976 / 7.84918。第 1000 步 2.35982 / 2.36794。最大绝对差 0.069，在第 662 步。差没有第 1 行后段那么大。

grad_norm 全段 MAE 0.048、相对误差 7.24%，不过线。不影响精度判定。

### 性能 — 达标（双侧已测）

2026-11-02 双侧补了同配方只加倍 DP 的两档（200 步，丢掉前 100 步）：

| 端 | CONFIG | 网格 | step>100 平均 tps |
| --- | --- | --- | --- |
| NPU | `muse_spark_debugmodel_with_cp_perf_dp1` | `dp_shard=1, cp=2, tp=2, ep=2`（4 卡） | **148.53** |
| NPU | `muse_spark_debugmodel_with_cp_perf_dp2` | `dp_shard=2, cp=2, tp=2, ep=2`（8 卡） | **144.21** |
| GPU | `muse_spark_debugmodel_with_cp_perf_dp1` | 同上（4 卡） | **485.99** |
| GPU | `muse_spark_debugmodel_with_cp_perf_dp2` | 同上（8 卡） | **478.41** |

NPU 线性度 144.21 / 148.53 = **0.9709**。GPU 线性度 478.41 / 485.99 = **0.9844**。差 |0.9709 − 0.9844| × 100 = **1.35 个百分点**，≤ 2，**达标**。GPU 前后半段 481.2 / 490.8，平稳。日志 `logs/with_cp_perf_dp1_npu.log`、`logs/with_cp_perf_dp2_npu.log`、`logs/with_cp_perf_dp1_gpu.log`、`logs/with_cp_perf_dp2_gpu.log`。

### 内存 — 不达标（双侧已测，GPU 预留显著更高）

| 端 | 单卡 `with_cp_mem_1device`（step 2–20） | 4 卡 `with_cp_perf_dp1` 稳定段 |
| --- | --- | --- |
| NPU | **0.25GiB**（`logs/with_cp_mem_1device_npu.log`） | **0.10GiB** |
| GPU | **1.47–3.85GiB，随步单调爬升，无稳定值**（末步 3.85；`logs/with_cp_mem_1device_gpu.log`） | **0.65GiB**（step>100 末值；`logs/with_cp_perf_dp1_gpu.log`） |

单卡 GPU 预留约为 NPU 的 6–15 倍且随步爬升，4 卡约 6.5 倍，不满足「同量级」，判不达标。两个可能方向需要 NPU 侧一起核查：GPU 端 caching allocator（`expandable_segments:True`）对多形状 flex attention 的预留增长，以及 CP→非 CP 内核回退路径的激活驻留差异。注意第 1 行 `mem_1device`（无 CP 内核）GPU 是平稳的 0.24GiB，与 NPU 0.26 同量级——说明爬升与 with_cp 这条配置路径相关，不是 GPU 整机口径问题。

#### 原因分析（内存 A：GPU 预留爬升）

1. 比的是日志 `memory`（窗口最大**预留**），不是整卡占用。GPU 单卡随步**单调爬升、无平台**；NPU 同配置几乎是常数。
2. 对照第 1 行 `mem_1device`（**无 CP**）：GPU 0.24 平稳 ≈ NPU 0.26 → **达标**。说明不是「GPU 整机口径坏了」，而是 **带 CP / 多形状 flex 的路径** 上 CUDA caching allocator 预留行为与 NPU 不同。
3. 机制解读（待口径确认后复测，不是已证实的算子泄漏）：
   - flex attention 按 mask / 序列形状编译；新 shape 再申请后，CUDA 常把块先占着不还（reserved 涨）。
   - `mem_1device` 把并行度打成 1（无 CP），与验收网格的 CP 路径不完全同一条内核。
   - 多模态行末值到 ~14GiB，还叠了视觉长度变化引起的重编译尖峰；`grid_sample` 反传走 CPU 也会进预留账本。
4. 本项不达标 = 同口径「预留」对不上量级，不宜写成「NPU 更省显存」或「GPU 泄漏」。复测建议：两侧统一 allocator 策略，取稳定平台而非末值再比。


### 确定性 — NPU 达标

各自两次打印值相同。

---

## 第 2 行② `muse_spark_debugmodel_fsdp2_cp2_ep4`（4 卡，AllGatherKV）

### 精度 — 达标

step 0：`outputs/checkpoint-fsdp2_cp2_ep4/step-0`。GPU `docs/gpu_baseline/fsdp2_cp2_ep4_1000.txt`，NPU `docs/npu_baseline/fsdp2_cp2_ep4_1000.txt`。2026-10-31 09:22–09:59 UTC 跑完，退出码 0。

| 窗口 | loss MAE | loss 平均相对误差 |
| --- | --- | --- |
| 1–1000 | 0.01422 | 0.58% |
| 1–100 | 0.00623 | 0.20% |
| 101–500 | 0.01235 | 0.47% |
| 501–1000 | 0.01732 | 0.73% |

全段 MAE 和相对误差都过线。第 1 步 7.84027 / 7.84010。第 1000 步 2.32123 / 2.33701。最大绝对差 0.065，在第 909 步。后段误差变大，和前两行一样，但 1000 步平均仍过线。

grad_norm 全段 MAE 0.046、相对误差 7.97%，不过线。不影响精度判定。10 步各自初始化的 loss 差大约 0.04～0.12，不是这项。

### 性能 — 达标（已闭环，差 1.17pp）

GPU（2026-11-02，`fsdp2_perf_dp2` / `fsdp2_perf_dp4`，200 步，step>100 各 100 步均值，趋势平稳）：

| 端 | CONFIG | 网格 | step>100 平均 tps |
| --- | --- | --- | --- |
| GPU | `muse_spark_debugmodel_fsdp2_cp2_ep4_perf_dp2` | `dp_shard=2, cp=2, tp=1, ep=4`（4 卡） | **335.02** |
| GPU | `muse_spark_debugmodel_fsdp2_cp2_ep4_perf_dp4` | `dp_shard=4, cp=2, tp=1, ep=4`（8 卡） | **324.08** |

GPU 线性度 324.08 / 335.02 = **0.9673**（前后半段 335.3 / 334.7）。NPU 同 CONFIG（2026-11-02）：121.64 / 116.24，线性度 **0.9556**。差 |0.9556 − 0.9673| × 100 = **1.17 个百分点**，≤ 2，**达标**。日志 `logs/fsdp2_perf_dp2_gpu.log`、`logs/fsdp2_perf_dp4_gpu.log`、`logs/fsdp2_perf_dp2_npu.log`、`logs/fsdp2_perf_dp4_npu.log`。

### 内存 — 不达标（GPU 单卡预留显著更高且随步爬升）

GPU 单卡（`fsdp2_mem_1device`，step2–20）：**1.46–3.85GiB，随步爬升，无稳定值**（末步 3.85）。GPU 4 卡（perf_dp2 step>100 末值）：**0.49GiB**。NPU 单卡 **0.25GiB**、4 卡 **0.12GiB**。单卡约 6–15 倍、4 卡约 4 倍，判不达标；爬升现象与第 2① 一致，建议两侧一起核查 allocator 口径后复测。

#### 原因分析（内存 A：GPU 预留爬升）

1. 比的是日志 `memory`（窗口最大**预留**），不是整卡占用。GPU 单卡随步**单调爬升、无平台**；NPU 同配置几乎是常数。
2. 对照第 1 行 `mem_1device`（**无 CP**）：GPU 0.24 平稳 ≈ NPU 0.26 → **达标**。说明不是「GPU 整机口径坏了」，而是 **带 CP / 多形状 flex 的路径** 上 CUDA caching allocator 预留行为与 NPU 不同。
3. 机制解读（待口径确认后复测，不是已证实的算子泄漏）：
   - flex attention 按 mask / 序列形状编译；新 shape 再申请后，CUDA 常把块先占着不还（reserved 涨）。
   - `mem_1device` 把并行度打成 1（无 CP），与验收网格的 CP 路径不完全同一条内核。
   - 多模态行末值到 ~14GiB，还叠了视觉长度变化引起的重编译尖峰；`grid_sample` 反传走 CPU 也会进预留账本。
4. 本项不达标 = 同口径「预留」对不上量级，不宜写成「NPU 更省显存」或「GPU 泄漏」。复测建议：两侧统一 allocator 策略，取稳定平台而非末值再比。


### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。loss 差大约 0.04～0.12。第 10 步 grad_norm 差收到 0.003 以内，轨迹没有散开，打印值仍然不同。

#### 原因分析（确定性对 GPU）

NPU 自洽（连跑两次打印值相同）已达标。跨设备要求打印值一致；bf16 + 不同后端本就不保证 bit-identical，第 1 步起即可有差、10 步内不发散。共享 step-0 的 1000 步精度项另算。本项按协议字面对 GPU 不达标，**不表示训练发散**。


---

## 第 2 行③ `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`（8 卡）

### 精度 — 不达标

step 0：`outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0`。GPU `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`，NPU `docs/npu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`。2026-10-31 09:59–11:04 UTC 跑完，退出码 0。这组数不能填第 4 行。

| 窗口 | loss MAE | loss 平均相对误差 |
| --- | --- | --- |
| 1–1000 | 0.03100 | 16.82% |
| 1–100 | 0.03596 | 2.73% |
| 101–500 | 0.04964 | 12.41% |
| 501–1000 | 0.01510 | 23.17% |

两条门槛都没过：MAE 0.031 大于 0.02，相对误差 16.82% 大于 2%。第 1 步 7.79217 / 7.78986，加载是对齐的。第 1000 步两边都掉到 0.00186 / 0.00340，绝对差很小。最大绝对差 0.270，在第 402 步。后 500 步 MAE 0.015 单独看能过绝对门槛，但全段平均过不了；后段相对误差被 GPU loss 掉到约 0.003 放大了。

grad_norm 全段 MAE 0.648、相对误差 41.6%，最大绝对差 8.41 在第 396 步。不进精度判定。10 步各自初始化的差不是这项。

#### 原因分析（精度）

1. 第 1 步两端 loss 已对齐到 ~1e-3，**权重加载正确**，不是各自初始化。
2. 全段 MAE≈0.031 > 0.02，MRE≈17–20% > 2%，两条都不过。后 500 步 MAE≈0.014–0.015 **单独能过绝对门槛**，但 loss 已到 ~1e-3，相对误差被小分母放大到 20%+。
3. 根因：**debug 宽度多模态过拟合**——绝对差变小、相对门槛变严。与「算子整层算错」不符（第 5 行真实尺度精度 MAE 0.000835 很好）。
4. 最大绝对差落在视觉长度变化 / flex 重编译附近；两边都有编译步 tps 掉到个位数，不能单边当 NPU 故障。
5. GPU 同 step-0 复验见 `docs/gpu_agent_precision_reverify_todo.md`；按现门槛预期仍不达标。改窗口或只认 MAE 是改口径。
6. 第 2③ 与第 4 行曲线**不可互填**。


**复验结果（2026-11-02，实测）**：GPU step-0 重跑 1000 步完成——`docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_r2.txt` 与 r1 **逐位相同**（MAE = 0），r2 对 NPU MAE 0.031 / MRE 16.82%，与原判定一致，不达标结论稳定可复现（原因分析第 5 点的「预期」已被实测确认）。日志 `logs/mm_fsdp2_precision_r2_gpu.log`。
### 性能 — 达标（已闭环）

GPU（2026-11-02，`mm_fsdp2_perf_dp1` / `mm_fsdp2_perf_dp2`，150 步，step>100 共 50 步均值，前后半段 136.0/132.4 与 111.3/107.5，平稳）：

| 端 | CONFIG | 网格 | step>100 平均 tps |
| --- | --- | --- | --- |
| GPU | `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_perf_dp1` | `dp_shard=1, cp=2, tp=2, ep=4`（4 卡） | **134.18** |
| GPU | `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_perf_dp2` | `dp_shard=2, cp=2, tp=2, ep=4`（8 卡） | **109.40** |

GPU 线性度 109.40 / 134.18 = **0.8153**。NPU 同 CONFIG（2026-11-02）：`mm_fsdp2_perf_dp1` / `perf_dp2` step>100 平均 tps **47.94** / **38.86**，线性度 **0.8106**。差 |0.8106 − 0.8153| × 100 = **0.47 个百分点**，≤ 2，**达标**（两侧都受视觉长度重编译影响，比值高度一致）。日志 `logs/mm_fsdp2_perf_dp1_gpu.log`、`logs/mm_fsdp2_perf_dp2_gpu.log`、`logs/mm_fsdp2_perf_dp1_npu.log`、`logs/mm_fsdp2_perf_dp2_npu.log`。

### 内存 — 不达标（已闭环）

| 端 | 单卡 `mm_fsdp2_mem_1device`（step2–20） | 4 卡 `perf_dp1` 稳定段 |
| --- | --- | --- |
| NPU | **0.28GiB**（`logs/mm_fsdp2_mem_1device_npu.log`） | **0.15GiB** |
| GPU | **2.39–14.15GiB，随步爬升，无稳定值**（末 14.15；`logs/mm_fsdp2_mem_1device_gpu.log`） | **2.24GiB** |

单卡约 8–50 倍、4 卡约 15 倍，不同量级。GPU 爬升与第 2①② 同类；`grid_sample` 反传走 CPU 也计入预留。

#### 原因分析（内存 A：GPU 预留爬升）

1. 比的是日志 `memory`（窗口最大**预留**），不是整卡占用。GPU 单卡随步**单调爬升、无平台**；NPU 同配置几乎是常数。
2. 对照第 1 行 `mem_1device`（**无 CP**）：GPU 0.24 平稳 ≈ NPU 0.26 → **达标**。说明不是「GPU 整机口径坏了」，而是 **带 CP / 多形状 flex 的路径** 上 CUDA caching allocator 预留行为与 NPU 不同。
3. 机制解读（待口径确认后复测，不是已证实的算子泄漏）：
   - flex attention 按 mask / 序列形状编译；新 shape 再申请后，CUDA 常把块先占着不还（reserved 涨）。
   - `mem_1device` 把并行度打成 1（无 CP），与验收网格的 CP 路径不完全同一条内核。
   - 多模态行末值到 ~14GiB，还叠了视觉长度变化引起的重编译尖峰；`grid_sample` 反传走 CPU 也会进预留账本。
4. 本项不达标 = 同口径「预留」对不上量级，不宜写成「NPU 更省显存」或「GPU 泄漏」。复测建议：两侧统一 allocator 策略，取稳定平台而非末值再比。


### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。跨设备差不发散，打印值不同。

#### 原因分析（确定性对 GPU）

NPU 自洽（连跑两次打印值相同）已达标。跨设备要求打印值一致；bf16 + 不同后端本就不保证 bit-identical，第 1 步起即可有差、10 步内不发散。共享 step-0 的 1000 步精度项另算。本项按协议字面对 GPU 不达标，**不表示训练发散**。


---

## 第 3 行 Dist-MoE bf16

### 精度、性能、内存 — 不可做

A100 是 SM80。官方 Dist-MoE / mxfp8 要 SM100，没有 GPU 1000 步，也没有同配方第二档卡数，也没有 GPU 预留。不编造对照。NPU 自己的 10 步：loss 从 7.68369 到 5.36020，稳定段约 355 tps，预留 0.09GiB。第 1 步离 `ln(2048) = 7.6246` 差 0.059，只说明 NPU 损失尺度，不是 MAE。

### 确定性 — NPU 达标，对 GPU 不可做

`HCCL_DETERMINISTIC=true` 下连跑两次，10 步 loss 和 grad_norm 逐项相同。没有 GPU 同策略结果。

---

## 第 4 行 `muse_spark_debugmodel_mm`（8 卡，EP2，Ulysses）

### 精度 — 不达标

step 0：`outputs/checkpoint-debugmodel_mm/step-0`。GPU `docs/gpu_baseline/debugmodel_mm_1000.txt`，NPU `docs/npu_baseline/debugmodel_mm_1000.txt`。2026-10-31 11:04–11:26 UTC 跑完，退出码 0。不要用第 2 行③的曲线填这里。

| 窗口 | loss MAE | loss 平均相对误差 |
| --- | --- | --- |
| 1–1000 | 0.03067 | 19.81% |
| 1–100 | 0.03704 | 2.64% |
| 101–500 | 0.04971 | 13.94% |
| 501–1000 | 0.01417 | 27.95% |

两条门槛都没过：MAE 0.031 大于 0.02，相对误差 19.81% 大于 2%。第 1 步 7.79205 / 7.79068。第 1000 步 0.00604 / 0.00211。最大绝对差 0.272，在第 187 步。后 500 步 MAE 0.014 单独看能过绝对门槛，全段平均过不了；后段相对误差被 GPU loss 掉到约 0.002 放大了。

grad_norm 全段 MAE 0.665、相对误差 67.5%，最大绝对差 12.87 在第 443 步。不进精度判定。10 步各自初始化的差不是这项。

#### 原因分析（精度）

1. 第 1 步两端 loss 已对齐到 ~1e-3，**权重加载正确**，不是各自初始化。
2. 全段 MAE≈0.031 > 0.02，MRE≈17–20% > 2%，两条都不过。后 500 步 MAE≈0.014–0.015 **单独能过绝对门槛**，但 loss 已到 ~1e-3，相对误差被小分母放大到 20%+。
3. 根因：**debug 宽度多模态过拟合**——绝对差变小、相对门槛变严。与「算子整层算错」不符（第 5 行真实尺度精度 MAE 0.000835 很好）。
4. 最大绝对差落在视觉长度变化 / flex 重编译附近；两边都有编译步 tps 掉到个位数，不能单边当 NPU 故障。
5. GPU 同 step-0 复验见 `docs/gpu_agent_precision_reverify_todo.md`；按现门槛预期仍不达标。改窗口或只认 MAE 是改口径。
6. 第 2③ 与第 4 行曲线**不可互填**。


**复验结果（2026-11-02，实测）**：GPU step-0 重跑 1000 步完成——`docs/gpu_baseline/debugmodel_mm_1000_r2.txt` 与 r1 **逐位相同**（MAE = 0），r2 对 NPU MAE 0.031 / MRE 19.81%，与原判定一致，不达标结论稳定可复现（原因分析第 5 点的「预期」已被实测确认）。日志 `logs/debugmodel_mm_precision_r2_gpu.log`。
### 性能 — 达标（已闭环，差 1.31pp）

GPU（2026-11-02，`mm_perf_dp1` / `mm_perf_dp2`，150 步，step>100 共 50 步均值，前后半段 383.2/365.5 与 353.5/362.7）：

| 端 | CONFIG | 网格 | step>100 平均 tps |
| --- | --- | --- | --- |
| GPU | `muse_spark_debugmodel_mm_perf_dp1` | `dp_shard=1, cp=2, tp=2, ep=2`（4 卡） | **374.36** |
| GPU | `muse_spark_debugmodel_mm_perf_dp2` | `dp_shard=2, cp=2, tp=2, ep=2`（8 卡） | **358.08** |

GPU 线性度 358.08 / 374.36 = **0.9565**。NPU 同 CONFIG（2026-11-02）：`mm_perf_dp1` / `mm_perf_dp2` step>100 平均 tps **118.56** / **114.96**，线性度 **0.9696**。差 |0.9696 − 0.9565| × 100 = **1.31 个百分点**，≤ 2，**达标**。日志 `logs/mm_perf_dp1_gpu.log`、`logs/mm_perf_dp2_gpu.log`、`logs/mm_perf_dp1_npu.log`、`logs/mm_perf_dp2_npu.log`。

### 内存 — 不达标（GPU 单卡预留显著更高且随步爬升）

GPU 单卡（`mm_mem_1device`，step2–20）：**2.34–14.15GiB，随步爬升，无稳定值**（末步 14.15）。GPU 4 卡（perf_dp1 step>100 末值）：**3.19GiB**。NPU 单卡 **0.28GiB**、4 卡 **0.15GiB**。单卡约 8–50 倍、4 卡约 21 倍，判不达标；爬升现象与第 2①② 一致。

#### 原因分析（内存 A：GPU 预留爬升）

1. 比的是日志 `memory`（窗口最大**预留**），不是整卡占用。GPU 单卡随步**单调爬升、无平台**；NPU 同配置几乎是常数。
2. 对照第 1 行 `mem_1device`（**无 CP**）：GPU 0.24 平稳 ≈ NPU 0.26 → **达标**。说明不是「GPU 整机口径坏了」，而是 **带 CP / 多形状 flex 的路径** 上 CUDA caching allocator 预留行为与 NPU 不同。
3. 机制解读（待口径确认后复测，不是已证实的算子泄漏）：
   - flex attention 按 mask / 序列形状编译；新 shape 再申请后，CUDA 常把块先占着不还（reserved 涨）。
   - `mem_1device` 把并行度打成 1（无 CP），与验收网格的 CP 路径不完全同一条内核。
   - 多模态行末值到 ~14GiB，还叠了视觉长度变化引起的重编译尖峰；`grid_sample` 反传走 CPU 也会进预留账本。
4. 本项不达标 = 同口径「预留」对不上量级，不宜写成「NPU 更省显存」或「GPU 泄漏」。复测建议：两侧统一 allocator 策略，取稳定平台而非末值再比。


### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。跨设备从第 1 步不同，差不发散。

#### 原因分析（确定性对 GPU）

NPU 自洽（连跑两次打印值相同）已达标。跨设备要求打印值一致；bf16 + 不同后端本就不保证 bit-identical，第 1 步起即可有差、10 步内不发散。共享 step-0 的 1000 步精度项另算。本项按协议字面对 GPU 不达标，**不表示训练发散**。


---

## 第 5 行 `muse_spark_debugmodel_ultradata_lora_sft`（8 卡，32K LoRA）

### 精度 — 达标

两边加载同一份 step 0：`outputs/checkpoint-ultradata_lora/step-0`（sha256 `54f674d008d41141e38a4c971e669d3b22f80a464e3567e88c811237e0b31102`）。GPU `docs/gpu_baseline/ultradata_lora_1000.txt`，NPU `docs/npu_baseline/ultradata_lora_1000.txt`。2026-10-31 11:26–18:26 UTC 跑完，退出码 0。`ln(202048) ≈ 12.216`，第 1 步尺度符合只训 LoRA 的大词表，不是 debug 词表 2048。

| 窗口 | loss MAE | loss 平均相对误差 |
| --- | --- | --- |
| 1–1000 | 0.000835 | 0.0075% |
| 1–100 | 0.000362 | 0.0030% |
| 101–500 | 0.000876 | 0.0077% |
| 501–1000 | 0.000897 | 0.0082% |

两条门槛都过了：MAE 0.000835 小于 0.02，相对误差 0.0075% 小于 2%。第 1 步 12.3629 / 12.3628。第 1000 步 10.1016 / 10.1023。最大绝对差 0.00401，在第 800 步。

grad_norm 全段 MAE 0.0106、相对误差 2.73%，最大绝对差 0.453 在第 845 步。不进精度判定。10 步跨设备 |loss| ≤ 0.027 只说明量级，正式数是这份 MAE。

### 性能 — 不达标（已闭环，差 6.62pp）

| 端 | CONFIG | 网格 | step>100 平均 tps |
| --- | --- | --- | --- |
| NPU | `..._ultradata_lora_sft_perf_dp1` | `dp_shard=1, cp=2, tp=2, ep=4`（4 卡，120 步） | **378.25** |
| NPU | `..._ultradata_lora_sft_perf_dp2` | `dp_shard=2, cp=2, tp=2, ep=8`（8 卡，120 步） | **340.05** |
| GPU | 同 CONFIG | 同上 | **10230.50** / **9874.95** |

NPU 线性度 340.05 / 378.25 = **0.8990**；GPU **0.9652**。差 |0.8990 − 0.9652| × 100 = **6.62 个百分点 > 2**，不达标。绝对 tps 口径与第 1 行不同，不跨行比吞吐。NPU 日志：`logs/ultradata_perf_dp1_npu.log`、定稿 8 卡 `logs/ultradata_perf_dp2_npu.ep8.log`（`Training completed`；其后有人覆盖重跑 `ultradata_perf_dp2_npu.log`，验收以 ep8 定稿文件为准）。GPU：`logs/ultradata_perf_dp1_gpu.log`、`logs/ultradata_perf_dp2_gpu.log`。

#### 原因分析（性能）

1. NPU 线性度 **0.8990**，GPU **0.9652**，差 **6.62pp > 2**。本行是 **NPU 加倍 DP（且 EP 4→8）掉得更多**，与第 1 行「GPU 掉更多」方向相反。
2. 绝对 tps（GPU ~1e4 vs NPU ~350）不可与第 1 行横比，也不进本项门槛。
3. 配置特点：32K 上下文 + LoRA + `perf_dp1` EP=4 / `perf_dp2` EP=8。扩展变差更可能来自 **长序列通信 / EP 加倍 / HCCL 路径**，不是短序列 debug 那套。
4. 8 卡定稿用 `logs/ultradata_perf_dp2_npu.ep8.log`；同名 `ultradata_perf_dp2_npu.log` 曾被中途重跑覆盖，验收以 ep8 为准。
5. 后续：对 NPU 4/8 卡做通信与计算拆分 profiling；确认 EP 折减与 GPU 一致后再判是否实现问题。


### 内存 — 不达标（已闭环）

| 端 | 单卡 `ultradata_mem_1device`（step2–20） | 4 卡 `perf_dp1` 稳定段 |
| --- | --- | --- |
| NPU | **13.25GiB**（平稳；`logs/ultradata_mem_1device_npu.log`） | **4.32GiB** |
| GPU | **0.24–0.54GiB，随步爬升**（末 0.54；`logs/ultradata_mem_1device_gpu.log`） | **0.44GiB** |

单卡 NPU 约为 GPU 末值的 24 倍，4 卡约 10 倍，不同量级。与第 2/4 行「GPU 预留偏高」方向相反，32K LoRA 路径两侧预留行为都需单独核查，不能和 debug 短序列行混谈。

#### 原因分析（内存 B：方向反过来）

1. 与第 2/4 行相反：这里是 **NPU 预留远高于 GPU**（单卡 13.25 vs 0.24–0.54，约 24×；4 卡约 10×）。
2. 不能套用「CUDA caching allocator 虚高」：本行 GPU 数字反而更小，且只有轻微爬升。
3. 本行是 **32K 长序列 + LoRA**，激活 / KV / 适配器驻留与短序列 debug 行不是同一量级问题；NPU 平稳 13.25 更像「真的多占了一块」或「长序列路径统计口径不同」，需要单独查（activation checkpoint、LoRA 参数驻留、预留字段语义），不能和 CP 爬升行混谈。
4. 按协议「同量级」字面仍不达标；复测前先确认两侧 `memory` 字段含义与是否含编译缓存。


### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。交叉差最小，仍不满足打印值一致。

#### 原因分析（确定性对 GPU）

NPU 自洽（连跑两次打印值相同）已达标。跨设备要求打印值一致；bf16 + 不同后端本就不保证 bit-identical，第 1 步起即可有差、10 步内不发散。共享 step-0 的 1000 步精度项另算。本项按协议字面对 GPU 不达标，**不表示训练发散**。


---

## 验证要点里补做的核对

这些格子原先写着没有单独构造、没有打印、没有逐层量过。2026-10-31 在 CPU 上补了，没有占用 1000 步队列的卡。脚本是 `/workspace/verify_open_points.py`。

| 要点 | 核对结果 |
| --- | --- |
| 无媒体 batch 的 dummy | 无 pixel、无 waveform 时视觉和音频 dummy 的绝对值和都是 0，加到 hidden 上的差是 0 |
| 模态是否每步都出现 | 静态集合是 vision 和 audio。10 个 microbatch 里图像 7、视频 5、音频 3，缺的走 dummy |
| bank index 与 token 分片 | 下标是切分前的绝对行号。CP=2 的 rank1 从 3 接着数，不从 0 重计 |
| pooler 布局 | weight 在 dp/cp/tp 上复制，激活按序列并行切 |
| 序列并行两种布局 | 打开时 token 轴是 (dp, cp, tp)，关掉时是 (dp, cp)，bank index 跟着 token 轴 |
| 梯度累积延迟归约 | with_cp、fsdp2、debugmodel_mm 的 accumulation steps 都是 1，defer 关闭，延迟归约进不去 |
| 滑窗 / 全局 NoPE | 层 3 和 7 无 RoPE、无窗。其余窗宽 128。滑窗是因果再加距离限制，谓词在 CPU 上对过 |
| assistant-only 标签 | 前 3 条有效 label 4626、6512、5426。system、user、tool 的监督数是 0 |
| LoRA 梯度 | 40 个投影的基座梯度都是 0。lora_B 有 39 个非 0。12 个 lora_A 为 0，因为对应的 B 初值是 0 |

CPU 上的 flex attention 没有反向，LoRA 是逐层投影反传，不是 8 卡整网一步。队列已经结束。**第 2①②③、4、5 行的 GPU perf/mem 已于 2026-11-02 全部补齐**（见 `docs/gpu_baseline_perf_mem.md`「GPU 实测结果（2026-11-02）」节），NPU 对应行已于 2026-11-02 补齐并完成对照判定。

## 现在在跑的 NPU 任务

2026-10-31 用 `ENTRYPOINT=torchtitan_ascend.train` 重跑。第一次用 `torchtitan.train` 时，`debugmodel` 在第 5 步段错误，`with_cp` / `fsdp2` 在编译 loss 时撞上 `make_reduction(..., strict_reduction=...)`。那三份不作数。

| 顺序 | 状态 |
| --- | --- |
| `debugmodel` | 完成。精度达标，见上 |
| `with_cp` | 完成。精度达标，见上 |
| `fsdp2_cp2_ep4` | 完成。精度达标，MAE 0.01422，相对 0.58% |
| `mm_fsdp2_tp2_cp2_ep4` | 完成。精度不达标，MAE 0.03100，相对 16.82% |
| `debugmodel_mm` | 完成。精度不达标，MAE 0.03067，相对 19.81% |
| `ultradata_lora` | 完成。精度达标，MAE 0.000835，相对 0.0075% |

每份导出后写入 `/workspace/npu_load_1000/<名>/mae.txt`，六份都已抄到 `docs/npu_baseline/`。队列已于 18:26 UTC 结束。第 2① with_cp 的 NPU 性能/单卡内存已于 2026-11-02 另跑；**GPU / NPU 五行 perf/mem 均已于 2026-11-02 闭环**（第 1 行此前已闭环）。

## 和旧段落的关系

此前总表把六行精度都写成「不达标」，并把 step 0 写成还在 GPU 机上。那是权重入库之前的状态。第 1 行性能、内存在 `docs/gpu_baseline_perf_mem.md` 末节已经闭环，总表后部却仍写「GPU 同配方未跑」。以本文总表和第 1 行、第 2 行①的计算为准。逐项原始表仍留在 `docs/npu_gpu_row1_perf_mem_analysis.md`，不在那里改判定。
