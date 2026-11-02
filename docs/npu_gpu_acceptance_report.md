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
| 2① | `with_cp` 4 卡 | **达标**（MAE 0.019，相对 0.73%） | NPU 已测（线性度 0.9709），待 GPU | NPU 已测（单卡 0.25 / 4 卡 0.10），待 GPU | 达标 / 不达标 |
| 2② | `fsdp2_cp2_ep4` 4 卡 | **达标**（MAE 0.014，相对 0.58%） | 未测 | 未测（无单卡） | 达标 / 不达标 |
| 2③ | `mm_fsdp2_tp2_cp2_ep4` 8 卡 | **不达标**（MAE 0.031，相对 16.8%） | 未测 | 未测（无单卡） | 达标 / 不达标 |
| 3 | Dist-MoE bf16 | 不可做（无 GPU 基线） | 不可做 | 不可做 | 达标 / 不可做 |
| 4 | `debugmodel_mm` 8 卡 | **不达标**（MAE 0.031，相对 19.8%） | 未测 | 未测（无单卡） | 达标 / 不达标 |
| 5 | `ultradata_lora_sft` 8 卡 | **达标**（MAE 0.000835，相对 0.0075%） | 未测 | 未测（无单卡） | 达标 / 不达标 |

「未测」是协议要求的对照还没跑，不是测完失败。「不达标」是按字面已经可以下结论。1000 步队列已结束。第 2① with_cp 的 NPU 性能/单卡内存已于 2026-11-02 补跑；GPU 对照以及第 4、5 行仍缺。

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

### 性能 — NPU 已测，待 GPU

2026-11-02 补了同配方只加倍 DP 的两档（200 步，丢掉前 100 步）：

| 端 | CONFIG | 网格 | step>100 平均 tps |
| --- | --- | --- | --- |
| NPU | `muse_spark_debugmodel_with_cp_perf_dp1` | `dp_shard=1, cp=2, tp=2, ep=2`（4 卡） | **148.53** |
| NPU | `muse_spark_debugmodel_with_cp_perf_dp2` | `dp_shard=2, cp=2, tp=2, ep=2`（8 卡） | **144.21** |

NPU 线性度 144.21 / 148.53 = **0.9709**。日志 `logs/with_cp_perf_dp1_npu.log`、`logs/with_cp_perf_dp2_npu.log`。GPU 同 CONFIG 还没跑，差几个百分点判不了，不能标达标或失败。

### 内存 — NPU 已测，待 GPU

| 端 | 单卡 `with_cp_mem_1device`（step 2–20） | 4 卡 `with_cp_perf_dp1` 稳定段 |
| --- | --- | --- |
| NPU | **0.25GiB**（`logs/with_cp_mem_1device_npu.log`） | **0.10GiB** |
| GPU | 未跑 | 约 0.10GiB（旧 10 步预留，不能代替单卡） |

单卡是全并行度 1、无 CP。GPU 同 CONFIG 未跑，不能标达标。

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

### 性能 — 未测

稳定段大约 NPU 123 tps、GPU 319–331 tps，这是绝对吞吐，而且不满 100 步，没有同配方 8 卡。不能算线性度。

### 内存 — 未测

单卡没跑。4 卡稳定预留两边都是 **0.12GiB**。同量级，缺单卡档。

### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。loss 差大约 0.04～0.12。第 10 步 grad_norm 差收到 0.003 以内，轨迹没有散开，打印值仍然不同。

---

## 第 2 行③ `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`（8 卡）

### 精度 — 不达标

#### 精度原因分析

正确跑法：加载同一份 `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0`，NPU / GPU 各 ≥1000 步；MoE 门槛为 MAE≤0.02 **或** MRE≤2%（全段平均）。曲线：`docs/npu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`、`docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`。摘要：`docs/npu_baseline/mm_fsdp2_tp2_cp2_ep4_mae.txt`。

| 窗口 | loss MAE | loss MRE | 读法 |
| --- | --- | --- | --- |
| 1–1000 | 0.03100 | 16.82% | 全段：两条门槛都不过 |
| 1–100 | 0.03596 | 2.73% | 前段相对误差已略超 2% |
| 101–500 | 0.04964 | 12.41% | 中段绝对差最大档 |
| 501–1000 | 0.01510 | 23.17% | MAE 单独可过 0.02，MRE 反而更高 |

**结论：不达标。**

排除「跑法/加载错误」：

- 第 1 步 NPU 7.79217 / GPU 7.78986（差约 2.3e-3）。各自随机初始化的 10 步差是 0.03～0.15 量级，不是同一回事。
- 训练日志有 `Loading the checkpoint from .../checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0`。退出码 0；loss / grad_norm 全程有限。

主因（按正确 1000 步曲线）：

1. debug 多模态尺度下两端都过拟合：第 1000 步 NPU ≈0.00186、GPU ≈0.00340。绝对差很小，但分母到 1e-3 后，MRE 被放大。
2. 后 500 步最能说明问题：MAE 0.015 单独能过绝对门槛，MRE 23.17%——「绝对更近、相对更差」，是小 loss 分母效应，不是前 100 步把全段平均拉爆。
3. 最大绝对差 0.270 约在第 402 步，与视觉打包长度变化、flex 重编译步重合。两边都会出现编译步 tps 掉到个位数，不能单独当成 NPU 算子故障。
4. GPU 路径上 `grid_sample` 反传曾走 CPU，数值栈与 NPU 不完全同构，会贡献一部分设备差；即便忽略这一层，按现门槛 MRE 仍难进 2%。

边界：

- 本行曲线**不得**填第 4 行 `muse_spark_debugmodel_mm`（另一套配置）。
- 同 step-0 的 GPU 曲线复验见 `docs/gpu_agent_precision_reverify_todo.md`，用于确认可复现；按现门槛预期仍不达标。若只改判定窗口或标准，那是改口径，不是重验自动翻盘。

`grad_norm` 全段 MAE 0.648、MRE 41.6%，不进入精度门槛。

### 性能 — 未测

没有同配方 4 卡。第 5、8 步 tps 掉到十来个，是视觉 token 长度触发重新编译，GPU 同样掉到 15。这不是线性度失败，是编译，两边都有。

### 内存 — 未测

单卡没跑。日志预留 NPU 稳定约 0.12–0.26GiB，GPU 约 0.14GiB。GPU `nvidia-smi` 约 6.8GiB/卡是整卡占用，不能拿来判这项。多模态 GPU 上 `grid_sample` 反传走了 CPU，loss 和预留都含这一层，所以即使用 10 步预留「同量级」也不能写成达标。

### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。跨设备差不发散，打印值不同。

---

## 第 3 行 Dist-MoE bf16

### 精度、性能、内存 — 不可做

A100 是 SM80。官方 Dist-MoE / mxfp8 要 SM100，没有 GPU 1000 步，也没有同配方第二档卡数，也没有 GPU 预留。不编造对照。NPU 自己的 10 步：loss 从 7.68369 到 5.36020，稳定段约 355 tps，预留 0.09GiB。第 1 步离 `ln(2048) = 7.6246` 差 0.059，只说明 NPU 损失尺度，不是 MAE。

### 确定性 — NPU 达标，对 GPU 不可做

`HCCL_DETERMINISTIC=true` 下连跑两次，10 步 loss 和 grad_norm 逐项相同。没有 GPU 同策略结果。

---

## 第 4 行 `muse_spark_debugmodel_mm`（8 卡，EP2，Ulysses）

### 精度 — 不达标

#### 精度原因分析

正确跑法：加载同一份 `outputs/checkpoint-debugmodel_mm/step-0`，各 ≥1000 步；门槛同上。曲线：`docs/npu_baseline/debugmodel_mm_1000.txt`、`docs/gpu_baseline/debugmodel_mm_1000.txt`。摘要：`docs/npu_baseline/debugmodel_mm_mae.txt`。

| 窗口 | loss MAE | loss MRE |
| --- | --- | --- |
| 1–1000 | 0.03067 | 19.81% |
| 1–100 | 0.03704 | 2.64% |
| 101–500 | 0.04971 | 13.94% |
| 501–1000 | 0.01417 | 27.95% |

**结论：不达标。** 第 1 步 7.79205 / 7.79068，权重已对齐。第 1000 步两端 loss 到 ~1e-3；后 500 步 MAE 0.014 可单独过绝对门槛，但 MRE 27.95%。机制与第 2 行③相同：debug 多模态过拟合后相对误差被分母放大。**不得与第 2 行③曲线混用。** GPU 复验见 `docs/gpu_agent_precision_reverify_todo.md`。`grad_norm` MAE 0.665 / MRE 67.5% 不进精度门槛。

### 性能 — 未测

没有同配方 4 卡，10 步也不满 100 步稳定段。NPU 稳定约 120 tps，GPU 约 82–90，只能看见绝对吞吐，不能算线性度。

### 内存 — 未测

单卡没跑。稳定预留 NPU 0.12GiB，GPU 约 0.10GiB，同量级，缺单卡档。

### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。跨设备从第 1 步不同，差不发散。

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

### 性能 — 未测

没有第二档卡数，也没有丢掉前 100 步的同配方比值。NPU 10 步稳定约 347 tps。GPU 稳定档可以到很高的 tps，序列和批的计量不同，不能直接相除。

### 内存 — 未测

单卡没跑。NPU 稳定预留约 4.3GiB（两次略有不同），GPU 日志约 **3.46GiB**。32K 预留比 debug 配方高一个量级，两边仍是同一量级，但缺单卡档，不能标达标。

### 确定性 — NPU 达标，对 GPU 不达标

各自两次相同。交叉差最小，仍不满足打印值一致。

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

CPU 上的 flex attention 没有反向，LoRA 是逐层投影反传，不是 8 卡整网一步。队列已经结束。第 2① with_cp 的 NPU 性能/单卡内存已补跑（见上）；第 4、5 行以及 with_cp 的 GPU 对照仍缺。

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

每份导出后写入 `/workspace/npu_load_1000/<名>/mae.txt`，六份都已抄到 `docs/npu_baseline/`。队列已于 18:26 UTC 结束。第 2① with_cp 的 NPU 性能/单卡内存已于 2026-11-02 另跑；GPU 同 CONFIG 以及第 4 / 5 行变体仍缺。第 1 行那三组 GPU 必跑已经做完。

## 和旧段落的关系

此前总表把六行精度都写成「不达标」，并把 step 0 写成还在 GPU 机上。那是权重入库之前的状态。第 1 行性能、内存在 `docs/gpu_baseline_perf_mem.md` 末节已经闭环，总表后部却仍写「GPU 同配方未跑」。以本文总表和第 1 行、第 2 行①的计算为准。逐项原始表仍留在 `docs/npu_gpu_row1_perf_mem_analysis.md`，不在那里改判定。
