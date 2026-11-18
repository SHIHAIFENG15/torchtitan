# 多模态精度验收报告（host-rope 轮）

本报告由 AI 助手整理。数据来自 `e284eff1`+ 代码下 GPU（8xA100，
`417802a1`）与 NPU（8xAscend950DT，A5 `a5-muse-accept`）的同一套
step-0、同一套 `run_acc.py` 各 1000 步跑批，以及 step-1 逐位 dump 对比。

## 结论摘要

- 多模态两条配置（AllGatherKV `mm_fsdp2_tp2_cp2_ep4`、Ulysses
  `debugmodel_mm`）的**系统性数值误差已归零**：进 attention 前的全部
  关键张量（`attention_input`、`raw_q`、`raw_k`、`value`）与 RoPE 产物
  （`rope_cache`、`query`、`key`）在 8 个 rank 上逐位一致。
- 剩余差异全部为 bf16 设备核的固有舍入噪声，打在路由近平局上。
- 旧判据（全程逐点 loss MAE<=0.02 或 MRE<=2%）**低于该配置的噪声地板
  （约 0.03），物理上不可达**；同设备仅改 1 ulp 的两次跑批全程 MAE
  也是 0.030。建议改用收敛段 MAE + 收敛步一致性 + step-1 逐位诊断
  判收（见下）。

## 定位与修复链

| 轮次 | 根因 | 修复 | 提交 | 路径 |
| --- | --- | --- | --- | --- |
| pairwise | TP 分片下 QKV 投影归约顺序不同 | 共享 pairwise fp32 GEMM | `00cd5bc2`..`484ddeca` | 调试门控（`MUSE_PAIRWISE_GEMM=1`），验收跑不启用 |
| qk norm + RoPE | 每侧各自实现 | 共享 fp32 q/k norm 链 + fp32 旋转，局部张量执行 | `68279566`..`c578caf7` | 模型路径，验收生效 |
| host polar | 设备端 `polar` 的 sin/cos 跨设备差 ~1.5e-5 | `polar` 移到 host | `781ac7f0`, `8733147a` | 被 `e284eff1` 取代 |
| 本轮 | 角度表里 CUDA `pow` 比 CPU/NPU 少 1 ulp，被 t<=511 放大成 1.5e-5 角度差 | inv_freq + scaling + 外积 + polar 全部在 host 构建 | `e284eff1` | 模型路径，验收生效 |
| 顺带抢救 | NPU `F.grid_sample` backward 产生非有限值（A5 工作区未提交修复） | 纯 torch 双线性实现入库 | `1628fc9` | 模型路径，NPU 正确性 |

`e284eff1` 的验证：host 重建的 cache 与 NPU dump **逐位相同**（0.0），
与修复前 GPU dump 差 1.5303e-5，即整段跨设备差异全部来自 CUDA `pow`。

必要的模型修复为三组：共享 fp32 q/k norm 链、全 host RoPE cache、NPU
grid_sample 绕过。pairwise fp32 GEMM 与各 step-1 dump 脚本不改变验收
运行的模型行为，是定位与验证工具（证据链交付件）；验收 1000 步跑不设
`MUSE_PAIRWISE_GEMM`，QKV 投影仍走各设备原生的 mm 核，其归约差归入
噪声项。

## Step-1 逐位验证（`39308c19` GPU dump vs A5 NPU `step1_cpu_cache`）

| 张量 | 8 rank 逐位一致 |
| --- | --- |
| `attention_input` / `block_input` / `positions` / `tokens` / `padding_mask` | 是 |
| `raw_q` / `raw_k` / `value` | 是 |
| `rope_cache` / `query` / `key` | 是（本轮修复目标） |
| `ffn_input`（bf16） | 否：约 6% 元素差，最小差 7.4e-6，全部是最后一位舍入 |

路由翻转 108/16384（修复前 111）。翻转槽位的 topk 分数差中位数
4.7e-4、94% 小于 1e-2；该配置 topk=2，翻转槽位权重占 token 路由总分
约 0.5。来源为 CUDA/NPU 注意力核在 fp32 上的累积顺序差，落到 bf16
只剩最后一位。

## 1000 步验收数据（同一代码）

| 指标 | AllGatherKV | Ulysses |
| --- | --- | --- |
| 全程 loss MAE | 0.030428 | 0.029314 |
| 全程 loss MRE（MAE/均值） | 6.5% | 6.2% |
| **最后 100 步 loss MAE** | **0.001520** | **0.002605** |
| 最后 100 步 grad_norm MAE | 0.038 | 0.049 |
| 第 1 步 loss 差 | 1.6e-3 | 8.3e-4 |
| 第 1 步 loss（GPU/NPU） | 7.792013 / 7.790420 | 7.791249 / 7.790420 |
| loss 跌破 0.5 | 41 步 / 41 步 | 41 步 / 41 步 |
| loss 跌破 0.1 | 341 步 / 413 步 | 341 步 / 341 步 |
| 第 1000 步 loss | 0.00450 / 0.00274 | 0.00221 / 0.00568 |

说明：

- 第 1 步差 1.6e-3 不是 RoPE 的（该链逐位一致），是视觉/音频编码器
  与其余层设备核的 bf16 舍入固有差。
- AllGatherKV 跌破 0.1 两边差 72 步；loss 在 0.1 附近抖动时过线时间
  对噪声敏感（旧 NPU 轮同样配置为 341 步），属噪声量级。

## 噪声地板对照（判据不可达的证据）

同一 GPU、同一代码，仅 `e284eff1` 这一个 1 ulp 级改动，前后两轮
1000 步逐点比较：

| 对比 | 全程 loss MAE |
| --- | --- |
| GPU 旧（`_r2`）vs GPU 新（`_host_rope`），mm | 0.0300 |
| GPU 旧 vs GPU 新，Ulysses | 0.0305 |
| GPU 新 vs NPU 新，mm | 0.0304 |
| GPU 新 vs NPU 新，Ulysses | 0.0293 |

跨设备差距与同设备 1 ulp 改动的自差异同量级：该模型 340 步内 loss
从 7.8 降至 0.1，逐点曲线处于混沌放大区，任何最后一位差异都会被
放大到 ~0.03。全程逐点 MAE<=0.02 量不出实现差异。

## 判据建议

1. **收敛段 loss MAE<=0.02**（最后 100 步）：本轮 0.0015 / 0.0026，
   通过；
2. **收敛步一致性**：loss<0.5 必须 41/41；loss<0.1 允许小抖动，差值
   以百步内为界：本轮 Ulysses 341/341，AllGatherKV 341/413；
3. **step-1 逐位诊断**：`raw_q/raw_k/value/rope_cache/query/key` 逐位
   相同（防系统性回归，最灵敏）。

是否采用由验收负责人拍板。若沿用旧判据，则任何实现（包括同设备重跑）
都无法通过。

## 数据文件

- GPU：`docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_host_rope.txt`、
  `docs/gpu_baseline/debugmodel_mm_1000_host_rope.txt`（`417802a1`）
- NPU：`docs/npu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_host_rope.txt`、
  `docs/npu_baseline/debugmodel_mm_1000_host_rope.txt`（本提交）
- step-1 dump：GPU `outputs/step1_cpu_cache_gpu/`（`39308c19`），
  NPU `outputs/step1_cpu_cache/`（A5，待入库，见备份清单）
- 旧轮 MAE 记录：`docs/npu_baseline/*_mae.txt`（rsqrt 轮 0.031 FAIL）

## 环境

- GPU：8xA100，torch 2.14.0a0，见 `docs/gpu_baseline_setup.md`
- NPU：8xAscend950DT，CANN 9.2.0，torch_npu 2.15.0+git1a25e6b，
  复现见 `docs/a5_env_repro_guide.md`
