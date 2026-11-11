# 多模态第 1 步第 0 层激活（GPU）

只跑 AllGatherKV 这一条，只跑第 1 步。上一份 `outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4/` 已经说明 router 权重和 bias 两边按位相同，专家选错来自第 0 层 `gate_input`。这一次把产生这份 `gate_input` 的第 0 层中间结果存下来。

在已经交出 gate 分数的同一棵 torchtitan 树上执行。不要新克隆。不要改 `hi_mid_lo_linear.py`、`vision_encoder.py`、attention 或 router。不要重做 step-0。不要跑 Ulysses、文本配置、Dist-MoE、mxfp8，也不要跑 1000 步。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 不存在。
- `outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4/rank0.pt` 到 `rank7.pt` 不齐全。
- venv 里把 `aten::grid_sampler_2d_backward` 转到 CPU 的垫片没了。垫片在 `usercustomize.py` 就保持那样，不要再盖一层 `sitecustomize.py`。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
test -f scripts/dump_step1_layer0_acts.py
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
test -f outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4/rank7.pt
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。

## 环境

沿用 gate 分数导出的那个 venv。不要安装 `torchtitan_ascend`，不要设置 `ENTRYPOINT`。不要设置 `MM_DTYPE_REDUCED_PRECISION`、`HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑这一个 torchrun。

脚本在这三个变量还没设置时会补上，和上次基线相同：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

## 跑这一条

仓库根目录。网格是 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。输出写到新目录，不要覆盖 `outputs/step1_gate_scores/` 或 `outputs/step1_expert_ids/`。

```bash
python scripts/dump_step1_layer0_acts.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_layer0_acts/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_layer0_acts/mm_fsdp2_tp2_cp2_ep4 \
  --expect-gate outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_layer0_acts_mm_fsdp2.log 2>&1
echo EXIT:$?
```

对照 `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt` 的第 1 步：loss `7.789858818054199`，grad_norm `9.11008358001709`。打印值和这两个数相差要在 `1e-4` 以内。差得更多，或脚本最后没有打出 `STEP1_LAYER0_DUMP check ok`，就保留日志和 `.pt`，不要改代码重跑。

进程如果在 `layer0 dump missed` 处抛错，同样保留日志，不要改脚本重跑，把 traceback 带回来。

## 每个 rank 要有的张量

8 个文件，`rank0.pt` 到 `rank7.pt`。键在 `layer0` 下面。张量都是该 rank 的本地 shard，不要在保存前 all-gather。dtype 保持计算时的 dtype，不要额外转成 bf16。

| 键 | 含义 |
| --- | --- |
| `tokens` | 这一步的 token id |
| `positions` | 送进第 0 层的 position |
| `image_bank_indices_T` | 有视觉 token 才有 |
| `video_bank_indices_T` | 有视频 token 才有 |
| `padding_mask` | 与 token 轴对齐 |
| `block_input` | 进第 0 层的残差，fp32。embedding 和视觉写回之后 |
| `attention_input` | attention 真正读到的输入，norm 并转成 bf16 之后 |
| `query` `key` | q/k norm、温度、RoPE 之后，attention 读到的那一对 |
| `value` | qkv 投影给出的 value |
| `attn_out` | inner attention 的输出，`wo` 之前 |
| `attention_output` | `wo` 之后的 attention 输出 |
| `attention_branch` | `post_attention_norm` 的输出 |
| `stream` | `alpha * x + beta * branch`，fp32 |
| `ffn_input` | 进 MoE 的 bf16 激活。必须和已有 `layers.0` 的 `gate_input` 逐元素相同 |

`ffn_input` 对上了，这份中间结果就是上次那次 step 1。对不上，说明 checkpoint、批次或代码和上次不是同一次，不能拿来和 950 配对。

## 怎样算这次有效

日志里有 `Training completed`，没有 `ChildFailedError`。8 个 `.pt` 都在。日志末尾有 `ffn_input mismatch 0/` 和 `STEP1_LAYER0_DUMP check ok`。进程退出码是 0。

```bash
find outputs/step1_layer0_acts/mm_fsdp2_tp2_cp2_ep4 -type f | sort
python - <<'PY'
import re, pathlib
pat = re.compile(r"step:\s+1\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)")
text = re.sub(r"\x1b\[[0-9;]*m", "", pathlib.Path("logs/step1_layer0_acts_mm_fsdp2.log").read_text(errors="replace"))
print(pat.findall(text)[:1])
print("check", [line for line in text.splitlines() if "ffn_input mismatch" in line or "STEP1_LAYER0_DUMP check" in line])
PY
git rev-parse HEAD
```

把 step 1 的 loss、grad_norm、`git rev-parse HEAD` 和 `outputs/step1_layer0_acts/mm_fsdp2_tp2_cp2_ep4/` 留下。`.pt` 不要删。不要提交 tensorboard 目录和日志。

本说明由 AI 助手按第 0 层前向和已提交的 gate 分数整理。

## 交数（2026-11-11）

- HEAD：`98a44bfb`（含 `scripts/dump_step1_layer0_acts.py`）
- 是否独占：swy 的 Wan2.2 单 prompt 大批次视频生成占满 8 卡（每卡 72GB）从 15:00 起到 18:17，共 ~3h20m。看门狗 60 秒一轮监测 8 卡全空 + 无 compute-apps，18:17:17 命中 GPU-FREE
- step-1 数值对照（阈值 1e-4）：loss `7.78986`（diff 1.18e-06）、grad_norm `9.1101`（diff 1.64e-05）→ **有效**
- **torchrun 退出码 3**（收尾阶段某个 rank 报错；rank 0 输出正常：`Training completed` + `Process group destroyed` 都在）；因此脚本自身的 `_check` 分支**未跑到**，`STEP1_LAYER0_DUMP check ok` 那行也没打出来。**未修改脚本重跑**（按文档要求）——改由助手用现成 `.pt` + 上次 gate 分数文件手动跑 `_check` 等价逻辑：
  ```
  rank0..rank7 ffn_input (bf16)  vs gate.layers.0.gate_input
  全部 same=True；TOTAL mismatch: 0/262144
  ```
  → 与已交出 gate 分数的 step 1 是**同一次**（128×256 bf16 逐位一致）
- 8 rank 输出：`outputs/step1_layer0_acts/mm_fsdp2_tp2_cp2_ep4/rank0-7.pt`；每 rank 14 个键（`tokens/positions/padding_mask/block_input/attention_input/query/key/value/attn_out/attention_output/attention_branch/stream/ffn_input/video_bank_indices_T`），视频 token 存在故无 `image_bank_indices_T`，符合文档
- 日志：`/home/y00839695/torchtitan/logs/step1_layer0_acts_mm_fsdp2.log`；看门狗日志 `/home/y00839695/l0_watch.log`
- `.pt` 保留在 GPU 机；不上传 tb_step1_layer0_acts/ 和日志（按文档）
