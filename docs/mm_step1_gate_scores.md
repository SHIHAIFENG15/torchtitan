# 多模态第 1 步 gate 操作数和分数（GPU）

只跑 AllGatherKV 这一条，只跑第 1 步。目的是把 CUDA 上 router gate 的输入、一次 GEMM 的 fp32 结果，以及 top-k 真正用来选专家的分数存下来。已有的 `outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4/` 只有选中的专家编号，不够判断分差来自 GEMM 还是来自进 GEMM 之前的激活。

在已经交出那份专家编号的同一棵 torchtitan 树上执行。不要新克隆。不要改 `hi_mid_lo_linear.py`、`vision_encoder.py` 或 router。不要重做 step-0。不要跑 Ulysses、文本配置、Dist-MoE、mxfp8，也不要跑 1000 步。

`torch.mm(..., out_dtype=float32)` 只在这次真实 forward 里调用一次。不要在脚本外面再用同一批输入重算一遍当作 `gate_logits`。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 不存在。
- `outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4/rank0.pt` 到 `rank7.pt` 不齐全。新结果必须和这 8 个文件里的 `expert_ids` 逐元素相同。
- venv 里把 `aten::grid_sampler_2d_backward` 转到 CPU 的垫片没了。上次如果垫片在 `usercustomize.py`，保持那样，不要再盖一层 `sitecustomize.py`。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
test -f scripts/dump_step1_gate_scores.py
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
test -f outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4/rank7.pt
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。

## 环境

沿用 1000 步基线和专家编号导出的那个 venv。不要安装 `torchtitan_ascend`，不要设置 `ENTRYPOINT`。不要设置 `MM_DTYPE_REDUCED_PRECISION`、`HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑这一个 torchrun。

脚本在这三个变量还没设置时会补上，和上次基线相同：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

## 跑这一条

仓库根目录。网格是 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。输出写到新目录，不要覆盖 `outputs/step1_expert_ids/`。

```bash
python scripts/dump_step1_gate_scores.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_gate_scores/mm_fsdp2_tp2_cp2_ep4 \
  --expect-ids outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_gate_scores_mm_fsdp2.log 2>&1
echo EXIT:$?
```

对照 `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt` 的第 1 步：loss `7.789858818054199`，grad_norm `9.11008358001709`。打印值和这两个数相差要在 `1e-4` 以内。差得更多，或脚本最后没有打出 `STEP1_GATE_DUMP check ok`，就保留日志和 `.pt`，不要改代码重跑，把 step 1 原行和 `expert_ids mismatch` 那一行带回来。

## 每个 rank 要有的张量

8 个文件，`rank0.pt` 到 `rank7.pt`。每个文件 8 个 router，键是 `layers.0.feed_forward.router` 到 `layers.7.feed_forward.router`。张量都是该 rank 的本地 token，不要在保存前 all-gather。

上次 rank0 的本地 token 数是 128，专家数是 8，top-k 是 2。脚本按各张量彼此对齐来检查，不把 128 写死。

| 键 | 含义 | dtype |
| --- | --- | --- |
| `gate_input` | 进 HiMidLo `_linear` 的 bf16 激活 | bfloat16 |
| `gate_weight` | 同一次调用的 bf16 权重 | bfloat16 |
| `gate_logits` | 这次 CUDA `mm(out_dtype=fp32)` 的输出，sigmoid 之前 | float32 |
| `scores` | `MuseSparkRouterSigmoid` 的输出，还没加 bias，也还没做 route_norm | float32 |
| `expert_bias` | 选择用的 bias | float32 |
| `scores_for_choice` | `scores + expert_bias`，`topk` 读的就是它 | float32 |
| `topk_ids` | router 返回的专家编号，`sorted=False`，不排序 | int32 |
| `topk_scores` | router 返回的分数，route_norm 和 route_scale 之后，与 `topk_ids` 对齐 | float32 |
| `expert_ids` | `topk_ids` 在 K 维排序后的结果，用来对上一次的专家编号 | int32 |
| `padding_mask` | 与 token 轴对齐的 bool mask | bool |

`expert_ids` 必须和 `outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4/` 里同一 rank、同一层逐元素相同。对上了，这份 gate 张量就是上次那次 step 1，不用再跑。对不上，说明 checkpoint、批次或代码和上次不是同一次，后面的分数不能和 950 的 gate 操作数配对。

## 怎样算这次有效

日志里有 `Training completed`，没有 `ChildFailedError`。8 个 `.pt` 都在。日志末尾有 `expert_ids mismatch 0/` 和 `STEP1_GATE_DUMP check ok`。进程退出码是 0。

```bash
find outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4 -type f | sort
python - <<'PY'
import re, pathlib
pat = re.compile(r"step:\s+1\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)")
text = re.sub(r"\x1b\[[0-9;]*m", "", pathlib.Path("logs/step1_gate_scores_mm_fsdp2.log").read_text(errors="replace"))
print(pat.findall(text)[:1])
print("mismatch", [line for line in text.splitlines() if "expert_ids mismatch" in line or "STEP1_GATE_DUMP check" in line])
PY
git rev-parse HEAD
```

把 step 1 的 loss、grad_norm、`git rev-parse HEAD` 和 `outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4/` 留下。`.pt` 不要删。不要提交 tensorboard 目录和日志。

本说明由 AI 助手按 950 上已有的 gate 操作数 dump 和本分支已提交的专家编号整理。
