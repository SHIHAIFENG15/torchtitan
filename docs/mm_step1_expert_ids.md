# 多模态第 1 步专家编号（GPU）

只跑 AllGatherKV 这一条。目的是把第 1 步每个 token 选中的专家编号存下来，和 950 按 token 对。上次的梯度 `.pt` 里没有这个编号，不能代替这次。

在已经跑出 1000 步 GPU 基线、并且跑过 `outputs/step1_grads/` 的那棵 torchtitan 树上执行。不要新克隆。不要改 `vision_encoder.py`，不要重做 step-0，不要跑 Ulysses，不要跑 1000 步。

`topk` 是 `sorted=False`。脚本把每个 token 的 K 个专家编号排好序再保存，比较时不用再排。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 不存在。
- venv 里把 `aten::grid_sampler_2d_backward` 转到 CPU 的垫片没了。上次如果垫片在 `usercustomize.py`，保持那样，不要再盖一层 `sitecustomize.py`。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
test -f scripts/dump_step1_expert_ids.py
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。

## 环境

沿用 1000 步基线和上次梯度导出的那个 venv。进程启动前设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

不要设置 `HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑这一个 torchrun。

## 跑这一条

仓库根目录。网格是 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。

```bash
python scripts/dump_step1_expert_ids.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_expert_ids/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_expert_ids_mm_fsdp2.log 2>&1
```

对照 `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt` 的第 1 步：loss `7.789858818054199`，grad_norm `9.11008358001709`。打印值和这两个数相差要在 `1e-4` 以内。差得更多就删掉 `outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4`，不要换代码重跑，把 step 1 原行带回来。

## 怎样算这次有效

日志里有 `Training completed`，没有 `ChildFailedError`。输出目录有 `rank0.pt` 到 `rank7.pt`。日志里有 8 行 `STEP1_EXPERT_DUMP`。

每个 `.pt` 里 `routers` 的键是层名，值是该 rank 本地 token 的 `expert_ids`，形状 `(T, K)`，`int32`，K 维已排序。有 padding 时同时有 `padding_mask`。不要只回日志。

```bash
find outputs/step1_expert_ids -type f | sort
python - <<'PY'
import re, pathlib
pat = re.compile(r"step:\s+1\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)")
text = re.sub(r"\x1b\[[0-9;]*m", "", pathlib.Path("logs/step1_expert_ids_mm_fsdp2.log").read_text(errors="replace"))
print(pat.findall(text)[:1])
print(text.count("STEP1_EXPERT_DUMP"))
PY
```

把 step 1 的 loss、grad_norm、`git rev-parse HEAD` 和 `outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4/` 留下。`.pt` 不要删。

## 交数（2026-11-10）

- HEAD：`63bd433c`（含 `scripts/dump_step1_expert_ids.py`）
- step 1 对照（阈值 1e-4）：loss `7.78986`（diff 1.18e-06）、grad_norm `9.1101`（diff 1.64e-05）→ **有效**
- 输出：`outputs/step1_expert_ids/mm_fsdp2_tp2_cp2_ep4/rank0-7.pt`（8 文件，rank0 抽查：8 层 router、每层 `(128, 2)` int32、K 维已排序、含 `padding_mask`、rank=0/world=8）
- 日志 `STEP1_EXPERT_DUMP` 行数为 1（`LOG_RANK=0` 只回显 rank0；8 个 `.pt` 齐全）
- **脚本修复（已入本提交）**：`dump_step1_expert_ids.py` 原样在 GPU 上触发 `SpmdTypeError: No valid sharding strategy for cpu`（expert_ids 为 spmd_types 包装的 DTensor，`.cpu()` 无合法分片策略）。修复：`to_local()` 取本地分片 + 整段导出包进 `spmd.local()`（spmd_types 官方逃生门，用于全局传播无法表达的块）。NPU 侧同样会踩到，建议拉取
- `.pt` 保留在 GPU 机 `/home/y00839695/torchtitan/outputs/step1_expert_ids/`
