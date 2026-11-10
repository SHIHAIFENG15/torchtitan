# 多模态第 1 步梯度导出（GPU）

在已经跑出 1000 步 GPU 基线的那棵 torchtitan 树上执行。不要新克隆一份，新克隆没有 step-0 权重，也没有原来的 `sitecustomize.py`。

导出的是反向结束、裁剪和优化器更新之前的局部梯度。两条都是 8 卡，各跑 1 步。一次只跑一个 torchrun。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。不要丢弃本地改动。
- 下面两个目录有一个不存在：
  - `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0`
  - `outputs/checkpoint-debugmodel_mm/step-0`
- venv 里的 `sitecustomize.py` 不再把 `aten::grid_sampler_2d_backward` 转到 CPU。不要删这个文件，也不要另写一个 `sitecustomize.py` 盖住它。
- 想改 `vision_encoder.py`、配方、学习率或 seed。都不要改。

先在这棵树上快进本分支，让 `scripts/dump_step1_grads.py` 出现。快进之后再确认上面两个 step-0 目录还在。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
git status --short
git rev-parse HEAD
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
test -d outputs/checkpoint-debugmodel_mm/step-0
python -c "import sitecustomize,inspect; print(inspect.getfile(sitecustomize))"
```

`sitecustomize` 的源码里要能看到 `grid_sampler_2d_backward`。看不到就停。

## 环境

沿用 1000 步基线的那个 venv。进程启动前设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

不要设置 `HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。不要设 `deterministic_warn_only=True`。8 张卡都要可见。

## 跑这两条

在仓库根目录执行。

AllGatherKV，`dp_shard=2, cp=2, tp=2, ep=4`：

```bash
python scripts/dump_step1_grads.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_grads/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_grads/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_grads_mm_fsdp2.log 2>&1
```

对照 `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt` 的第 1 步：loss `7.789858818054199`，grad_norm `9.11008358001709`。

Ulysses，`dp_shard=2, cp=2, tp=2, ep=2`：

```bash
python scripts/dump_step1_grads.py \
  --module torchtitan_recipes.tests.models.muse_spark \
  --config muse_spark_debugmodel_mm \
  --ngpu 8 \
  --load outputs/checkpoint-debugmodel_mm/step-0 \
  --out outputs/step1_grads/debugmodel_mm \
  --tb tb_step1_grads/debugmodel_mm \
  > logs/step1_grads_debugmodel_mm.log 2>&1
```

对照 `docs/gpu_baseline/debugmodel_mm_1000.txt` 的第 1 步：loss `7.790676116943359`，grad_norm `9.093783378601074`。

## 怎样算这次有效

日志里有 `Training completed`，没有 `ChildFailedError`。每个输出目录有 `rank0.pt` 到 `rank7.pt`，以及同名的 `.json`。每条日志里有 8 行 `STEP1_GRAD_DUMP`。

打印的 step 1 loss、grad_norm 和上面两个全文数字相差要在 `1e-4` 以内。差得更多就删掉这次的 `outputs/step1_grads/<配置>`，不要换代码重跑，把日志里的 step 1 原行带回来。

`.json` 是每个参数的局部梯度范数、最大绝对值和均值。`.pt` 里的 `grads` 是 fp32 CPU 上的局部分片，给后续和 950 逐元素相减。不要只回 json。

跑完后执行：

```bash
find outputs/step1_grads -type f | sort
python - <<'PY'
import re, pathlib
pat = re.compile(r"step:\s+1\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)")
for path in ("logs/step1_grads_mm_fsdp2.log", "logs/step1_grads_debugmodel_mm.log"):
    text = re.sub(r"\x1b\[[0-9;]*m", "", pathlib.Path(path).read_text(errors="replace"))
    print(path, pat.findall(text)[:1])
PY
```

把这两行 loss、`git rev-parse HEAD`，以及两个输出目录留下。`.pt` 不要删。
