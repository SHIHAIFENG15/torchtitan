# 多模态 1000 步精度：音频 RMSNorm 换了同一套 rsqrt（GPU）

只重跑上次精度没过的两条多模态，各 1000 步。代码必须包含 `fd76aee9`（音频 RMSNorm 用整数初值加 4 步不融合的 Newton `rsqrt`，算完立刻舍回 bf16）。文本配置、Dist-MoE、mxfp8、性能和显存都不要跑。

不要重做 step-0。不要覆盖这些已有文件：

- `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`
- `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_r2.txt`
- `docs/gpu_baseline/debugmodel_mm_1000.txt`
- `docs/gpu_baseline/debugmodel_mm_1000_r2.txt`
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/`
- `outputs/checkpoint-debugmodel_mm/`
- `outputs/step1_audio_rsqrt/`

旧的 `*_1000.txt` 是改 rsqrt 之前的 loss，不能拿来和这次的 NPU 比。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。常驻的 `atk server`（每卡大约几百 MB，不到 2GB）可以忽略。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `git merge-base --is-ancestor fd76aee9 HEAD` 失败。
- `audio_encoder.py` 里没有 `0x5F3759DF`。
- 下面两个 step-0 目录有一个不存在。
- venv 里把 `aten::grid_sampler_2d_backward` 转到 CPU 的垫片没了。垫片在 `usercustomize.py` 就保持那样，不要再盖一层 `sitecustomize.py`。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
git merge-base --is-ancestor fd76aee9 HEAD
grep -n 0x5F3759DF torchtitan/models/muse_spark/audio_encoder.py
test -f run_acc.py
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
test -d outputs/checkpoint-debugmodel_mm/step-0
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。

## 环境

沿用上次导出的那个 venv。不要安装 `torchtitan_ascend`，不要调用 `torchtitan_ascend.setup()`，不要设置 `MUSE_SETUP_ASCEND`。不要设置 `ENTRYPOINT`。不要设置 `MM_DTYPE_REDUCED_PRECISION`、`HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑一个 torchrun，第一条结束再跑第二条。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

## 跑这两条

仓库根目录。两条都是 8 卡、1000 步，加载已有 step-0，`load_only`，不会改 checkpoint。

AllGatherKV，网格 `dp_shard=2, cp=2, tp=2, ep=4`：

```bash
ACC_NGPU=8 \
ACC_MODULE=tests.integration_tests.models \
ACC_CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
ACC_LOAD=outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
ACC_STEPS=1000 \
ACC_TB=tb_acc_mm_fsdp2_rsqrt \
python3 run_acc.py > logs/metrics_acc_mm_fsdp2_rsqrt.log 2>&1
echo EXIT:$?
```

Ulysses，配置函数 `muse_spark_debugmodel_mm`：

```bash
ACC_NGPU=8 \
ACC_MODULE=torchtitan_recipes.tests.models.muse_spark \
ACC_CONFIG=muse_spark_debugmodel_mm \
ACC_LOAD=outputs/checkpoint-debugmodel_mm/step-0 \
ACC_STEPS=1000 \
ACC_TB=tb_acc_mm_rsqrt \
python3 run_acc.py > logs/metrics_acc_mm_rsqrt.log 2>&1
echo EXIT:$?
```

日志里要有 `Training completed` 和 `Process group destroyed`。第 1 步打印的 loss 大约是 `7.79030`，grad_norm 大约是 `9.1101`。这只是日志里的四舍五入，差一个显示位可以继续。差到 `1e-2` 以上就停，保留日志，不要改代码重跑。

## 导出 loss

从 TensorBoard 取出全精度的 `loss_metrics/global_avg_loss` 和 `grad_norm`，写成和旧基线一样的三列。不要从日志里的四舍五入数字抄。

```bash
python3 - << 'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

jobs = [
    (
        "tb_acc_mm_fsdp2_rsqrt",
        "muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4",
        "docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_rsqrt.txt",
    ),
    (
        "tb_acc_mm_rsqrt",
        "muse_spark_debugmodel_mm",
        "docs/gpu_baseline/debugmodel_mm_1000_rsqrt.txt",
    ),
]
tags = ("loss_metrics/global_avg_loss", "grad_norm")
for folder, config, out in jobs:
    acc = EventAccumulator(folder, size_guidance={"scalars": 0})
    acc.Reload()
    series = []
    for tag in tags:
        series.append({e.step: e.value for e in acc.Scalars(tag)})
    steps = sorted(series[0])
    if steps != list(range(1, 1001)) or sorted(series[1]) != steps:
        raise SystemExit(f"{folder} does not have steps 1..1000 for both tags")
    lines = [
        f"# config: {config}",
        "# ngpu: 8",
        "# step loss grad_norm",
    ]
    for step in steps:
        lines.append(f"{step} {series[0][step]!r} {series[1][step]!r}")
    Path(out).write_text("\n".join(lines) + "\n")
    print(out, len(steps))
PY
```

## 怎样算交数完成

两个 txt 都有 1000 行数据。把 `git rev-parse HEAD` 和两条日志的 `EXIT` 留下。只提交这两个 txt：

- `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_rsqrt.txt`
- `docs/gpu_baseline/debugmodel_mm_1000_rsqrt.txt`

不要提交 tensorboard 目录和日志。不要覆盖旧的 `*_1000.txt`。推到 `npu-muse-spark-debug-r04`，不要 force-push。

本说明由 AI 助手整理。NPU 已用同一份 `run_acc.py`、同一份 step-0 跑完 AllGatherKV；Ulysses 在同一条队列里接着跑。MAE 要等这两份 GPU loss 和 NPU 的 TensorBoard 对齐后再算。
