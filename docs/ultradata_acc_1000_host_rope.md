# UltraData 32K LoRA 1000 步精度：host-rope 后复跑（GPU）

补齐验收表中未执行的 UltraData 32K LoRA 行。只跑这一条，1000 步。代码必须包含 `e8b4527`（含 RoPE cache 全 host 构建 `e284eff1`、NPU grid_sample 绕过 `1628fc9`、pairwise 树按行分块——32K 序列下不分块的 pairwise 广播约需 200 GiB，A100 会同样 OOM）。文本、多模态、性能和显存都不要跑。

不要重做 step-0。不要覆盖这些已有文件：

- `docs/gpu_baseline/ultradata_lora_1000.txt`（旧代码基线，不能和这次的 NPU 比）
- `outputs/checkpoint-ultradata_lora/`
- `outputs/tb_acc_ultradata*`（已有目录）

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。常驻的 `atk server` 可以忽略。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `git merge-base --is-ancestor e8b4527 HEAD` 失败。
- `torchtitan/models/common/rope.py` 里没有 `ambient = torch.empty(0).device`。
- step-0 目录不存在。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
git merge-base --is-ancestor e8b4527 HEAD
grep -n "ambient = torch.empty(0).device" torchtitan/models/common/rope.py
test -f run_acc.py
test -d outputs/checkpoint-ultradata_lora/step-0
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。

## 环境

沿用上次导出的那个 venv。不要设置 `MM_DTYPE_REDUCED_PRECISION`、`HCCL_DETERMINISTIC`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑一个 torchrun。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

配方自带 `train_timeout_seconds=1800`、`local_compile_regions=["loss"]`，不用手动设置。

## 跑这一条

仓库根目录。8 卡、1000 步，加载已有 step-0，`load_only`，不会改 checkpoint。

```bash
ACC_NGPU=8 \
ACC_MODULE=torchtitan_recipes.tests.models.muse_spark \
ACC_CONFIG=muse_spark_debugmodel_ultradata_lora_sft \
ACC_LOAD=outputs/checkpoint-ultradata_lora/step-0 \
ACC_STEPS=1000 \
ACC_TB=tb_acc_ultradata_host_rope \
python3 run_acc.py > logs/metrics_acc_ultradata_host_rope.log 2>&1
echo EXIT:$?
```

日志里要有 `Training completed` 和 `Process group destroyed`。第 1 步 loss 没有新的参照值，与旧基线第 1 步差在 `1e-2` 以内就继续。若进程被外部杀掉（如机器回收），保留日志，不要改代码重跑。

## 导出 loss

从 TensorBoard 取出全精度的 `loss_metrics/global_avg_loss` 和 `grad_norm`，写成和旧基线一样的三列。不要从日志里的四舍五入数字抄。

```bash
python3 - << 'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

folder = "tb_acc_ultradata_host_rope"
config = "muse_spark_debugmodel_ultradata_lora_sft"
out = "docs/gpu_baseline/ultradata_lora_1000_host_rope.txt"
tags = ("loss_metrics/global_avg_loss", "grad_norm")
acc = EventAccumulator(folder, size_guidance={"scalars": 0})
acc.Reload()
series = [{e.step: e.value for e in acc.Scalars(tag)} for tag in tags]
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

txt 有 1000 行数据。把 `git rev-parse HEAD` 和日志的 `EXIT` 留下。只提交这一个 txt：

- `docs/gpu_baseline/ultradata_lora_1000_host_rope.txt`

不要提交 tensorboard 目录和日志。不要覆盖旧的 `ultradata_lora_1000.txt`。推到 `npu-muse-spark-debug-r04`，不要 force-push。

本说明由 AI 助手整理。NPU 侧用同一份 `run_acc.py`、同一份 step-0 在 A5 排队跑（输出 `logs/metrics_acc_ultradata_host_rope.log` 与 `tb_acc_ultradata_host_rope`）。指标口径与多模态轮一致：全程逐点 MAE 受约 0.03 固有随机波动封顶，按收敛段 MAE、收敛步一致性判收；完整依据见 `docs/mm_acc_report_host_rope.md`。
