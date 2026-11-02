# GPU agent 任务：精度曲线复验（防挑战）

给 `node-5-15`（或同等 8×A100）上的另一个 agent。

目的：对**精度已判定不通过 / 易被挑战**的配置，用**同一份 step 0**再跑一遍 GPU 1000 步，确认基线可复现，而不是另建权重。原因分析见 `docs/npu_debugmodel_verification.md` 与 `docs/npu_gpu_acceptance_report.md` 中该配置的精度小节。

分支：先 `git pull` 到含本文件的 `npu-muse-spark-debug-r04`。

## 硬约束

1. **必须加载仓库里已有的 step 0**，不要让 `loss_compare` 再 seed 一份新权重。新 seed 会让与 NPU 的对照失效。
2. 导出文件用新名字（`*_1000_r2.txt`），**不要覆盖**现有 `docs/gpu_baseline/*_1000.txt`。
3. 独占 GPU；多模态 8 卡很重，视觉长度变化会触发 flex 重编译，属预期。
4. 跑完把曲线、日志路径、与旧 GPU 曲线的逐步 MAE、以及相对 NPU 重算的 MAE/MRE 写回本文件「交数」节，并更新验收/验证文档。

## 环境

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
cd /path/to/torchtitan
mkdir -p logs docs/gpu_baseline
ls outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
ls outputs/checkpoint-debugmodel_mm/step-0
ls outputs/checkpoint-debugmodel/step-0
```

入口用默认 `torchtitan.train`（不要 `torchtitan_ascend.train`）。

## 必跑 A：第 2③（精度不通过，主目标）

旧 GPU 曲线：`docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`  
step 0：`outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0`  
NPU 对照：`docs/npu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt`（现 MAE 0.031 / MRE 16.82%）

按与 NPU 1000 步队列相同方式：`scripts/loss_compare.py`，**加载上述 step-0**，`--steps=1000`，`--baseline-ngpus=8`，module=`tests.integration_tests.models`，config=`muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`。

```bash
# 训练日志必须出现：
# Loading the checkpoint from .../outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
# 若脚本默认会重新 seed：停掉，改成只 load 已有 step-0（与 NPU 队列 run_note 一致）
python scripts/loss_compare.py . . \
  --baseline-module=tests.integration_tests.models \
  --baseline-config=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --baseline-ngpus=8 \
  --steps=1000 \
  --metrics=loss,grad_norm \
  --export-result=docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_r2.txt \
  --output-folder=outputs/gpu_reverify_mm_fsdp2_r2 \
  2>&1 | tee logs/mm_fsdp2_precision_r2_gpu.log
```

交数前自检：

```bash
python3 - <<'PY'
from pathlib import Path
def load(p):
    out = {}
    for line in Path(p).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        a = line.split()
        out[int(a[0])] = float(a[1])
    return out
old = load('docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt')
new = load('docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_r2.txt')
npu = load('docs/npu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt')
keys = sorted(set(old) & set(new) & set(npu))
mae_gpu = sum(abs(new[k] - old[k]) for k in keys) / len(keys)
mae_npu = sum(abs(npu[k] - new[k]) for k in keys) / len(keys)
mre_npu = sum(abs(npu[k] - new[k]) / max(abs(new[k]), 1e-12) for k in keys) / len(keys)
print('steps', len(keys), 'GPU_r2_vs_GPU_r1_MAE', mae_gpu, 'r2_vs_NPU_MAE', mae_npu, 'MRE', mre_npu)
print('step1', new[1], old[1], npu[1])
PY
```

期望：

- 第 1 步与旧 GPU / NPU 仍在 ~1e-3 量级对齐（证明加载同一权重）。
- `GPU_r2_vs_GPU_r1_MAE` 应很小（可复现）；若突然变大，先查是否误建了新 seed。
- `r2_vs_NPU` 的 MAE/MRE **不必**变达标；本任务是复验基线，不是强行翻盘。

## 必跑 B：第 4 行 `debugmodel_mm`（同类精度失败）

旧曲线：`docs/gpu_baseline/debugmodel_mm_1000.txt`  
step 0：`outputs/checkpoint-debugmodel_mm/step-0`  
export 到 `docs/gpu_baseline/debugmodel_mm_1000_r2.txt`，`--baseline-ngpus=8`，module=`torchtitan_recipes.tests.models.muse_spark`，config=`muse_spark_debugmodel_mm`。

## 可选 C：第 1 行 debugmodel（精度已通过，防「MAE 高」质疑）

时间紧可跳过。若跑：load `outputs/checkpoint-debugmodel/step-0`，4 卡，export `debugmodel_1000_r2.txt`。

## 不要做

- 不要为翻盘去改模型数学或门槛。
- 不要用新 seed 的 1000 步去替代现有 GPU 基线。
- 不要跑 Dist-MoE（A100 SM80）。
- 第 1 行**性能**独占复测是另一件事；本文件只覆盖精度曲线。

## 交数

```text
日期 / HEAD / 是否独占: 2026-11-02 / b188a02c（复验开始时）+ ultradata EP 修复 / 是（daemon 等待全空后顺序执行，一次一个 torchrun）
2③ r2 日志: logs/mm_fsdp2_precision_r2_gpu.log（1000 步，Training completed，无 ChildFailedError）
2③ step1 loss (r2 / 旧GPU / NPU): 7.78986 / 7.78986 / 7.79217
2③ GPU_r2 vs GPU_r1 MAE: 0.000000（1000 步逐位相同）
2③ r2 vs NPU MAE / MRE / 结论: 0.030999 / 16.82% / 与 r1 对 NPU 的数完全一致；基线可复现，按协议维持不达标
第4行同上: 日志 logs/debugmodel_mm_precision_r2_gpu.log（1000 步，Training completed）；step1 7.79068 / 7.79068 / 7.79205；GPU_r2 vs GPU_r1 MAE 0.000000；r2 vs NPU 0.030674 / 19.81%；基线可复现，维持不达标
可选 C（第 1 行）: 已跑。日志 logs/debugmodel_precision_r2_gpu.log（1000 步）；step1 7.84006 / 7.84006 / 7.84031；GPU_r2 vs GPU_r1 MAE 0.000000；r2 vs NPU 0.031049 / 1.28%；达标结论不变
是否误建新 seed（训练日志有无 Loading .../step-0）: 否。三份训练日志各出现一次 `Loading the checkpoint from .../outputs/checkpoint-*/step-0`，且 GPU_r2_vs_GPU_r1_MAE = 0 证明加载的就是既有 step-0 权重；通过 TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH 注入，未让 loss_compare 另建 seed 参与 1000 步
```

补充：导出文件 `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000_r2.txt`、`docs/gpu_baseline/debugmodel_mm_1000_r2.txt`、`docs/gpu_baseline/debugmodel_1000_r2.txt`；输出目录 `outputs/gpu_reverify_*_r2/`。复验用的检查点目录已重排为 `outputs/checkpoint-<配置名>/step-0/`（与本文档 `ls` 预一致；此前为无 step-0 层的平铺布局，`TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH` 指向 `outputs/checkpoint-<配置名>`）。

完成后：更新 `docs/gpu_baseline/`（仅新增 r2 文件）、本文件交数节、`docs/npu_gpu_acceptance_report.md` 对应精度小节加一句「GPU 曲线已复验」，以及 `docs/npu_debugmodel_verification.md` 对应配置的精度小节补一行复验数。
