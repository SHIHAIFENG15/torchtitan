# GPU 基线还要补的部分

对照 `docs/npu_debugmodel_verification.md` 里已经记下来的 NPU 数，和 2026-10-30 在 `node-5-15`（8×A100-80GB，容器 `gpu-z00824525-tt`）上的 `docs/gpu_debugmodel_baseline.md`。GPU 那次检出是 `bdab0793`。本文写两件事：1000 步为什么写不出 step 0，以及还没跑的 GPU 命令。

这台 Mac 到不了 `node-5-15`。下面的命令要在 GPU 容器里执行。不要在 NPU 上跑。

## 已经有的，不要重跑

第 1 行、第 2 行①②③的 10 步，GPU 每个配置连跑两次，loss 和 grad_norm 各自相同。和 NPU 从第 1 步就不同，差没有越拉越大。这是设备差异，不是某一次没跑完。不要为了让 10 步逐位相同再跑。

第 1 行 1000 步已经导出：`docs/gpu_baseline/debugmodel_1000.txt`。step 0 权重在 `outputs/checkpoint/step-0/`，不在 `outputs/gpu_baseline_debugmodel/` 里面。把这两样拷出来即可。NPU 还没加载，MAE 还没算。不要再跑第 1 行的 1000 步。

线性度不用测。这几条不是同一配方把数据并行加倍。不要把第 1 行改成 `NGPU=8`。

Dist-MoE bf16 和 mxfp8 不要在这台 A100 上跑。A100 是 SM80，官方路径要 SM100。

## 1000 步失败的原因

`scripts/loss_compare.py` 写 step 0 时调用 `scripts/_checkpoint_test_config.py` 里的 seed 模式。它把六种并行度都改成 1，`NGPU=1`，用来写一份没有分片的权重。配方在构建模型时已经把注意力换成了 CP 内核（Ulysses 或 AllGatherKV）。并行度改成 1 之后，内核类型没有换回来。校验在 `torchtitan/config/validation.py`：

```text
ValueError: layers.0.attention.inner_attention is CPInnerAttention but the
context parallel degree is 1. Select a non-CP kernel.
```

第 2 行①②③因此停在 `create_seed_checkpoint`。10 步结果仍然有效。第 4 行、第 5 行也是 `cp=2`，用同一脚本写 step 0 会遇到同一个错误。

`convert_config_type` 要求替换类的 Config 是现有 Config 的子类，只能往 CP 子类转，不能转回 `FlexInnerAttention`。AllGatherKV 还有一个 CP 专用字段 `reduce_dtype`，基类配置里没有。seed 阶段按基类字段另建一份配置，丢掉 `reduce_dtype`。这不改参数，也不改 1000 步训练用的内核。`mode == "load"` 的那次不会调用这个函数，正式 1000 步仍是原配方的 CP。

改动在 `scripts/_checkpoint_test_config.py` 的生成模块里，函数 `_select_non_cp_kernel`。覆盖的四对是 Muse 配方实际用的：

| CP 内核 | seed 时换成 |
| --- | --- |
| `UlyssesFlexInnerAttention` | `FlexInnerAttention` |
| `UlyssesSlidingWindowFlexInnerAttention` | `SlidingWindowFlexInnerAttention` |
| `AllGatherKVFlexInnerAttention` | `FlexInnerAttention` |
| `AllGatherKVSlidingWindowFlexInnerAttention` | `SlidingWindowFlexInnerAttention` |

遇到表外的 `CPInnerAttention` 会直接报错，不会悄悄换成另一种算法。不要改模型数学，也不要把 `deterministic_warn_only` 设成 True。不要用 `--no-seed-checkpoint` 代替这一步，那样两边就不是同一份 step 0 权重。

GPU 容器先拉包含这个函数的 `npu-muse-spark-debug-r04`。停在 `bdab0793` 再跑，还是原来的 `ValueError`。

## 还要在 GPU 上跑的

环境沿用 2026-10-30 那次：`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`TORCH_DEVICE_BACKEND_AUTOLOAD=1`、`MUSE_DEBUG_DETERMINISTIC=1`。多模态两条（第 4 行，以及若重跑第 2 行③）仍要保留 `aten::grid_sampler_2d_backward` 走 CPU 确定性实现的垫片，否则 `use_deterministic_algorithms(True)` 会在 `grid_sample` 上失败。一次只跑一个 torchrun。`NGPU` 必须等于配方世界大小。

日志里要有 `Training completed` 和 `Process group destroyed`，不能有 `ChildFailedError`。10 步同一配置连跑两次，只比较 loss 和 grad_norm。

### 1. 第 4 行 10 步，跑两次

这是清单上的 `muse_spark_debugmodel_mm`：8 卡 `dp_shard=2, cp=2, tp=2, ep=2`，Ulysses。不是第 2 行③的 EP4、AllGatherKV。NPU 已有两次对齐的 10 步。

`bdab0793` 上这条配方没有把 `train_timeout_seconds` 设成 1800。视觉长度一变就要重编 flex attention，默认 100 秒会把先跑完的 rank 打成 HCCL watchdog 超时。当前分支在 `muse_spark_debugmodel_mm` 里已经设了 1800。拉分支之后再跑。日志里应能看到 HCCL execution timeout `1836000ms`，或进程组超时被调到 30 分钟。启动超时仍是 300 秒。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_mm \
  ./run_train.sh
```

网格应打出 `pp=1, dp_replicate=1, dp_shard=2, cp=2, tp=2, ep=2`。两次都跑完再记表。不要把第 2 行③的 GPU 数填进第 4 行。

### 2. 第 2 行①②③的 1000 步

先确认 `grep -n _select_non_cp_kernel scripts/_checkpoint_test_config.py` 有结果。把 2026-10-30 失败的 `outputs/gpu_baseline_*/seed_training.log` 留着，输出目录换一个后缀，避免盖掉 traceback。

第 2 行①：

```bash
python scripts/loss_compare.py . . \
  --baseline-module=torchtitan_recipes.tests.models.muse_spark \
  --baseline-config=muse_spark_debugmodel_with_cp \
  --baseline-ngpus=4 \
  --steps=1000 \
  --metrics=loss,grad_norm \
  --export-result=docs/gpu_baseline/with_cp_1000.txt \
  --output-folder=outputs/gpu_baseline_with_cp_cpfix
```

第 2 行②：module 换成 `tests.integration_tests.models`，config 换成 `muse_spark_debugmodel_fsdp2_cp2_ep4`，`--baseline-ngpus=4`。

第 2 行③：module 仍是 `tests.integration_tests.models`，config 换成 `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`，`--baseline-ngpus=8`。

`--baseline-ngpus` 的默认值是 8。4 卡配方写成 8 会在建网格时失败。

seed 日志里不应再出现 `Select a non-CP kernel`。若下一步是别的报错，把完整 traceback 留下，不要改 1000 步配方去绕。step 0 权重以 `loss_compare` 实际写出的目录为准。第 1 行那次写在 `outputs/checkpoint/step-0/`，不在 `--output-folder` 下面。

这 1000 步加载的是 step 0 权重，step 1 的 loss 不必等于 NPU 10 步的第 1 步。不要用这个不相等判断跑坏了。

### 3. 第 4 行的 1000 步

10 步两次对齐之后再跑。同样依赖 `_select_non_cp_kernel`。`--baseline-ngpus=8`。

```bash
python scripts/loss_compare.py . . \
  --baseline-module=torchtitan_recipes.tests.models.muse_spark \
  --baseline-config=muse_spark_debugmodel_mm \
  --baseline-ngpus=8 \
  --steps=1000 \
  --metrics=loss,grad_norm \
  --export-result=docs/gpu_baseline/debugmodel_mm_1000.txt \
  --output-folder=outputs/gpu_baseline_debugmodel_mm
```

### 4. 第 5 行

NPU 已经跑完 10 步。GPU 还没有。配方 `muse_spark_debugmodel_ultradata_lora_sft`，8 卡 `dp_shard=2, cp=2, tp=2, ep=8`。当前分支读本地 JSONL，不再走 Hugging Face 流。

要先放好两样文件，不要下 816B 权重，也不要下全部 32 份 JSONL：

- tokenizer 放到 `./assets/hf/Muse-Spark-Open-816B-A42B/`。官方仓库要手动授权。NPU 用的是公开副本 `unsloth/Muse-Spark-Open-816B-A42B` 的 tokenizer 文件。`tokenizer.json` 最大 id 是 201817，配置里的词表宽度仍是 202048。
- `openbmb/UltraData-SFT-Agent-2609` 的 General-Agent 第 1 份，放到 `./assets/datasets/muse_spark_ultradata/General_Agent_part-01-of-32.jsonl`。大小必须是 `1575113443` 字节。`repeat=True`，10 步够用。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel_ultradata_lora_sft \
  ./run_train.sh
```

跑两次。loss 会停在 12.2 附近，因为 `ln(202048)` 约 12.216，而且只训 LoRA。不要和第 1 行 7.x 的 loss 比。

10 步两次对齐之后，再跑 1000 步。这条同样是 `cp=2`，要有 `_select_non_cp_kernel`。32K、全量重算，会比第 1 行慢很多。`--baseline-ngpus=8`。

```bash
python scripts/loss_compare.py . . \
  --baseline-module=torchtitan_recipes.tests.models.muse_spark \
  --baseline-config=muse_spark_debugmodel_ultradata_lora_sft \
  --baseline-ngpus=8 \
  --steps=1000 \
  --metrics=loss,grad_norm \
  --export-result=docs/gpu_baseline/ultradata_lora_1000.txt \
  --output-folder=outputs/gpu_baseline_ultradata_lora
```

## 跑完记什么

10 步：逐步 loss、grad_norm，以及 GPU 与 NPU 的差。第 4 行的 NPU loss 是 `7.69125, 6.81713, 5.68443, 6.06900, 4.36464, 5.61337, 4.97801, 3.39901, 3.96171, 4.57818`，grad_norm 是 `9.7405, 6.5383, 4.2879, 4.9316, 5.0541, 3.8197, 3.1647, 3.0696, 2.7995, 2.7223`。第 5 行 NPU loss 是 `12.31145, 12.31274, 12.31039, 12.30229, 12.27736, 12.28796, 12.27721, 12.26652, 12.28093, 12.28428`，grad_norm 是 `0.2715, 0.2676, 0.3027, 0.3203, 0.3633, 0.3613, 0.3223, 0.3691, 0.3398, 0.2656`。

1000 步：导出文件路径和 step 0 目录。NPU 还没加载之前，MAE 写「基线已导出，NPU 未加载」。不要在 GPU 上就算 MAE。
