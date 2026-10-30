# Muse Spark GPU 基线：环境搭建和验收

给执行这份文档的 agent：先读完再装环境、再启动训练。命令在 GPU 机器上执行。不要在 `179.173.2.3` 上启动这里的训练，不要改 NPU 上已经写好的验收数，不要把 `docs/npu_pta_fixes.md` 里的 site-packages 补丁打进 CUDA 环境。

人要做的事和 agent 相同：把当前这棵带未提交改动的 torchtitan 拷到 GPU 机器，装 CUDA 版 PyTorch，按下面四个配置各取一份 GPU 基线。

## 缺的是什么

NPU 验收写在 `docs/npu_debugmodel_verification.md`。前两行四个配置已经有确定性的 10 步 loss / grad_norm。下面四项没有 GPU 数，不能编：

| 验收项 | 标准 | 现在缺什么 |
| --- | --- | --- |
| 精度 | 同一份 step 0 权重，两边各跑不少于 1000 步。MoE 相对 GPU 基线的平均绝对误差不超过 0.02，或平均相对误差不超过 2% | 没有这份共享权重，也没有 GPU 的 1000 步 loss。10 步对 10 步不是这项 |
| 性能 | 丢掉前 100 步，用稳定段算同一配方的 8 卡 tps / 4 卡 tps，再和 GPU 的同一比值比，线性度差距不超过 2 个百分点 | 现有四个配置的网格互不相同，不能拿第 1 行的 4 卡 tps 去除第 2 行③的 8 卡 tps。NPU 这边也没有满 100 步 |
| 内存 | 日志里的 `memory` 是这一步窗口的最大预留，不是整卡占用。先比单卡和同场景 GPU，再比 FSDP2 多卡 | 没有 GPU 的预留数。当前配方不是单卡 |
| 确定性 | 同一配置连跑两次，loss 和 grad_norm 的打印值逐项相同，并且和 GPU 的同策略结果一致 | NPU 自己的两次已经对齐。GPU 的两次还没有 |

这次 GPU 机器要交出两层结果：

1. 和 NPU 同一条启动路径的 10 步，每个配置连跑两次。用来并排看 loss / grad_norm / memory / tps。两次 GPU 的 loss 和 grad_norm 必须相同。不要求等于下面的 NPU 数字。差多少就记多少。
2. 精度那一项用 `scripts/loss_compare.py` 从同一份 step 0 权重跑 1000 步，只在 GPU 上先把基线文件存下来。NPU 还没加载过这份权重，所以这一步先不要算 MAE。

第 3 到第 5 行（Dist-MoE、清单上 EP2 的多模态、32K LoRA）NPU 也没有样本。不要在 GPU 上用第 2 行的结果代替它们。

## 必须带着的代码

分支 `npu-muse-spark-debug-r04`，已提交 HEAD `d0f5101f`，基于 `release/0.4` 的 `45e33cd4`。只 `git clone` 或只 checkout 这个 HEAD 不够。下面这些改动还在工作区里，GPU 基线必须带上，否则网格和编译区会和 NPU 的 10 步不是同一次配置：

- `torchtitan_recipes/tests/models/muse_spark.py`
  - `local_compile_regions = ["loss", "fp32_to_bf16_split"]`，写在 `muse_spark_debugmodel` 里，`debugmodel_mm` 换过 model config 之后再写一次
  - `MUSE_DEBUG_DETERMINISTIC=1` 时把 `debug.seed` 设为 42，`debug.deterministic` 设为 True
  - `muse_spark_debugmodel_with_cp` 的网格是 `dp_shard=1, cp=2, tp=2, ep=2`。不要退回基类的 `dp_shard=2`，那会变成 8 卡
  - `with_cp` 清空 `override.imports`。融合 SwiGLU 这条覆盖只注册了 CUDA。NPU 清掉之后走 `silu(gate) * up`。GPU 基线保持清空，才和 NPU 是同一条算子路径
- `docs/npu_debugmodel_verification.md`
- 本文件

已提交的 6 个 NPU 提交可以留在这棵树上跑 CUDA。RoPE 和视觉编码器里对 complex 的改写只在 `device.type == "npu"` 时生效。融合 SwiGLU 只在设备不是 CUDA 时退回普通 SiLU。词表并行损失里少掉的那次 device assert 只发生在编译过程中。这些都不要为了「跑 GPU」再改回去。

不要带上：

- 容器里的 `.venv`
- `/usr/local/Ascend`
- `docs/npu_pta_fixes.md` 描述的已安装包补丁。那是 910B 的 PTA，不是 CUDA 环境的一部分

源目录在 NPU 主机上：

`/home/npu-work/npu-y00839695-torchtitan-20261028/dev/torchtitan`

拷到 GPU 机器之后先核对：

```bash
git rev-parse HEAD
git status -sb
grep -n "local_compile_regions" torchtitan_recipes/tests/models/muse_spark.py
grep -n "data_parallel_shard_degree=1" torchtitan_recipes/tests/models/muse_spark.py
```

HEAD 必须是 `d0f5101f`。`grep` 两处都要有命中。`git status` 里应能看到 `muse_spark.py` 被修改。对不上就停止，不要用 `release/0.4` 的干净树代替。

## 安装

项目要求 Python 3.11 及以上。NPU 那台是 3.12。GPU 用 3.12 即可。

先看本机 CUDA，再选对应的 PyTorch 轮子。README 里的 `cu132` 只是例子，不要在 CUDA 12.1 的机器上照抄。

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

按 `nvidia-smi` 右上角的 CUDA 版本选一个 nightly 索引，例如驱动支持 CUDA 12.8 时用 `cu128`。装完必须是带 CUDA 的 torch，不是 `+cpu`，也不是 `torch_npu`。

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
pip install -e . --no-deps
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

`torch.cuda.is_available()` 必须是 True。卡数少于 4 就停止。卡数少于 8 时，三个 4 卡配置可以跑，8 卡多模态先记下跳过，不要把网格改成 4 卡来凑。

`requirements.txt` 里的 `datasets` 第一次读本地 json 时可能要拉数据集脚本。`tests/assets/c4_test/data.json` 和 `tests/assets/tokenizer` 已经在树里，不需要下载 C4 全文。

不要执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

不要设置 `HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`TORCHNPU_PRECOMPILE_THREADS`。`TORCHINDUCTOR_COMPILE_THREADS=1` 是 910B 编译进程池段错误的规避，CUDA 上不要预先加上。

确定性需要的 CuBLAS 工作区要在进程启动前设好。训练代码里也会再设一次，但那时候动态库可能已经初始化。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

`MUSE_DEBUG_DETERMINISTIC=1` 没有对应的命令行开关。`ConfigLoader` 的 `--override` 只接受已经注册的 override，不能写 `--override training.steps=1000`。`deterministic_warn_only` 保持默认 False，不要改成 True。

`run_train.sh` 自己会设 `PYTORCH_ALLOC_CONF=expandable_segments:True`，并用 `torchrun --nproc_per_node=$NGPU`。`NGPU` 必须等于配方的世界大小，不是空闲卡的张数。

## 四个配置

启动前用 `nvidia-smi` 确认要占用的卡没有别的训练。被占用就停，不要杀掉别人的进程。

| 清单 | 配置 | 模块 | NGPU | 网格 |
| --- | --- | --- | --- | --- |
| 第 1 行 | `muse_spark_debugmodel` | `torchtitan_recipes.tests.models.muse_spark` | 4 | `dp_shard=2, tp=2, ep=2` |
| 第 2 行① | `muse_spark_debugmodel_with_cp` | `torchtitan_recipes.tests.models.muse_spark` | 4 | `dp_shard=1, cp=2, tp=2, ep=2` |
| 第 2 行② | `muse_spark_debugmodel_fsdp2_cp2_ep4` | `tests.integration_tests.models` | 4 | `dp_shard=2, cp=2, tp=1, ep=4` |
| 第 2 行③ | `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` | `tests.integration_tests.models` | 8 | `dp_shard=2, cp=2, tp=2, ep=4` |

日志里应出现 `Trainer is initialized with 512 tokens per DP rank`。第 1 行和第 2 行②是每步 1024 token（2 个 DP rank）。第 2 行①只有 1 个 DP rank，每步 512 token。对不上就停止这次任务。

10 步确定性，每个配置两次。把 `CONFIG`、`MODULE`、`NGPU` 换成表里的值：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
cd /path/to/torchtitan
NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark CONFIG=muse_spark_debugmodel \
  ./run_train.sh > logs/gpu_det_debugmodel_r1.log 2>&1
NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark CONFIG=muse_spark_debugmodel \
  ./run_train.sh > logs/gpu_det_debugmodel_r2.log 2>&1
```

另外三个配置同样各跑 r1、r2。8 卡那条 `NGPU=8`。一次只跑一个 torchrun。日志里要有 `Training completed`，不能有 `ChildFailedError`。

日志带 ANSI 色码。先去掉再取数：

```python
import re
text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, encoding="utf-8", errors="replace").read())
pat = re.compile(
    r"step:\s+(\d+)\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)\s+memory:\s+([0-9.]+)GiB.*?tps:\s+([0-9,]+)"
)
```

同一次配置的 r1 和 r2：10 步 loss 和 grad_norm 必须逐项相同。tps 和第 1 步 memory 可以不同。不同就写明第一次分叉的 step，不要把这次记成确定性通过。

NPU 上同一路径、同一 `MUSE_DEBUG_DETERMINISTIC=1` 的 10 步如下。GPU 数字写在旁边，不要改这组 NPU 数。

第 1 行 loss：`7.68353, 6.88134, 5.98353, 5.61493, 6.05757, 5.39870, 5.23369, 5.46803, 5.07188, 4.98734`

grad_norm：`4.7419, 3.2349, 2.3663, 1.9338, 1.7812, 1.6968, 1.6693, 1.5549, 1.7912, 1.6464`

第 2 行① loss：`7.72229, 6.93197, 6.03953, 5.71570, 5.73440, 5.44686, 5.28071, 5.90602, 5.17893, 5.12831`

grad_norm：`5.1450, 3.9766, 2.8388, 2.2161, 2.1739, 1.8826, 1.9517, 2.0066, 2.0813, 1.8717`

第 2 行② loss：`7.73262, 6.89378, 5.99186, 5.64755, 6.07865, 5.43435, 5.27176, 5.50165, 5.10669, 5.01443`

grad_norm：`4.6582, 3.2025, 2.3524, 1.9698, 1.8299, 1.7440, 1.6869, 1.5714, 1.8020, 1.6591`

第 2 行③ loss：`7.67449, 6.84383, 5.69314, 6.02861, 4.36879, 5.64743, 4.94917, 3.33959, 3.95203, 4.55447`

grad_norm：`10.1172, 6.4551, 4.1008, 5.0077, 5.0873, 3.7888, 2.9002, 3.2298, 2.8241, 2.8045`

这组 NPU 数来自各卡自己的分片初始化和 seed 42，没有加载 checkpoint。GPU 的 10 步也走这条路径，所以可以逐项相减。差的是设备实现，不是某一次跑失败。

## 1000 步基线

`scripts/loss_compare.py` 会做三件事：把并行度临时改成全 1，在 1 张卡上写出不含分片的 step 0 权重；再按原配方的网格把这份权重加载进来；并把 `training.steps` 和 `debug.deterministic`、`debug.seed=42` 设上。`--baseline-ngpus` 的默认值是 8。4 卡配方必须显式写成 4，否则世界大小对不上。

这条路径和第 1 节的 10 步不是同一次初始化。step 1 的 loss 不必等于 `7.68353`。不要用这个不相等来判断 GPU 训练坏了。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
python scripts/loss_compare.py . . \
  --baseline-module=torchtitan_recipes.tests.models.muse_spark \
  --baseline-config=muse_spark_debugmodel \
  --baseline-ngpus=4 \
  --steps=1000 \
  --metrics=loss,grad_norm \
  --export-result=docs/gpu_baseline/debugmodel_1000.txt \
  --output-folder=outputs/gpu_baseline_debugmodel
```

第 2 行①把 config 换成 `muse_spark_debugmodel_with_cp`，ngpus 仍是 4。第 2 行②的 module 换成 `tests.integration_tests.models`，config 换成 `muse_spark_debugmodel_fsdp2_cp2_ep4`，ngpus 是 4。第 2 行③的 config 换成 `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`，ngpus 是 8。

导出的文件和 `outputs/gpu_baseline_*/` 里的 step 0 checkpoint 留在 GPU 机器上。NPU 要复现精度时，加载的是这一份，而不是各自再初始化一次。在 NPU 还没跑这 1000 步之前，MAE 一栏写「基线已导出，NPU 未加载」。

`loss_compare.py` 造 seed checkpoint 时会把 `checkpointer` 从 None 改成 `CheckpointManager`。如果这一步失败，保留完整 traceback，已经跑完的 10 步仍然有效。不要为了让它通过去改模型数学，也不要把 `deterministic_warn_only` 设成 True。

## 性能和内存怎么记

从 10 步日志，以及 1000 步那次的训练日志里，按 step 记下 `memory` 和 `tps`。

`tps` 是单卡 token/s：`每步每 DP rank 的配置 token 数 / (这一步耗时 × CP × TP × PP)`。debugmodel 的配置 token 数是 512。它不含编译。同一次确定性的两次跑，tps 可以差一截，不能拿 tps 判断确定性。

`tflops` / `mfu` 在 NPU 上因为认不出 910B 峰值，分母落到了 A100 的 `3.120e14`，mfu 打出来是 0.00%。GPU 上如果日志给出了本卡峰值，把峰值和 mfu 原样记下。不要用 NPU 那个 0.00% 去比。

线性度这一项先空着。四个配置的 8 卡和 4 卡不是同一配方把 DP 加倍。不要把第 1 行改成 `NGPU=8`，那个配方的世界大小是 4。

内存同时记两列：日志里的 `memory`（最大预留），以及 `nvidia-smi` 看到的卡型号和总容量。NPU 日志里的容量是 Ascend910B4 29.49GiB。第 1 步往往含编译，比后面高。比较时写明是第几步。

## 结果写到哪里

在 GPU 这棵树上新建 `docs/gpu_debugmodel_baseline.md`，不要改写 `docs/npu_debugmodel_verification.md` 里的 NPU 表。每份配置写：

- `git rev-parse HEAD`，以及 `muse_spark.py` 是否仍含上面的未提交改动
- torch 版本、`nvidia-smi` 的卡型号和张数
- 两次 10 步的 loss、grad_norm、memory、tps
- 和上一节 NPU 数列的逐项差。第一个不相同的 step 单独写出来
- 1000 步导出文件的路径。没跑成就写失败阶段和 traceback 路径
- 性能线性度写「未测」，并写明原因是没有同一配方的 8 卡和 4 卡

跑完把 `docs/gpu_debugmodel_baseline.md`、`docs/gpu_baseline/*.txt` 和对应的 `outputs/gpu_baseline_*/` checkpoint 留在 GPU 机器上。精度对照要等 NPU 加载同一份 step 0 权重再跑 1000 步之后才能算。
