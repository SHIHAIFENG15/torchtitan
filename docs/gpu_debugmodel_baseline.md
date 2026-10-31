# Muse Spark GPU debugmodel 基线结果

按 `docs/gpu_baseline_setup.md` 的要求在 GPU 机器（node-5-15，8×A100-80GB）上执行并记录。
执行日期：2026-10-30。

## 代码与环境状态

- 分支 `npu-muse-spark-debug-r04`，`git rev-parse HEAD` = `bdab079330dab1d707c027b7055e1a91460d9cb1`
- 文档要求的代码验证全部命中：
  - `grep -n "local_compile_regions" torchtitan_recipes/tests/models/muse_spark.py` → 152、248、338 行
  - `grep -n "data_parallel_shard_degree=1" ...` → 167 行（with_cp 网格保持 `dp_shard=1, cp=2, tp=2, ep=2`）
  - `MUSE_DEBUG_DETERMINISTIC` 处理在 153 行（seed=42，deterministic=True）
- `git status` 干净：文档所述「工作区未提交改动」已随 `bdab0793` 提交（该提交相对 `d0f5101f` 恰好就是这批 muse_spark.py 改动 + 文档，代码状态等价）
- torch `2.14.0a0+git69231fe`（CUDA 12.8），`torch.cuda.is_available()=True`，`device_count=8`
- GPU：NVIDIA A100-SXM4-80GB × 8（driver 580.178.04，CUDA 13.0）
- Python 3.12.3（容器 `gpu-z00824525-tt`，镜像 `pytorch-gpu-test:latest`，`--gpus all`），venv `--system-site-packages`
- 未设置 NPU 相关变量（无 HCCL_*、ASCEND_*、TORCHNPU_*、TORCHINDUCTOR_COMPILE_THREADS）；已设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`TORCH_DEVICE_BACKEND_AUTOLOAD=1`、`MUSE_DEBUG_DETERMINISTIC=1`

### 环境偏差（与文档默认流程不同处，均为环境层，不改模型数学）

1. torch 未另装 nightly，直接使用镜像内置的 cu128 构建（NVIDIA 内部 fork）。因此开源 torchvision 与其 C++ ABI 不匹配。
2. venv 内安装 `torchvision==0.27.0.dev20260407+cu128`，并在其 `_meta_registrations.py` 打了 `register_fake` 失败降级垫片（仅影响假注册，不影响算子数学）。
3. `cuda-bindings` 固定 13.0.0（cutlass-dsl 4.8.1 需要 `cuda.bindings.runtime.cudaLibrary_t`，13.4.x 已移除）。
4. `sitecustomize.py`（venv site-packages）注册了 `aten::grid_sampler_2d_backward` 的 CUDA kernel：把该反传路由到 CPU 原生实现（CPU 路径本身确定性）。起因：mm 配置的数据增广（torchvision transforms v2）走 `grid_sample`，`torch.use_deterministic_algorithms(True)` 下 CUDA 无确定性实现会直接抛错。CPU 实现与 CUDA 数学一致，只有 mm 配置触发。
5. torchrun 用系统 python 启动，通过 `PYTHONPATH=<venv site-packages>:/opt/pytorch-gpu-image/pytorch` 让其看到 venv 依赖。
6. pip 安装需 `PIP_NO_INDEX=0` 覆盖镜像默认值。

## 10 步确定性结果（每配置连跑两次）

四次配置的 r1 与 r2 的 loss、grad_norm **逐项完全相同**（见下表，r2 未重复列出）。
日志：`logs/gpu_det_{debugmodel,with_cp,fsdp2_ep4,mm}_{r1,r2}.log`（ANSI 已在解析时去除）。
`Training completed` 全部出现，无 `ChildFailedError`。

### 第 1 行 `muse_spark_debugmodel`（NGPU=4, dp_shard=2, tp=2, ep=2）

| step | loss (GPU) | loss (NPU) | diff | grad_norm (GPU) | grad_norm (NPU) | diff | memory (GiB) | tps |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.78319 | 7.68353 | +0.09966 | 4.5083 | 4.7419 | -0.2336 | 0.13 | 12 |
| 2 | 6.98204 | 6.88134 | +0.10070 | 3.0766 | 3.2349 | -0.1583 | 0.11 | 198 |
| 3 | 6.05718 | 5.98353 | +0.07365 | 2.3356 | 2.3663 | -0.0307 | 0.11 | 192 |
| 4 | 5.72283 | 5.61493 | +0.10790 | 1.8723 | 1.9338 | -0.0615 | 0.11 | 188 |
| 5 | 6.17705 | 6.05757 | +0.11948 | 1.8080 | 1.7812 | +0.0268 | 0.11 | 193 |
| 6 | 5.52639 | 5.39870 | +0.12769 | 1.6744 | 1.6968 | -0.0224 | 0.11 | 187 |
| 7 | 5.35266 | 5.23369 | +0.11897 | 1.6648 | 1.6693 | -0.0045 | 0.11 | 206 |
| 8 | 5.58669 | 5.46803 | +0.11866 | 1.5742 | 1.5549 | +0.0193 | 0.11 | 185 |
| 9 | 5.14160 | 5.07188 | +0.06972 | 1.7994 | 1.7912 | +0.0082 | 0.11 | 185 |
| 10 | 5.08906 | 4.98734 | +0.10172 | 1.6531 | 1.6464 | +0.0067 | 0.11 | 186 |

### 第 2 行① `muse_spark_debugmodel_with_cp`（NGPU=4, dp_shard=1, cp=2, tp=2, ep=2）

| step | loss (GPU) | loss (NPU) | diff | grad_norm (GPU) | grad_norm (NPU) | diff | memory (GiB) | tps |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.76121 | 7.72229 | +0.03892 | 4.9706 | 5.1450 | -0.1744 | 0.12 | 7 |
| 2 | 7.04560 | 6.93197 | +0.11363 | 3.7637 | 3.9766 | -0.2129 | 0.10 | 337 |
| 3 | 6.14399 | 6.03953 | +0.10446 | 2.7675 | 2.8388 | -0.0713 | 0.10 | 359 |
| 4 | 5.76702 | 5.71570 | +0.05132 | 2.1134 | 2.2161 | -0.1027 | 0.10 | 360 |
| 5 | 5.83972 | 5.73440 | +0.10532 | 2.0797 | 2.1739 | -0.0942 | 0.10 | 333 |
| 6 | 5.56806 | 5.44686 | +0.12120 | 1.8371 | 1.8826 | -0.0455 | 0.10 | 343 |
| 7 | 5.37893 | 5.28071 | +0.09822 | 1.8808 | 1.9517 | -0.0709 | 0.10 | 340 |
| 8 | 6.00469 | 5.90602 | +0.09867 | 1.9120 | 2.0066 | -0.0946 | 0.10 | 357 |
| 9 | 5.21235 | 5.17893 | +0.03342 | 1.9980 | 2.0813 | -0.0833 | 0.10 | 361 |
| 10 | 5.22134 | 5.12831 | +0.09303 | 1.8066 | 1.8717 | -0.0651 | 0.10 | 359 |

### 第 2 行② `muse_spark_debugmodel_fsdp2_cp2_ep4`（NGPU=4, dp_shard=2, cp=2, tp=1, ep=4）

| step | loss (GPU) | loss (NPU) | diff | grad_norm (GPU) | grad_norm (NPU) | diff | memory (GiB) | tps |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.77261 | 7.73262 | +0.03999 | 4.4130 | 4.6582 | -0.2452 | 0.10 | 12 |
| 2 | 6.97454 | 6.89378 | +0.08076 | 3.0704 | 3.2025 | -0.1321 | 0.12 | 319 |
| 3 | 6.04940 | 5.99186 | +0.05754 | 2.3049 | 2.3524 | -0.0475 | 0.12 | 329 |
| 4 | 5.73287 | 5.64755 | +0.08532 | 1.8938 | 1.9698 | -0.0760 | 0.12 | 326 |
| 5 | 6.19677 | 6.07865 | +0.11812 | 1.7841 | 1.8299 | -0.0458 | 0.12 | 329 |
| 6 | 5.54783 | 5.43435 | +0.11348 | 1.6787 | 1.7440 | -0.0653 | 0.12 | 330 |
| 7 | 5.35267 | 5.27176 | +0.08091 | 1.6436 | 1.6869 | -0.0433 | 0.12 | 326 |
| 8 | 5.58104 | 5.50165 | +0.07939 | 1.5637 | 1.5714 | -0.0077 | 0.12 | 331 |
| 9 | 5.14174 | 5.10669 | +0.03505 | 1.7834 | 1.8020 | -0.0186 | 0.12 | 327 |
| 10 | 5.09205 | 5.01443 | +0.07762 | 1.6563 | 1.6591 | -0.0028 | 0.12 | 331 |

### 第 2 行③ `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`（NGPU=8, dp_shard=2, cp=2, tp=2, ep=4）

| step | loss (GPU) | loss (NPU) | diff | grad_norm (GPU) | grad_norm (NPU) | diff | memory (GiB) | tps |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.66228 | 7.67449 | -0.01221 | 9.6677 | 10.1172 | -0.4495 | 0.07 | 5 |
| 2 | 6.78664 | 6.84383 | -0.05719 | 6.3877 | 6.4551 | -0.0674 | 0.14 | 14 |
| 3 | 5.78175 | 5.69314 | +0.08861 | 4.1806 | 4.1008 | +0.0798 | 0.14 | 108 |
| 4 | 6.13758 | 6.02861 | +0.10897 | 5.1661 | 5.0077 | +0.1584 | 0.14 | 112 |
| 5 | 4.30295 | 4.36879 | -0.06584 | 5.0362 | 5.0873 | -0.0511 | 0.14 | 15 |
| 6 | 5.76427 | 5.64743 | +0.11684 | 3.7439 | 3.7888 | -0.0449 | 0.14 | 75 |
| 7 | 5.09090 | 4.94917 | +0.14173 | 2.8541 | 2.9002 | -0.0461 | 0.14 | 109 |
| 8 | 3.43795 | 3.33959 | +0.09836 | 3.4150 | 3.2298 | +0.1852 | 0.14 | 15 |
| 9 | 4.05134 | 3.95203 | +0.09931 | 2.8106 | 2.8241 | -0.0135 | 0.14 | 108 |
| 10 | 4.67039 | 4.55447 | +0.11592 | 2.8055 | 2.8045 | +0.0010 | 0.14 | 111 |

### 与 NPU 数列的差异小结

- 四个配置从 **step 1 起** GPU 与 NPU 的 loss/grad_norm 即不相同（第一个不相同的 step = 1）。
- loss：GPU 系统性偏高，差值范围约 +0.03 ~ +0.14（除 mm 的 step 1/2/5 略低）。
- grad_norm：差值范围约 -0.45 ~ +0.19，多数 step 为 GPU 略低。
- 差异量级在各配置间一致，符合「设备实现差异」的预期，未发现某一次跑失败的异常点。

## 1000 步基线（精度项的共享权重）

| 配置 | 结果 |
| --- | --- |
| 第 1 行 `muse_spark_debugmodel` | ✅ 导出成功：`docs/gpu_baseline/debugmodel_1000.txt`（1003 行，step/loss/grad_norm × 1000） |
| 第 2 行① `with_cp` | ❌ 失败（见下） |
| 第 2 行② `fsdp2_cp2_ep4` | ❌ 失败（见下） |
| 第 2 行③ `mm_fsdp2_tp2_cp2_ep4` | ❌ 失败（见下） |

- 第 1 行产物：
  - step 0 共享权重：`outputs/checkpoint/step-0/`（fp32 model-only，约 40MB `.distcp` 分片；`loss_compare.py` 实际写入 `--output-dir=outputs` 下的 `checkpoint/step-0`，而非 `outputs/gpu_baseline_debugmodel/` 内）
  - 训练日志：`outputs/gpu_baseline_debugmodel/baseline_training.log`、seed 日志 `outputs/gpu_baseline_debugmodel/seed_training.log`
- 第 2 行①②③ 失败阶段：`create_seed_checkpoint`（`scripts/loss_compare.py` 在 NGPU=1 上构建模型写 step 0 权重时）：

  ```
  ValueError: layers.0.attention.inner_attention is CPInnerAttention but the
  context parallel degree is 1. Select a non-CP kernel.
  ```

  这三个配置的 attention 内核强制要求 cp≥2，与 seed 阶段「并行度临时改成全 1」冲突。
  完整 traceback 保留在：
  - `outputs/gpu_baseline_debugmodel_with_cp/seed_training.log`（另见 `logs/lc_with_cp.log`）
  - `outputs/gpu_baseline_fsdp2_cp2_ep4/seed_training.log`（另见 `logs/lc_fsdp2.log`）
  - `outputs/gpu_baseline_mm_fsdp2_tp2_cp2_ep4/seed_training.log`（另见 `logs/lc_mm.log`）

  按文档要求未修改模型数学/内核选择来强行通过；这三个配置的 10 步结果仍然有效。
  若要让 2①②③ 也产出 1000 步基线，需要上游明确 seed 阶段对 CP 内核的处理方式（例如 seed 允许 cp=2 的非分片写盘，或提供非 CP 内核的等价路径），建议与 NPU 侧同学对齐后再做。

- MAE 一栏：**基线已导出（第 1 行），NPU 未加载**。第 2 行①②③ 无基线文件。

## 性能线性度

未测——四个配置不是「同一配方把 DP 加倍」的关系（第 1 行世界大小固定为 4，不存在对应的 8 卡同配方），不满足线性度比较的前提。

## 内存

- 日志 `memory`（该步窗口最大预留）：见上表 memory 列。第 1 步通常含编译，偏高属正常。
- `nvidia-smi`：NVIDIA A100-SXM4-80GB × 8（对比 NPU 侧 Ascend910B4 29.49GiB）。
- mm 8 卡 10 步实测每卡约 6.8GiB（nvidia-smi）。

## 留在 GPU 机器上的产物清单

- 本文件：`docs/gpu_debugmodel_baseline.md`
- 导出基线：`docs/gpu_baseline/debugmodel_1000.txt`
- step 0 共享权重：`outputs/checkpoint/step-0/`
- 10 步日志：`logs/gpu_det_*.log`（8 个）
- 1000 步日志：`outputs/gpu_baseline_debugmodel/*.log`、`logs/lc_*.log`、失败 traceback：`outputs/gpu_baseline_*/seed_training.log`
- 解析脚本与结构化数据：`/home/y00839695/parse_logs.py`、`outputs/structured_logs/`


---

# 附录：GPU 基线补跑（followup）结果

对应 `docs/gpu_baseline_followup.md` 的任务，于 2026-10-31 执行完毕。此时分支 HEAD 已更新到 `d33e0152`（新增 `_select_non_cp_kernel` 等）。

## 第 4 行 `muse_spark_debugmodel_mm`（NGPU=8, dp_shard=2, cp=2, tp=2, ep=2, Ulysses）10 步 ×2

r1 与 r2 的 loss、grad_norm 逐项相同。日志：`logs/gpu_det_mm_ulysses_{r1,r2}.log`，均含 `Training completed`。

| step | loss (GPU) | loss (NPU) | diff | grad_norm (GPU) | grad_norm (NPU) | diff | memory (GiB) | tps |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.68172 | 7.69125 | -0.00953 | 9.6199 | 9.7405 | -0.1206 | 0.07 | 4 |
| 2 | 6.72927 | 6.81713 | -0.08786 | 6.0757 | 6.5383 | -0.4626 | 0.10 | 34 |
| 3 | 5.74722 | 5.68443 | +0.06279 | 4.1215 | 4.2879 | -0.1664 | 0.10 | 83 |
| 4 | 6.15960 | 6.06900 | +0.09060 | 5.0550 | 4.9316 | +0.1234 | 0.10 | 88 |
| 5 | 4.31937 | 4.36464 | -0.04527 | 4.8039 | 5.0541 | -0.2502 | 0.10 | 39 |
| 6 | 5.72301 | 5.61337 | +0.10964 | 3.5429 | 3.8197 | -0.2768 | 0.10 | 82 |
| 7 | 5.08826 | 4.97801 | +0.11025 | 3.1745 | 3.1647 | +0.0098 | 0.10 | 82 |
| 8 | 3.40201 | 3.39901 | +0.00300 | 3.0659 | 3.0696 | -0.0037 | 0.10 | 35 |
| 9 | 3.99220 | 3.96171 | +0.03049 | 2.8626 | 2.7995 | +0.0631 | 0.10 | 87 |
| 10 | 4.64057 | 4.57818 | +0.06239 | 2.8362 | 2.7223 | +0.1139 | 0.10 | 90 |

与 NPU 第一个不相同的 step = 1；loss 差 -0.09 ~ +0.11，grad_norm 差 -0.46 ~ +0.12，量级与其他配置一致。

## 1000 步基线导出（全部完成）

`_select_non_cp_kernel`（`scripts/_checkpoint_test_config.py`）生效后，seed 阶段不再出现
`Select a non-CP kernel` 报错。五个导出文件全部生成：

| 配置 | 导出文件 | step 0 权重目录 |
| --- | --- | --- |
| 第 1 行 `muse_spark_debugmodel` (4卡) | `docs/gpu_baseline/debugmodel_1000.txt` | `outputs/checkpoint-debugmodel/step-0` |
| 第 2 行① `with_cp` (4卡) | `docs/gpu_baseline/with_cp_1000.txt` | `outputs/checkpoint-with_cp/step-0` |
| 第 2 行② `fsdp2_cp2_ep4` (4卡) | `docs/gpu_baseline/fsdp2_cp2_ep4_1000.txt` | `outputs/checkpoint-fsdp2_cp2_ep4/step-0` |
| 第 2 行③ `mm_fsdp2_tp2_cp2_ep4` (8卡) | `docs/gpu_baseline/mm_fsdp2_tp2_cp2_ep4_1000.txt` | `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` |
| 第 4 行 `debugmodel_mm` (8卡) | `docs/gpu_baseline/debugmodel_mm_1000.txt` | `outputs/checkpoint-debugmodel_mm/step-0` |
| 第 5 行 `ultradata_lora_sft` (8卡) | `docs/gpu_baseline/ultradata_lora_1000.txt` | `outputs/checkpoint-ultradata_lora/step-0` |

说明：
- `loss_compare.py` 实际把 step 0 写到全局 `outputs/checkpoint/step-0`（会被下一次运行覆盖），上面每个配置的目录是在该配置跑完后立即拷贝留存的。
- MAE 一栏全部写「**基线已导出，NPU 未加载**」。
- 第 1 行的 1000 步首跑于 2026-10-30（`bdab0793`，无 `_select_non_cp_kernel` 但第 1 行不触发该路径），其余均为 2026-10-31（`d33e0152`）。导出文件为确定性输出，重跑内容不变。

## 第 5 行 `muse_spark_debugmodel_ultradata_lora_sft`（NGPU=8, dp_shard=2, cp=2, tp=2, ep=8）10 步 ×2

数据文件（本机经 hf-mirror 下载后上传，非服务器直连）：
- `assets/hf/Muse-Spark-Open-816B-A42B/`：`tokenizer.json`（28,085,854 B）、`tokenizer_config.json`、`special_tokens_map.json`
- `assets/datasets/muse_spark_ultradata/General_Agent_part-01-of-32.jsonl`（**1,575,113,443 B**，与文档要求一致）

r1 与 r2 逐项相同。日志：`logs/gpu_det_ultradata_{r1,r2}.log`，均含 `Training completed`。

| step | loss (GPU) | loss (NPU) | diff | grad_norm (GPU) | grad_norm (NPU) | diff | memory (GiB) | tps |
|---|---|---|---|---|---|---|---|---|
| 1 | 12.32300 | 12.31145 | +0.01155 | 0.2734 | 0.2715 | +0.0019 | 3.46 | 61 |
| 2 | 12.30694 | 12.31274 | -0.00580 | 0.2676 | 0.2676 | +0.0000 | 3.46 | 13,189 |
| 3 | 12.30219 | 12.31039 | -0.00820 | 0.2949 | 0.3027 | -0.0078 | 3.46 | 1,483 |
| 4 | 12.28745 | 12.30229 | -0.01484 | 0.3125 | 0.3203 | -0.0078 | 3.46 | 16,274 |
| 5 | 12.27585 | 12.27736 | -0.00151 | 0.3652 | 0.3633 | +0.0019 | 3.46 | 5,899 |
| 6 | 12.26657 | 12.28796 | -0.02139 | 0.3555 | 0.3613 | -0.0058 | 3.46 | 16,061 |
| 7 | 12.28624 | 12.27721 | +0.00903 | 0.3242 | 0.3223 | +0.0019 | 3.46 | 2,765 |
| 8 | 12.26244 | 12.26652 | -0.00408 | 0.3652 | 0.3691 | -0.0039 | 3.46 | 16,331 |
| 9 | 12.25394 | 12.28093 | -0.02699 | 0.3359 | 0.3398 | -0.0039 | 3.46 | 4,064 |
| 10 | 12.28451 | 12.28428 | +0.00023 | 0.2617 | 0.2656 | -0.0039 | 3.46 | 16,285 |

与 NPU 第一个不相同的 step = 1，但差异量级显著小于其他配置（loss ≤0.027，grad ≤0.008）。
loss 量级符合预期（ln(202048)≈12.216，仅训 LoRA）。

## 结论

- 10 步确定性：6 个配置（第 1、2①②③、4、5 行）GPU 各自两次逐项相同，全部通过。
- 与 NPU 对照：全部从 step 1 起存在小幅设备差异，无异常发散点。
- 1000 步基线：6 份导出文件 + 6 份 step 0 权重全部就绪，等待 NPU 加载后计算 MAE。
- 性能线性度：仍未测（前提不满足，见上文）。
