# 多模态第 1 步：音频 RMSNorm 改完后再导出（GPU）

只跑 AllGatherKV 这一条，只跑第 1 步。这次模型有一处改动：音频塔的 RMSNorm 改为 fp32 两两相加，最后再舍入回 bf16。文本的 qk norm、output norm、perception norm 没动。conv 和 linear 没动。

不要再改模型。不要重做 step-0。不要跑 Ulysses、文本配置、Dist-MoE、mxfp8，也不要跑 1000 步。不要覆盖这些目录：

- `outputs/step1_fp32_attn/`
- `outputs/step1_audio_acts/`
- `outputs/step1_layer0_acts/`
- `outputs/step1_gate_scores/`
- `outputs/step1_expert_ids/`

旧 loss `7.789858818054199` 不要拿来当通过条件。上一次 fp32 attention 的 GPU 数是 loss `7.79107`、grad_norm `9.0969`。这次 norm 改了，这两个数会变，变了是预期的。如果又回到 `7.78986`，说明树上没有 fp32 attention 那次改动，停下来，保留日志。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。常驻的 `atk server`（每卡大约几百 MB，不到 2GB）可以忽略。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 不存在。
- venv 里把 `aten::grid_sampler_2d_backward` 转到 CPU 的垫片没了。垫片在 `usercustomize.py` 就保持那样，不要再盖一层 `sitecustomize.py`。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
test -f scripts/dump_step1_audio_acts.py
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。

## 环境

沿用上次导出的那个 venv。不要安装 `torchtitan_ascend`，不要调用 `torchtitan_ascend.setup()`，不要设置 `MUSE_SETUP_ASCEND`。不要设置 `ENTRYPOINT`。不要设置 `MM_DTYPE_REDUCED_PRECISION`、`HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑这一个 torchrun。

脚本在这三个变量还没设置时会补上：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

## 跑这一条

仓库根目录。网格是 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。输出目录是新的。目录里如果已经有 `rank0.pt`，脚本会拒绝覆盖并退出。

```bash
python scripts/dump_step1_audio_acts.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_audio_rms/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_audio_rms/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_audio_rms_mm_fsdp2.log 2>&1
echo EXIT:$?
```

日志里要有 `Training completed` 和 `STEP1_AUDIO check ok`。记下 step 1 的 loss、grad_norm。

若进程在 `step1 audio dump missed` 处抛错，或检查没打出 `STEP1_AUDIO check ok`，保留日志和已经写下的 `.pt`，不要改代码重跑。

## 怎样算交数完成

8 个 `.pt` 都在。日志有 `Training completed` 和 `STEP1_AUDIO check ok`。把 step 1 的 loss、grad_norm、`git rev-parse HEAD`，以及 `outputs/step1_audio_rms/mm_fsdp2_tp2_cp2_ep4/` 留下。`.pt` 提交到本分支。不要提交 tensorboard 目录和日志。不要用旧 expert id 或旧 loss 判断这次过不过。

本说明由 AI 助手根据音频塔 RMSNorm 的一档 bf16 舍入差整理。上次对照里，波形、linear 和 conv 按位相同，分叉在音频 RMSNorm。
