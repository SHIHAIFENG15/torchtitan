# 多模态第 1 步：音频塔中间结果（GPU）

只跑 AllGatherKV 这一条，只跑第 1 步。模型代码和上一次 fp32 attention 是同一份。这次只多记音频塔每一级的输出，用来看 28 个 token 是从哪一级开始和 NPU 分开的。

不要改模型。不要重做 step-0。不要跑 Ulysses、文本配置、Dist-MoE、mxfp8，也不要跑 1000 步。不要覆盖 `outputs/step1_fp32_attn/`、`outputs/step1_layer0_acts/`、`outputs/step1_gate_scores/`、`outputs/step1_expert_ids/`。

旧 loss `7.789858818054199` 是 bf16 qkv 的数，不要拿来当通过条件。上一次 fp32 attention 的 GPU 数是 loss `7.79107`、grad_norm `9.0969`。这次模型没改，loss 和 grad_norm 应靠近这两个数。如果又回到 `7.78986`，说明树上没有 fp32 attention 那次改动，停下来，保留日志。

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
  --out outputs/step1_audio_acts/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_audio_acts/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_audio_acts_mm_fsdp2.log 2>&1
echo EXIT:$?
```

日志里要有 `Training completed` 和 `STEP1_AUDIO check ok`。记下 step 1 的 loss、grad_norm，以及每个 rank 那行 `dummy=`、`audio_embeds` 的 shape 和 dtype。

若进程在 `step1 audio dump missed` 处抛错，或检查没打出 `STEP1_AUDIO check ok`，保留日志和已经写下的 `.pt`，不要改代码重跑。

## 每个 rank 要有的张量

8 个文件，`rank0.pt` 到 `rank7.pt`。`tensors` 里每个键是一个列表，按这一步里的调用顺序排列。一段波形调用一次。dtype 保持模块实际输出，不要转成 bf16。

- `audio_encoder.waveforms`、`audio_encoder.lengths`：进编码器的波形和每段的 padded 长度。
- `audio_encoder.layers.{0,1}.patchifier`，以及它的 `.linear`、`.norm`。
- `audio_encoder.layers.{0,1}.block{1,2}`：该 block 加完残差之后的输出。
- 同一个 block 下的 `.conv`：depthwise causal conv 的输出，还没进后面的 linear。
- 同一个 block 下的 `.linear1`、`.norm`、`.linear2`。
- `audio_encoder.norm`、`audio_encoder.o_proj`、`audio_encoder.dual_norm`。
- `audio_encoder`：编码器返回值，也就是 dual_norm 之后的特征。
- `audio_embeds`：写回 residual 之前的向量，perception norm 再乘上 embedding multiplier。
- `audio_bank_indices_T`：本地 token 到这条 bank 的下标。没有真实音频的 rank 可以没有这一项。
- `block_input`：第 0 层入口，token 维在前。上一次 fp32 dump 里对过的下标就在这一维上。

`audio_is_dummy` 为 true 的 rank 跑的是占位波形，输出不会写进 hidden。比较时跳过这些 rank。

上一次已经对上的位置，供之后对齐，这次不用在 GPU 上分析：rank2 的 85 和 93 到 104，rank3 的 44，rank4 的 87 和 95 到 106，rank5 的 46。这些位置的 `audio_bank_indices_T` 是非负的，`video_bank` 是 -1 或没有。

## 怎样算交数完成

8 个 `.pt` 都在。日志有 `Training completed` 和 `STEP1_AUDIO check ok`。把 step 1 的 loss、grad_norm、`git rev-parse HEAD`、每个 rank 的 `dummy` / `audio_embeds` 那一行，以及 `outputs/step1_audio_acts/mm_fsdp2_tp2_cp2_ep4/` 留下。`.pt` 提交到本分支。不要提交 tensorboard 目录和日志。不要用旧 expert id 或旧 loss 判断这次过不过。

本说明由 AI 助手根据上一次 fp32 attention 对照整理。对照里，attention 输入按位相同的 rank 上，value 的差已经落到 fp32 累加量级。剩下的第 0 层之前的差在音频写回，不在视觉。
