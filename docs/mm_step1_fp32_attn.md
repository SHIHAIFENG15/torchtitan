# 多模态第 1 步：fp32 attention 对照（GPU）

只跑 AllGatherKV 这一条，只跑第 1 步。这次测的是代码改动本身：qkv、output gate、wo 改为 fp32 累加并保持 fp32 输出，q、k、v 以 fp32 进入 flex attention。视觉和音频 GEMM 也是 fp32 累加，输出再舍入回原来的 dtype。

旧的 step 1（loss `7.789858818054199`，grad_norm `9.11008358001709`，以及 `outputs/step1_expert_ids/`、`outputs/step1_gate_scores/`、`outputs/step1_layer0_acts/`）是改动前的 bf16 qkv 模型。那些数会变。不要拿它们当通过条件，也不要为了对齐旧数再跑一次。

在已经交出第 0 层激活的同一棵 torchtitan 树上执行。不要新克隆。不要改模型代码。不要重做 step-0。不要跑 Ulysses、文本配置、Dist-MoE、mxfp8，也不要跑 1000 步。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 不存在。
- venv 里把 `aten::grid_sampler_2d_backward` 转到 CPU 的垫片没了。垫片在 `usercustomize.py` 就保持那样，不要再盖一层 `sitecustomize.py`。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
test -f scripts/dump_step1_fp32_attn.py
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

仓库根目录。网格是 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。输出写到新目录，不要覆盖旧的 expert、gate、layer0 目录。

```bash
python scripts/dump_step1_fp32_attn.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_fp32_attn/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_fp32_attn/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_fp32_attn_mm_fsdp2.log 2>&1
echo EXIT:$?
```

日志里要有 `Training completed`。记下 step 1 的 loss 和 grad_norm，这就是新的 GPU 基线。和旧的 `7.78986` 不同是预期的。

脚本末尾应打出每个 rank 的 `query`/`key`/`value`/`attn_out`/`attention_output` dtype，以及 `STEP1_FP32_ATTN check ok`。这五个张量应为 `float32`。若 dtype 不是 float32，或进程在 `fp32 attn dump missed` 处抛错，保留日志和已经写下的 `.pt`，不要改代码重跑。

## 每个 rank 要有的张量

8 个文件，`rank0.pt` 到 `rank7.pt`。张量都是该 rank 的本地 shard。

`layer0` 的键与上次第 0 层导出相同：`tokens`、`positions`、`padding_mask`、`block_input`、`attention_input`、`query`、`key`、`value`、`attn_out`、`attention_output`、`attention_branch`、`stream`、`ffn_input`。有视觉或音频 token 时还会有对应的 bank index。

`routers` 下有 `layers.0` 到 `layers.7` 的 `feed_forward.router`。每个里面有排好序的 `expert_ids`、`topk_scores`，以及该层的 `padding_mask`。

`padding_mask` 的长度可以和 `ffn_input` 的 token 维不同。不要因为这个长度重跑。

## 这次改了什么，这次不覆盖什么

文本 attention 的 qkv、output gate、wo 是 bf16 乘、fp32 累加、fp32 输出。q、k、v 和 attention 输出保持 fp32。CUDA 上走 `mm(out_dtype=float32)`，非 CUDA 上把输入和权重升到 fp32 再做 mm。

视觉和音频的 GEMM 也是 fp32 累加，但输出会再舍入一次回到输入的 dtype。NPU 的 LayerNorm 不接受 fp32 激活配 bf16 权重；`ca1d9b5c` 会在视觉 `ln_pre` 上报这个错。用这一次的 HEAD，不要用 `ca1d9b5c`。

没有改的部分：router gate 本来就是 fp32 输出，保持不动。MoE 的 grouped expert GEMM 仍把权重收成 bf16，pooler 和 unpooler 仍是普通 Linear。进 attention 和进 MoE 之前的 `.to(bf16)` 还在。第 0 层之前若仍有 token 不一致，会留在 `block_input` 里。

## 怎样算交数完成

8 个 `.pt` 都在。日志有 `Training completed` 和 `STEP1_FP32_ATTN check ok`。把 step 1 的 loss、grad_norm、`git rev-parse HEAD`、dtype 那几行，以及 `outputs/step1_fp32_attn/mm_fsdp2_tp2_cp2_ep4/` 留下。`.pt` 提交到本分支。不要提交 tensorboard 目录和日志。不要和旧 expert id 做通过与否的判断。

本说明由 AI 助手根据已定位的 qkv 舍入差整理。
