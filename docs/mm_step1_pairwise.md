# 多模态第 1 步：成对乘加对照（GPU）

只跑 AllGatherKV 这一条，只跑第 1 步。不要跑 `docs/mm_acc_1000_rsqrt.md`，不要跑 1000 步，不要跑 Ulysses。

NPU 已经用同一份代码跑完这一步。`wo` 改成各卡本地成对相加之后，layer0 的 Q、attention 输出、`ffn_input` 和 8 层 expert id 与上一份 NPU dump 逐位相同。和旧的 GPU dump（`9934c577`，CUDA `mm(out_dtype=float32)`）比，专家选择仍差 57 个 token。那份旧 dump 不能当目标。这次要的是 GPU 在 `ddf5dd7b` 或更新的 HEAD 上重跑，和 NPU 的 `outputs/step1_pairwise_wo/` 比。

在已经有 `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 的同一棵树上执行。不要新克隆。不要改模型代码。不要重做 step-0。

## 先停下来的情况

- `nvidia-smi` 上有别人的训练。不要杀。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 不存在。
- venv 里把 `aten::grid_sampler_2d_backward` 转到 CPU 的垫片没了。垫片在 `usercustomize.py` 就保持那样，不要再盖一层 `sitecustomize.py`。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
test "$(git rev-parse HEAD)" != ""
git merge-base --is-ancestor ddf5dd7b HEAD
test -f scripts/dump_step1_fp32_attn.py
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。HEAD 必须包含 `ddf5dd7b`。导入 Muse 模型时会执行 `os.environ.setdefault("MUSE_PAIRWISE_GEMM", "1")`，不要把它关掉，也不要另设 `MM_DTYPE_REDUCED_PRECISION`。

## 环境

沿用上次导出的那个 venv。不要安装 `torchtitan_ascend`，不要调用 `torchtitan_ascend.setup()`，不要设置 `MUSE_SETUP_ASCEND`。不要设置 `ENTRYPOINT`。不要设置 `MM_DTYPE_REDUCED_PRECISION`、`HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑这一个 torchrun。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

## 跑这一条

仓库根目录。网格是 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。写到新目录，不要覆盖 `outputs/step1_fp32_attn/`、`outputs/step1_expert_ids/` 或任何已有 dump。

```bash
python scripts/dump_step1_fp32_attn.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_pairwise_gpu/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_pairwise_gpu/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_pairwise_gpu_mm_fsdp2.log 2>&1
echo EXIT:$?
```

日志里要有 `STEP1_FP32_ATTN check ok`。`query`、`key`、`value`、`attn_out`、`attention_output` 应为 `float32`。若进程在 `fp32 attn dump missed` 或 `pairwise local shape` 处抛错，保留日志，不要改代码重跑。

## 怎样算交数完成

8 个 `.pt` 都在 `outputs/step1_pairwise_gpu/mm_fsdp2_tp2_cp2_ep4/`。记下 step 1 的 loss、grad_norm 和 `git rev-parse HEAD`。`.pt` 可以提交到本分支。不要提交 tensorboard 目录和日志。不要和 `9934c577` 的旧 expert id 判断通过与否。NPU 对照文件在 A5 的 `outputs/step1_pairwise_wo/mm_fsdp2_tp2_cp2_ep4/`。

本说明由 AI 助手根据 NPU 已跑完的第 1 步整理。
