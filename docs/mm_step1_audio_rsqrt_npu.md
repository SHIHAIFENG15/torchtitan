# 多模态第 1 步：音频 Newton rsqrt，在另一台 A5 上跑并和 GPU 对比

只跑 AllGatherKV 这一条，只跑第 1 步。代码已经在本分支：音频 RMSNorm 用整数初值加 4 步不融合的 Newton `rsqrt`，算完立刻舍回 bf16。文本的 norm 没动。不要再改模型，不要重做 step-0，不要跑 Ulysses、文本配置、Dist-MoE、mxfp8，也不要跑 1000 步。

GPU 已经跑完，8 个 `.pt` 在：

`outputs/step1_audio_rsqrt/mm_fsdp2_tp2_cp2_ep4/rank{0-7}.pt`

提交 `bdf58493`。那个目录里已经有 `rank0.pt`，脚本会拒绝覆盖。NPU 必须写到另一个目录。

蓝区 `177.45.28.1` 目前 SSH 不通。那边同一份代码的第 1 步曾经跑完，loss `7.79030`、grad_norm `9.1101`，只作参照。这次以新机器上的 `.pt` 和分支里的 GPU `.pt` 逐张量对比为准。loss 不必等于 `7.79030`，也不要拿 `7.78986` 或 `7.79063` 当通过条件。

## 先停下来的情况

- 8 张卡里任何一张还有别的计算进程。不要杀。等全部空出来再跑。空卡上大约几 GB 的 HBM 底噪可以忽略，以 `npu-smi` 进程表为准。不要对 `npu-smi` 的管道做 `grep -q`，管道被提前关掉时 `npu-smi` 会挂住。
- 工作区不干净，或者当前分支不是 `npu-muse-spark-debug-r04`。
- `git merge-base --is-ancestor fd76aee9 HEAD` 失败。`fd76aee9` 才是 Newton `rsqrt` 那次代码。
- `outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0` 不存在。用原来的 step-0，不要重新初始化。
- 当前 Python 不能 `import torch`、`import torch_npu`、`import torchtitan_ascend`。不要为了这次任务重编 `torch_npu`，也不要新装一套环境。

```bash
git fetch origin npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only origin npu-muse-spark-debug-r04
git merge-base --is-ancestor fd76aee9 HEAD
test -f scripts/dump_step1_audio_acts.py
test -f outputs/step1_audio_rsqrt/mm_fsdp2_tp2_cp2_ep4/rank0.pt
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
test ! -f outputs/step1_audio_rsqrt_npu/mm_fsdp2_tp2_cp2_ep4/rank0.pt
```

远端名是 `shihaifeng` 时，把上面的 `origin` 换成 `shihaifeng`。

## 环境

在仓库根目录跑。用这台机器上已经能跑 Muse 的那个 Python，`torchrun` 要在 `PATH` 里。下面这个变量必须设置，worker 才会调用 `torchtitan_ascend.setup()`：

```bash
export MUSE_SETUP_ASCEND=1
```

不要设置 `ENTRYPOINT`、`MM_DTYPE_REDUCED_PRECISION`、`HCCL_DETERMINISTIC`、`ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。8 张卡都要可见。一次只跑这一个 torchrun。

脚本在这三个变量还没设置时会补上：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export MUSE_DEBUG_DETERMINISTIC=1
```

## 跑这一条

网格是 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。

```bash
python scripts/dump_step1_audio_acts.py \
  --module tests.integration_tests.models \
  --config muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  --ngpu 8 \
  --load outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
  --out outputs/step1_audio_rsqrt_npu/mm_fsdp2_tp2_cp2_ep4 \
  --tb tb_step1_audio_rsqrt_npu/mm_fsdp2_tp2_cp2_ep4 \
  > logs/step1_audio_rsqrt_npu_mm_fsdp2.log 2>&1
echo EXIT:$?
```

日志里要有 `Training completed` 和 `STEP1_AUDIO check ok`。记下 step 1 的 loss、grad_norm。

若进程在 `step1 audio dump missed` 处抛错，或检查没打出 `STEP1_AUDIO check ok`，保留日志和已经写下的 `.pt`，不要改代码重跑。

## 和 GPU 对

两边都用本机的 Python。GPU 是分支里已有的目录，NPU 是刚写出的目录。

```bash
python - << 'PY'
import torch
from pathlib import Path

gpu = Path("outputs/step1_audio_rsqrt/mm_fsdp2_tp2_cp2_ep4")
npu = Path("outputs/step1_audio_rsqrt_npu/mm_fsdp2_tp2_cp2_ep4")

def load(path):
    return torch.load(path, weights_only=False)

def report(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"META {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
    if torch.equal(a, b):
        return "equal"
    d = (a.float() - b.float()).abs()
    return f"nnz={int((d != 0).sum())}/{d.numel()} max={float(d.max()):.4e}"

keys = (
    "audio_encoder.waveforms",
    "audio_encoder.layers.0.patchifier.linear",
    "audio_encoder.layers.0.block1.conv",
    "audio_encoder.layers.0.block1.norm",
    "audio_encoder.layers.0.block1.linear1",
    "audio_encoder.layers.0.block1",
    "audio_encoder.layers.0.block2.norm",
    "audio_encoder",
    "audio_embeds",
    "block_input",
)
g0 = load(gpu / "rank0.pt")
n0 = load(npu / "rank0.pt")
print("GPU git", g0.get("git_rev"))
print("NPU git", n0.get("git_rev"))
for rank in range(8):
    g = load(gpu / f"rank{rank}.pt")["tensors"]
    n = load(npu / f"rank{rank}.pt")["tensors"]
    print(f"rank{rank}")
    for key in keys:
        parts = []
        for i, (a, b) in enumerate(zip(g[key], n[key])):
            parts.append(f"[{i}]{report(a, b)}")
        print(f"  {key} " + " ".join(parts))
PY
```

`block1.conv` 应当 equal。这次要看的是 `block1.norm` 和 `audio_embeds`。norm 逐位相同，说明两边的 `rsqrt` 已经落到同一档 bf16。`audio_embeds` 仍有差，差就出在 norm 之后的 linear，把对比原文留下，不要改代码重跑。

## 怎样算交数完成

8 个 NPU `.pt` 都在 `outputs/step1_audio_rsqrt_npu/mm_fsdp2_tp2_cp2_ep4/`。日志有 `Training completed` 和 `STEP1_AUDIO check ok`。把 step 1 的 loss、grad_norm、`git rev-parse HEAD`，以及上面这段对比输出留下。NPU 的 `.pt` 提交到本分支。不要提交 tensorboard 目录和日志，也不要改 `outputs/step1_audio_rsqrt/` 里的 GPU 文件。

本说明由 AI 助手整理。
