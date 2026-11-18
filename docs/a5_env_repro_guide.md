# A5 NPU 验证环境复现指导

A5 机器（177.45.28.1，8 x Ascend950DT）可能被回收。本文说明如何在另一台
950 机器上完整复现 Muse Spark NPU 精度验证环境。截至 `1628fc9`，所有
代码改动均已并入本分支，clone 即得。

本说明由 AI 助手整理，逐项与 A5 实机核对过（镜像、挂载、版本号、7 个
工作区补丁文件的 sha1、`_torch_compat` 的 blob id）。

## 1. 硬件与宿主机

- 8 x Ascend950DT，宿主机驱动 `npu-smi 26.2.0.b020`，8 卡全部可见。
- 不需要额外安装 CANN toolkit（在镜像里，9.2.0）。

## 2. 容器

镜像（华为公开仓库，可重新拉取）：

```
swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.13.0-cann9.2.0-950-openeuler24.03-py3.12
```

等效启动（A5 上是 `--privileged` + host 网络 + 以下挂载；设备由
privileged 覆盖，显式列表为 /dev/davinci0-7、/dev/davinci_manager、
/dev/hisi_hdc、/dev/ummu/tid、/dev/uburma/*）：

```bash
mkdir -p /work/a5-muse
docker run -dit --name muse-accept \
  --privileged --network host --ipc host \
  --shm-size 128g \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /work/a5-muse:/tmp/a5-muse \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.13.0-cann9.2.0-950-openeuler24.03-py3.12 \
  sleep infinity
```

容器内 Python 3.12.15，镜像自带（不要升级）：

| 包 | 版本 |
| --- | --- |
| torch | 2.15.0+cpu |
| torch_npu | 2.15.0+git1a25e6b |
| triton / triton_ascend | 3.6.0 |
| spmd_types | 0.3.1 |
| torchao | 0.19.0.dev20261109+cpu |
| torchvision | 0.29.2 |
| numpy | 2.5.4 |
| torch_checkpointing / torch_remat | 0.1.1 / 0.2.2 |

## 3. 代码（两个仓库，都 editable 安装）

```bash
cd /tmp/a5-muse
git clone -b npu-muse-spark-debug-r04 git@github.com:SHIHAIFENG15/torchtitan.git
cd torchtitan && python3 -m pip install -e . && cd ..
git clone https://gitcode.com/Ascend/torchtitan-ascend.git
cd torchtitan-ascend && git checkout b508510a && python3 -m pip install -e . && cd ..
```

- torchtitan 必须包含 `1628fc9`。关键修复清单：RoPE cache 全程 host
  构建（`e284eff1`，跨设备逐位一致的前提）、NPU bilinear grid_sample
  纯 torch 绕过（`1628fc9`，防 backward 非有限）、音频 Newton rsqrt
  （`fd76aee9`）、共享 fp32 q/k norm（`68279566`）等。
- A5 旧工作区（`019fc43` + 7 个未提交补丁）已逐文件 sha1 核对，
  与本分支完全等价，无需保留。
- `torchtitan_ascend` 提供训练入口 `torchtitan_ascend.train`（run_train.sh
  的 `ENTRYPOINT`）。A5 上锁定在 gitcode `b508510a`，工作区干净。

## 4. 环境变量

```bash
export HCCL_DETERMINISTIC=true
export MUSE_DEBUG_DETERMINISTIC=1
export ENTRYPOINT=torchtitan_ascend.train
```

不要设置 `MM_DTYPE_REDUCED_PRECISION`、`ASCEND_RT_VISIBLE_DEVICES`。
一次只跑一个 torchrun。

## 5. 数据资产

已入库（clone 即得，勿重跑 step-0）：

- `outputs/checkpoint-*`（6 份 step-0，含 433M 的 ultradata 分片）
- `docs/gpu_baseline/*.txt`（GPU 各轮 1000 步 loss，含 `*_host_rope.txt`）
- `docs/npu_baseline/*.txt`（NPU 1000 步 loss 与 MAE 记录）
- `outputs/step1_*_gpu/`、`outputs/step1_audio_*/`、`outputs/step1_grads/`、
  `outputs/structured_logs/`（已入库部分）

仅存在于 A5、需要回收前备份的（约 60M + 今日新产物）：

- `outputs/step1_cpu_cache/`（11M，本轮 NPU step-1 对齐 dump 的 NPU 侧）
- `outputs/step1_raw/`、`outputs/step1_raw2/`、`outputs/step1_qk/`、
  `outputs/step1_pairwise/`、`outputs/step1_pairwise_wo/`
- `outputs/tb_acc_*/`（NPU TensorBoard，loss 导出的源数据）
- 今日 host_rope 轮新产物：`tb_acc_mm_fsdp2_host_rope/`、
  `tb_acc_mm_host_rope/` 及导出的 txt（跑完后提交到
  `docs/npu_baseline/`）。

## 6. 复现 1000 步验收

```bash
cd /tmp/a5-muse/torchtitan
ACC_NGPU=8 \
ACC_MODULE=tests.integration_tests.models \
ACC_CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
ACC_LOAD=/tmp/a5-muse/torchtitan/outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0 \
ACC_STEPS=1000 ACC_TB=tb_acc_mm_fsdp2 \
python3 run_acc.py > logs/metrics_acc_mm_fsdp2.log 2>&1

ACC_NGPU=8 \
ACC_MODULE=torchtitan_recipes.tests.models.muse_spark \
ACC_CONFIG=muse_spark_debugmodel_mm \
ACC_LOAD=/tmp/a5-muse/torchtitan/outputs/checkpoint-debugmodel_mm/step-0 \
ACC_STEPS=1000 ACC_TB=tb_acc_mm \
python3 run_acc.py > logs/metrics_acc_mm.log 2>&1
```

loss 导出、指标口径与门槛讨论见 `docs/mm_acc_1000_host_rope.md` 和
`docs/mm_acc_1000_rsqrt.md`（TensorBoard 三列导出脚本在里面）。
MAE = 逐点 |GPU - NPU| 的均值；MRE = MAE / 参考曲线均值。
注意：全程逐点 MAE<=0.02 的门槛已证明被噪声卡死（同设备 1 ulp 改动
也会产生 ~0.03），验收判据建议改为收敛段 MAE + 收敛步一致性 +
step-1 逐位诊断，详见 host_rope 轮的分析。

## 7. 复现 step-1 逐位诊断（可选）

```bash
# 与 docs/mm_step1_pairwise.md 相同的 fp32 attention dump
# NPU 侧产物在 outputs/step1_cpu_cache/mm_fsdp2_tp2_cp2_ep4/rank*.pt
# GPU 侧基线在 outputs/step1_cpu_cache_gpu/（39308c19 起）
# 对齐预期：raw_q/raw_k/value/rope_cache/query/key 全部逐位相同；
# 剩余 expert 翻转（~108/16384）为 bf16 注意力核噪声打在近平局上。
```

## 8. 重建后自检清单

```bash
npu-smi info                                   # 8 卡可见
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python3 -c "import torchtitan"                 # 应打印 backports [199884, 200238, 200199]
grep -n "ambient = torch.empty(0).device" torchtitan/models/common/rope.py
grep -n "0x5F3759DF" torchtitan/models/muse_spark/audio_encoder.py
grep -n "_bilinear_grid_sample" torchtitan/models/muse_spark/vision_encoder.py
test -d outputs/checkpoint-mm_fsdp2_tp2_cp2_ep4/step-0
test -d outputs/checkpoint-debugmodel_mm/step-0
# 冒烟：跑 1 步，mm_fsdp2 第 1 步 loss 约 7.79217、grad_norm 约 9.10471
```

GPU 侧环境另有差异（CUDA 8xA100、torch 2.14.0a0、`usercustomize.py`
里的 grid_sampler_2d_backward CPU 垫片），见 `docs/gpu_baseline_setup.md`，
不在本文范围。
