# A5 跑通操作文档（给另一个 agent）

在 **A5** NPU 上验证 Muse Spark 清单 **第 1–5 行共 8 个配置能跑完**。  
**不要**跑精度 1000 步、性能线性度、内存对照；那些只在 A2/GPU 闭环。

| 代号 | 角色 |
| --- | --- |
| **A2** | 已跑通并完成指标验收的参考环境（`179.173.2.3` / `npu-y00839695-torchtitan-20261028`） |
| **A5** | 本文件目标：只验证「训练循环能走完」 |

踩坑对照与补丁说明：`docs/npu_pta_fixes.md`；flex 补丁：`docs/npu_910b_patches/`。

---

## 0. 硬规则

1. 只在 A5 NPU 跑；分支 `npu-muse-spark-debug-r04`（拉最新 tip）。
2. **必须** `ENTRYPOINT=torchtitan_ascend.train`。
3. 成功标准：日志有 `Training completed` 且 `Process group destroyed`，退出码 0。确定性连跑两次、MAE、tps 线性度 **一律不做**。
4. 一次一个 torchrun；8 卡任务前确认卡空闲。
5. 日志写入 `logs/*_a5.log`，不要覆盖 A2 的 `*_npu.log`。
6. 不要改核心代码除非用户授权；失败先查 `docs/npu_pta_fixes.md` 文首对照表。

---

## 1. A5 机器信息（开跑前填）

| 项 | A5 填写 |
| --- | --- |
| 主机 / 容器 | |
| 仓库路径 | |
| 卡（期望 8×910B4） | |
| `torch` / `torch_npu` | 目标 `2.15.0+cpu` / `2.15.0.dev20261005+git14dc54e` |
| ascend 可 import | |
| flex 补丁 MD5 已核对 | |
| UltraData JSONL（第 5 行） | |

---

## 2. 环境

```bash
grep -q "$(hostname)" /etc/hosts || echo "127.0.0.1 $(hostname)" >> /etc/hosts
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /path/to/.venv/bin/activate   # 按 A5 实际改

cd /path/to/torchtitan
git fetch shihaifeng npu-muse-spark-debug-r04
git checkout npu-muse-spark-debug-r04
git pull --ff-only shihaifeng npu-muse-spark-debug-r04
git log -1 --oneline

bash docs/npu_910b_patches/apply_npu_910b_patches.sh
source docs/npu_910b_patches/env_npu_910b.sh
# 期望 MD5：
# flex_attention.py  659bf1dbc6c9652d8a5fdf9dad9bedeb
# triton_heuristics.py  e0fcd810ba31c602be44fb7c5e49b058

export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHNPU_PRECOMPILE_THREADS=1
export HCCL_DETERMINISTIC=true
export MUSE_DEBUG_DETERMINISTIC=1
export ENTRYPOINT=torchtitan_ascend.train
mkdir -p logs
```

数据：仓库内 `tests/assets/` 即可。第 5 行需要本地 UltraData JSONL（可用 `MUSE_SPARK_ULTRADATA_JSONL_PATTERN`）；不要依赖 huggingface 出网。

---

## 3. 八个配置（5 行）

| # | 清单行 | CONFIG | NGPU | MODULE |
| --- | --- | --- | --- | --- |
| 1 | 1 | `muse_spark_debugmodel` | 4 | `torchtitan_recipes.tests.models.muse_spark` |
| 2 | 2① | `muse_spark_debugmodel_with_cp` | 4 | 同上 |
| 3 | 2② | `muse_spark_debugmodel_fsdp2_cp2_ep4` | 4 | `tests.integration_tests.models` |
| 4 | 2③ | `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` | 8 | `tests.integration_tests.models` |
| 5 | 3 | `muse_spark_debugmodel_dist_moe_bf16` | 4 | `torchtitan_recipes.tests.models.muse_spark` |
| 6 | 3 | `muse_spark_debugmodel_dist_moe_mxfp8` | 4 | 同上 |
| 7 | 4 | `muse_spark_debugmodel_mm` | 8 | 同上 |
| 8 | 5 | `muse_spark_debugmodel_ultradata_lora_sft` | 8 | 同上 |

**关于 #6 mxfp8**：A2 上未适配（配置期要 SM100/`dist_moe` CuTe / torchao），**预期跑不通**。A5 仍按清单点一次：若同样在配置期失败，记「预期跳过/失败」+ 首段报错即可，**不要**为跑通改配方。其余 7 个必须 `Training completed`。

---

## 4. 命令（每个配置跑 1 次即可）

```bash
SPARK=torchtitan_recipes.tests.models.muse_spark
IT=tests.integration_tests.models

# 1
NGPU=4 MODULE=$SPARK CONFIG=muse_spark_debugmodel \
  ./run_train.sh | tee logs/run_debugmodel_a5.log

# 2
NGPU=4 MODULE=$SPARK CONFIG=muse_spark_debugmodel_with_cp \
  ./run_train.sh | tee logs/run_with_cp_a5.log

# 3
NGPU=4 MODULE=$IT CONFIG=muse_spark_debugmodel_fsdp2_cp2_ep4 \
  ./run_train.sh | tee logs/run_fsdp2_a5.log

# 4
NGPU=8 MODULE=$IT CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 \
  ./run_train.sh | tee logs/run_mm_fsdp2_a5.log

# 5
NGPU=4 MODULE=$SPARK CONFIG=muse_spark_debugmodel_dist_moe_bf16 \
  ./run_train.sh | tee logs/run_dist_moe_bf16_a5.log

# 6 — 预期可能配置期失败，仍执行并留日志
NGPU=4 MODULE=$SPARK CONFIG=muse_spark_debugmodel_dist_moe_mxfp8 \
  ./run_train.sh | tee logs/run_dist_moe_mxfp8_a5.log || true

# 7
NGPU=8 MODULE=$SPARK CONFIG=muse_spark_debugmodel_mm \
  ./run_train.sh | tee logs/run_debugmodel_mm_a5.log

# 8
NGPU=8 MODULE=$SPARK CONFIG=muse_spark_debugmodel_ultradata_lora_sft \
  ./run_train.sh | tee logs/run_ultradata_a5.log
```

多模态（#4/#7）前几步 tps 掉到个位数是 flex 重编译，属正常；等编完即可。不要为此改 timeout 以外的逻辑（配方里已有 `train_timeout_seconds=1800`）。

---

## 5. 交付

在回复或 `docs/npu_a5_run_results.md` 中给出：

1. A5 主机/容器、`git log -1`、补丁 MD5、`ENTRYPOINT`
2. 下表：

| # | CONFIG | 结果 | 日志 |
| --- | --- | --- | --- |
| 1–8 | … | 通过 / 失败 / 预期跳过 | `logs/…` |

3. 失败项：贴关键报错前 30 行，并注明是否已对照 `docs/npu_pta_fixes.md`

**不要**提交精度曲线、不要改 A2 验收总表，除非用户另要求。

---

## 6. 最小冒烟

整表来不及时先交 #1：

```bash
NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark \
  CONFIG=muse_spark_debugmodel \
  ./run_train.sh | tee logs/run_debugmodel_a5_smoke.log
```

通过后再按第 4 节顺序跑完其余配置。
