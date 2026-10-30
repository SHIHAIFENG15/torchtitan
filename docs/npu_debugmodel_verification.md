# Muse Spark debugmodel 验收

对照 `docs/muse模型适配专项/muse模型适配工作项清单.xlsx` 的 **Muse Spark** 表，按第 1 到第 5 行写。Glimmer 不跑。每一行把配置、验证要点的分析、跑通数据、这一行的验收指标放在一起。

机器：`179.173.2.3`（容器 `npu-y00839695-torchtitan-20261028`）为主；8 卡多模态也曾在 `179.173.2.2` 对齐复验。代码分支当前为 `npu-muse-spark-debug-r04`（基于 `pytorch/torchtitan` 的 `release/0.4` + NPU 提交）。site-packages 三处 PTA 补丁仍不在 git 里，见 `docs/npu_pta_fixes.md`。

日志指标含义（各行共用，按代码而不是按名字猜）：

| 字段 | 含义 |
| --- | --- |
| loss | 全部有效 token 上的全局平均交叉熵 |
| grad_norm | `optim_step()` 的返回值，梯度裁剪之后的范数 |
| memory | 这一步窗口里的最大预留显存。第 1 步把编译也算进去，所以往往比后面高。容量按日志是 Ascend910B4 29.49GiB |
| tps | 单卡 token/s：`每步每 DP rank 的配置 token 数 / (这一步耗时 × CP × TP × PP)`。debugmodel 的配置 token 数是 512。它不是墙钟吞吐，也不含编译 |
| tflops / mfu | 由每 token FLOPs 乘 tps 得到。MFU 分母回退为 A100 峰值 `3.120e14`（日志：`Peak flops undefined for: Ascend910B4`）。debugmodel 太小，mfu 打出来是 0.00%，不能当 910B 利用率 |

没有同场景 GPU 基线时，不编造对比数。验收标准本身是四项：精度、性能、内存、确定性。下面写在对应行里，标准全文不另起一章重复。

GPU 机器上的环境、四项缺数和取数步骤写在 `docs/gpu_baseline_setup.md`。NPU 上的表不要改成 GPU 的数。

---

## 2026-10-29 切到 release/0.4 之后的重跑

`pytorch/torchtitan` 的 `release/0.4` 顶端仍是 v0.4.0（`45e33cd4`，2026-10-28）。本地分支 `npu-muse-spark-debug-r04` 就在这个点上，另有 6 个 NPU 提交，HEAD `d0f5101f`。旧验收跑在 `npu-muse-spark-debug`（`0861159b`，基线 `688704a6`）上。两棵树的 Muse Spark 模型不同：v0.4.0 默认打开 RoPE、RMSNorm、attention 编译区，`torch_remat` 为 0.2.2，flex attention 为 `fullgraph=True`。

直接在新默认上跑 4 卡 `with_cp`（`with_cp_4card_r2.log`）会在第 1 步反向失败：`torch.utils.checkpoint.CheckpointError`，重算张量从 position 30 起和正向保存的错位，NPU 上的 `[245, 1]` float32 重算成 CPU 标量。debug 配方因此把 `local_compile_regions` 收回到旧的 `loss`、`fp32_to_bf16_split`（`torchtitan_recipes/tests/models/muse_spark.py`，尚未提交）。`debugmodel_mm` 会换掉 model config，同样收回。确定性重跑另加 `HCCL_DETERMINISTIC=true`、`MUSE_DEBUG_DETERMINISTIC=1`（`debug.seed=42`，`debug.deterministic=True`）。卡是 0、2、3、6。1 号卡从约 15:12 CST 起被 `pytest test_nested_reduction.py` 占着，没有动它。

### 四个配置的确定性

判定只看 loss 和 grad_norm：同一配置、同一环境（`HCCL_DETERMINISTIC=true`，`MUSE_DEBUG_DETERMINISTIC=1`，seed 42）连续跑两次，10 步打印值逐项相同。tps 和第一步 memory 允许不同。四组都对上了。

| 配置 | 网格 | 日志 | 结果 |
| --- | --- | --- | --- |
| 第 1 行 `muse_spark_debugmodel` | 4 卡 `dp_shard=2, tp=2, ep=2` | `det_debugmodel_r04_r1/r2.log` | 10 步 loss、grad_norm 相同，并与旧日志 `item2_logs/det_debugmodel_r1.log` 相同 |
| 第 2 行① `muse_spark_debugmodel_with_cp` | 4 卡 `dp_shard=1, cp=2, tp=2, ep=2` | `det_with_cp_r04_r1/r2.log` | 10 步 loss、grad_norm 相同。旧分支没有跑完，没有旧数 |
| 第 2 行② `muse_spark_debugmodel_fsdp2_cp2_ep4` | 4 卡 `dp_shard=2, cp=2, tp=1, ep=4` | `det_fsdp2_r04_r1/r2.log` | 10 步 loss、grad_norm 相同，并与旧日志 `item2_logs/det_fsdp2_r1.log` 相同 |
| 第 2 行③ `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` | 8 卡 `dp_shard=2, cp=2, tp=2, ep=4` | `det_mm_r04_r1/r2.log` | 10 步 loss、grad_norm 相同。旧多模态日志没有设 seed，不逐项比 |


### 第 1 行

`muse_spark_debugmodel`，网格 `dp_shard=2, tp=2, ep=2`。`det_debugmodel_r04_r1/r2.log`，两次 loss、grad_norm 逐项相同，并且和旧日志 `item2_logs/det_debugmodel_r1.log` 的 10 步完全相同。

| step | loss | grad_norm | r1 tps | r2 tps |
| --- | --- | --- | --- | --- |
| 1 | 7.68353 | 4.7419 | 18 | 18 |
| 2 | 6.88134 | 3.2349 | 315 | 260 |
| 3 | 5.98353 | 2.3663 | 327 | 266 |
| 4 | 5.61493 | 1.9338 | 327 | 246 |
| 5 | 6.05757 | 1.7812 | 323 | 290 |
| 6 | 5.39870 | 1.6968 | 326 | 282 |
| 7 | 5.23369 | 1.6693 | 316 | 287 |
| 8 | 5.46803 | 1.5549 | 304 | 270 |
| 9 | 5.07188 | 1.7912 | 307 | 283 |
| 10 | 4.98734 | 1.6464 | 318 | 288 |

memory 第 1 步 0.09GiB，其后 0.11GiB。tps 两次不一致，和以前一样，只说明计时不同。

### 第 2 行 ① with_cp

网格 `dp_shard=1, cp=2, tp=2, ep=2`，4 卡。先有一趟没开确定性的 `with_cp_4card_r3.log`（17:33 CST，`Training completed`）：loss 7.71652，6.99130，6.11190，5.75051，5.82153，5.51257，5.34266，6.02426，5.30019，5.21196；grad_norm 4.9245 到 1.8415。这趟只证明反向能走完。

正式对照是 `det_with_cp_r04_r1/r2.log`。两次 loss、grad_norm 逐项相同。

| step | loss | grad_norm |
| --- | --- | --- |
| 1 | 7.72229 | 5.1450 |
| 2 | 6.93197 | 3.9766 |
| 3 | 6.03953 | 2.8388 |
| 4 | 5.71570 | 2.2161 |
| 5 | 5.73440 | 2.1739 |
| 6 | 5.44686 | 1.8826 |
| 7 | 5.28071 | 1.9517 |
| 8 | 5.90602 | 2.0066 |
| 9 | 5.17893 | 2.0813 |
| 10 | 5.12831 | 1.8717 |

稳定步 tps 大约 130–145。r1 第 1 步 memory 0.24GiB（编译），r2 第 1 步 0.08GiB，其后都是 0.10GiB。旧分支上这组没有跑完，没有旧数可对。

### 第 2 行 ② FSDP2×CP2×EP4

`tests.integration_tests.models` 的 `muse_spark_debugmodel_fsdp2_cp2_ep4`，网格 `dp_shard=2, cp=2, ep=4`。`det_fsdp2_r04_r1/r2.log` 两次逐项相同，并且和旧日志 `item2_logs/det_fsdp2_r1.log` 的 10 步 loss、grad_norm 完全相同：7.73262，6.89378，5.99186，5.64755，6.07865，5.43435，5.27176，5.50165，5.10669，5.01443；grad_norm 4.6582 到 1.6591。memory 稳定在 0.12GiB。稳定步 tps 大约 112–124。

### 第 2 行 ③ 8 卡多模态

`muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`，网格 `dp_shard=2, cp=2, tp=2, ep=4`，8 卡。1 号卡上的 `pytest test_nested_reduction.py`（pid 203951）没有停，8 卡任务和它叠在同一张卡上跑。pytest 在训练结束之后仍在。

`det_mm_r04_r1.log`：17:50 CST 进入第 1 步，17:55 `Training completed`。`det_mm_r04_r2.log`：18:13 CST 进入第 1 步，18:14 `Training completed`。两次都开了 `HCCL_DETERMINISTIC=true` 和 `MUSE_DEBUG_DETERMINISTIC=1`。10 步 loss、grad_norm 逐项相同，下表两趟共用。r2 的 memory 更低（第 1 步 0.07GiB，其后 0.12GiB），因为编译缓存已经热了；tps 也不相同。旧的多模态日志没有设 seed，不和这次逐项比。

| step | loss | grad_norm | memory | tps |
| --- | --- | --- | --- | --- |
| 1 | 7.67449 | 10.1172 | 0.23GiB | 1 |
| 2 | 6.84383 | 6.4551 | 0.26GiB | 2 |
| 3 | 5.69314 | 4.1008 | 0.26GiB | 32 |
| 4 | 6.02861 | 5.0077 | 0.26GiB | 36 |
| 5 | 4.36879 | 5.0873 | 0.26GiB | 5 |
| 6 | 5.64743 | 3.7888 | 0.26GiB | 36 |
| 7 | 4.94917 | 2.9002 | 0.26GiB | 37 |
| 8 | 3.33959 | 3.2298 | 0.26GiB | 5 |
| 9 | 3.95203 | 2.8241 | 0.26GiB | 37 |
| 10 | 4.55447 | 2.8045 | 0.26GiB | 38 |

第 1、2、5、8 步 tps 掉到个位数，对应 flex attention 编译。第 6 步前有一条 `Sample length 633 > training max_context_length=512`，该样本被跳过。r1 稳定步 memory 0.26GiB；r2 稳定步 0.12GiB。loss 和 grad_norm 不受这个差别影响。

---

## 第 1 行：基础 debugmodel 文本训练

| 项 | 内容 |
| --- | --- |
| 工作项 | 跑通文本 next-token 训练，验证数据加载、损失计算、反向传播与参数更新 |
| 配方 | `muse_spark_debugmodel`（`torchtitan_recipes/tests/models/muse_spark.py`） |
| 清单并行 | 单卡；dim 256 / 8 层 / 8 experts；seq_len 512；10 步 |
| 实际网格 | 跟配方默认，不是单卡：`pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2`，4 卡。世界大小 `1 × 2 × 1 × 2 × 1 = 4`。EP 从这 4 个进程里分专家，不另加进程。序列并行跟着张量并行，也不另加进程 |
| 数据 / 优化 | `tests/assets/c4_test/data.json`，`shuffle=False`；AdamW lr=`8e-4`，warmup 2；词表 2048；`checkpointer=None` |
| 结果 | 训练循环跑通。正式对照用确定性两次：`det_debugmodel_r1/r2.log`（`HCCL_DETERMINISTIC=true`，`seed=42`，`debug.deterministic=True`，`deterministic_warn_only` 保持默认 False） |

配置核对：

| 清单 | 这次 | 依据 |
| --- | --- | --- |
| dim 256 | 满足 | `_DEBUG_MODEL_DIM = 256` |
| 8 层 | 满足 | `_debugmodel` 里 `num_layers = 8` |
| 8 experts | 满足 | `num_experts=8`，每个 token 再选 top-2 |
| seq_len 512 | 满足 | debugmodel 默认训练长度 512，microbatch 也是 512 |
| 10 步 | 满足 | `training.steps=10`，日志到 `step: 10` 后 `Training completed` |
| 卡数 | 按配方是 4 卡，不是清单写的单卡 | 默认网格 `dp_shard=2 × tp=2` |

2026-10-27 那次用 0、1、3、4 号 910B4，`NGPU=4`，10 步打完，退出码 0。rank 0 和 rank 1 每一步的 loss、grad_norm 相同，这是 loss 网格归约之后的全局值。那一趟没有开确定性，日志已经不在磁盘上，数记在下面第一张表。

### 验证要点

#### 数据加载

路径是 `GrainDataLoader` + `ConcatThenSplitPackingConfig` + `DATASETS["c4_test"]`。日志里 tokenizer 从 `tests/assets/tokenizer/tokenizer.json` 读入，datasets 库去拉 json 加载器。10 步都打出了有限 loss，说明每步都拿到了 batch，而且标签里有参与损失的 token。标签若全是 `ignore_index=-100`，交叉熵不会停在 7 附近。

4 卡时样本按 FSDP 的 2 个数据分片切开，这是这套网格的正常切法。两边 loss 经过归约后一致，说明两个数据分片都进了同一次全局损失。初始化日志是每 DP rank 512 token、每步 1024 token、warmup 2 步。

2026-10-30 用同一套加载器把 10 步的 token 和 `data.json` 对过。文件里有 2000 条，`shuffle=False`，两个 DP rank 各 1000 条。每条都是 `encode(text, add_bos=True, add_eos=True)` 之后 `input_ids=tokens[:-1]`、`labels=tokens[1:]`，2000 条的 token id 没有一条对不上。

10 步、每个 rank 512 token、2 个 rank，一共 10240 个位置。这些位置的 `(input_ids, labels)` 按文档顺序接上原文的错位对，错 0 个。这 10 步里没有 `-100`。打包把多篇文档拼进同一行，两个 rank 一共 44 处是文档切段的末位：那里的标签是原文的下一个 token，不是拼进去的下一篇的开头。这套测试 tokenizer 的 `decode` 不把空格打回来，对照用的是 token id。

结论：加载器在 4 卡上每步都喂进了带有效标签的 batch。这 10 步的 10240 个位置和原文错位对全部相同。

#### 损失计算

损失是 `ChunkedLossWrapper`，8 段，里面是词表并行交叉熵。TP=2 时每个 rank 只有词表的一半，前向用 all-reduce 拼 log-sum-exp。日志里的 loss 是按有效 token 平均后再做跨 rank 归约。若是 512 个 token 的和，数量级会到几千。

确定性那次第 1 步 7.68353。随机初始化时，2048 类上的均匀猜测是 `ln(2048) = 7.6246`，差 0.059。第 1 步的预测接近均匀，损失尺度像平均交叉熵。后面降到 4.98734，第 5 步和第 8 步又升过。`shuffle=False`，每步句子不同，所以下降本身不能单独证明公式算对了。

2026-10-30 在 CPU 上用 `F.cross_entropy` 对过损失公式。输入是合成的，词表 2048，和 debugmodel 一样。8 段 `ChunkedLossWrapper`（序列 512，每 17 个标签里有一个 `-100`）的标量损失和整段 `F.cross_entropy` 的差是 0，`lm_head.weight` 的梯度最大绝对差是 `9.3e-9`。TP=2 的词表并行交叉熵（序列 64，含 `-100` 和词表最后一类）两个 rank 的损失差都是 0，各自词表分片上的 logit 梯度最大绝对差是 `8.4e-9` 和 `6.1e-9`。这是公式和最后一层线性的对拍。训练那一步的 NPU logits 没有再拿出来重算，RoPE、SwiGLU、flex attention 也没有和 CUDA 核对齐。

结论：8 段损失和词表并行损失都等于同一批 logits 上的 `F.cross_entropy`。这不等于整网和 GPU 对齐。

#### 反向传播

`grad_norm` 10 步都是有限值。确定性那次从 4.7419 落到大约 1.6，没有 NaN，也没有变成 0。`trainer.train_step` 里这个数是 `engine.optim_step()` 的返回值，所以反向图产生了优化器能用的梯度。rank 0 和 rank 1 一致，说明 FSDP/TP 的梯度归约没有让两边各算各的。

`lm_head` 和词表并行交叉熵的梯度已经和 `F.cross_entropy` 对过，最大绝对差在 `1e-8` 这一档，数在损失那一节。910B4 上 flex attention 走 mask-in，反向分块是 16；更大的分块会在真实反向里段错误，所以没有和未改分块的结果比。FSDP 还警告过 `FSDPMuseSparkTransformerBlock` 返回了 view。若后面有就地修改，pre-backward 的 all-gather 会被跳过。这次没有顺着这条警告查出是哪一次算子。

结论：4 卡反向在跑，梯度有限且两边一致。`lm_head` 这一层的梯度对上了。flex attention 整层还没有和未改分块的实现对拍。

#### 参数更新

`optim_step()` 在打日志之前被调用。学习率 `8e-4`，warmup 2 步，226 个参数组，梯度范数不是 0，AdamW 会改权重。10 步都打出来了，说明更新函数返回了。

2026-10-30 用和确定性验收相同的环境又跑了 10 步，loss 仍是 7.68353 到 4.98734。每步 `optim_step()` 前后把参数拷到 CPU 做差，再在 4 个 rank 上取最大绝对差。学习率 warmup 2 步：第 1 步 `4e-4`，第 2 步 `8e-4`，之后线性降到第 10 步 `1e-4`。

第 1 步最大绝对差 `5.77e-4`，落在 `tok_embeddings.embedding.weight`，rank 0 本地元素的平均绝对差是 `3.66e-4`。10 步累计最大绝对差 `5.46e-3`，平均 `1.87e-3`，仍是这张 embedding。rank 0 本地 2,757,024 个元素里，2,636,928 个离开了初始值，120,096 个十步后仍相同。没有另存 checkpoint 文件。loss 中间的回升仍然掺着换句子，不能单独当成参数学对了。

结论：参数每步都在改。10 步后离初始值最远的是 embedding，最大绝对差 `5.46e-3`。

| 验证要点 | 4 卡这次能说的 | 还缺的 |
| --- | --- | --- |
| 形状和步数 | dim 256、8 层、8 专家、序列 512、10 步都对上 | |
| 网格 | `dp_shard=2 × tp=2`，EP=2 复用这 4 个进程。清单写单卡，实际跟配方 | |
| 数据加载 | 10240 个位置和原文错位对全部相同，这 10 步没有 `-100` | `decode` 不保留空格，对照用的是 token id |
| 损失计算 | 8 段损失和词表并行损失与 `F.cross_entropy` 差为 0 | 不是训练步 logits 回放，也不是 GPU 模型 |
| 反向传播 | `lm_head` 和词表并行 logit 梯度对上了 | flex attention 分块 16 没有和未改分块比；FSDP view 警告没往下追 |
| 参数更新 | 10 步累计最大绝对差 `5.46e-3`，在 embedding | 没有另存 checkpoint 文件 |

### 跑通数据

早先一趟没有开确定性（日志已不在磁盘上）。和下面确定性两次不是同一次随机种子。

| step | loss | grad_norm |
| --- | --- | --- |
| 1 | 7.68154 | 4.7997 |
| 2 | 6.79467 | 3.2132 |
| 3 | 5.90877 | 2.2807 |
| 4 | 5.55777 | 1.8885 |
| 5 | 6.04875 | 1.7536 |
| 6 | 5.41199 | 1.6882 |
| 7 | 5.23373 | 1.6302 |
| 8 | 5.41660 | 1.4989 |
| 9 | 5.01979 | 1.7646 |
| 10 | 4.99761 | 1.6373 |

确定性两次，loss / grad_norm 在日志打印的位数上相同。模型参数初始化占用 0.01GiB。

| step | loss | grad_norm | r1 memory | r1 tps | r2 memory | r2 tps |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.68353 | 4.7419 | 0.25GiB (0.84%) | 1 | 0.09GiB (0.31%) | 4 |
| 2 | 6.88134 | 3.2349 | 0.11GiB (0.38%) | 277 | 0.11GiB (0.38%) | 170 |
| 3 | 5.98353 | 2.3663 | 0.11GiB (0.38%) | 285 | 0.11GiB (0.38%) | 177 |
| 4 | 5.61493 | 1.9338 | 0.11GiB (0.38%) | 307 | 0.11GiB (0.38%) | 215 |
| 5 | 6.05757 | 1.7812 | 0.11GiB (0.38%) | 316 | 0.11GiB (0.38%) | 177 |
| 6 | 5.39870 | 1.6968 | 0.11GiB (0.38%) | 316 | 0.11GiB (0.38%) | 138 |
| 7 | 5.23369 | 1.6693 | 0.11GiB (0.38%) | 320 | 0.11GiB (0.38%) | 136 |
| 8 | 5.46803 | 1.5549 | 0.11GiB (0.38%) | 318 | 0.11GiB (0.38%) | 164 |
| 9 | 5.07188 | 1.7912 | 0.11GiB (0.38%) | 321 | 0.11GiB (0.38%) | 177 |
| 10 | 4.98734 | 1.6464 | 0.11GiB (0.38%) | 312 | 0.11GiB (0.38%) | 219 |

第 2 到第 10 步 r1 的 tps 平均 308，对应墙钟约 `512 / (308 × 2) ≈ 0.83` 秒。r2 平均 175。loss 相同，tps 差一截，所以 tps 不能当确定性指标。tflops 是 0.01 或 0.00，mfu 是 0.00%。

### 本行验收指标

精度。标准：同一份 step 0 权重，NPU 和 GPU 各预训练不少于 1000 步。MoE 相对基线的平均绝对误差不超过 0.02，或者平均相对误差不超过 2%。这台机器上没有这份权重，也没有 GPU 的 1000 步 loss，MAE 算不出来。能核对的只有 NPU 自己的尺度：第 1 步 7.68353，与 `ln(2048)` 差 0.059；第 10 步 4.98734；grad_norm 全程有限。这不等于和 GPU 对齐。

性能。标准：丢掉前 100 步，用稳定段算 8 卡 tps / 4 卡 tps，再和 GPU 的同一比值比，线性度差距不超过 2 个百分点。本行只有 10 步，也没有同一配方的 8 卡对照。稳定段 r1 单卡 tps 约 308。r2 同 loss 只有 175，墙钟不稳。

内存。标准：先比单卡和同场景 GPU 的显存，再开 FSDP2 比多卡。日志里的 memory 是最大预留，不是整卡占用。没有 GPU 数，也没有按清单改成单卡再跑。第 1 步 r1 0.25GiB、r2 0.09GiB（含编译）；第 2 步之后 0.11GiB，占日志容量 0.38%。参数本身 0.01GiB。

确定性。标准：`HCCL_DETERMINISTIC=true`，加上 `debug.deterministic`，同一配置连跑两次，精度完全一致。本行 10 步 loss、grad_norm 打印值相同。两次的 tps 和第 1 步 memory 不一致，那是计时和编译期分配。这还不是验收全文：验收还要和 GPU 的同策略结果一致。GPU 那一侧这次没有。

---

## 第 2 行：DP / TP / EP / SP / CP

| 项 | 内容 |
| --- | --- |
| 工作项 | 适配数据并行、张量并行、专家并行、序列并行与上下文并行。验证压缩 MoE 参数分片与 pooler/unpooler 重分布、多机专家计算与不均匀 token、序列切分布局一致性、梯度累积期间的延迟归约，以及长序列下滑窗层与全局 NoPE 层的混合语义 |
| 配方 | ① `muse_spark_debugmodel_with_cp` ② `muse_spark_debugmodel_fsdp2_cp2_ep4` ③ `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` |
| 清单并行 | ① 4 卡 TP2×EP2×CP2 ② 4 卡 FSDP2×CP2×EP4 ③ 8 卡 FSDP2×TP2×CP2×EP4+SP |
| 备注 | TP/EP/SP 三者绑定启用；CP 可独立或叠加；TP 需整除 KV heads（debug 为 4）；Ulysses 需 `n_heads` / `n_kv_heads` 整除 TP×CP |

三条配方覆盖同一列验证要点，程度不同：

| 要点 | ① with_cp | ② fsdp2_cp2_ep4 | ③ mm_fsdp2… |
| --- | --- | --- | --- |
| 压缩 MoE 参数分片 | 代码已改成清单上的 4 卡网格。旧 8 卡跑法没走到 10 步 | 跑完。`dp_shard=2, cp=2, tp=1, ep=4` | 跑完。`dp_shard=2, cp=2, tp=2, ep=4` |
| pooler/unpooler 重分布 | 旧 8 卡停在 `_StridedShard(dim=0) + Shard(dim=0)` 到 `Replicate` 的两次 all_gather | 10 步走完，没有单独打印 pooler 布局 | 同左 |
| 多机 / 不均匀 token | 只有这一台机器。文本长度固定 512 | 同样是单机、定长 | 单机 8 卡。图、音频、视频每步长度不同，第 5、8 步为新长度重新编译 |
| 序列切分布局一致性 | CP=2 的网格能建。旧 8 卡的切分 all_gather 没走完；无 TP 的对照跑完了，没有和另一份布局对过 | CP=2 的 10 步走完。没有和另一份布局对过 | 同左，外加视觉序列 |
| 梯度累积延迟归约 | 配置为 accumulation steps 1，延迟归约没有被碰到 | 同左 | 同左 |
| 滑窗 / 全局 NoPE | 无 TP 的对照跑完了 10 步，没有单独和 GPU 对语义。带 TP 的 4 卡重跑还没有 step loss | 两类层跟着 10 步跑了，没有单独和 GPU 对语义 | 同上，外加视觉 |

这三条的 tps 不能相除。网格、数据、有没有视觉编译都不一样，不能拿来算 8 卡 / 4 卡线性度。

### ① `muse_spark_debugmodel_with_cp`

| 项 | 内容 |
| --- | --- |
| 清单 | 4 卡：TP2 × EP2 × CP2；Ulysses CP |
| 代码原先 | 在基类 `dp_shard=2, tp=2, ep=2` 上再设 `cp=2`，世界大小 `2 × 2 × 2 = 8`，EP 不另加进程。和清单的 4 卡不一致 |
| 代码修正 | 已改为 `dp_shard=1, tp=2, ep=2, cp=2, SP=True`，世界大小 4。`train_timeout_seconds=1800`；`override.imports=[]`（去掉 CUDA Helion/SwiGLU） |
| CP 算法 | Ulysses（`_muse_spark_context_parallel_transform`） |

旧 8 卡跑法没有跑完。2026-10-28 07:14 UTC 用 8 张卡启动（`NGPU=8 ./run_train.sh`，日志 `item2_logs/with_cp.log`）。进程起来后停在 `_StridedShard(dim=0) + Shard(dim=0)` 到 `Replicate` 的两次 all_gather，日志从 07:22 UTC 起不再增长。8 个 rank 的 AICore 一直是 0，CPU 在 HCCL 里空转。4 号卡当时被别人的 `rn50_grad_probe.py` 占满（AICore 99%，`ASCEND_RT_VISIBLE_DEVICES=4`，停在 inductor autotune）。这次任务没有跑完 10 步，已在约 08:25 UTC 停掉。0 号卡上还有别人的 `pytest test_perf.py`。随后在 2.3 上又启动过一次（`with_cp_rerun.log`），网格同样是 8 卡，flex attention 反向编译时 rank 5 SIGSEGV，其余 rank 被 SIGTERM。

对齐环境之后，旧 8 卡网格能进第 1 步，但 loss 有限、grad_norm 是 nan（`with_cp_2.2_r4.log`：`loss≈1.94, grad_norm=nan`）。对照诊断：去掉 TP 的 Ulysses 4 卡（`dp_shard=2, cp=2, tp=1`，`with_cp_no_tp_r1.log`）可以跑完，梯度有限。所以卡住和 nan 出在 TP 与 Ulysses CP 叠在一起的那条路径，不是 CP 本身建不了网格。

按清单改成 4 卡之后：`with_cp_4card_r1.log` 在导入 `_MUSE_SPARK_HELION_ROPE_OVERRIDE` 时失败（`override.imports=[]` 之前的那次）。`with_cp_4card_r2.log` 已经打出 `dp_shard=1, cp=2, tp=2, ep=2` 并进入第 1 步，日志停在 inductor 编译，没有 step loss，也没有 `Training completed`。这一条还不能填 10 步表。

本配方的验收指标：精度、性能、内存、确定性都还没有 10 步结果。无 TP 的对照只说明 Ulysses 在 `tp=1` 时梯度有限，不能代替清单上的 TP2×EP2×CP2。

### ② `muse_spark_debugmodel_fsdp2_cp2_ep4`

| 项 | 内容 |
| --- | --- |
| 网格 | `dp_shard=2, cp=2, tp=1, ep=4`，4 卡；AllGatherKV CP |
| 结果 | 跑通。2026-10-28 先有一趟非确定性：2、3、4、5 号卡，10 步退出码 0，loss 7.76808→5.02892，grad_norm 4.6650→1.6260。后来按和第 1 行相同的确定性开关连跑两次：`det_fsdp2_r1/r2.log`，10 步 loss、grad_norm 打印值相同 |

| step | loss | grad_norm | r1 memory | r1 tps | r2 memory | r2 tps |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.73262 | 4.6582 | 0.26GiB (0.89%) | 1 | 0.11GiB (0.36%) | 14 |
| 2 | 6.89378 | 3.2025 | 0.12GiB (0.42%) | 122 | 0.12GiB (0.42%) | 117 |
| 3 | 5.99186 | 2.3524 | 0.12GiB (0.42%) | 121 | 0.12GiB (0.42%) | 120 |
| 4 | 5.64755 | 1.9698 | 0.12GiB (0.42%) | 121 | 0.12GiB (0.42%) | 116 |
| 5 | 6.07865 | 1.8299 | 0.12GiB (0.42%) | 123 | 0.12GiB (0.42%) | 121 |
| 6 | 5.43435 | 1.7440 | 0.12GiB (0.42%) | 123 | 0.12GiB (0.42%) | 121 |
| 7 | 5.27176 | 1.6869 | 0.12GiB (0.42%) | 124 | 0.12GiB (0.42%) | 120 |
| 8 | 5.50165 | 1.5714 | 0.12GiB (0.42%) | 123 | 0.12GiB (0.42%) | 119 |
| 9 | 5.10669 | 1.8020 | 0.12GiB (0.42%) | 123 | 0.12GiB (0.42%) | 120 |
| 10 | 5.01443 | 1.6591 | 0.12GiB (0.42%) | 125 | 0.12GiB (0.42%) | 120 |

第 2 到第 10 步 tps：r1 平均 123，r2 平均 119。非数据并行规模是 CP×TP×PP=2，`512 / (123 × 2)` 约 2.1 秒一步，和日志时间间隔一致。第 1 步 loss 7.73262，离 `ln(2048)` 差 0.108。tflops 0.00，mfu 0.00%。比第 1 行慢，是因为 CP=2、TP=1，和第 1 行的 TP=2、CP=1 不是同一网格。

精度：没有 GPU 的 1000 步基线。第 1 步接近均匀猜测，grad_norm 全程有限（4.6582 到 1.6591）。性能：稳定段单卡 tps 约 123，没有同配方 8 卡，线性度算不出来。内存：第 1 步 r1 0.26GiB、r2 0.11GiB；之后 0.12GiB，占日志容量 0.42%。确定性：两次 loss、grad_norm 一致；tps 和第 1 步 memory 不一致。和 GPU 同策略对齐这次没有。

### ③ `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`

| 项 | 内容 |
| --- | --- |
| 网格 | `pp=1, dp_replicate=1, dp_shard=2, cp=2, tp=2, ep=4`，8 卡。`2 × 2 × 2 = 8`，EP 从这 8 个进程里分专家，不另加进程。序列并行跟着 TP |
| CP | AllGatherKV |
| 模型 | 文本 dim 256、8 层、8 专家、词表 2048、序列 512、10 步。视觉是 NaViT，head_dim 16 |
| 数据 | `tests/assets/cc12m_test/cc12m-train-0000.tar`，加合成音频、视频。每 DP rank 512 token、每步 1024 token |
| 超时 | 启动用的 `init_timeout_seconds` 保持 300。第 1 步之后这条配方自己把 `train_timeout_seconds` 设为 1800。日志是 `Synchronizing and adjusting timeout for all ProcessGroups to 0:30:00` |
| 结果 | **2.3** 跑通：`item2_logs/mm_r12.log`，2026-10-28 11:02–11:07 UTC，`Training completed`，`Process group destroyed`，没有 `ChildFailedError`。**2.2** 对齐后复验：`mm_2.2_8card.log`，2026-10-29。两趟都没有开确定性种子 |

2.3（`mm_r12.log`）：

| step | loss | grad_norm | memory | tps | tflops | mfu |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.66202 | 9.4633 | 0.07GiB (0.24%) | 3 | 0.00 | 0.00% |
| 2 | 6.83399 | 6.2513 | 0.12GiB (0.42%) | 15 | 0.00 | 0.00% |
| 3 | 5.79272 | 4.1767 | 0.12GiB (0.42%) | 40 | 0.00 | 0.00% |
| 4 | 6.22080 | 5.1368 | 0.12GiB (0.42%) | 41 | 0.00 | 0.00% |
| 5 | 4.33259 | 4.8085 | 0.24GiB (0.81%) | 2 | 0.00 | 0.00% |
| 6 | 5.80127 | 3.5880 | 0.24GiB (0.81%) | 37 | 0.00 | 0.00% |
| 7 | 5.17229 | 2.9730 | 0.24GiB (0.81%) | 39 | 0.00 | 0.00% |
| 8 | 3.40500 | 2.9641 | 0.24GiB (0.81%) | 1 | 0.00 | 0.00% |
| 9 | 3.98423 | 2.8151 | 0.24GiB (0.81%) | 38 | 0.00 | 0.00% |
| 10 | 4.68848 | 3.0236 | 0.24GiB (0.81%) | 39 | 0.00 | 0.00% |

去掉第 1、5、8 步（编译把 tps 打到 3、2、1），剩下 6 步 tps 平均 39。非数据并行规模是 4，`512 / (39 × 4)` 约 3.3 秒，和这 6 步的日志间隔一致。第 5 步和第 8 步 tps 掉到个位数，是新的视觉 token 长度在重新编译 flex attention，不是算子挂住。第 1 步 7.66202，和 `ln(2048)` 差 0.037，尺度仍像平均交叉熵。后面有降有升，打包的图、音频、视频每步长度不同，不能单独当成和参考实现对上了。`checkpointer` 是空的，没有存权重，参数差没有直接量。10 步 grad_norm 都是有限值，没有变成 0。

2.2 对齐复验。2 号卡上当时有别人的 `runner_warmup.py`，约占 7GB，没有停它，8 卡任务仍然跑完。03:25–04:02 UTC，`Training completed`。loss 和 2.3 不是逐位相同，因为两边都没有设 `debug.seed`，也没有开 `debug.deterministic`。

| step | loss | grad_norm | memory | tps |
| --- | --- | --- | --- | --- |
| 1 | 7.85128 | 9.7895 | 0.23GiB (0.77%) | 0 |
| 2 | 6.88970 | 6.3927 | 0.26GiB (0.89%) | 0 |
| 3 | 5.88565 | 4.4378 | 0.26GiB (0.89%) | 38 |
| 4 | 6.15249 | 5.1606 | 0.26GiB (0.89%) | 39 |
| 5 | 4.45755 | 4.7667 | 0.26GiB (0.89%) | 1 |
| 6 | 5.80792 | 3.5132 | 0.26GiB (0.89%) | 34 |
| 7 | 5.12224 | 2.8415 | 0.26GiB (0.89%) | 39 |
| 8 | 3.47310 | 3.1467 | 0.26GiB (0.89%) | 1 |
| 9 | 4.01132 | 2.8866 | 0.26GiB (0.89%) | 39 |
| 10 | 4.63377 | 2.8766 | 0.26GiB (0.89%) | 38 |

第 1 步 loss 7.85128，离 `ln(2048)` 差 0.227。去掉第 1、2、5、8 步，稳定段 tps 平均 38，和 2.3 的 39 同一档。

这次能跑完，靠的是下面几处改动，细节在 `docs/npu_pta_fixes.md`：

- 视觉 RoPE 的 complex64 下标不走 `rope.py` 那一处，910B 的 index 没有 complex64。
- `torch_remat` 在 NPU 上不再用空指针 storage 做占位张量。空指针在反向重算里会段错误。
- 8 个 rank 一起给 flex attention 反向选块时，会有一个 rank 段错误。选块改成跨进程串行。
- 非 Ascend 950 不用 flex decoding。短视觉序列会另编一套核，编译时间超过原来的 100 秒训练超时，先跑完的 rank 把 all-gather 拖超时，HCCL watchdog 以 SIGABRT 退出。
- 这条配方的 `train_timeout_seconds` 设为 1800，盖住第 5 步、第 8 步那种新长度编译。
- 单帧 `torch.cat` 得到的是原张量。视觉编码器返回前做了 `contiguous()`，避免 FSDP 把输出当成 view 丢掉 pre-backward all-gather。
- loss 在编译期跳过 `_assert_async`。910B 的 inductor 给这个 0 维 device assert 做代码生成时会挂住。

没有和一份参考 loss 对过。这次只说明 8 卡多模态 debugmodel 的 10 步训练循环能走完。

精度：没有 GPU 基线。2.3 第 1 步与 `ln(2048)` 差 0.037，第 10 步 4.68848，grad_norm 9.4633 到 3.0236，全程有限。性能：稳定段单卡 tps 约 39，不能和第 1、2 行的 tps 相除。内存：第 1 步 0.07GiB；第 2–4 步 0.12GiB；第 5 步起 0.24GiB，占日志容量 0.81%。2.2 稳态约 0.26GiB。确定性：两趟都没开确定性开关，loss 也不逐位相同，所以这项还没做。

#### 换机器复现这条 8 卡

只拉分支不够。8 卡能跑完，还靠三份装在 site-packages 里的文件，它们不在 git 里。换机器时先装同一份 wheel 和 `torch_remat==0.2.2`，再把这三份盖上去。本机路径前缀是 `/home/npu-work/npu-y00839695-torchtitan-20261028/`，容器里是 `/workspace/`。

| 文件 | md5 | 盖住之后的行为 |
| --- | --- | --- |
| `torch_npu/_inductor/kernel/flex_attention.py` | `659bf1dbc6c9652d8a5fdf9dad9bedeb` | 非 950 不用 mask-out，反向分块默认 16；`_use_flex_decoding` 在非 950 上返回 false |
| `torch_npu/_inductor/runtime/triton_heuristics.py` | `e0fcd810ba31c602be44fb7c5e49b058` | 非 950 不为 `simd_simt_mix` 生成 `simt_only`；名字里带 `flex_attention` 的 autotune 用 `fcntl` 锁 `/tmp/npu_flex_attention_autotune.lock` |
| `torch_remat/_placeholder.py` | `f3d63fe46a039b58da3c39bdd3152e6a` | NPU 上占位张量用 `empty_strided`，不再把空指针 `set_` 成真实形状 |

和 8 卡有关的提交，按当时尖端 `0861159b` 往前看：`49cbb4d3` 处理 `rope.py` 的 complex64 下标，并把非 CUDA 的 SwiGLU 改成 `silu(gate) * up`；`986e693c` 在 Dist-MoE 导入失败时跳过，并探测 FSDP `MixedPrecisionPolicy`；`16db70a0` 处理视觉 RoPE 的 complex64 下标、loss 里编译期跳过 `_assert_async`、视觉输出 `contiguous()`、这条配方的 `train_timeout_seconds=1800`；`0861159b` 把启动用的 `init_timeout_seconds` 保持在 300，1800 只写在配方里。当前分支名是 `npu-muse-spark-debug-r04`，基于 `release/0.4`。2.2 复验时分支从 `38347913` 对齐到 `0861159b`，三份文件 md5 与上表相同。

环境（训练在容器里，不在宿主机的 Python 里）：

| 项 | 本机 |
| --- | --- |
| 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-910b-ubuntu22.04-py3.12-devel` |
| 容器系统 | Ubuntu 22.04.5，Python 3.12.15。宿主机是 openEuler 22.03 aarch64，没有这份能用的 venv |
| CANN | 9.2.0，`/usr/local/Ascend/cann-9.2.0` |
| 驱动 / 固件 | 26.1.1 / 7.8.0.6.201 |
| 卡 | 8 × Ascend 910B4，单卡 HBM 32768MB。日志里的容量是 29.49GiB |
| torch | `2.15.0+cpu` |
| torch_npu | `2.15.0.dev20261005+git14dc54e`，wheel 在 `/mnt/wheels/`，安装时 `--no-deps` |
| torchvision | `0.30.0.dev20261007+cpu` |
| triton | 3.6.0（带 CANN 扩展，没有单独的 `triton_ascend` 包） |
| numpy | 1.26.4 |
| torch_remat | 0.2.2 |
| torchtitan | 目录里 `pip install -e .` |

启动前在容器里做和 `env.sh` 一样的事：

```bash
grep -q "$(hostname)" /etc/hosts || echo "127.0.0.1 $(hostname)" >> /etc/hosts
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /path/to/.venv/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHNPU_PRECOMPILE_THREADS=1
cd /path/to/torchtitan
```

`TORCHINDUCTOR_COMPILE_THREADS` 和 `TORCHNPU_PRECOMPILE_THREADS` 都要是 1。910B4 上编译进程池和 flex attention 反向的并行选块会段错误。8 张卡都要空着。`NGPU=8` 会用 0 到 7。有卡被占时，用 `ASCEND_RT_VISIBLE_DEVICES` 挑 8 张空卡，再设 `NGPU=8`。

数据在仓库里：`tests/assets/tokenizer/tokenizer.json`，`tests/assets/cc12m_test/cc12m-train-0000.tar`。视频和音频是配方里的合成数据，不用另下。

```bash
source /workspace/env.sh
NGPU=8 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 ./run_train.sh
```

网格应打出 `pp=1, dp_replicate=1, dp_shard=2, cp=2, tp=2, ep=4`。第 1 步之后应出现 `Synchronizing and adjusting timeout for all ProcessGroups to 0:30:00`。启动超时仍是 300 秒。跑完的标志是 rank 0 打出 `step: 10`，然后 `Training completed` 和 `Process group destroyed`。`LOG_RANK` 默认 0，日志里只有 rank 0。第 5 步和第 8 步 tps 会掉到个位数，那是新的视觉长度在编译，等它编完即可。

---

## 第 3 行：Dist-MoE 融合

| 项 | 内容 |
| --- | --- |
| 工作项 | 适配 dispatch、expert compute、combine 的融合执行，支持 bf16 与 mxfp8，验证与压缩 MoE 的兼容 |
| 配方 | `muse_spark_debugmodel_dist_moe_bf16`、`muse_spark_debugmodel_dist_moe_mxfp8`（`muse_spark.py` 约 164、175 行） |
| 清单并行 | 4 卡（沿用基类 TP2 × EP2），另开 CUDA graphs |
| 备注 | 上游为 Blackwell 依赖（sm100+），且需额外安装 dist_moe 包。NPU 上需评估替代方案 |
| 结果 | 没跑 |

| 要点 | 结果 |
| --- | --- |
| 融合执行的数值正确性 | 没有数。当前环境 `import dist_moe` 失败，`torchao` 也不在 |
| 与压缩 MoE 的兼容性 | 没有跑到专家计算。910B 上这两条配方没有替代实现可跑 |

精度、性能、内存、确定性都没有样本。提交 `986e693c` 只是导入失败时跳过，避免别的配方被这条依赖拖死，不是这条本身跑通了。

---

## 第 4 行：NaViT 视觉与音频编码器

| 项 | 内容 |
| --- | --- |
| 工作项 | 适配 NaViT 视觉编码器与音频编码器，验证 `dataset_modalities` 静态参与、dummy 输入零值依赖，以及 bank indices 与 token 分片对齐 |
| 配方 | `muse_spark_debugmodel_mm`（`muse_spark.py` 约 195 行） |
| 清单并行 | 8 卡：DP2 × TP2 × CP2 × EP2，CP 固定为 2 |
| 备注 | 数据为 CC12M 真实图像 + 合成视频 + 合成音频。编码器每步必执行，无媒体时走 dummy 输入 |
| 结果 | 这条名字本身没跑。跑完的是第 2 行③里 EP=4 的多模态变体，数据路径同一套 |

| 要点 | 结果 |
| --- | --- |
| dataset_modalities 静态参与 | 变体跑完 10 步，说明图像、视频、音频打包进了训练。没有打印每个模态是否每步都出现 |
| dummy 输入零值依赖 | 没有单独构造「无媒体」batch，也没有核对 dummy 是否为 0 |
| bank indices 与 token 分片对齐 | 没有打印 bank index 和 token 分片。10 步 loss 有限只说明没有在这里崩溃 |

清单上的网格是 EP2，和第 2 行③的 EP4 不是同一条。变体的 loss 表、编译步和环境见第 2 行③，不能把那次 10 步算成这一行验收通过。精度、性能、内存、确定性都还没有按这一行的网格量过。

---

## 第 5 行：32K、真实 tokenizer、renderer 与 LoRA

| 项 | 内容 |
| --- | --- |
| 工作项 | 适配 32K 序列、真实 tokenizer、renderer 与 LoRA，验证 ChatProcessor 的 assistant-only label masking 与全量激活重算路径 |
| 配方 | `muse_spark_debugmodel_ultradata_lora_sft`（`muse_spark.py` 约 284 行） |
| 清单并行 | 8 卡：DP2 × TP2 × CP2 × EP8；seq_len 32768 |
| 备注 | 唯一使用真实 MuseSparkTokenizer 的 debug 配置（vocab 202048）。强制 `FullActivationRematConfig`。这条会调用 `with_cp(seq_len=32768)` |
| 结果 | 没跑 |

| 要点 | 结果 |
| --- | --- |
| assistant-only label masking | 没有日志 |
| 真实 tokenizer | 没有加载发布版 MuseSparkTokenizer。前面几行用的是 `tests/assets/tokenizer/tokenizer.json`，词表 2048 |
| renderer | 没有跑 |
| LoRA | 没有跑 |
| 全量激活重算 | 没有跑。debugmodel 默认是 `DefaultActivationRematConfig`，不是这条的 `FullActivationRematConfig` |

精度、性能、内存、确定性都没有样本。第 2 行的 CP 和多模态结果不能外推到 32K 和词表 202048。
