# muse_spark_debugmodel 验证要点分析

对照清单第 1 项。这一项要的是文本 next-token 训练能走完，并核对数据加载、损失计算、反向传播、参数更新。模型形状是 dim 256、8 层、8 个专家、序列 512、10 步。

4 卡按这个配方的默认网格跑，算这一项的正式配置。配方把 `data_parallel_shard_degree` 和 `tensor_parallel_degree` 都写成 2，世界大小是 `1 × 2 × 1 × 2 × 1 = 4`。`expert_parallel_degree=2` 从这 4 个进程里分专家，不另加进程。序列并行跟着张量并行，也不另加进程。2026-10-27 那次用 0、1、3、4 号 910B4，`NGPU=4`，日志是：

```text
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
```

10 步打完，退出码 0。rank 0 和 rank 1 每一步的 loss、grad_norm 相同，这是 loss 网格归约之后的全局值。

## 配置核对

| 清单 | 这次 | 依据 |
| --- | --- | --- |
| dim 256 | 满足 | `_DEBUG_MODEL_DIM = 256` |
| 8 层 | 满足 | `_debugmodel` 里 `num_layers = 8` |
| 8 experts | 满足 | `num_experts=8`，每个 token 再选 top-2 |
| seq_len 512 | 满足 | debugmodel 默认训练长度 512，microbatch 也是 512 |
| 10 步 | 满足 | `training.steps=10`，日志到 `step: 10` 后 `Training completed` |
| 4 卡 | 满足 | 默认网格 `dp_shard=2 × tp=2`，进程数 4 |

数据是 `tests/assets/c4_test/data.json`，`shuffle=False`。优化器是 AdamW，学习率 `8e-4`，warmup 2 步。词表 2048。

rank 0 的数：

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

## 1. 数据加载

路径是 `GrainDataLoader` + `ConcatThenSplitPackingConfig` + `DATASETS["c4_test"]`。日志里 tokenizer 从 `tests/assets/tokenizer/tokenizer.json` 读入，datasets 库去拉 json 加载器。10 步都打出了有限 loss，说明每步都拿到了 batch，而且标签里有参与损失的 token。标签若全是 `ignore_index=-100`，交叉熵不会停在 7 附近。

4 卡时样本按 FSDP 的 2 个数据分片切开，这是这套网格的正常切法。两边 loss 经过归约后一致，说明两个数据分片都进了同一次全局损失。

还没有把一个 batch 的 token id 和 `data.json` 原文对过，也没有单独打印 next-token 错位：输入是 `tokens[:-1]`，标签是 `tokens[1:]`，打包边界处是否标成 -100。

结论：加载器在 4 卡上每步都喂进了带有效标签的 batch。样本内容和标签对齐没有逐条看过。

## 2. 损失计算

损失是 `ChunkedLossWrapper`，8 段，里面是词表并行交叉熵。TP=2 时每个 rank 只有词表的一半，前向用 all-reduce 拼 log-sum-exp。日志里的 loss 是按有效 token 平均后再做跨 rank 归约。若是 512 个 token 的和，数量级会到几千。

第 1 步 7.68154。随机初始化时，2048 类上的均匀猜测是 `ln(2048) = 7.6246`，差 0.057。第 1 步的预测接近均匀，损失尺度像平均交叉熵。后面降到 4.99761，第 5 步和第 8 步又升过。`shuffle=False`，每步句子不同，所以下降本身不能单独证明公式算对了。

还没有拿同一批 logits 和标签，在 CPU 上用 `F.cross_entropy` 再算一遍，也没有核对 8 段 chunk 加起来是否等于整段平均。NPU 上 RoPE 用实数下标再变回复数，SwiGLU 用 `silu(gate) * up`，flex attention 反向分块固定为 16。这些和上游 CUDA 核不同，这次没有做 loss 对齐。

结论：4 卡上的损失尺度像词表并行的平均交叉熵，两个 rank 的归约结果一致。数值还没有和参考实现对拍。

## 3. 反向传播

`grad_norm` 10 步都是有限值，从 4.7997 落到大约 1.5–1.8，没有 NaN，也没有变成 0。`trainer.train_step` 里这个数是 `engine.optim_step()` 的返回值，所以反向图产生了优化器能用的梯度。rank 0 和 rank 1 一致，说明 FSDP/TP 的梯度归约没有让两边各算各的。

还没有和一份参考实现比某一层的梯度。910B4 上 flex attention 走 mask-in，反向分块是 16；更大的分块会在真实反向里段错误。缩小例子里这块梯度不是 0，但没有和未改分块的结果比过。FSDP 还警告过 `FSDPMuseSparkTransformerBlock` 返回了 view。若后面有就地修改，pre-backward 的 all-gather 会被跳过。这次没有顺着这条警告查出是哪一次算子。

结论：4 卡反向在跑，梯度有限且两边一致。梯度数值是否正确还没有证明。

## 4. 参数更新

`optim_step()` 在打日志之前被调用。学习率 `8e-4`，warmup 2 步，梯度范数不是 0，AdamW 会改权重。10 步都打出来了，说明更新函数返回了。

这次 `checkpointer=None`，没有存第 0 步和第 10 步的权重，也没有抽一个参数看 `max(abs(after - before))`。loss 从 7.68 到 5.00 中间有两次回升，在不打乱的数据上，这掺着换句子，不能单独当成参数学对了。

结论：4 卡上每步都执行了优化器更新。参数实际改了多少，没有直接量过。

## 总评

| 验证要点 | 4 卡这次能说的 | 还缺的 |
| --- | --- | --- |
| 形状和步数 | dim 256、8 层、8 专家、序列 512、10 步都对上 | |
| 4 卡网格 | `dp_shard=2 × tp=2`，EP=2 复用这 4 个进程 | |
| 数据加载 | c4_test 和 tokenizer 被读到，每步有带标签的 batch | 和原文逐条对 token、错位、ignore |
| 损失计算 | 第 1 步接近 `ln(2048)`，两 rank 归约一致 | 和本地 `cross_entropy` 对拍 |
| 反向传播 | grad_norm 有限且两 rank 一致 | 某一层梯度和参考实现对比 |
| 参数更新 | 每步都调用了 `optim_step()` | 更新前后的参数差 |

第 1 项的配置和整条训练循环，按 4 卡默认配方已经跑通。四个验证要点都有运行迹象，都还没有和参考数值对上。

清单第 3 到第 5 项（Dist-MoE、多模态、32K LoRA）这次没有跑。第 2 项的三条配方里，`muse_spark_debugmodel_fsdp2_cp2_ep4` 已在 2026-10-28 用 2、3、4、5 号卡跑完 10 步，退出码 0。网格是 `dp_shard=2, cp=2, tp=1, ep=4`。rank 0 的 loss 从 7.76808 到 5.02892，grad_norm 从 4.6650 到 1.6260，10 步都是有限值。另外两条按代码是 8 卡（`with_cp` 在基类的 `dp_shard=2, tp=2` 上再加 `cp=2`；多模态那条是 `dp_shard=2, tp=2, cp=2`），当时 0 号和 1 号卡被别的任务占用，没有启动。


`muse_spark_debugmodel_with_cp` 随后在 2026-10-28 07:14 UTC 用 8 张卡启动（`NGPU=8 ./run_train.sh`，日志 `item2_logs/with_cp.log`）。进程起来后停在 `_StridedShard(dim=0) + Shard(dim=0)` 到 `Replicate` 的两次 all_gather，日志从 07:22 UTC 起不再增长。8 个 rank 的 AICore 一直是 0，CPU 在 HCCL 里空转。4 号卡当时被别人的 `rn50_grad_probe.py` 占满（AICore 99%，`ASCEND_RT_VISIBLE_DEVICES=4`，停在 inductor autotune）。这次 8 卡任务没有跑完 10 步，已在约 08:25 UTC 停掉。0 号卡上还有别人的 `pytest test_perf.py`。4 号卡空出来之前，这条 8 卡配方不能重跑。多模态那条随后单独跑通，见下一节。

## 8 卡多模态 debugmodel

清单第 2 项里标成 8 卡的配方是 `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4`。2026-10-28 11:02–11:07 UTC 在 8 张 910B4 上跑完 10 步。日志 `item2_logs/mm_r12.log` 以 `Training completed` 和 `Process group destroyed` 结束，没有 `ChildFailedError`。

启动：

```text
source /workspace/env.sh
NGPU=8 MODULE=tests.integration_tests.models \
  CONFIG=muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4 ./run_train.sh
```

网格是 `pp=1, dp_replicate=1, dp_shard=2, cp=2, tp=2, ep=4`。EP 从这 8 个进程里分专家，不另加进程。`2 × 2 × 2 = 8`。序列 512，10 步。文本 dim 256、8 层、8 专家、词表 2048。视觉是 NaViT，head_dim 16。数据是 `tests/assets/cc12m_test/cc12m-train-0000.tar` 加合成音频、视频。第 1 步之后进程组超时从默认 100 秒改成 1800 秒，日志是 `Synchronizing and adjusting timeout for all ProcessGroups to 0:30:00`。

rank 0 的数。loss 和 grad_norm 是归约后的全局值。第 5 步和第 8 步 tps 掉到 2 和 1，是新的视觉 token 长度在重新编译 flex attention，不是算子挂住。

| step | loss | grad_norm |
| --- | --- | --- |
| 1 | 7.66202 | 9.4633 |
| 2 | 6.83399 | 6.2513 |
| 3 | 5.79272 | 4.1767 |
| 4 | 6.22080 | 5.1368 |
| 5 | 4.33259 | 4.8085 |
| 6 | 5.80127 | 3.5880 |
| 7 | 5.17229 | 2.9730 |
| 8 | 3.40500 | 2.9641 |
| 9 | 3.98423 | 2.8151 |
| 10 | 4.68848 | 3.0236 |

第 1 步 7.66202，和 `ln(2048) = 7.6246` 差 0.037，尺度仍像平均交叉熵。后面有降有升，打包的图、音频、视频每步长度不同，不能单独当成和参考实现对上了。`checkpointer` 是空的，没有存权重，参数差没有直接量。10 步 grad_norm 都是有限值，没有变成 0。

这次能跑完，靠的是下面几处改动，细节在 `docs/npu_pta_fixes.md` 第 10 节起：

- 视觉 RoPE 的 complex64 下标不走 `rope.py` 那一处，910B 的 index 没有 complex64。
- `torch_remat` 在 NPU 上不再用空指针 storage 做占位张量。空指针在反向重算里会段错误。
- 8 个 rank 一起给 flex attention 反向选块时，会有一个 rank 段错误。选块改成跨进程串行。
- 非 Ascend 950 不用 flex decoding。短视觉序列会另编一套核，编译时间超过原来的 100 秒训练超时，先跑完的 rank 把 all-gather 拖超时，HCCL watchdog 以 SIGABRT 退出。
- 这条配方的 `train_timeout_seconds` 设为 1800，盖住第 5 步、第 8 步那种新长度编译。
- 单帧 `torch.cat` 得到的是原张量。视觉编码器返回前做了 `contiguous()`，避免 FSDP 把输出当成 view 丢掉 pre-backward all-gather。

没有和一份参考 loss 对过。这次只说明 8 卡多模态 debugmodel 的 10 步训练循环能走完。
