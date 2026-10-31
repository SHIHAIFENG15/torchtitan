# Muse Spark 在 910B 上的 NPU 修复记录

记录 `muse_spark_debugmodel` 及相关清单配方在容器里的失败原因和修复。只记本仓库仍要保留的改动。

`make_reduction(..., strict_reduction=...)`、`NPUCombinedScheduling` 与 remat 空指针 storage，已经由 `torchtitan-ascend` 的 `refactor/npu-override` 覆盖：

- 运行时：`torchtitan_ascend/patches/inductor.py`、`torchtitan_ascend/patches/storage.py`
- 安装期：`fixes/torch_npu/storage.patch`

本文件不再展开这三处，代码侧也不再维护对应手工改法和 `torch_remat` 占位补丁，直接用上面的方案。

## 可复用补丁（请先看这里）

910B4 上仍要单独打的 **flex / heuristics** site-packages 改动在：

- 目录：`docs/npu_910b_patches/`
- 应用：`bash docs/npu_910b_patches/apply_npu_910b_patches.sh`
- 环境变量：`source docs/npu_910b_patches/env_npu_910b.sh`
- 说明与版本钉扎：`docs/npu_910b_patches/README.md`

换机器时：检出带这些文件的分支 → 装同版本 `torch_npu` → 跑 apply 脚本 → source env → 再跑训练。Inductor 接口对齐和 remat storage 走 `torchtitan-ascend`，不要再手工改 `lowering.py` / `npu_combined_scheduling.py` / `torch_remat/_placeholder.py`。

环境：

- `torch==2.15.0+cpu`
- 修复前 `torch_npu==2.15.0+git95c7e46`（本地源码编出的 wheel，源码目录已不在）
- 修复后 `torch_npu==2.15.0.dev20261005+git14dc54e`

## 1. 注册 NPU inductor 时导入 flex attention 失败

第一步建 attention mask 时，`create_attention_mask` 会走 `torch.compile(create_block_mask)`。进入 inductor 后 `torch_npu` 加载 `torch_npu._inductor`，旧包在 `torch_npu/_inductor/kernel/flex_attention.py` 里先导入：

```python
from torch._inductor.kernel.flex.flex_attention import (
    get_float32_precision,
)
```

失败后再退回已经不存在的旧模块：

```python
from torch._inductor.kernel.flex_attention import (
    get_float32_precision,
)
```

当前 PyTorch 把函数改名为 `set_float32_precision`，模块也挪到了 `torch._inductor.kernel.flex.flex_attention`。第一段导入抛 `ImportError`，回退路径抛 `ModuleNotFoundError`，外面包成 `BackendCompilerFailed`。另外几个 rank 的 `SIGTERM` 是 rank 2 退出后被 `torchrun` 杀掉的。

处理：换成 `/mnt/wheels/torch_npu-2.15.0.dev20261005+git14dc54e-cp312-cp312-linux_aarch64.whl`。`2.14` 和 `2.13` 的 wheel 分别要求 `torch==2.14.0.*`、`torch==2.13.0.*`，不能用。安装时加了 `--no-deps`，没有动 PyTorch。

这份 wheel 里的兼容代码：

```python
get_float32_precision = None
_upstream_set_float32_precision = None
try:
    from torch._inductor.kernel.flex.flex_attention import (
        get_bwd_subgraph_outputs,
        process_joint_outputs,
        validate_joint_graph,
    )
    try:
        from torch._inductor.kernel.flex.flex_attention import (
            get_float32_precision,
        )
    except ImportError:
        from torch._inductor.kernel.flex.flex_attention import (
            set_float32_precision as _upstream_set_float32_precision,
        )
except ImportError:
    from torch._inductor.kernel.flex_attention import (
        get_float32_precision,
    )


def _set_flex_attention_float32_precision(kernel_options, dtype):
    if "FLOAT32_PRECISION" in kernel_options:
        return
    if get_float32_precision is not None:
        kernel_options["FLOAT32_PRECISION"] = get_float32_precision()
    else:
        _upstream_set_float32_precision(kernel_options, dtype)
```

文件：`torch_npu/_inductor/kernel/flex_attention.py`。

## 2. 融合时把 `[4, 4]` 送给 sympy

`09:35:53` 的训练日志，root cause 仍是 rank 2，发生在 `create_attention_mask` 的编译里：

```text
torch._inductor.exc.InductorError: SympifyError: SympifyError: [4, 4]
```

512 长度、块大小 128 时，query 和 key 方向各有 4 个块。NPU 用 `NumelList((4, 4))` 表示这组循环长度。`SizeVarAllocator.simplify` 被 NPU 改成：遇到 tuple 或 list 就返回一个 Python 列表。上游 `SIMDKernel._split_iteration_ranges` 随后执行 `sympy.Eq([4, 4], 1)`，sympy 不能把列表当成一个数。

NPU 自己的 `NPUTritonKernel._split_iteration_ranges` 会先把这种列表乘成 `4 * 4`。这次崩溃走的是上游 `SIMDKernel.is_compatible`，没有经过那个函数。

本地在 `torch_npu/_inductor/codegen/_sizevars.py` 里包了一层 `SIMDKernel._split_iteration_ranges`：进入上游逻辑之前，把 list 或 tuple 形式的 group 换成 `NumelList(...).numels()`。

## 3. RoPE 用 complex64 做下标

`09:59:14` 的训练已经走过 attention mask 编译，停在第一层 RoPE：

```text
rope.py: rope_cache = rope_cache[positions]
RuntimeError: aclnnIndex failed, error code 161002
Tensor self not implemented for DT_COMPLEX64
```

`ComplexRoPE` 的频率表是 `complex64`。NPU 的 index 只支持 float、bfloat16、整型和 bool，不支持 complex64。

`torchtitan/models/common/rope.py` 的 `_reshape_for_broadcast` 在 NPU 上改为先 `view_as_real`，用 float 下标取出，再 `view_as_complex` 变回去。CUDA 仍走原来的复数下标。

## 4. 文档 mask 里的下标不能放进逐点子图

`10:02:16` 的训练停在 `flex_attention` 编译 mask 子图：

```text
SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph
While executing aten.index.Tensor
attention.py: return kv_idx >= offsets[document_id[q_idx]]
```

打包文档的 mask 要按 query 位置取出该文档的起始下标。910B 上间接访存的 codegen 是关的，`aten.index` 被登记成 AclNN fallback。fallback 会分配一块 buffer，逐点的 mask 子图不允许这样做。

DVM 后端已经在 `torch_npu/_inductor/kernel/flex_attention.py` 里改回上游的逐点 index。默认 Triton 后端没有。本地把这个改回也用到 `inductor_indirect_memory_mode` 为空的情况。

改完之后，缩小的 flex attention 能走过这个下标，但 910B4 上 mask-out 内核编译失败：一块是 UB 溢出，另一块是 `bishengir-compile` 直接 abort。同一段 mask 走 mask-in 可以编译并跑出结果。因此 `_supports_flex_attention_mask_out` 在非 Ascend 950 上改为返回 false，训练使用 mask-in。

## 5. 融合 SwiGLU 没有 NPU 实现

`10:14:12` 的训练停在 MoE 专家的激活：

```text
moe.py: hidden_RF = self.activation_fn(gate_up_R2F, offsets=offsets_E)
fused_swiglu.py: return silu_and_mul_op(gate_up, offsets)
NotImplementedError: Could not run 'torchtitan::silu_and_mul' with arguments from the 'CPU' backend
```

`torchtitan::silu_and_mul` 只登记了 CUDA。张量在 NPU 上时，PrivateUse1 的 fallback 会改去调 CPU 实现，而 CPU 也没有这个算子。

`FusedSwiGLU.__call__` 在非 CUDA 设备上改为 `silu(gate) * up`。有效行的结果和 Triton 核一致，反向走普通 autograd。CUDA 仍走原来的融合核。

## 6. flex attention 反向在 910B4 上段错误

四卡训练走到 flex attention 反向后，rank 3 以 SIGSEGV（exit -11）退出，没有 Python 异常。自动选块挑中了大于 16 的 `BLOCK_M1/BLOCK_N1`。同一段 mask 把反向分块固定成 16 时，前向和反向都能跑完，梯度不是 0。

`torch_npu/_inductor/kernel/flex_attention.py` 的 mask-in 反向在非 Ascend 950 上把 `BLOCK_M1`、`BLOCK_N1`、`BLOCK_M2`、`BLOCK_N2` 默认设为 16。调用方如果已经指定了这些值，不会被覆盖。

分块改成 16 之后，训练能打完前几步，但 inductor 的编译子进程池在后面的 step 里段错误退出（日志里有 `SubprocPool unclean exit`）。`env.sh` 里设置 `TORCHINDUCTOR_COMPILE_THREADS=1`，编译改在本进程做。这样 `muse_spark_debugmodel` 的 10 步可以跑完。

`TORCHINDUCTOR_COMPILE_THREADS` 管的是 PyTorch 自己的编译进程池。`torch_npu` 另有 `precompile_thread_num`，默认至少 32，环境变量是 `TORCHNPU_PRECOMPILE_THREADS`。`muse_spark_debugmodel_fsdp2_cp2_ep4` 在 flex attention 反向选块时，这个池子开了 128 个线程去测 `triton_per_fused_flex_attention_backward_0` 的 68 组配置，rank 0 段错误。`env.sh` 里把 `TORCHNPU_PRECOMPILE_THREADS` 也设成 1。

## 7. 上下文并行的 block mask 求和被编成 simt_only

`muse_spark_debugmodel_fsdp2_cp2_ep4` 在 `_cp_shard` 里编译 `create_block_mask`。query 被 CP 切开后，mask 先垫到 128 的倍数，再按 `128 x 128` 的块求和。这个 reduction 走 `NPUTritonKernel`，元数据里的 `npu_kernel_type` 被写成 `simd_simt_mix`。选块时会先生成 `compile_mode="simt_only"`。910B4 不是 Ascend 950，Triton 直接拒绝这个模式，于是整组配置都算失败：

```text
NoTritonConfigsError: No valid triton configs for kernel
triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_3
ValueError: compile_mode="simt_only" is supported only on Ascend 950 targets.
```

孤立的 `sum` 复现也会停在同一句话，但训练失败是自己走到了这个 kernel。

`torch_npu/_inductor/runtime/triton_heuristics.py` 里，非 Ascend 950 不再为 `simd_simt_mix` 生成 `simt_only` 配置，只保留 `simd_simt_template`。910B 上这个模式走普通 SIMD。用动态长度 256、key 长度 512 的 bool mask 编译同一段 view、permute、sum，结果和 eager 一致。

## 8. 视觉 RoPE 的 complex64 下标

8 卡多模态配方 `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` 第一轮停在：

```text
RuntimeError: index_high_dims_op_api aclnnIndex 161002
DT_COMPLEX64
vision_encoder.py: freqs_cis_GD[sp_perm]
```

`49cbb4d` 修的是 `rope.py` 的 `_reshape_for_broadcast`。视觉编码器在 sparse attention 里直接对 complex64 的频率缓存做下标，不经过那个函数。910B 的 index 没有 complex64。

`torchtitan/models/muse_spark/vision_encoder.py` 在 NPU 上改成先 `view_as_real`，按下标取出实部虚部，再 `view_as_complex`。其它设备仍走原来的下标。

## 9. 反向重算里的空指针占位张量

不再在本仓库记录改法，也不再打 `torch_remat` 占位补丁。用 `torchtitan-ascend` 的 `torchtitan_ascend/patches/storage.py`和安装期 `fixes/torch_npu/storage.patch`（跟踪 Ascend/pytorch !47944）。

## 10. 8 个 rank 同时给 flex attention 反向选块

占位张量改完之前，还有一种段错误没有 Python 异常：一个 rank 在 `triton_per_fused_flex_attention_backward_0` 的几十组候选里退出（exit -11），其余 rank 被 SIGTERM。同一形状在单卡上前向反向都能算出有限梯度。视觉序列一例是 1460，head_dim 16，2 个本地 head。文本一例是 query `1x4x256x32`、KV `1x2x512x32`。崩的 rank 每次不一样。

尝试过在选块时直接拿最小的 runtime tile。launcher 和 runtime block 对不上，进程以 `realloc(): invalid pointer` / `munmap_chunk(): invalid pointer` 中止。这处改动已撤掉。

留下的改法在 `torch_npu/_inductor/runtime/triton_heuristics.py`：名字里带 `flex_attention` 的 autotune 用 `fcntl.flock` 锁住 `/tmp/npu_flex_attention_autotune.lock`，同一时刻只有一个进程在跑候选。同进程重入用深度计数，避免自己把自己锁死。候选本身仍按原来的计时挑选，不改分块。

## 11. 短视觉序列的 flex decoding 和 100 秒超时

第 1 到第 4 步能打完之后，第 5 步视觉 token 数变成 4。inductor 改走 flex decoding（query 长度小于 128）。rank 0 还在编译，另一个 rank 已经发出 all-gather。第 1 步之后进程组超时是 `train_timeout_seconds=100`。大约 100 秒后：

```text
Watchdog caught collective operation timeout:
WorkHCCL(OpType=ALLGATHER, NumelIn=15856, NumelOut=31712, Timeout(ms)=100000)
```

HCCL watchdog 把还在集合通信里的 rank 打成 SIGABRT（exit -6）。`NumelOut` 是 `NumelIn` 的 2 倍，对应 TP=2 的 all-gather。

两处一起改：

- `torch_npu/_inductor/kernel/flex_attention.py` 的 `_use_flex_decoding` 在非 Ascend 950 上直接返回 false。短序列继续用已经能跑的 mask-in flex attention，不再编一套 decoding 核。
- `tests/integration_tests/models.py` 里这条 8 卡配方把 `comm.train_timeout_seconds` 设为 1800。新的视觉长度仍会重新编译，第 5 步和第 8 步各花了大约一分钟，100 秒盖不住，1800 秒盖得住。

## 12. 词表断言和 FSDP 的 view 输出

`torchtitan/components/loss.py` 里，`_LossParallelCrossEntropy` 的 `torch._assert_async` 在 `torch.compiler.is_compiling()` 时跳过。910B 的 inductor 编不出这个 0 维 device 断言，会停在编译里。eager 时检查还在。

`NaViTEncoder.forward` 末尾，单帧时 `torch.cat` 返回的就是那一帧，FSDP 把它看成 view，就地算子会丢掉 pre-backward all-gather。返回值改成 `torch.cat(outputs).contiguous()`。音频编码器和文本 block 的同类警告这次没有改调用点。

## 13. 8 卡多模态 10 步

上面几处都在的那次运行是 2026-10-28 的 `item2_logs/mm_r12.log`。10 步结束，日志有 `Training completed`。rank 0 的 loss 从 7.66202 到 4.68848，grad_norm 从 9.4633 到 3.0236，逐步数字在 `docs/npu_debugmodel_verification.md`。没有和参考 loss 对过。

## 14. Dist-MoE bf16 在 910B 上的专家计算

清单第 3 行的 `muse_spark_debugmodel_dist_moe_bf16` 原先过不了设备检查：

```text
ValueError: Dist-MoE requires an SM100-or-newer CUDA device
```

`torchtitan/models/common/dist_moe/runtime.py` 要求 `device.type == "cuda"` 且计算能力主版本至少 10。910B 的 `device.type` 是 `npu`。可选包 `dist_moe` 和 `torchao` 都没装，装上包也过不了这道检查。mxfp8 配方在配置期就会调用 `dist_moe.BlockScaledConfig()`，这次没有改，也没有跑。

bf16 的改动都在 torchtitan 里，不改 site-packages。CUDA 设备仍走原来的包和 SM100 检查。

`runtime.py` 在 NPU 上不再调用 `dist_moe.create_context`。换成 `_NpuDistMoeContext`，只实现 `select_activation_slot`、`reset`、`close`，并记下 `max_num_local_input_tokens`。流水线注册和关闭用的是这几个方法。专家的 scratch 不在这个对象里。

`routed_experts.py` 在 `x_TD.device.type == "npu"` 时调用 `npu_dist_moe_forward`，包在保存区域 `feed_forward.routed_experts.dist_moe` 里（`recompute=False`）。`num_local_tokens_per_expert_E` 要传进去，不能先 `del`。

`torchtitan/models/common/dist_moe/npu.py` 做三件事：

- 专家 GEMM 用 `npu_grouped_matmul`。权重在模块里是 `[E, out, in]`，调用前转成 `[E, in, out]`。`group_list` 是 int64 前缀和，`split_item=3`、`group_type=0`、`group_list_type=0`。直接传 `[E, out, in]` 会在 `aclnnGroupedMatmulV5` 报 161002。非方阵对照（F=8，2F=16）和 `rows @ W.T` 的最大绝对差是 0.0304，是 bf16 误差。FSDP 权重先 `to_local()`。
- 激活是普通 SwiGLU 时用 `npu_swiglu`，按最后一维对半，前一半 gate、后一半 up。ClampedSwiGLU 仍走 Python 公式。debug 模型用的是普通 SwiGLU。
- token 交换用现成的 `AllToAllTokenDispatcher._dispatch` / `_combine`，不是 `npu_moe_distribute_dispatch`。dispatcher 用 `object.__setattr__` 挂在模块上，不进子模块，避免改 CUDA 那条的 state dict 和 FSDP 树。分数仍在 combine 里做加权 scatter-add。

第一次 4 卡启动（`/workspace/dist_moe_bf16_r1.log`）没有 step，报：

```text
RuntimeError: remat.region 'dispatch' was called with recompute=True while nested inside a recompute=False (SAVE) region.
```

公共 `dispatch()` / `combine()` 会再开默认重算的 `dispatch`、`combine` 区域。外层 `dist_moe` 是保存区域，里面不能再套重算区域。NPU 路径改成直接调 `_dispatch` / `_combine`，元数据在这一层组装，不再开内层区域。

`muse_spark_debugmodel_dist_moe_bf16` 在 `_with_dist_moe` 之后把 `disable_cuda_graphs` 设回 True。910B 没有 CUDA graph。`local_compile_regions` 收回 `loss`、`fp32_to_bf16_split`，`train_timeout_seconds=1800`。网格仍是基类的 4 卡 `dp_shard=2, tp=2, ep=2`。

2026-10-30 在 `179.173.2.3` 上，`HCCL_DETERMINISTIC=true`、`MUSE_DEBUG_DETERMINISTIC=1`、`NGPU=4`，同一配置连跑两次（`/workspace/dist_moe_bf16_r2.log`、`r3.log`）。10 步 loss、grad_norm 逐项相同，loss 从 7.68369 到 5.36020，grad_norm 从 2.7578 到 1.6899。两趟都有 `Training completed` 和 `Process group destroyed`。逐步数字在 `docs/npu_debugmodel_verification.md` 第 3 行。没有 GPU 1000 步基线。mxfp8 和压缩 MoE 没有样本。

## 15. 第 4 行多模态配方的训练超时

`muse_spark_debugmodel_mm` 是清单第 4 行：8 卡 `dp_shard=2, tp=2, cp=2, ep=2`，Ulysses。视觉 token 长度一变，flex attention 就要重新编译。第 2 行③那条 EP4、AllGatherKV 的变体已经在 `tests/integration_tests/models.py` 里把 `train_timeout_seconds` 设成 1800，因为编译超过默认 100 秒时，先跑完的 rank 会把 all-gather 拖到 HCCL watchdog 超时。

清单上的这份配方原来没有设这个超时。`torchtitan_recipes/tests/models/muse_spark.py` 的 `muse_spark_debugmodel_mm` 在返回前补上 `config.comm.train_timeout_seconds = 1800`。其它字段没改，融合 SwiGLU override 仍在。

2026-10-30 在 `179.173.2.3` 上，`HCCL_DETERMINISTIC=true`、`MUSE_DEBUG_DETERMINISTIC=1`、`NGPU=8`，连跑两次（`/workspace/det_mm_ep2_r1.log`、`r2.log`）。网格是 `dp_shard=2, cp=2, tp=2, ep=2`。10 步 loss、grad_norm 逐项相同，loss 从 7.69125 到 4.57818，grad_norm 从 9.7405 到 2.7223。逐步数字在 `docs/npu_debugmodel_verification.md` 第 4 行。没有单独打印每个模态、dummy 是否为 0、bank index 和 token 分片。

## 16. 第 5 行改读本地 UltraData JSONL

`muse_spark_debugmodel_ultradata_lora_sft` 原来调用 `_ultradata_sft_dataloader()`，不传 source 时内部是 `HuggingFaceStreamingSource`，数据集 `openbmb/UltraData-SFT-Agent-2609`。这台机器上 huggingface.co:443 会被连接重置，流式数据起不来。816B 配方已经支持本地 JSONL，debug 配方没有走那条。

debug 配方在 `_use_released_tokenizer_with_debug_model` 之后同样读 `MUSE_SPARK_ULTRADATA_JSONL_PATTERN`，默认 `_ULTRADATA_LOCAL_JSONL_PATTERN`，用 `IndexedJsonlSource` 交给原来的 `_ultradata_sft_dataloader`。ChatProcessor、MuseSparkRenderer、LoRA、32K 和 `FullActivationRematConfig` 没改。`.gitignore` 加上 `assets/datasets/`，JSONL 不进 git。

官方 tokenizer 仓库要手动授权。放到 `assets/hf/Muse-Spark-Open-816B-A42B` 的是公开副本里的 tokenizer 文件。`tokenizer.json` 最大 id 是 201817，配置里的词表宽度仍是 202048。没有下 816B 权重。数据只放了 General-Agent 第 1 份，1575113443 字节。

2026-10-30 在 `179.173.2.3` 上，`HCCL_DETERMINISTIC=true`、`MUSE_DEBUG_DETERMINISTIC=1`、`NGPU=8`，连跑两次（`/workspace/det_ultradata_r1.log`、`r2.log`）。网格是 `dp_shard=2, cp=2, tp=2, ep=8`。10 步 loss、grad_norm 逐项相同，loss 从 12.31145 到 12.28428，grad_norm 从 0.2715 到 0.2656。逐步数字在 `docs/npu_debugmodel_verification.md` 第 5 行。没有单独打印 loss_mask 里有多少个监督 token。

## 17. 第 1 行性能线性度与单卡内存对照配方

验收标准要求：性能用同一配方只加倍 DP，比 8 卡 tps / 4 卡 tps；内存先比单卡日志预留。原 `muse_spark_debugmodel` 世界大小写死为 4、只有 10 步，凑不齐这两项。

在 `torchtitan_recipes/tests/models/muse_spark.py` 增加三个配方，不改原 10 步验收配置：

- `muse_spark_debugmodel_perf_dp2`：网格与基类相同（`dp_shard=2, tp=2, ep=2`），`steps=200`
- `muse_spark_debugmodel_perf_dp4`：只把 `data_parallel_shard_degree` 改成 4，世界大小 8，`steps=200`
- `muse_spark_debugmodel_mem_1device`：全部并行度为 1，关掉 sequence parallel，`steps=20`

跑法、记数表和达标判定写在 `docs/gpu_baseline_perf_mem.md`。NPU 与 GPU 必须用同一 CONFIG。跑完后把数填进 `docs/npu_debugmodel_verification.md` 第 1 行和 `docs/npu_gpu_acceptance_report.md`。
