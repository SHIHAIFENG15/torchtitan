# NPU PTA 与 PyTorch 2.15 的两处接口不一致

记录 `muse_spark_debugmodel` 在容器 `npu-y00839695-torchtitan-20261027` 里启动失败的原因，以及已经做的修复。

环境：

- `torch==2.15.0+cpu`
- 修复前 `torch_npu==2.15.0+git95c7e46`（本地源码编出的 wheel，源码目录已不在）
- 修复后 `torch_npu==2.15.0.dev20261005+git14dc54e`
- 远程仓库 [Ascend/pytorch](https://gitcode.com/Ascend/pytorch) 的 `master` 里，`make_reduction` 仍然没有 `strict_reduction`

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

## 2. 编译 block mask 时 `make_reduction` 不接受 `strict_reduction`

换包之后，失败点变成 `create_block_mask` 里的：

```python
mask = mask.view(B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE)
mask = mask.permute(0, 1, 2, 4, 3, 5)
mask_block_sum = mask.sum(dim=[-2, -1])
```

终端里 `sliding_window_mod` 的栈只是这块 bool mask 的来源。真正的异常是：

```text
TypeError: make_reduction() got an unexpected keyword argument 'strict_reduction'
target: aten.sum.dim_IntList
args[1]: [-2, -1]
```

eager 的同一段 `view + permute + sum` 可以算出 `int64` 结果。`torch.compile` 会走到 PyTorch 2.15 的 `sum` / `prod` lowering，它们这样调用：

```python
fn = make_reduction(
    "sum", override_return_dtype=dtype, strict_reduction=strict_reduction
)
```

`torch_npu/_inductor/__init__.py` 把 `torch._inductor.lowering.make_reduction` 换成了自己的函数，签名只有 `(reduction_type, override_return_dtype)`。远程 master 也是这个签名。

本地改的是容器里实际加载的这份文件：

`/workspace/dev/.venv/lib/python3.12/site-packages/torch_npu/_inductor/lowering.py`

宿主机同一路径是 `/home/npu-work/npu-y00839695-torchtitan-20261027/dev/.venv/lib/python3.12/site-packages/torch_npu/_inductor/lowering.py`。

`ascend_npu_ir/.../inductor_patch/lowering.py` 里还有一份同名函数。这次崩溃没有走到它，所以没有改。

修复后的函数：

```python
def make_reduction(
    reduction_type: ReductionType,
    override_return_dtype=None,
    *,
    strict_reduction: bool = False,
):
    # PyTorch 2.15 sum/prod pass strict_reduction into make_reduction.
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )
        if npu_config.dump_fx_graph:
            node_name = f'reduction_{next(node_id)}'
            input_graphs = fetch_graphs([x, axis if axis is not None else list(range(len(x.get_size())))])
            new_graph = merge_traced_graphs(input_graphs, reduction_type_to_aten_fn[reduction_type],
                                            node_name, keepdim=keepdims)
            result = Reduction.create(reduction_type=reduction_type,
                                      input_node=x,
                                      node_name=node_name,
                                      traced_graph=new_graph,
                                      strict_reduction=strict_reduction,
                                      **kwargs)
        else:
            result = Reduction.create(reduction_type=reduction_type,
                                      input_node=x,
                                      strict_reduction=strict_reduction,
                                      **kwargs)
        if isinstance(result.data.data, Reduction):
            result.realize()
        return result

    return inner
```

`torch._inductor.ir.Reduction.create` 已经有 `strict_reduction: bool = False`，这里只是把参数传进去。

重装 `torch_npu` wheel 会覆盖这个修改。

## 最小复现

```python
import torch

def f(mask):
    mask = mask.view(1, 1, 4, 128, 4, 128)
    mask = mask.permute(0, 1, 2, 4, 3, 5)
    return mask.sum(dim=[-2, -1])

mask = torch.randint(0, 2, (1, 1, 512, 512), device="npu", dtype=torch.bool)
print(torch.compile(f)(mask).shape)
```

单独编译这段 `sum` 时，`strict_reduction` 已经不再报错。那次小复现接着停在 Triton 的 `compile_mode="simt_only"`，只说明这个孤立 kernel 的编译选项不适合当前机器。它不是训练终端里的失败。

## 3. 训练终端实际停在 `NPUCombinedScheduling`

`09:30:31` 的训练日志，root cause 是 rank 2：

```text
torch._inductor.exc.InductorError: AttributeError: 'NPUCombinedScheduling' object has no attribute '_nv_universal_gemm_scheduling'
```

调用链是 inductor 做 fusion 时走到父类方法：

```text
torch_npu/_inductor/scheduler.py: fuse_nodes_once
  -> Scheduler.can_fuse
    -> CUDACombinedScheduling.can_fuse_reduction_pair
         self._nv_universal_gemm_scheduling.has_conflicting_epilogue_reductions(...)
```

`NPUCombinedScheduling` 继承了 `CUDACombinedScheduling`，但 `__init__` 只调用 `BaseScheduling.__init__`，自己创建 `_triton_scheduling` 和 `_catlass_scheduling`。父类 `__init__` 里才会赋值 `_nv_universal_gemm_scheduling`，所以子类没有这个属性。`can_fuse_reduction_pair` 没有在 NPU 侧重写，融合 reduction 时直接访问了这个 NVIDIA GEMM scheduler。`get_fusion_pair_priority` 在 Welford 关闭时也会 `super()` 到同一处。

本地改的是：

`/workspace/dev/.venv/lib/python3.12/site-packages/torch_npu/_inductor/codegen/npu_combined_scheduling.py`

NPU 上没有 NVIDIA GEMM，所以 reduction pair 不再查那个 scheduler，epilogue 融合按父类对非 GEMM 节点的结果返回 `False`。融合优先级改走 Triton scheduler，避免再调用父类。

```python
def can_fuse_reduction_pair(self, node1, node2) -> bool:
    del node1, node2
    return True

def can_fuse_reduction_epilogue(self, node1, node2) -> bool:
    del node1, node2
    return False

def get_fusion_pair_priority(self, node1, node2) -> int:
    if not npu_config.enable_welford:
        return self._triton_scheduling.get_fusion_pair_priority(node1, node2)
    if is_same_welford_output_group(node1, node2):
        return 0
    if is_welford_epilogue_fusion(node1, node2):
        return 1
    return self._triton_scheduling.get_fusion_pair_priority(node1, node2) + 2
```

## 4. 融合时把 `[4, 4]` 送给 sympy

`09:35:53` 的训练日志，root cause 仍是 rank 2，发生在 `create_attention_mask` 的编译里：

```text
torch._inductor.exc.InductorError: SympifyError: SympifyError: [4, 4]
```

512 长度、块大小 128 时，query 和 key 方向各有 4 个块。NPU 用 `NumelList((4, 4))` 表示这组循环长度。`SizeVarAllocator.simplify` 被 NPU 改成：遇到 tuple 或 list 就返回一个 Python 列表。上游 `SIMDKernel._split_iteration_ranges` 随后执行 `sympy.Eq([4, 4], 1)`，sympy 不能把列表当成一个数。

NPU 自己的 `NPUTritonKernel._split_iteration_ranges` 会先把这种列表乘成 `4 * 4`。这次崩溃走的是上游 `SIMDKernel.is_compatible`，没有经过那个函数。

本地在 `torch_npu/_inductor/codegen/_sizevars.py` 里包了一层 `SIMDKernel._split_iteration_ranges`：进入上游逻辑之前，把 list 或 tuple 形式的 group 换成 `NumelList(...).numels()`。

## 5. RoPE 用 complex64 做下标

`09:59:14` 的训练已经走过 attention mask 编译，停在第一层 RoPE：

```text
rope.py: rope_cache = rope_cache[positions]
RuntimeError: aclnnIndex failed, error code 161002
Tensor self not implemented for DT_COMPLEX64
```

`ComplexRoPE` 的频率表是 `complex64`。NPU 的 index 只支持 float、bfloat16、整型和 bool，不支持 complex64。

`torchtitan/models/common/rope.py` 的 `_reshape_for_broadcast` 在 NPU 上改为先 `view_as_real`，用 float 下标取出，再 `view_as_complex` 变回去。CUDA 仍走原来的复数下标。

## 6. 文档 mask 里的下标不能放进逐点子图

`10:02:16` 的训练停在 `flex_attention` 编译 mask 子图：

```text
SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph
While executing aten.index.Tensor
attention.py: return kv_idx >= offsets[document_id[q_idx]]
```

打包文档的 mask 要按 query 位置取出该文档的起始下标。910B 上间接访存的 codegen 是关的，`aten.index` 被登记成 AclNN fallback。fallback 会分配一块 buffer，逐点的 mask 子图不允许这样做。

DVM 后端已经在 `torch_npu/_inductor/kernel/flex_attention.py` 里改回上游的逐点 index。默认 Triton 后端没有。本地把这个改回也用到 `inductor_indirect_memory_mode` 为空的情况。

改完之后，缩小的 flex attention 能走过这个下标，但 910B4 上 mask-out 内核编译失败：一块是 UB 溢出，另一块是 `bishengir-compile` 直接 abort。同一段 mask 走 mask-in 可以编译并跑出结果。因此 `_supports_flex_attention_mask_out` 在非 Ascend 950 上改为返回 false，训练使用 mask-in。

## 7. 融合 SwiGLU 没有 NPU 实现

`10:14:12` 的训练停在 MoE 专家的激活：

```text
moe.py: hidden_RF = self.activation_fn(gate_up_R2F, offsets=offsets_E)
fused_swiglu.py: return silu_and_mul_op(gate_up, offsets)
NotImplementedError: Could not run 'torchtitan::silu_and_mul' with arguments from the 'CPU' backend
```

`torchtitan::silu_and_mul` 只登记了 CUDA。张量在 NPU 上时，PrivateUse1 的 fallback 会改去调 CPU 实现，而 CPU 也没有这个算子。

`FusedSwiGLU.__call__` 在非 CUDA 设备上改为 `silu(gate) * up`。有效行的结果和 Triton 核一致，反向走普通 autograd。CUDA 仍走原来的融合核。

## 8. flex attention 反向在 910B4 上段错误

四卡训练走到 flex attention 反向后，rank 3 以 SIGSEGV（exit -11）退出，没有 Python 异常。自动选块挑中了大于 16 的 `BLOCK_M1/BLOCK_N1`。同一段 mask 把反向分块固定成 16 时，前向和反向都能跑完，梯度不是 0。

`torch_npu/_inductor/kernel/flex_attention.py` 的 mask-in 反向在非 Ascend 950 上把 `BLOCK_M1`、`BLOCK_N1`、`BLOCK_M2`、`BLOCK_N2` 默认设为 16。调用方如果已经指定了这些值，不会被覆盖。

分块改成 16 之后，训练能打完前几步，但 inductor 的编译子进程池在后面的 step 里段错误退出（日志里有 `SubprocPool unclean exit`）。`env.sh` 里设置 `TORCHINDUCTOR_COMPILE_THREADS=1`，编译改在本进程做。这样 `muse_spark_debugmodel` 的 10 步可以跑完。

`TORCHINDUCTOR_COMPILE_THREADS` 管的是 PyTorch 自己的编译进程池。`torch_npu` 另有 `precompile_thread_num`，默认至少 32，环境变量是 `TORCHNPU_PRECOMPILE_THREADS`。`muse_spark_debugmodel_fsdp2_cp2_ep4` 在 flex attention 反向选块时，这个池子开了 128 个线程去测 `triton_per_fused_flex_attention_backward_0` 的 68 组配置，rank 0 段错误。`env.sh` 里把 `TORCHNPU_PRECOMPILE_THREADS` 也设成 1。

## 9. 上下文并行的 block mask 求和被编成 simt_only

`muse_spark_debugmodel_fsdp2_cp2_ep4` 在 `_cp_shard` 里编译 `create_block_mask`。query 被 CP 切开后，mask 先垫到 128 的倍数，再按 `128 x 128` 的块求和。这个 reduction 走 `NPUTritonKernel`，元数据里的 `npu_kernel_type` 被写成 `simd_simt_mix`。选块时会先生成 `compile_mode="simt_only"`。910B4 不是 Ascend 950，Triton 直接拒绝这个模式，于是整组配置都算失败：

```text
NoTritonConfigsError: No valid triton configs for kernel
triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_3
ValueError: compile_mode="simt_only" is supported only on Ascend 950 targets.
```

第 2 节里那个孤立 `sum` 复现也停在同一句话，但当时训练在更前面就退出了，所以那次不是训练失败原因。这次训练自己走到了这个 kernel。

`torch_npu/_inductor/runtime/triton_heuristics.py` 里，非 Ascend 950 不再为 `simd_simt_mix` 生成 `simt_only` 配置，只保留 `simd_simt_template`。910B 上这个模式走普通 SIMD。用动态长度 256、key 长度 512 的 bool mask 编译同一段 view、permute、sum，结果和 eager 一致。

## 10. 视觉 RoPE 的 complex64 下标

8 卡多模态配方 `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` 第一轮停在：

```text
RuntimeError: index_high_dims_op_api aclnnIndex 161002
DT_COMPLEX64
vision_encoder.py: freqs_cis_GD[sp_perm]
```

`49cbb4d` 修的是 `rope.py` 的 `_reshape_for_broadcast`。视觉编码器在 sparse attention 里直接对 complex64 的频率缓存做下标，不经过那个函数。910B 的 index 没有 complex64。

`torchtitan/models/muse_spark/vision_encoder.py` 在 NPU 上改成先 `view_as_real`，按下标取出实部虚部，再 `view_as_complex`。其它设备仍走原来的下标。

## 11. 反向重算里的空指针占位张量

视觉下标改完之后，训练能打完第 1 步，随后某个 rank 以 SIGSEGV（exit -11）退出。`faulthandler` 的当前线程停在：

```text
torch_remat/_placeholder.py: _make_placeholder_tensor
  placeholder.set_(null_storage, ...)
torchtitan/models/common/linear/linear.py: maybe_gather_tp_input
```

这是 TP 的 remat 区域在反向重算时，给一个没有保存的输出造占位张量。占位张量用 `torch._C._construct_storage_from_data_pointer(0, device, nbytes)`，再 `set_` 成真实形状。CPU 和 CUDA 上这个空指针只在读数据时抛错。910B 的缓存分配器要给每块存储做 32 字节对齐，会碰到这个空指针，进程段错误。单独在一张卡上对同样的形状调用 `set_` 可以成功，所以不是某个 shape 写错了。4 卡那条没有 TP，不会走进 `maybe_gather_tp_input`，所以之前没撞上。

`torch_remat/_placeholder.py` 在 `device.type == "npu"` 时改为 `empty_strided` 分配真实存储，再用原来的 `_set_storage_data_ptr_access_error_msg` 禁止读数据。读到占位张量仍然抛 RuntimeError。其它设备仍用空指针，不占那块显存。

## 12. 8 个 rank 同时给 flex attention 反向选块

占位张量改完之前，还有一种段错误没有 Python 异常：一个 rank 在 `triton_per_fused_flex_attention_backward_0` 的几十组候选里退出（exit -11），其余 rank 被 SIGTERM。同一形状在单卡上前向反向都能算出有限梯度。视觉序列一例是 1460，head_dim 16，2 个本地 head。文本一例是 query `1x4x256x32`、KV `1x2x512x32`。崩的 rank 每次不一样。

尝试过在选块时直接拿最小的 runtime tile。launcher 和 runtime block 对不上，进程以 `realloc(): invalid pointer` / `munmap_chunk(): invalid pointer` 中止。这处改动已撤掉。

留下的改法在 `torch_npu/_inductor/runtime/triton_heuristics.py`：名字里带 `flex_attention` 的 autotune 用 `fcntl.flock` 锁住 `/tmp/npu_flex_attention_autotune.lock`，同一时刻只有一个进程在跑候选。同进程重入用深度计数，避免自己把自己锁死。候选本身仍按原来的计时挑选，不改分块。

## 13. 短视觉序列的 flex decoding 和 100 秒超时

第 1 到第 4 步能打完之后，第 5 步视觉 token 数变成 4。inductor 改走 flex decoding（query 长度小于 128）。rank 0 还在编译，另一个 rank 已经发出 all-gather。第 1 步之后进程组超时是 `train_timeout_seconds=100`。大约 100 秒后：

```text
Watchdog caught collective operation timeout:
WorkHCCL(OpType=ALLGATHER, NumelIn=15856, NumelOut=31712, Timeout(ms)=100000)
```

HCCL watchdog 把还在集合通信里的 rank 打成 SIGABRT（exit -6）。`NumelOut` 是 `NumelIn` 的 2 倍，对应 TP=2 的 all-gather。

两处一起改：

- `torch_npu/_inductor/kernel/flex_attention.py` 的 `_use_flex_decoding` 在非 Ascend 950 上直接返回 false。短序列继续用已经能跑的 mask-in flex attention，不再编一套 decoding 核。
- `tests/integration_tests/models.py` 里这条 8 卡配方把 `comm.train_timeout_seconds` 设为 1800。新的视觉长度仍会重新编译，第 5 步和第 8 步各花了大约一分钟，100 秒盖不住，1800 秒盖得住。

## 14. 词表断言和 FSDP 的 view 输出

`torchtitan/components/loss.py` 里，`_LossParallelCrossEntropy` 的 `torch._assert_async` 在 `torch.compiler.is_compiling()` 时跳过。910B 的 inductor 编不出这个 0 维 device 断言，会停在编译里。eager 时检查还在。

`NaViTEncoder.forward` 末尾，单帧时 `torch.cat` 返回的就是那一帧，FSDP 把它看成 view，就地算子会丢掉 pre-backward all-gather。返回值改成 `torch.cat(outputs).contiguous()`。音频编码器和文本 block 的同类警告这次没有改调用点。

## 15. 8 卡多模态 10 步

上面几处都在的那次运行是 2026-10-28 的 `item2_logs/mm_r12.log`。10 步结束，日志有 `Training completed`。rank 0 的 loss 从 7.66202 到 4.68848，grad_norm 从 9.4633 到 3.0236，逐步数字在 `docs/npu_debugmodel_verification.md`。没有和参考 loss 对过。

## 16. Dist-MoE bf16 在 910B 上的专家计算

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

## 17. 第 4 行多模态配方的训练超时

`muse_spark_debugmodel_mm` 是清单第 4 行：8 卡 `dp_shard=2, tp=2, cp=2, ep=2`，Ulysses。视觉 token 长度一变，flex attention 就要重新编译。第 2 行③那条 EP4、AllGatherKV 的变体已经在 `tests/integration_tests/models.py` 里把 `train_timeout_seconds` 设成 1800，因为编译超过默认 100 秒时，先跑完的 rank 会把 all-gather 拖到 HCCL watchdog 超时。

清单上的这份配方原来没有设这个超时。`torchtitan_recipes/tests/models/muse_spark.py` 的 `muse_spark_debugmodel_mm` 在返回前补上 `config.comm.train_timeout_seconds = 1800`。其它字段没改，融合 SwiGLU override 仍在。

2026-10-30 在 `179.173.2.3` 上，`HCCL_DETERMINISTIC=true`、`MUSE_DEBUG_DETERMINISTIC=1`、`NGPU=8`，连跑两次（`/workspace/det_mm_ep2_r1.log`、`r2.log`）。网格是 `dp_shard=2, cp=2, tp=2, ep=2`。10 步 loss、grad_norm 逐项相同，loss 从 7.69125 到 4.57818，grad_norm 从 9.7405 到 2.7223。逐步数字在 `docs/npu_debugmodel_verification.md` 第 4 行。没有单独打印每个模态、dummy 是否为 0、bank index 和 token 分片。
