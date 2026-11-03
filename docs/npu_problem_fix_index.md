# Muse Spark NPU：问题 → 改动对照表

分支：`npu-muse-spark-debug-r04`。只列「跑任务时踩过、并有对应改动」的项；验收指标不达标（精度/线性度/内存对照）见 `docs/npu_gpu_acceptance_report.md`，不进本表。

## 环境（A2）

本对照表对应的实测环境记为 **A2**：

| 项 | 值 |
| --- | --- |
| 环境代号 | **A2** |
| 机器 / 容器 | `179.173.2.3` / `npu-y00839695-torchtitan-20261028`（多模态也曾在 `179.173.2.2`） |
| 卡 | 8 × Ascend **910B4** |
| 镜像 | CANN 9.2.0 910B Ubuntu22.04 py3.12 |
| torch / torch_npu | `2.15.0+cpu` / `2.15.0.dev20261005+git14dc54e` |
| 训练入口 | `ENTRYPOINT=torchtitan_ascend.train`（勿用裸 `torchtitan.train`） |
| 必打补丁 | `docs/npu_910b_patches/` + ascend Inductor/remat patches |
| 编译线程 | `TORCHINDUCTOR_COMPILE_THREADS=1`，`TORCHNPU_PRECOMPILE_THREADS=1` |

换机复现：检出本分支 → 同版本 wheel → `apply_npu_910b_patches.sh` → `source env_npu_910b.sh` → ascend 入口。细节见 `docs/npu_pta_fixes.md`、`docs/npu_910b_patches/README.md`。

---

## 对照表

| # | 现象 / 失败 | 根因（一句话） | 改动落点 | 代表 commit / 文档 |
| --- | --- | --- | --- | --- |
| 1 | flex 注册失败 `get_float32_precision` / `ModuleNotFoundError` | 旧 `torch_npu` 与 PyTorch 2.15 flex API 不匹配 | 换 `torch_npu` wheel `…git14dc54e` | `npu_pta_fixes.md` §1 |
| 2 | Inductor `SympifyError: [4, 4]` | NPU `NumelList` 被当成 list 送给 sympy | 当时本地改 sizevars；现行靠新 wheel + ascend | `npu_pta_fixes.md` §2 |
| 3 | RoPE `aclnnIndex` / complex64 不支持 | 910B index 无 complex64 | `rope.py` NPU 路径 `view_as_real` 再下标；视觉 RoPE 另绕开 | `f2b8b9d6` 一带；verification 第 2③ |
| 4 | SwiGLU / Helion CUDA 路径在 NPU 挂 | 融合 override 绑 CUDA | 非 CUDA 回退 `silu(gate)*up`；with_cp 清空 `override.imports` | 同 NPU 适配提交 |
| 5 | remat 反向段错误（空指针 storage） | NPU 上 null storage 占位不可用 | **torchtitan-ascend** `patches/storage.py`；本仓不再维护手工改 | `b3135c09`；`npu_pta_fixes.md` 文首 |
| 6 | `make_reduction(..., strict_reduction=...)` 编译挂 | Inductor API 与 torch_npu 不一致 | **torchtitan-ascend** `patches/inductor.py` | 同上 |
| 7 | 8 卡 flex 反向多 rank 选块段错误 | 并行选块竞态 | flex 反向分块默认 16；选块串行；autotune 文件锁 | `docs/npu_910b_patches/`（`5a0f0ddf`） |
| 8 | 非 950 短视觉序列重编 → HCCL watchdog | flex decoding / mask-out 另编核，超时 | 非 950 关 decoding/mask-out；`train_timeout_seconds=1800` | patches + `103706ec` / `f2d26e1e` |
| 9 | 编译进程池 + flex 段错误 | 多线程编译不安全 | `COMPILE_THREADS=1` / `PRECOMPILE_THREADS=1` | `env_npu_910b.sh` |
| 10 | loss 编译期 `_assert_async` 卡住 | 910B inductor 对 0 维 device assert 代码生成挂 | 配方/loss 路径编译期跳过 | `103706ec` 一带 |
| 11 | FSDP 丢掉视觉输出 pre-backward all-gather | 单帧 `cat` 得到 view | 视觉编码器返回前 `contiguous()` | 同上 |
| 12 | 全局 `init_timeout=1800` 过猛 | 启动与训练超时混用 | `init` 保持 300；1800 只写配方 `train_timeout` | `d0f5101f` |
| 13 | Dist-MoE 官方包要 SM100，NPU 过不了设备检查 | CuTe annex 不在 / 设备类型拒绝 | `dist_moe/npu.py`（grouped_matmul + swiglu）；runtime 跳过 SM100；NPU 走 all-to-all 内部接口避免嵌套 remat 区 | **`fc3aa0cf`（已入库）** |
| 14 | Dist-MoE 导入 / FSDP `MixedPrecisionPolicy` 在现装 torch 失败 | 可选依赖探测不够软 | 导入失败时跳过 + 探测 policy | `f1c0e844` / `986e693c` |
| 15 | UltraData 配方拉 huggingface 被重置 | 容器出网受限 | 读本地 `MUSE_SPARK_ULTRADATA_JSONL` | `f99ea981` |
| 16 | `ultradata_perf_dp1` 4 卡 `EP8` ValueError | `ep` 必须整除 `dp*cp*tp` | 基函数加 `expert_parallel_degree`；线性度两侧 **EP4** | `b188a02c` |
| 17 | 用 `torchtitan.train` 跑 1000 步：debugmodel 第 5 步段错误；CP 行撞 `make_reduction` | 未走 ascend 补丁入口 | 一律 `torchtitan_ascend.train` | acceptance 报告末尾说明 |
| 18 | 8 卡首步 / 视觉变长仍 watchdog | 默认 100s/300s 不够编核 | 相关配方 `train_timeout_seconds=1800`（必要时 init 也 1800） | perf 配方；`gpu_baseline_perf_mem.md` |

---

## 未做 / 不可做（无「修复提交」）

| 项 | 说明 |
| --- | --- |
| Dist-MoE **mxfp8** / 压缩 MoE | 配方未改、未跑；配置期就要 `dist_moe` / torchao |
| Dist-MoE **GPU 对照** | A100 SM80，官方要 SM100 → 精度/性能/内存「不可做」 |
| flex 整层与未改分块对拍 | 分块 16 是稳定性选择，未做数值对拍 |
| FSDP view 警告追根 | 有日志警告，未单独立项修 |

---

## 相关文档

| 文档 | 用途 |
| --- | --- |
| `docs/npu_pta_fixes.md` | 失败原因长文与补丁边界 |
| `docs/npu_910b_patches/` | flex/heuristics 可复用补丁 |
| `docs/npu_debugmodel_verification.md` | 按清单行的跑通数据与验收 |
| `docs/npu_gpu_acceptance_report.md` | NPU↔GPU 四指标闭环与不达标原因 |
| `docs/muse_acceptance_report_row1_2/` | 第 1–5 行验收报告 + 图 |
