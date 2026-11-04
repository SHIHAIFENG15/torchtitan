# A5 跑通结果（Muse Spark 8 配置 · 2026-11-02/03 · 更新 2026-11-04）

按配置写入验收正文：`docs/npu_debugmodel_verification.md`（各配置「跑通」表的 A2 / A5 行）。本文件保留 A5 环境、队列与阻塞证据；**无额外代码适配**（未用 910B flex 补丁）。

## 结论摘要（2026-11-04 更新）

- A5 机器：`141.61.49.192`，**8 × Ascend 950PR**（131GB HBM），CANN 9.2.0，openEuler 24.03 aarch64
- **4 卡配置 4/4 全部完成**（2026-11-04，`Training completed` 已入日志）：
  - ✅ `muse_spark_debugmodel`（R1）
  - ✅ `muse_spark_debugmodel_with_cp`（R2）
  - ✅ `muse_spark_debugmodel_fsdp2_cp2_ep4`（R3）
  - ✅ `muse_spark_debugmodel_dist_moe_bf16`（R5）
- ✅ `dist_moe_mxfp8`（R6）预期配置期失败（与 A2 一致，按文档记录）
- ⏸ 三个 8 卡配置（R4/R7/R8）**被 crm 用户的 vLLM 常驻服务阻塞**：该服务占 NPU7 ~118GB 并持有设备上下文，8 卡 HCCL 通信域初始化必然失败（`hcclCommInitRootInfoConfig error 4`，已用最小 8 进程 HCCL 测试独立复现，测试脚本 `/home/root_a5/hccl_test.sh`）
- 期间机器经历了一次**驱动升级**（950 驱动 25.6.rc2 + 固件 9.2.0.105，由 crm 用户执行）：升级窗口内设备节点消失导致全部运行失败；升级完成后 4 卡配置立即跑通
- 队列持续自愈重试中（`/home/root_a5/a5_queue5.sh`）；vLLM 停止后 8 卡配置自动补跑

## 环境准备（全部就绪）

| 项 | 值 |
| --- | --- |
| 仓库 | `/home/root_a5/torchtitan`（`npu-muse-spark-debug-r04`，随远端更新） |
| venv | `/home/torchtitan_env/repos/torchtitan-ascend/.venv`（python3.12 链接已修复→3.12.3） |
| torch / torch_npu | `2.15.0.dev20261005+cpu` / `2.15.0.dev20261005+git2c8682d`（A5 文档目标 `git14dc54e`，本机为 `git2c8682d`，与 `torchtitan-ascend/INSTALL.md` pinned 一致） |
| triton-ascend / grain | 3.6.0 / 0.2.18 |
| 910B 补丁 | **未采用**：apply 后 hunks 全部成功但 MD5 与 A2 基准不一致（`2ef66332…` vs `659bf1db…`），且当时（驱动故障窗口）运行触发 `ERR99999`；已 restore。修复索引 #8 指明「非 950 才关 decoding/mask-out」，950PR 属 950 系，原生路径更合适 |
| UltraData JSONL | `/home/root_a5/General_Agent_part-01-of-32.jsonl`（1,575,113,443 B，sha256 `656b3d83…` 与 GPU 侧一致；从 node-5-15 断点续传中转） |
| 队列 | `/home/root_a5/a5_queue5.sh`（自愈重试：每配置最多 8–12 次；4 卡任务只等卡 0–3，8 卡任务等全部空闲；`MUSE_SPARK_ULTRADATA_JSONL_PATTERN` 已指本地数据） |

## 运行结果（最终）

| # | CONFIG | NGPU | 状态 | 成功日志 |
|---|---|---|---|---|
| 1 | `muse_spark_debugmodel` | 4 | ✅ Training completed | `logs/run_debugmodel_a5.log` |
| 2 | `muse_spark_debugmodel_with_cp` | 4 | ✅ Training completed | `logs/run_with_cp_a5.log` |
| 3 | `muse_spark_debugmodel_fsdp2_cp2_ep4` | 4 | ✅ Training completed | `logs/run_fsdp2_a5.log` |
| 5 | `muse_spark_debugmodel_dist_moe_bf16` | 4 | ✅ Training completed | `logs/run_dist_moe_bf16_a5.log` |
| 6 | `muse_spark_debugmodel_dist_moe_mxfp8` | 4 | 预期配置期失败（A2 同） | `logs/run_dist_moe_mxfp8_a5.log` |
| 4 | `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` | 8 | ⏸ 等待 vLLM 停止 | `logs/run_mm_fsdp2_a5.log`（历史尝试） |
| 7 | `muse_spark_debugmodel_mm` | 8 | ⏸ 同上 | `logs/run_debugmodel_mm_a5.log`（历史尝试） |
| 8 | `muse_spark_debugmodel_ultradata_lora_sft` | 8 | ⏸ 同上（数据已就位） | `logs/run_ultradata_a5.log`（历史尝试） |

日志目录：`/home/root_a5/torchtitan/logs/run_*_a5.log`（每次尝试覆盖写；历史尝试的失败信息同时进 `/home/root_a5/a5q5.log`）。

## 占用与阻塞证据

- 1 秒粒度进程监视：`/home/root_a5/monitor.log`
- 外部终止记录：`/home/root_a5/a5q5.log` 及 `/home/torchtitan_env/a5.log` 中 `Signal 15 (SIGTERM) received by PID …`
- 占用方：`torchbench.py`（zjx，lazy_fusion/cm3leon 对比基准循环，已停止）、`vllm serve Qwen3-30B` → 重启为 `vllm_serve_qwen38_27b.sh`（NPU7，crm 用户）、zfr 的 `megamoe_a5` 分析脚本组
- 8 卡阻塞复现：`/home/root_a5/hccl_test.sh`（最小 8 进程 HCCL 测试，`hcclCommInitRootInfoConfig error 4`）

## 后续

三个 8 卡配置需要 crm 用户的 vLLM 服务停止（或迁移）后由队列自动补跑；届时回填本文件「状态」列。A5 与 A2 的环境差异（950PR vs 910B4、torch_npu build、补丁不适用、驱动升级）已如上记录，供 A2 维护者参考。
