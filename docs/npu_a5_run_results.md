# A5 跑通结果（Muse Spark 8 配置 · 2026-11-02/03）

## 结论摘要

- A5 机器：`141.61.49.192`，**8 × Ascend 950PR**（131GB HBM），CANN 9.2.0，openEuler 24.03 aarch64
- 该机**并非独占**：vLLM 常驻推理服务（占 NPU7 ~118GB）、zjx 的 torchbench 基准循环（`--accuracy --train --iterations 50`，周期性清理 NPU 进程）、zfr 的 MegaMoE A5 分析套件三者持续占用
- 我方环境已完整就绪；**8 个配置 0 个完成**——所有运行在启动后 60–120 秒内被外部 SIGTERM（1 秒粒度监视器证据：`/home/root_a5/monitor.log`，a5.log 中 `Signal 15 (SIGTERM) received by PID 1495627/28/29`）
- 早期 queue3 阶段 R1/R2/R3 曾在干净窗口完成（`Training completed`），但完成日志随第三方清理被删（`/home/torchtitan_env/repos/torchtitan_a5` 整目录被移除）
- 队列已重启为自愈模式持续重试（`/home/root_a5/a5_queue5.sh`）；一旦出现独占窗口即可完成全部 8 配置并更新本文件

## 环境准备（全部就绪）

| 项 | 值 |
| --- | --- |
| 仓库 | `/home/root_a5/torchtitan` @ `14abeeb2`（`npu-muse-spark-debug-r04` tip，含 A5 文档） |
| venv | `/home/torchtitan_env/repos/torchtitan-ascend/.venv`（python3.12 链接已修复→3.12.3） |
| torch / torch_npu | `2.15.0.dev20261005+cpu` / `2.15.0.dev20261005+git2c8682d`（A5 文档目标 `git14dc54e`，本机为 `git2c8682d`，与 `torchtitan-ascend/INSTALL.md` pinned 一致） |
| triton-ascend / grain | 3.6.0 / 0.2.18 |
| 910B 补丁 | **未采用**：apply 后 hunks 全部成功但 MD5 与 A2 基准不一致（`2ef66332…` vs `659bf1db…`），且运行触发 `ERR99999` 编译崩溃；已 restore。修复索引 #8 指明「非 950 才关 decoding/mask-out」，950PR 属 950 系，原生路径更合适 |
| UltraData JSONL | `/home/root_a5/General_Agent_part-01-of-32.jsonl`（1,575,113,443 B，sha256 `656b3d83…` 与 GPU 侧一致；从 node-5-15 分段中转） |
| 队列 | `/home/root_a5/a5_queue5.sh`（自愈重试：每配置最多 8–12 次；4 卡任务只等卡 0–3，8 卡任务等全部空闲；`MUSE_SPARK_ULTRADATA_JSONL_PATTERN` 已指本地数据） |

## 运行尝试记录

| # | CONFIG | NGPU | 状态 | 备注 |
|---|---|---|---|---|
| 1 | `muse_spark_debugmodel` | 4 | ✗ 被外部 SIGTERM（queue3 曾完成，日志被第三方清理删除） | 编译期被杀；queue5 重试 12 次 |
| 2 | `muse_spark_debugmodel_with_cp` | 4 | ✗ 同上 | |
| 3 | `muse_spark_debugmodel_fsdp2_cp2_ep4` | 4 | ✗ 同上 | |
| 4 | `muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4` | 8 | 未跑 | 等 NPU7 空闲（vLLM 118GB 常驻） |
| 5 | `muse_spark_debugmodel_dist_moe_bf16` | 4 | 重试中 | |
| 6 | `muse_spark_debugmodel_dist_moe_mxfp8` | 4 | 预期配置期失败（A2 同），未跑到 | 到达后按文档记「预期跳过/失败」 |
| 7 | `muse_spark_debugmodel_mm` | 8 | 未跑（同 4） | |
| 8 | `muse_spark_debugmodel_ultradata_lora_sft` | 8 | 未跑（同 4；数据已就位） | |

日志目录：`/home/root_a5/torchtitan/logs/run_*_a5.log`（每次尝试覆盖写；历史尝试的失败信息同时进 `/home/root_a5/a5q5.log`）。

## 占用证据

- 1 秒粒度进程监视：`/home/root_a5/monitor.log`
- 外部终止记录：`/home/root_a5/a5q5.log` 及 `/home/torchtitan_env/a5.log` 中 `Signal 15 (SIGTERM) received by PID …`
- 占用方：`torchbench.py`（zjx，lazy_fusion/cm3leon 对比基准循环）、`vllm serve Qwen3-30B`（NPU7，3.5 天）、zfr 的 `megamoe_a5` 分析脚本组

## 后续

需要管理员协调独占窗口（或等三方活动停歇）。窗口出现后队列自动完成全部 8 配置，并回填本文件的「状态」列与成功日志路径。A5 与 A2 的环境差异（950PR vs 910B4、torch_npu build、补丁不适用）已如上记录，供 A2 维护者参考。
