#!/usr/bin/env bash
# Queue remaining Muse Spark NPU perf/mem recipes. Run inside the training
# container after `source /workspace/env.sh`.
set -u
cd /workspace/dev/torchtitan
mkdir -p logs
export HCCL_DETERMINISTIC="${HCCL_DETERMINISTIC:-true}"
export MUSE_DEBUG_DETERMINISTIC="${MUSE_DEBUG_DETERMINISTIC:-1}"
export ENTRYPOINT="${ENTRYPOINT:-torchtitan_ascend.train}"
STATUS_LOG=logs/npu_perf_mem_queue_status.txt
: >"$STATUS_LOG"

run_one() {
  local ngpu="$1"
  local module="$2"
  local config="$3"
  local log="$4"
  echo "=== START ${config} NGPU=${ngpu} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$STATUS_LOG"
  if NGPU="$ngpu" MODULE="$module" CONFIG="$config" ./run_train.sh >"$log" 2>&1; then
    echo "=== OK ${config} $(date -u +%Y-%m-%dT%H:%M:%SZ) log=${log} ===" | tee -a "$STATUS_LOG"
  else
    local ec=$?
    echo "=== FAIL ${config} exit=${ec} $(date -u +%Y-%m-%dT%H:%M:%SZ) log=${log} ===" | tee -a "$STATUS_LOG"
  fi
}

RECIPES_SPARK=torchtitan_recipes.tests.models.muse_spark
RECIPES_IT=tests.integration_tests.models

# Fast mem first, then 4-card text, then 8-card text, then mm, then ultradata.
run_one 1 "$RECIPES_IT" muse_spark_debugmodel_fsdp2_cp2_ep4_mem_1device \
  logs/fsdp2_mem_1device_npu.log
run_one 1 "$RECIPES_SPARK" muse_spark_debugmodel_dist_moe_bf16_mem_1device \
  logs/dist_moe_bf16_mem_1device_npu.log
run_one 1 "$RECIPES_IT" muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_mem_1device \
  logs/mm_fsdp2_mem_1device_npu.log
run_one 1 "$RECIPES_SPARK" muse_spark_debugmodel_mm_mem_1device \
  logs/mm_mem_1device_npu.log

run_one 4 "$RECIPES_IT" muse_spark_debugmodel_fsdp2_cp2_ep4_perf_dp2 \
  logs/fsdp2_perf_dp2_npu.log
run_one 8 "$RECIPES_IT" muse_spark_debugmodel_fsdp2_cp2_ep4_perf_dp4 \
  logs/fsdp2_perf_dp4_npu.log

run_one 4 "$RECIPES_SPARK" muse_spark_debugmodel_dist_moe_bf16_perf_dp2 \
  logs/dist_moe_bf16_perf_dp2_npu.log
run_one 8 "$RECIPES_SPARK" muse_spark_debugmodel_dist_moe_bf16_perf_dp4 \
  logs/dist_moe_bf16_perf_dp4_npu.log

run_one 4 "$RECIPES_IT" muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_perf_dp1 \
  logs/mm_fsdp2_perf_dp1_npu.log
run_one 8 "$RECIPES_IT" muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4_perf_dp2 \
  logs/mm_fsdp2_perf_dp2_npu.log

run_one 4 "$RECIPES_SPARK" muse_spark_debugmodel_mm_perf_dp1 \
  logs/mm_perf_dp1_npu.log
run_one 8 "$RECIPES_SPARK" muse_spark_debugmodel_mm_perf_dp2 \
  logs/mm_perf_dp2_npu.log

run_one 1 "$RECIPES_SPARK" muse_spark_debugmodel_ultradata_lora_sft_mem_1device \
  logs/ultradata_mem_1device_npu.log
run_one 4 "$RECIPES_SPARK" muse_spark_debugmodel_ultradata_lora_sft_perf_dp1 \
  logs/ultradata_perf_dp1_npu.log
run_one 8 "$RECIPES_SPARK" muse_spark_debugmodel_ultradata_lora_sft_perf_dp2 \
  logs/ultradata_perf_dp2_npu.log

echo "=== QUEUE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$STATUS_LOG"
