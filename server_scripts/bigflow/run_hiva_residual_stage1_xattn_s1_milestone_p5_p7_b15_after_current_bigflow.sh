#!/usr/bin/env bash
set -euo pipefail

# Follow-on queue for the LP-MT HiVA P5/P7 S=1 milestone jobs, using the
# same hyperparameters as the current residual stage-1 xattn queue but with
# HIVA_RESIDUAL_NUM_BLOCKS=15 instead of the default 9.
#
# This script waits for the current P5/P7 tmux queue to finish before starting
# so it can be launched immediately without competing for GPUs 4,5,6,7.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

RUN_GROUP_STAMP="${RUN_GROUP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WAIT_TMUX_SESSION="${WAIT_TMUX_SESSION:-hiva_s1_milestone_p5_p7_gpu4567_20260513_105200}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/stage1_xattn_s1_milestone_p5_p7_b15_after_current_bigflow_${RUN_GROUP_STAMP}.queue.log}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date)] Queue log: ${QUEUE_LOG}"
echo "[$(date)] Waiting for tmux session '${WAIT_TMUX_SESSION}' to finish before launching b15 jobs."
while tmux has-session -t "${WAIT_TMUX_SESSION}" 2>/dev/null; do
  sleep 300
done
echo "[$(date)] Wait target '${WAIT_TMUX_SESSION}' is no longer active. Launching b15 jobs."

COMMON_ENV=(
  GPU_IDS=4,5,6,7
  NUM_GPUS=4
  NUM_PROCESSES=4
  BATCH_PER_GPU=1024
  BATCH_SIZE=1024
  S=1
  WANDB_ENABLE=true
  HIVA_RESIDUAL_NUM_BLOCKS=15
  HIVA_RESIDUAL_SCALE_TR=3.0
  HIVA_RESIDUAL_SCALE_ROT=3.0
  HIVA_RESIDUAL_SCALE_GRIP=0.3
  HIVA_DECODED_ACTION_LOSS_WEIGHT=1.0
  HIVA_DECODED_TR_LOSS_BETA=0.1
  HIVA_DECODED_ROT_LOSS_BETA=0.05
  HIVA_DECODED_GRIP_LOSS_BETA=0.1
)

run_one() {
  local degree="$1"
  local init_base="$2"
  local sidecar="$3"
  local summary="$4"

  local run_stamp="${RUN_GROUP_STAMP}_p${degree}_b15"
  local run_name="smolvla_hiva_coeff_lpmt_stage1_xattn_s1_milestone_b15_v5_d4_6_10_p${degree}_tr3_rot3_grip0p3_daw1_b1024_g4_s1_${RUN_GROUP_STAMP}"

  echo "===== $(date) launching P${degree} b15 stage-1 xattn S=1 ====="
  echo "INIT_HIVA_BASE=${init_base}"
  echo "SIDECAR=${sidecar}"
  echo "SIDECAR_SUMMARY=${summary}"
  echo "RUN_NAME=${run_name}"

  env "${COMMON_ENV[@]}" \
    RUN_STAMP="${run_stamp}" \
    RUN_NAME="${run_name}" \
    INIT_HIVA_BASE="${init_base}" \
    SIDECAR="${sidecar}" \
    SIDECAR_SUMMARY="${summary}" \
    HIVA_DEGREE="${degree}" \
    bash "${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_s1_milestone_ckpts_bigflow.sh"

  echo "===== $(date) finished P${degree} b15 stage-1 xattn S=1 ====="
}

run_one 5 \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_v5_d4_6_10_coeffpool_full_ce_mean_k10_p5_f15_bigflow_b128_s2_nores_nodl_20260511_184643/checkpoints/last/pretrained_model" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.parquet" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.summary.json"

run_one 7 \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_s2_nores_nodl_20260511_184643/checkpoints/last/pretrained_model" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.parquet" \
  "/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p7_f15_canonical_lp_mt.summary.json"

echo "All P5/P7 b15 S=1 milestone jobs completed at $(date)"
