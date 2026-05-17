#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train"
LOG_DIR="/home/jongwoopark/lerobot/outputs/train_logs"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_DIR}/move_hiva_train_to_add_disk3_${RUN_STAMP}.log"

mkdir -p "${DEST_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

move_dir_safe() {
  local src="$1"
  local dst="$2"
  local parent
  local tmp

  if [[ ! -d "${src}" ]]; then
    echo "ERROR: missing source directory: ${src}" >&2
    exit 2
  fi
  if [[ -e "${dst}" ]]; then
    echo "ERROR: destination already exists: ${dst}" >&2
    exit 2
  fi

  parent="$(dirname "${dst}")"
  tmp="${dst}.partial_${RUN_STAMP}"
  mkdir -p "${parent}"

  echo "===== $(date) moving ====="
  echo "SRC=${src}"
  echo "DST=${dst}"
  echo "TMP=${tmp}"
  du -sh "${src}" || true

  rsync -a --info=stats2 "${src}/" "${tmp}/"
  mv "${tmp}" "${dst}"
  rm -rf "${src}"

  echo "===== $(date) moved to ${dst} ====="
  du -sh "${dst}" || true
}

echo "===== HiVA train move to add_disk3 started at $(date) ====="
echo "DEST_ROOT=${DEST_ROOT}"
echo "LOG_PATH=${LOG_PATH}"
df -h /nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark /nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark

move_dir_safe \
  "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/finetuning_stage0" \
  "${DEST_ROOT}/finetuning_stage0"

move_dir_safe \
  "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/HiVA_train/Best_Models" \
  "${DEST_ROOT}/Best_Models"

move_dir_safe \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203" \
  "${DEST_ROOT}/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p5_steps5000_20260515_233203"

move_dir_safe \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520" \
  "${DEST_ROOT}/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520"

move_dir_safe \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459" \
  "${DEST_ROOT}/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p7_f15_bigflow_b128_g4_s0p5_steps5000_nores_nodl_20260515_044459"

move_dir_safe \
  "/home/jongwoopark/lerobot/outputs/train/smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_888072651_pid2567557" \
  "${DEST_ROOT}/finetuning_stage0/smolvla_original_bigcornea_s0p5_b64_g8_20260515_223843_888072651_pid2567557"

echo "===== HiVA train move to add_disk3 finished at $(date) ====="
df -h /nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark /nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark
