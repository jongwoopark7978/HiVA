#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_SCRIPT="${SCRIPT_DIR}/run_hiva_residual_stage1_xattn_bigcornea.sh"
RUN_STAMP="${RUN_STAMP:-20260512_112407}"
QUEUE_LOG="${REPO_ROOT}/outputs/train_logs/run_hiva_residual_stage1_xattn_b9_remaining_${RUN_STAMP}_$(date +%Y%m%d_%H%M%S).queue.log"

mkdir -p "$(dirname "${QUEUE_LOG}")"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

fmt_float() {
  printf '%s' "$1" | tr '.' 'p'
}

run_job() {
  local label="$1"
  local scale="$2"
  local daw="$3"
  local tr_beta="$4"
  local rot_beta="$5"
  local grip_beta="$6"

  local scale_tag daw_tag tr_tag rot_tag grip_tag run_name log_file
  scale_tag="$(fmt_float "${scale}")"
  daw_tag="$(fmt_float "${daw}")"
  tr_tag="$(fmt_float "${tr_beta}")"
  rot_tag="$(fmt_float "${rot_beta}")"
  grip_tag="$(fmt_float "${grip_beta}")"
  run_name="smolvla_hiva_coeff_lpmt_residual_stage1_xattn_b9_v5_d4_6_10_k10_f15_scale${scale_tag}_daw${daw_tag}_trb${tr_tag}_rotb${rot_tag}_gripb${grip_tag}_b128_s2_${RUN_STAMP}_${label}_scale${scale_tag}_daw${daw_tag}_trb${tr_tag}_rotb${rot_tag}_gripb${grip_tag}"
  log_file="${REPO_ROOT}/outputs/train_logs/run_hiva_residual_stage1_xattn_b9_${RUN_STAMP}_${label}_scale${scale_tag}_daw${daw_tag}_trb${tr_tag}_rotb${rot_tag}_gripb${grip_tag}.log"

  printf '===== %s starting %s scale=%s daw=%s tr_beta=%s rot_beta=%s grip_beta=%s =====\n' \
    "$(date)" "${label}" "${scale}" "${daw}" "${tr_beta}" "${rot_beta}" "${grip_beta}"

  set +e
  RUN_STAMP="${RUN_STAMP}" \
  HIVA_SCRIPT_SNAPSHOT=1 \
  HIVA_ORIGINAL_SCRIPT_PATH="${BASE_SCRIPT}" \
  RUN_NAME="${run_name}" \
  LOG_FILE="${log_file}" \
  HIVA_RESIDUAL_SCALE_MULT="${scale}" \
  HIVA_DECODED_ACTION_LOSS_WEIGHT="${daw}" \
  HIVA_DECODED_TR_LOSS_BETA="${tr_beta}" \
  HIVA_DECODED_ROT_LOSS_BETA="${rot_beta}" \
  HIVA_DECODED_GRIP_LOSS_BETA="${grip_beta}" \
  "${BASE_SCRIPT}"
  local status=$?
  set -e
  printf '===== %s status %s: %s =====\n' "$(date)" "${label}" "${status}"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi

  printf '===== %s finished %s =====\n' "$(date)" "${label}"
}

run_job job1_scale6p0 6.0 1.0 0.1 0.1 0.1
run_job job2_rotbeta0p03 3.0 1.0 0.1 0.03 0.1
run_job job2_rotbeta0p05 3.0 1.0 0.1 0.05 0.1
run_job job2_rotbeta0p07 3.0 1.0 0.1 0.07 0.1

printf 'Queue log: %s\n' "${QUEUE_LOG}"
