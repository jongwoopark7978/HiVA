#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-3}"
POLICY_PATH="${POLICY_PATH:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520/checkpoints/007500/pretrained_model}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
OUT_ROOT="${REPO_ROOT}/outputs/eval/smoke_async_full_gpu${GPU_ID}_s0p25_ckpt_007500_50eps_${TIMESTAMP}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/jongwoo_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"

RENAME_MAP='{"observation.images.image":"observation.images.agentview","observation.images.image2":"observation.images.wrist"}'

gpu_mem_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' '
}

run_one() {
  local bs="$1"
  local label="bs${bs}"
  local out_dir="${OUT_ROOT}/${label}/libero_object_taskids__0_"
  local log="${LOG_DIR}/smoke_async_full_gpu${GPU_ID}_s0p25_ckpt_007500_${label}_50eps_${TIMESTAMP}.log"
  local mem_log="${LOG_DIR}/smoke_async_full_gpu${GPU_ID}_s0p25_ckpt_007500_${label}_50eps_${TIMESTAMP}.mem.log"
  local summary="${OUT_ROOT}/${label}/smoke_result.txt"
  mkdir -p "${out_dir}" "$(dirname "${summary}")"

  local baseline start end status peak
  baseline="$(gpu_mem_mib)"
  start="$(date +%s)"
  echo "===== ${label} started at $(date) baseline_gpu_mem_mib=${baseline} =====" | tee -a "${summary}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.num_steps=10 \
    --policy.chunk_size=15 \
    --policy.n_action_steps=10 \
    --policy.use_duration_head=false \
    --policy.hiva_coeff_sidecar_path="${HIVA_COEFF_SIDECAR}" \
    --policy.hiva_coeff_sidecar_summary_path="${HIVA_COEFF_SUMMARY}" \
    --policy.hiva_duration_execution_map= \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids="[0]" \
    --env.control_mode=relative \
    --env.max_parallel_tasks=1 \
    --eval.batch_size="${bs}" \
    --eval.n_episodes=50 \
    --eval.use_async_envs=true \
    --eval.max_episodes_rendered=1 \
    --rename_map="${RENAME_MAP}" \
    --output_dir="${out_dir}" \
    --job_name="smoke_async_full_gpu${GPU_ID}_s0p25_ckpt_007500_${label}_${TIMESTAMP}" \
    > "${log}" 2>&1 &
  local pid="$!"

  peak="${baseline}"
  while kill -0 "${pid}" 2>/dev/null; do
    local used
    used="$(gpu_mem_mib || echo 0)"
    printf '%s gpu%s_mem_mib=%s\n' "$(date '+%F %T')" "${GPU_ID}" "${used}" >> "${mem_log}"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used > peak )); then
      peak="${used}"
    fi
    sleep 2
  done

  wait "${pid}"
  status="$?"
  end="$(date +%s)"

  local delta=$((peak - baseline))
  {
    echo "status=${status}"
    echo "elapsed_seconds=$((end - start))"
    echo "baseline_gpu_mem_mib=${baseline}"
    echo "peak_gpu_mem_mib=${peak}"
    echo "incremental_peak_gpu_mem_mib=${delta}"
    echo "log=${log}"
    echo "mem_log=${mem_log}"
    echo "output_dir=${out_dir}"
    echo "===== ${label} finished at $(date) ====="
  } | tee -a "${summary}"

  return "${status}"
}

main() {
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "POLICY_PATH=${POLICY_PATH}"
  echo "HIVA_COEFF_SIDECAR=${HIVA_COEFF_SIDECAR}"
  echo "HIVA_COEFF_SUMMARY=${HIVA_COEFF_SUMMARY}"

  local status=0
  run_one 10 || status="$?"
  if [[ "${status}" -eq 0 ]]; then
    run_one 25 || status="$?"
  fi
  exit "${status}"
}

main "$@"
