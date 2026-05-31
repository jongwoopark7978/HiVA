#!/usr/bin/env bash
set -euo pipefail

# Resume the two stage-0 LP-MT HiVA training jobs that were running on
# BigBrain GPUs 6-9 before shutdown.
#
# Jobs resumed:
#   - GPUs 6,7: S=0.25, degree 3, b256, total steps 10000
#   - GPUs 8,9: S=0.5, degree 5/P5, b256, total steps 5000
#
# Usage:
#   bash server_scripts/bigbrain/resume_current_stage0_gpu6_9_after_shutdown_bigbrain.sh
#   DRY_RUN=1 bash server_scripts/bigbrain/resume_current_stage0_gpu6_9_after_shutdown_bigbrain.sh
#   ONLY=s0p25 bash server_scripts/bigbrain/resume_current_stage0_gpu6_9_after_shutdown_bigbrain.sh
#   ONLY=p5 bash server_scripts/bigbrain/resume_current_stage0_gpu6_9_after_shutdown_bigbrain.sh

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/outputs/train_logs"
mkdir -p "${LOG_DIR}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigbrain/add_disk0/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
DATA_REPO_ID="${DATA_REPO_ID:-local/libero_lerobot_v3_lerobotkeys}"
INIT_SMOLVLA="${INIT_SMOLVLA:-/home/jongwoopark/lerobot/smolvla_libero}"

RENAME_MAP_JSON='{"observation.images.agentview":"observation.images.image","observation.images.wrist":"observation.images.image2"}'
TASKS="${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"

check_required_paths() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "ERROR: required path does not exist: ${path}" >&2
      exit 2
    fi
  done
}

latest_resume_checkpoint() {
  local output_dir="$1"
  find "${output_dir}/checkpoints" -mindepth 3 -maxdepth 3 -path '*/training_state/training_step.json' \
    | sed -E 's#/training_state/training_step.json$##' \
    | sort \
    | tail -1
}

check_resume_dir() {
  local output_dir="$1"
  if [[ ! -d "${output_dir}" ]]; then
    echo "ERROR: output dir does not exist, cannot resume: ${output_dir}" >&2
    exit 2
  fi
  if [[ ! -d "${output_dir}/checkpoints" ]]; then
    echo "ERROR: no checkpoints dir, cannot resume: ${output_dir}/checkpoints" >&2
    exit 2
  fi
  local latest
  latest="$(latest_resume_checkpoint "${output_dir}")"
  if [[ -z "${latest}" ]]; then
    echo "ERROR: no checkpoint with training_state/training_step.json under ${output_dir}/checkpoints" >&2
    exit 2
  fi
  echo "${latest}"
}

common_train_args() {
  local init_smolvla="$1"
  local sidecar="$2"
  local sidecar_summary="$3"
  local hiva_degree="$4"
  local hiva_duration_sigma="$5"
  local batch_size="$6"
  local total_steps="$7"
  local save_steps="$8"
  local scheduler_warmup_steps="$9"
  local scheduler_decay_steps="${10}"
  local output_dir="${11}"
  local job_name="${12}"

  TRAIN_ARGS=(
    --policy.type=smolvla_hiva_coeff
    --policy.push_to_hub=false
    --policy.load_vlm_weights=false
    --policy.init_smolvla_checkpoint_path="${init_smolvla}"
    --policy.hiva_coeff_sidecar_path="${sidecar}"
    --policy.hiva_coeff_sidecar_summary_path="${sidecar_summary}"
    --policy.hiva_duration_classes='[4,6,10]'
    --policy.hiva_dmax=10
    --policy.hiva_fit_horizon=15
    --policy.hiva_k=10
    --policy.hiva_degree="${hiva_degree}"
    --policy.hiva_degree_tr="${hiva_degree}"
    --policy.hiva_degree_rot="${hiva_degree}"
    --policy.hiva_degree_grip="${hiva_degree}"
    --policy.hiva_tr_loss_weight=1.0
    --policy.hiva_rot_loss_weight=1.0
    --policy.hiva_grip_loss_weight=1.0
    --policy.hiva_duration_noisy_loss_weight=1.0
    --policy.hiva_duration_clean_loss_weight=0.0
    --policy.hiva_duration_noisy_sigma="${hiva_duration_sigma}"
    --policy.hiva_duration_loss=ce_mean
    --policy.hiva_duration_prediction_type=categorical
    --policy.hiva_duration_readout=coeff_modality_pool
    --policy.hiva_duration_fm_loss_weight=1.0
    --policy.hiva_duration_cont_norm=bounded
    --policy.hiva_suffix_attention=full
    --policy.hiva_basis_mode=canonical_lp_mt
    --policy.hiva_duration_head_type=residual_ffn
    --policy.hiva_decoded_action_loss_weight=0.0
    --policy.hiva_decoded_tr_loss_weight=1.0
    --policy.hiva_decoded_rot_loss_weight=1.0
    --policy.hiva_decoded_grip_loss_weight=1.0
    --policy.hiva_decoded_prefix_weight=1.0
    --policy.hiva_decoded_post_duration_exec_weight=0.5
    --policy.hiva_decoded_preview_weight=0.1
    --policy.hiva_decoded_terminal_weight=0.0
    --policy.hiva_decoded_loss_beta=0.1
    --policy.hiva_residual_enabled=false
    --policy.hiva_residual_horizon=15
    --policy.hiva_residual_ffn_hidden_mult=4.0
    --policy.hiva_residual_token_time_hidden_mult=2.0
    --policy.hiva_residual_alpha_init=0.1
    --policy.hiva_residual_zero_init=true
    --batch_size="${batch_size}"
    --steps="${total_steps}"
    --log_freq=1
    --save_checkpoint=true
    --save_freq=0
    --save_steps="${save_steps}"
    --num_workers=0
    --seed=1000
    --policy.scheduler_warmup_steps="${scheduler_warmup_steps}"
    --policy.scheduler_decay_steps="${scheduler_decay_steps}"
    --policy.scheduler_decay_lr=2.5e-6
    --policy.device=cuda
    --policy.num_steps=10
    --policy.chunk_size=15
    --policy.n_action_steps=10
    --dataset.repo_id="${DATA_REPO_ID}"
    --dataset.root="${DATA_ROOT}"
    --rename_map="${RENAME_MAP_JSON}"
    --env.type=libero
    --env.control_mode=relative
    --env.task="${TASKS}"
    --env.max_parallel_tasks=1
    --output_dir="${output_dir}"
    --job_name="${job_name}"
    --resume=true
    --eval.batch_size=1
    --eval.n_episodes=1
    --eval_freq=0
    --wandb.enable=true
    --wandb.project=lerobot
    --wandb.disable_artifact=true
    --wandb.mode=online
  )
}

run_s0p25() {
  local job_name="smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigbrain_b256_g2_s0p25_steps10000_restart_20260517_230401"
  local output_dir="${REPO_ROOT}/outputs/train/${job_name}"
  local sidecar="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet"
  local sidecar_summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json"
  local latest
  latest="$(check_resume_dir "${output_dir}")"
  check_required_paths "${sidecar}" "${sidecar_summary}" "${DATA_ROOT}" "${INIT_SMOLVLA}" "${ACCELERATE_BIN}" "${LEROBOT_TRAIN_BIN}"

  export PATH="${CONDA_ENV_BIN}:${PATH}"
  export CUDA_VISIBLE_DEVICES=6,7
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache
  mkdir -p "${HF_DATASETS_CACHE}"

  common_train_args \
    "${INIT_SMOLVLA}" \
    "${sidecar}" \
    "${sidecar_summary}" \
    3 \
    0.25 \
    256 \
    10000 \
    '[1600,1800,1900,5000,6000,6250,7000,7500,8750,10000]' \
    300 \
    10000 \
    "${output_dir}" \
    "${job_name}"
  TRAIN_ARGS+=(--wandb.notes="Resume after shutdown. S=0.25 degree3 k10_f15 BigBrain run; GPUs=6,7; batch_per_gpu=256; effective_batch=512; steps=10000; save_steps=[1600,1800,1900,5000,6000,6250,7000,7500,8750,10000]; latest_checkpoint=${latest}")

  echo "Resuming s0p25 from ${latest}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: CUDA_VISIBLE_DEVICES=%q %q launch --num_processes=2 --main_process_port=29567 --mixed_precision=bf16 %q' \
      "${CUDA_VISIBLE_DEVICES}" "${ACCELERATE_BIN}" "${LEROBOT_TRAIN_BIN}"
    printf ' %q' "${TRAIN_ARGS[@]}"
    printf '\n'
    return 0
  fi

  "${ACCELERATE_BIN}" launch \
    --num_processes=2 \
    --main_process_port=29567 \
    --mixed_precision=bf16 \
    "${LEROBOT_TRAIN_BIN}" \
    "${TRAIN_ARGS[@]}"
}

run_p5() {
  local job_name="smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_p5_f15_bigbrain_b256_g2_s0p5_steps5000_restart_20260517_230045"
  local output_dir="${REPO_ROOT}/outputs/train/${job_name}"
  local sidecar="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.parquet"
  local sidecar_summary="${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_p5_f15_canonical_lp_mt.summary.json"
  local latest
  latest="$(check_resume_dir "${output_dir}")"
  check_required_paths "${sidecar}" "${sidecar_summary}" "${DATA_ROOT}" "${INIT_SMOLVLA}" "${ACCELERATE_BIN}" "${LEROBOT_TRAIN_BIN}"

  export PATH="${CONDA_ENV_BIN}:${PATH}"
  export CUDA_VISIBLE_DEVICES=8,9
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  source "/home/jongwoopark/lerobot/server_scripts/common_hf_cache.sh"
setup_hf_datasets_cache
  mkdir -p "${HF_DATASETS_CACHE}"

  common_train_args \
    "${INIT_SMOLVLA}" \
    "${sidecar}" \
    "${sidecar_summary}" \
    5 \
    0.25 \
    256 \
    5000 \
    '[1600,1800,1900,3125,3250,3375,3500,3625,3750,3875,4000,4375,5000]' \
    150 \
    5000 \
    "${output_dir}" \
    "${job_name}"
  TRAIN_ARGS+=(--wandb.notes="Resume after shutdown. P5 stage0 LP-MT BigBrain run; GPUs=8,9; batch_per_gpu=256; effective_batch=512; steps=5000; save_steps=[1600,1800,1900,3125,3250,3375,3500,3625,3750,3875,4000,4375,5000]; latest_checkpoint=${latest}")

  echo "Resuming p5 from ${latest}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: CUDA_VISIBLE_DEVICES=%q %q launch --num_processes=2 --main_process_port=29589 --mixed_precision=bf16 %q' \
      "${CUDA_VISIBLE_DEVICES}" "${ACCELERATE_BIN}" "${LEROBOT_TRAIN_BIN}"
    printf ' %q' "${TRAIN_ARGS[@]}"
    printf '\n'
    return 0
  fi

  "${ACCELERATE_BIN}" launch \
    --num_processes=2 \
    --main_process_port=29589 \
    --mixed_precision=bf16 \
    "${LEROBOT_TRAIN_BIN}" \
    "${TRAIN_ARGS[@]}"
}

if [[ "${1:-}" == "--internal-run" ]]; then
  case "${2:-}" in
    s0p25)
      run_s0p25
      ;;
    p5)
      run_p5
      ;;
    *)
      echo "Usage: $0 --internal-run {s0p25|p5}" >&2
      exit 2
      ;;
  esac
  exit 0
fi

ONLY="${ONLY:-all}"
DRY_RUN="${DRY_RUN:-0}"

case "${ONLY}" in
  all|s0p25|p5)
    ;;
  *)
    echo "ERROR: ONLY must be one of: all, s0p25, p5" >&2
    exit 2
    ;;
esac

echo "Resume launcher: ${SCRIPT_PATH}"
echo "RUN_STAMP=${RUN_STAMP}"
echo "ONLY=${ONLY}"
echo "DRY_RUN=${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ "${ONLY}" == "all" || "${ONLY}" == "s0p25" ]]; then
    DRY_RUN=1 run_s0p25
  fi
  if [[ "${ONLY}" == "all" || "${ONLY}" == "p5" ]]; then
    DRY_RUN=1 run_p5
  fi
  exit 0
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "s0p25" ]]; then
  outer_log="${LOG_DIR}/resume_s0p25_b256_gpu6_7_after_shutdown_${RUN_STAMP}.outer.log"
  RUN_STAMP="${RUN_STAMP}" nohup setsid bash "${SCRIPT_PATH}" --internal-run s0p25 > "${outer_log}" 2>&1 < /dev/null &
  echo "Launched s0p25 resume on GPUs 6,7: pid=$!, log=${outer_log}"
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "p5" ]]; then
  outer_log="${LOG_DIR}/resume_p5_s0p5_b256_gpu8_9_after_shutdown_${RUN_STAMP}.outer.log"
  RUN_STAMP="${RUN_STAMP}" nohup setsid bash "${SCRIPT_PATH}" --internal-run p5 > "${outer_log}" 2>&1 < /dev/null &
  echo "Launched p5 resume on GPUs 8,9: pid=$!, log=${outer_log}"
fi

echo "Resume launch commands submitted at $(date)."
