#!/usr/bin/env bash
set -euo pipefail

# Resume the BestS0p25 stage-0 LP-MT HiVA coefficient run from checkpoint 006250
# and save dense extra checkpoints 006300..007500. The trainer writes to a small
# resume output directory, while a watcher moves completed checkpoints into the
# requested ckeckpoints_extra directory so they stay separate from the original
# checkpoint set.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_DIR="${BASE_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/BestS0p25_80.75_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${BASE_DIR}/checkpoints/006250}"
CONFIG_PATH="${CONFIG_PATH:-${RESUME_CHECKPOINT}/pretrained_model/train_config.json}"
EXTRA_CKPT_DIR="${EXTRA_CKPT_DIR:-${BASE_DIR}/ckeckpoints_extra}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WORK_OUTPUT_DIR="${WORK_OUTPUT_DIR:-${BASE_DIR}/resume_extra_006250_to_007500_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/train_logs}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/resume_extra_006250_to_007500_${RUN_STAMP}.log}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29625}"
SAVE_STEPS_JSON="${SAVE_STEPS_JSON:-[6300,6400,6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500]}"
FINAL_STEP="${FINAL_STEP:-10000}"
TARGET_FINAL_STEP="${TARGET_FINAL_STEP:-7500}"
STOP_AFTER_TARGET_FINAL_STEP="${STOP_AFTER_TARGET_FINAL_STEP:-true}"
DATA_ROOT="${DATA_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys}"
SIDECAR="${SIDECAR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
SIDECAR_SUMMARY="${SIDECAR_SUMMARY:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

CONDA_ENV_BIN="${CONDA_ENV_BIN:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ENV_BIN}/accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ENV_BIN}/lerobot-train}"

mkdir -p "${EXTRA_CKPT_DIR}" "${WORK_OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

export PATH="${CONDA_ENV_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${BASE_DIR}/hf_datasets_cache}"
export TMPDIR="${TMPDIR:-${BASE_DIR}/tmp}"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

for required_path in "${CONFIG_PATH}" "${RESUME_CHECKPOINT}/training_state" "${DATA_ROOT}" "${SIDECAR}" "${SIDECAR_SUMMARY}" "${ACCELERATE_BIN}" "${LEROBOT_TRAIN_BIN}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

mapfile -t SAVE_STEP_NAMES < <(python - <<'PY'
steps = [6300,6400,6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500]
print("\n".join(f"{step:06d}" for step in steps))
PY
)

checkpoint_complete() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/pretrained_model/model.safetensors" ]] || return 1
  [[ -f "${checkpoint_dir}/pretrained_model/config.json" ]] || return 1
  [[ -f "${checkpoint_dir}/pretrained_model/train_config.json" ]] || return 1
  [[ -f "${checkpoint_dir}/training_state/training_step.json" ]] || return 1
  [[ -f "${checkpoint_dir}/training_state/rng_state.safetensors" ]] || return 1
  [[ -f "${checkpoint_dir}/training_state/scheduler_state.json" ]] || return 1
  [[ -f "${checkpoint_dir}/training_state/optimizer_param_groups.json" ]] || return 1
  [[ -f "${checkpoint_dir}/training_state/optimizer_state.safetensors" ]] || return 1
  return 0
}

move_completed_checkpoints() {
  local moved_any=0
  local latest=""
  for step_name in "${SAVE_STEP_NAMES[@]}"; do
    local src="${WORK_OUTPUT_DIR}/checkpoints/${step_name}"
    local dst="${EXTRA_CKPT_DIR}/${step_name}"
    if checkpoint_complete "${dst}"; then
      latest="${step_name}"
      continue
    fi
    if checkpoint_complete "${src}"; then
      echo "[$(date)] Moving completed checkpoint ${step_name} -> ${dst}"
      mv "${src}" "${dst}"
      latest="${step_name}"
      moved_any=1
    fi
  done

  if [[ -n "${latest}" ]]; then
    ln -sfn "${latest}" "${EXTRA_CKPT_DIR}/last"
  fi

  return "${moved_any}"
}

all_expected_checkpoints_moved() {
  local step_name
  for step_name in "${SAVE_STEP_NAMES[@]}"; do
    if ! checkpoint_complete "${EXTRA_CKPT_DIR}/${step_name}"; then
      return 1
    fi
  done
  return 0
}

watch_checkpoints() {
  local train_pid="$1"
  while kill -0 "${train_pid}" 2>/dev/null; do
    move_completed_checkpoints || true
    if [[ "${STOP_AFTER_TARGET_FINAL_STEP}" == "true" ]] && all_expected_checkpoints_moved; then
      echo "[$(date)] All requested checkpoints through ${TARGET_FINAL_STEP} are moved; stopping training pid ${train_pid} before later checkpoints."
      kill -TERM "${train_pid}" 2>/dev/null || true
      break
    fi
    sleep 30
  done
  move_completed_checkpoints || true
}

echo "===== BestS0p25 extra-checkpoint resume started at $(date) ====="
echo "Host: $(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "BASE_DIR=${BASE_DIR}"
echo "RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "EXTRA_CKPT_DIR=${EXTRA_CKPT_DIR}"
echo "WORK_OUTPUT_DIR=${WORK_OUTPUT_DIR}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "FINAL_STEP=${FINAL_STEP}"
echo "TARGET_FINAL_STEP=${TARGET_FINAL_STEP}"
echo "STOP_AFTER_TARGET_FINAL_STEP=${STOP_AFTER_TARGET_FINAL_STEP}"
echo "SAVE_STEPS_JSON=${SAVE_STEPS_JSON}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SIDECAR=${SIDECAR}"
echo "SIDECAR_SUMMARY=${SIDECAR_SUMMARY}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "TMPDIR=${TMPDIR}"
echo "QUEUE_LOG=${QUEUE_LOG}"

"${ACCELERATE_BIN}" launch \
  --num_processes="${NUM_PROCESSES}" \
  --main_process_port="${MAIN_PROCESS_PORT}" \
  --mixed_precision=bf16 \
  "${LEROBOT_TRAIN_BIN}" \
  --config_path="${CONFIG_PATH}" \
  --resume=true \
  --output_dir="${WORK_OUTPUT_DIR}" \
  --steps="${FINAL_STEP}" \
  --save_checkpoint=true \
  --save_freq=0 \
  --save_steps="${SAVE_STEPS_JSON}" \
  --policy.device=cuda \
  --dataset.root="${DATA_ROOT}" \
  --policy.hiva_coeff_sidecar_path="${SIDECAR}" \
  --policy.hiva_coeff_sidecar_summary_path="${SIDECAR_SUMMARY}" \
  --wandb.notes="Resume BestS0p25 from 006250; dense extra checkpoints 006300..007500 saved to ckeckpoints_extra." &
train_pid="$!"

watch_checkpoints "${train_pid}" &
watcher_pid="$!"

status=0
wait "${train_pid}" || status=$?
wait "${watcher_pid}" || true
move_completed_checkpoints || true

if [[ "${STOP_AFTER_TARGET_FINAL_STEP}" == "true" ]] && all_expected_checkpoints_moved; then
  status=0
fi

if [[ "${status}" -ne 0 ]]; then
  echo "===== training failed with status ${status} at $(date) =====" >&2
  exit "${status}"
fi

echo "===== extra checkpoint directory contents ====="
find "${EXTRA_CKPT_DIR}" -maxdepth 1 -mindepth 1 -printf "%f -> %l\n" | sort
echo "===== BestS0p25 extra-checkpoint resume finished at $(date) ====="
