#!/usr/bin/env bash
set -euo pipefail

# Full 50-episode LIBERO eval for dense extra checkpoints from the BestS0p25
# stage0 LP-MT HiVA run. Designed to run each GPU group independently:
#
#   GROUP=gpu0_3 -> checkpoints 006300..006900 on GPUs 0,1,2,3
#   GROUP=gpu4_7 -> checkpoints 007000..007500 on GPUs 4,5,6,7
#
# Each group repeats its full checkpoint queue REPEATS times. The lower-level
# helper uses staged LIBERO-10 assignment:
#   GPU0/4: object, GPU1/5: goal, GPU2/6: spatial, GPU3/7: libero_10 [0..6]
#   after short suites finish, GPUs 0/4,1/5,2/6 evaluate libero_10 [7],[8],[9].

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
GROUP="${GROUP:-gpu0_3}"
REPEATS="${REPEATS:-2}"

TRAIN_DIR="${TRAIN_DIR:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/BestS0p25_80.75_smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${TRAIN_DIR}/ckeckpoints_extra}"
MODEL_TAG="$(basename "${TRAIN_DIR}")"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/jongwoopark/hf_datasets_cache}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"

HELPER="${REPO_ROOT}/server_scripts/bigtoken/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigtoken_${MODEL_TAG}_extra_ckpts_6300_7500_repeat${REPEATS}_50eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"
mkdir -p "${SWEEP_OUTPUT_DIR}" "${LOG_DIR}" "${HF_DATASETS_CACHE}"

case "${GROUP}" in
  gpu0_3)
    GPU_IDS="${GPU_IDS:-0,1,2,3}"
    CKPTS=(${CKPTS_OVERRIDE:-006300 006400 006500 006600 006700 006800 006900})
    ;;
  gpu4_7)
    GPU_IDS="${GPU_IDS:-4,5,6,7}"
    CKPTS=(${CKPTS_OVERRIDE:-007000 007100 007200 007300 007400 007500})
    ;;
  *)
    echo "Unknown GROUP=${GROUP}; expected gpu0_3 or gpu4_7" >&2
    exit 1
    ;;
esac

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

policy_path_for_ckpt() {
  local ckpt="$1"
  local ckpt_dir="${CHECKPOINT_ROOT}/${ckpt}"
  if [[ -d "${ckpt_dir}/pretrained_model" ]]; then
    echo "${ckpt_dir}/pretrained_model"
  else
    echo "${ckpt_dir}"
  fi
}

write_manifest() {
  local ckpt_json
  ckpt_json="$(printf '"%s",' "${CKPTS[@]}" | sed 's/,$//')"
  cat > "${SWEEP_OUTPUT_DIR}/${GROUP}_manifest.json" <<JSON
{
  "timestamp": "${TIMESTAMP}",
  "group": "${GROUP}",
  "gpu_ids": "${GPU_IDS}",
  "train_dir": "${TRAIN_DIR}",
  "checkpoint_root": "${CHECKPOINT_ROOT}",
  "checkpoints": [${ckpt_json}],
  "repeats": ${REPEATS},
  "sweep_output_dir": "${SWEEP_OUTPUT_DIR}",
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "n_episodes": ${N_EPISODES},
  "expected_episode_count": ${EXPECTED_EPISODE_COUNT},
  "expected_video_count": ${EXPECTED_VIDEO_COUNT},
  "staged_libero10_after_short": true,
  "metrics": ["episode_metrics.json", "metrics_summary.json", "metrics_summary.csv", "overlay_eval_summary.json"]
}
JSON
}

run_checkpoint() {
  local rep="$1"
  local ckpt="$2"
  local policy_path
  policy_path="$(policy_path_for_ckpt "${ckpt}")"
  local label="BestS0p25_extra_${GROUP}_rep${rep}_ckpt_${ckpt}_50eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/${GROUP}/repeat_${rep}/ckpt_${ckpt}"

  require_dir "${policy_path}"

  echo "===== $(date) ${GROUP} repeat ${rep}/${REPEATS}: evaluating ckpt ${ckpt} on GPUs ${GPU_IDS} ====="
  echo "POLICY_PATH=${policy_path}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  TIMESTAMP="${TIMESTAMP}_${GROUP}_rep${rep}_ckpt${ckpt}" \
  POLICY_PATH="${policy_path}" \
  CHECKPOINT_LABEL="${label}" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  N_EPISODES="${N_EPISODES}" \
  TASK_IDS_ALL="${TASK_IDS_ALL}" \
  MAX_PARALLEL_TASKS=1 \
  MAX_EPISODES_RENDERED=1 \
  EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT}" \
  EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT}" \
  DATA_ROOT="${DATA_ROOT}" \
  HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR}" \
  HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  N_ACTION_STEPS="${N_ACTION_STEPS}" \
  NUM_STEPS="${NUM_STEPS}" \
  HIVA_DURATION_EXECUTION_MAP="" \
  STAGED_LIBERO10_AFTER_SHORT=1 \
  SPLIT_LIBERO10_ACROSS_GPUS=0 \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  BASE_OUTPUT_DIR="${base_output_dir}" \
  bash "${HELPER}"

  echo "===== $(date) ${GROUP} repeat ${rep}/${REPEATS}: finished ckpt ${ckpt} ====="
}

main() {
  require_dir "${TRAIN_DIR}"
  require_dir "${CHECKPOINT_ROOT}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"
  require_file "${HELPER}"
  write_manifest

  echo "===== BestS0p25 extra checkpoint full eval started at $(date) ====="
  echo "GROUP=${GROUP}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "CKPTS=${CKPTS[*]}"
  echo "REPEATS=${REPEATS}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "N_EPISODES=${N_EPISODES}"

  for rep in $(seq 1 "${REPEATS}"); do
    echo "===== ${GROUP} repeat ${rep}/${REPEATS} started at $(date) ====="
    for ckpt in "${CKPTS[@]}"; do
      run_checkpoint "${rep}" "${ckpt}"
    done
    echo "===== ${GROUP} repeat ${rep}/${REPEATS} finished at $(date) ====="
  done

  echo "===== BestS0p25 extra checkpoint full eval finished at $(date) ====="
}

main "$@"
