#!/usr/bin/env bash
set -euo pipefail

# Full LIBERO eval sweep for the Best0p25 LP-MT HiVA stage0 checkpoint with
# inference-time duration remapping. Each remap case runs all suites/tasks with
# 50 episodes per task. The GPU0-3 cases and GPU4-7 cases run concurrently, and
# cases within each group run sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

POLICY_PATH="${POLICY_PATH:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/HiVA_train/finetuning_stage0/smolvla_hiva_coeff_lpmt_stage0_v5_d4_6_10_coeffpool_full_ce_mean_k10_f15_bigcornea_b64_s0p25_20260515_171520/checkpoints/007000/pretrained_model}"
CHECKPOINT_LABEL_BASE="${CHECKPOINT_LABEL_BASE:-best0p25_d4_6_10_ckpt007000_execmap}"

SIDECAR_ROOT="${SIDECAR_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark}"
DATA_ROOT="${DATA_ROOT:-${SIDECAR_ROOT}/libero_lerobot_v3_lerobotkeys}"
HIVA_COEFF_SIDECAR="${HIVA_COEFF_SIDECAR:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.parquet}"
HIVA_COEFF_SUMMARY="${HIVA_COEFF_SUMMARY:-${SIDECAR_ROOT}/libero_hiva_coeff_sidecar_v5_d4_6_10_wide_commit6_k10_f15_canonical_lp_mt.summary.json}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
TASK_IDS_ALL="${TASK_IDS_ALL:-[0,1,2,3,4,5,6,7,8,9]}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-2000}"
EXPECTED_VIDEO_COUNT="${EXPECTED_VIDEO_COUNT:-40}"
# Use 10 for normal D={4,6,10} execution. Individual remap cases that request
# a larger destination horizon, such as 10:15, are automatically bumped in
# run_case so the mapped duration is not clipped by the policy action queue.
N_ACTION_STEPS_BASE="${N_ACTION_STEPS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-15}"
NUM_STEPS="${NUM_STEPS:-10}"

SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/full_bigtoken_${CHECKPOINT_LABEL_BASE}_50eps_bs${EVAL_BATCH_SIZE}_${TIMESTAMP}}"
LOG_DIR="${REPO_ROOT}/outputs/eval_logs"
HELPER="${REPO_ROOT}/server_scripts/bigtoken/eval_hiva_coeff_cleaner_suffix_s4_10eps_bigtoken.sh"
mkdir -p "${SWEEP_OUTPUT_DIR}" "${LOG_DIR}"

GPU0_3_CASES=(
  "map_4_6_10_to_4_10_10|4:4,6:10,10:10"
  "map_4_6_10_to_4_8_10|4:4,6:8,10:10"
  "map_4_6_10_to_4_2_10|4:4,6:2,10:10"
  "map_4_6_10_to_10_6_10|4:10,6:6,10:10"
)

GPU4_7_CASES=(
  "map_4_6_10_to_2_6_10|4:2,6:6,10:10"
  "map_4_6_10_to_4_6_15|4:4,6:6,10:15"
  "map_4_6_10_to_4_6_8|4:4,6:6,10:8"
  "map_4_6_10_to_4_6_6|4:4,6:6,10:6"
)

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

write_manifest() {
  CASES_JSON="$(N_ACTION_STEPS_BASE="${N_ACTION_STEPS_BASE}" python - "${GPU0_3_CASES[@]}" "${GPU4_7_CASES[@]}" <<'PY'
import json
import os
import sys

base_n_action_steps = int(os.environ["N_ACTION_STEPS_BASE"])
cases = []
for line in sys.argv[1:]:
    line = line.strip()
    if not line:
        continue
    label, mapping = line.split("|", 1)
    max_dst = max(
        int(item.split(":", 1)[1].strip())
        for item in mapping.split(",")
        if item.strip()
    )
    cases.append({
        "label": label,
        "hiva_duration_execution_map": mapping,
        "n_action_steps": max(base_n_action_steps, max_dst),
    })
print(json.dumps(cases, indent=2))
PY
)"
  GPU0_3_CASES_JSON="$(python - "${GPU0_3_CASES[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"
  GPU4_7_CASES_JSON="$(python - "${GPU4_7_CASES[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"
  cat > "${SWEEP_OUTPUT_DIR}/execmap_eval_manifest.json" <<JSON
{
  "timestamp": "${TIMESTAMP}",
  "policy_path": "${POLICY_PATH}",
  "checkpoint_label_base": "${CHECKPOINT_LABEL_BASE}",
  "data_root": "${DATA_ROOT}",
  "hiva_coeff_sidecar": "${HIVA_COEFF_SIDECAR}",
  "hiva_coeff_summary": "${HIVA_COEFF_SUMMARY}",
  "sweep_output_dir": "${SWEEP_OUTPUT_DIR}",
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "n_episodes": ${N_EPISODES},
  "expected_episode_count": ${EXPECTED_EPISODE_COUNT},
  "expected_video_count": ${EXPECTED_VIDEO_COUNT},
  "n_action_steps_base": ${N_ACTION_STEPS_BASE},
  "chunk_size": ${CHUNK_SIZE},
  "num_steps": ${NUM_STEPS},
  "gpu0_3_cases": ${GPU0_3_CASES_JSON},
  "gpu4_7_cases": ${GPU4_7_CASES_JSON},
  "cases": ${CASES_JSON}
}
JSON
}

run_case() {
  local label="$1"
  local duration_map="$2"
  local gpu_ids="$3"
  local case_n_action_steps
  case_n_action_steps="$(N_ACTION_STEPS_BASE="${N_ACTION_STEPS_BASE}" DURATION_MAP="${duration_map}" python - <<'PY'
import os

base = int(os.environ["N_ACTION_STEPS_BASE"])
mapping = os.environ["DURATION_MAP"].strip()
max_dst = base
if mapping:
    max_dst = max(
        int(item.split(":", 1)[1].strip())
        for item in mapping.split(",")
        if item.strip()
    )
print(max(base, max_dst))
PY
)"
  local checkpoint_label="${CHECKPOINT_LABEL_BASE}_${label}_50eps_bs${EVAL_BATCH_SIZE}"
  local base_output_dir="${SWEEP_OUTPUT_DIR}/${label}"

  echo "===== $(date) start ${label} on GPUs ${gpu_ids} ====="
  echo "POLICY_PATH=${POLICY_PATH}"
  echo "HIVA_DURATION_EXECUTION_MAP=${duration_map}"
  echo "CASE_N_ACTION_STEPS=${case_n_action_steps}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"

  TIMESTAMP="${TIMESTAMP}_${label}" \
  POLICY_PATH="${POLICY_PATH}" \
  CHECKPOINT_LABEL="${checkpoint_label}" \
  GPU_IDS="${gpu_ids}" \
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
  N_ACTION_STEPS="${case_n_action_steps}" \
  NUM_STEPS="${NUM_STEPS}" \
  HIVA_DURATION_EXECUTION_MAP="${duration_map}" \
  STAGED_LIBERO10_AFTER_SHORT=1 \
  SPLIT_LIBERO10_ACROSS_GPUS=0 \
  BASE_OUTPUT_DIR="${base_output_dir}" \
  bash "${HELPER}"

  echo "===== $(date) finished ${label} ====="
}

run_group() {
  local group_name="$1"
  local gpu_ids="$2"
  shift 2
  local cases=("$@")

  echo "===== ${group_name} started at $(date) on GPUs ${gpu_ids} ====="
  for case_spec in "${cases[@]}"; do
    local label="${case_spec%%|*}"
    local duration_map="${case_spec#*|}"
    run_case "${label}" "${duration_map}" "${gpu_ids}"
  done
  echo "===== ${group_name} finished at $(date) ====="
}

main() {
  require_dir "${POLICY_PATH}"
  require_dir "${DATA_ROOT}"
  require_file "${HIVA_COEFF_SIDECAR}"
  require_file "${HIVA_COEFF_SUMMARY}"
  require_file "${HELPER}"
  write_manifest

  echo "===== Best0p25 D={4,6,10} execution-map full eval sweep started at $(date) ====="
  echo "POLICY_PATH=${POLICY_PATH}"
  echo "SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "N_EPISODES=${N_EPISODES}"
  echo "N_ACTION_STEPS_BASE=${N_ACTION_STEPS_BASE}"
  echo "CHUNK_SIZE=${CHUNK_SIZE}"
  echo "NUM_STEPS=${NUM_STEPS}"

  run_group "gpu0_3_execmap_group" "0,1,2,3" "${GPU0_3_CASES[@]}" &
  pid_a=$!
  run_group "gpu4_7_execmap_group" "4,5,6,7" "${GPU4_7_CASES[@]}" &
  pid_b=$!

  wait "${pid_a}"
  wait "${pid_b}"

  echo "===== Best0p25 D={4,6,10} execution-map full eval sweep finished at $(date) ====="
}

main "$@"
