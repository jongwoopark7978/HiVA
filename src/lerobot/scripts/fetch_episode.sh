export MUJOCO_GL=egl
export DATA_ROOT=/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys
export DATA_REPO_ID=local/libero_lerobot_v3_lerobotkeys

SUITE=${SUITE:-libero_10}
TASK_ID=${TASK_ID:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/nfs/bigbrain/add_disk0/jongwoopark/libero_duration_list}

python /home/jongwoopark/lerobot/src/lerobot/scripts/replay_hiva_duration_in_libero.py \
  --dataset.root ${DATA_ROOT} \
  --suite ${SUITE} \
  --task-id ${TASK_ID} \
  --list-task-episodes \
  --output-dir ${OUTPUT_DIR}
