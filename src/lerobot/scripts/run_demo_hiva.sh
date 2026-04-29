export MUJOCO_GL=egl
export DATA_ROOT=/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys
export DATA_REPO_ID=local/libero_lerobot_v3_lerobotkeys

python /home/jongwoopark/lerobot/src/lerobot/scripts/replay_hiva_duration_in_libero.py \
  --dataset.root ${DATA_ROOT} \
  --sidecar /nfs/bigbrain/add_disk0/jongwoopark/debug_sidecars/libero10_task0_ep8,13,26,39,69,71,77.parquet \
  --suite libero_10 \
  --task-id 0 \
  --episode-indices 8,13,26,39,69,71,77 \
  --output-dir /nfs/bigbrain/add_disk0/jongwoopark/duration_replay/libero10_task0_ep8,13,26,39,69,71,77_v15
