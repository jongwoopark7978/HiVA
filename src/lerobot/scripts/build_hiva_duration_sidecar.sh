export MUJOCO_GL=egl
export DATA_ROOT=/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys
export DATA_REPO_ID=local/libero_lerobot_v3_lerobotkeys

python -m lerobot.scripts.build_hiva_duration_sidecar \
  --dataset.repo-id ${DATA_REPO_ID} \
  --dataset.root ${DATA_ROOT} \
  --output /nfs/bigbrain/add_disk0/jongwoopark/debug_sidecars/libero10_task0_ep8,13,26,39,69,71,77.parquet \
  --summary-json /nfs/bigbrain/add_disk0/jongwoopark/debug_sidecars/libero10_task0_ep8,13,26,39,69,71,77.summary.json \
  --episode-indices 8,13,26,39,69,71,77 \
  --labeler-version hiva_duration_debug

