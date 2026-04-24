export SIDECAR="/home/jongwoopark/lerobot/smolvla/build_hiva_duration_sidecar.py"
echo ${DATASET_REPO_ID}
echo ${DATA_ROOT}

python -m lerobot.scripts.build_hiva_duration_sidecar \
  --dataset.repo-id=${DATASET_REPO_ID} \
  --dataset.root=${DATA_ROOT} \
  --output=${SIDECAR} \
  --w1=4 \
  --w3=6 \
  --merge-window=3 \
  --labeler-version=hiva_duration_v1