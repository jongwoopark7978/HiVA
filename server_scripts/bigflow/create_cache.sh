export HF_DATASETS_CACHE="/nfs/bigquery.cs.stonybrook.edu/add_disk0/jongwoopark/tmp/jongwoo_hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="local/libero_lerobot_v3_lerobotkeys",
    root="/nfs/bigflow.cs.stonybrook.edu/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys",
)
print("len(ds) =", len(ds))
x = ds[0]
print("keys =", list(x.keys()))
PY