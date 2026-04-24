import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/nfs/bigquery.cs.stonybrook.edu/add_disk0/cristinam/libero/libero_lerobot_v3_lerobotkeys")

print("=== info.json features ===")
info = json.loads((root / "meta" / "info.json").read_text())
print(sorted(info["features"].keys()))

data_file = root / "data" / "chunk-000" / "file-000.parquet"
print("\n=== data parquet schema ===")
pf = pq.ParquetFile(data_file)
print(pf.schema)

wanted = [c for c in ["index", "dataset_index", "episode_index", "frame_index", "task_index", "timestamp"] if c in pf.schema.names]
print("\n=== first 10 rows of key id columns ===")
print(pq.read_table(data_file, columns=wanted).to_pandas().head(10))

ep_file = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
print("\n=== episode parquet schema ===")
ep_pf = pq.ParquetFile(ep_file)
print(ep_pf.schema)

ep_cols = [c for c in ["episode_index", "dataset_from_index", "dataset_to_index"] if c in ep_pf.schema.names]
print("\n=== first 10 rows of episode index mapping ===")
print(pq.read_table(ep_file, columns=ep_cols).to_pandas().head(10))