import os
import time
import faulthandler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from OpenGL import EGL

faulthandler.dump_traceback_later(30, repeat=True)

print("HF_DATASETS_CACHE =", os.environ.get("HF_DATASETS_CACHE"), flush=True)

root = "/nfs/bigquery.cs.stonybrook.edu/add_disk0/cristinam/libero/libero_lerobot_v3_lerobotkeys"
repo_id = "local/libero_lerobot_v3_lerobotkeys"

t0 = time.time()
print("before LeRobotDataset()", flush=True)
ds = LeRobotDataset(repo_id=repo_id, root=root)
print(f"after LeRobotDataset(): {time.time() - t0:.2f}s", flush=True)

t1 = time.time()
x = ds[0]
print(f"after ds[0]: {time.time() - t1:.2f}s", flush=True)
print("len(ds) =", len(ds), flush=True)
print("keys =", list(x.keys()), flush=True)

#EGL checking
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("EGL import OK")