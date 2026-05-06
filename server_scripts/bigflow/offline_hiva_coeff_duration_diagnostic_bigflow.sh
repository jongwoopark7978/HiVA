#!/usr/bin/env bash
set -euo pipefail

# Offline training-dataset diagnostic for HiVA coefficient SmolVLA duration accuracy.
#
# This script samples frames directly from the saved LeRobot parquet data, attaches the
# HiVA coefficient sidecar rows, samples fresh flow-matching noise/time values, and
# measures the noisy duration head without updating model weights.
#
# Example:
#   GPU_ID=3 \
#   POLICY_PATH=/home/jongwoopark/lerobot/outputs/train/smolvla_hiva_coeff_bigflow_b160_full_s2_20260504_052506/checkpoints/last/pretrained_model \
#   bash /home/jongwoopark/lerobot/server_scripts/bigflow/offline_hiva_coeff_duration_diagnostic_bigflow.sh
#
# Useful overrides:
#   NUM_SAMPLES=256
#   BATCH_SIZE=32
#   SEED=20260504
#   DATA_FILE=/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys/data/chunk-000/file-000.parquet
#   RESULT_PATH=/path/to/result.json

ROOT_DIR="/home/jongwoopark/lerobot"
PYTHON="${PYTHON:-/home/jongwoopark/miniconda3/envs/smolvla_libero/bin/python}"
GPU_ID="${GPU_ID:-3}"
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH to a HiVA coefficient pretrained_model directory.}"

DATASET_ROOT="${DATASET_ROOT:-/nfs/bigflow/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys}"
DATA_FILE="${DATA_FILE:-${DATASET_ROOT}/data/chunk-000/file-000.parquet}"
SIDECAR="${SIDECAR:-/nfs/bigflow/add_disk0/jongwoopark/libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-20260504}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  # POLICY_PATH is normally <run>/checkpoints/last/pretrained_model.
  OUTPUT_DIR="$(dirname "$(dirname "$(dirname "$POLICY_PATH")")")"
fi
mkdir -p "$OUTPUT_DIR"

RESULT_NAME="${RESULT_NAME:-offline_hiva_duration_noisy_diagnostic_file000_seed${SEED}_n${NUM_SAMPLES}.json}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_DIR}/${RESULT_NAME}}"
LOG_PATH="${LOG_PATH:-${RESULT_PATH%.json}.log}"

export POLICY_PATH DATASET_ROOT DATA_FILE SIDECAR NUM_SAMPLES BATCH_SIZE SEED RESULT_PATH
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "POLICY_PATH=${POLICY_PATH}"
echo "DATA_FILE=${DATA_FILE}"
echo "SIDECAR=${SIDECAR}"
echo "RESULT_PATH=${RESULT_PATH}"
echo "GPU_ID=${GPU_ID} NUM_SAMPLES=${NUM_SAMPLES} BATCH_SIZE=${BATCH_SIZE} SEED=${SEED}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "$PYTHON" -u - <<'PY' 2>&1 | tee "$LOG_PATH"
from __future__ import annotations

import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoTokenizer

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def png_bytes_to_tensor(blob: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(blob)).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def batch_iter(items, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def add_bucket(buckets: dict[str, list[float | int]], name: str, mask: torch.Tensor, correct: torch.Tensor) -> None:
    count = int(mask.sum().item())
    if count > 0:
        buckets[name][0] += float(correct[mask].sum().item())
        buckets[name][1] += count


checkpoint = Path(os.environ["POLICY_PATH"]).resolve()
dataset_root = Path(os.environ["DATASET_ROOT"])
data_file = Path(os.environ["DATA_FILE"])
sidecar_path = Path(os.environ["SIDECAR"])
result_path = Path(os.environ["RESULT_PATH"])
num_samples = int(os.environ["NUM_SAMPLES"])
batch_size = int(os.environ["BATCH_SIZE"])
seed = int(os.environ["SEED"])

random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"checkpoint={checkpoint}", flush=True)
print(f"data_file={data_file}", flush=True)
print(f"sidecar={sidecar_path}", flush=True)
print(f"device=cuda:0 cuda_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

cfg = TrainPipelineConfig.from_pretrained(checkpoint)
cfg.policy.pretrained_path = checkpoint
cfg.policy.device = "cuda:0"

print("loading dataset metadata/policy...", flush=True)
t0 = time.time()
ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=dataset_root, revision=cfg.dataset.revision)
policy = make_policy(cfg.policy, ds_meta=ds_meta)
policy.eval()
model = policy.model
print(f"policy_loaded_s={time.time() - t0:.1f}", flush=True)

print("loading direct parquet sample...", flush=True)
t0 = time.time()
columns = [
    "observation.images.agentview",
    "observation.images.wrist",
    "observation.state",
    "index",
    "episode_index",
    "frame_index",
    "task_index",
]
table = pq.read_table(data_file, columns=columns)
rows = table.to_pylist()
if len(rows) > num_samples:
    rows = random.sample(rows, num_samples)
raw_indices = [int(row["index"]) for row in rows]
print(
    f"rows={len(rows)} raw_index_range=[{min(raw_indices)}, {max(raw_indices)}] parquet_s={time.time() - t0:.1f}",
    flush=True,
)

print("loading filtered sidecar rows...", flush=True)
t0 = time.time()
sidecar_columns = [
    "dataset_index",
    "duration_class",
    "duration_label",
    "hiva_theta_tr_raw",
    "hiva_theta_rot_raw",
    "hiva_theta_grip_raw",
]
sidecar_rows = pq.read_table(
    sidecar_path,
    columns=sidecar_columns,
    filters=[("dataset_index", "in", raw_indices)],
).to_pylist()
sidecar_by_index = {int(row["dataset_index"]): row for row in sidecar_rows}
missing = [index for index in raw_indices if index not in sidecar_by_index]
if missing:
    raise RuntimeError(f"missing sidecar rows: {missing[:5]} total={len(missing)}")
print(f"sidecar_rows={len(sidecar_by_index)} sidecar_s={time.time() - t0:.1f}", flush=True)

print("loading task prompts and stats...", flush=True)
tasks = {
    int(row["task_index"]): row["task"]
    for row in pq.read_table(dataset_root / "meta/tasks.parquet").to_pylist()
}
with open(dataset_root / "meta/stats.json") as f:
    stats_json = json.load(f)
state_mean = torch.tensor(stats_json["observation.state"]["mean"], dtype=torch.float32).view(1, -1)
state_std = torch.tensor(stats_json["observation.state"]["std"], dtype=torch.float32).view(1, -1)
tokenizer = AutoTokenizer.from_pretrained(cfg.policy.vlm_model_name)

stats = {
    "n": 0,
    "correct": 0.0,
    "weighted_correct": 0.0,
    "weight_sum": 0.0,
    "ce_sum": 0.0,
    "weighted_loss_sum": 0.0,
}
buckets = {name: [0.0, 0] for name in ["t_lt_010", "t_lt_020", "t_lt_025", "t_025_050", "t_gt_050"]}

print("running diagnostic forward passes...", flush=True)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for batch_index, chunk in enumerate(batch_iter(rows, batch_size), start=1):
        current_batch_size = len(chunk)
        agent = torch.stack(
            [png_bytes_to_tensor(row["observation.images.agentview"]["bytes"]) for row in chunk]
        ).to("cuda:0")
        wrist = torch.stack(
            [png_bytes_to_tensor(row["observation.images.wrist"]["bytes"]) for row in chunk]
        ).to("cuda:0")
        state_raw = torch.tensor([row["observation.state"] for row in chunk], dtype=torch.float32)
        state = ((state_raw - state_mean) / (state_std + 1e-8)).to("cuda:0")
        prompts = [tasks[int(row["task_index"])] + "\n" for row in chunk]
        tokens = tokenizer(
            prompts,
            padding=cfg.policy.pad_language_to,
            truncation=True,
            max_length=cfg.policy.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokens["input_ids"].to("cuda:0")
        lang_masks = tokens["attention_mask"].to("cuda:0", dtype=torch.bool)
        duration_target = torch.tensor(
            [int(sidecar_by_index[int(row["index"])]["duration_class"]) for row in chunk],
            dtype=torch.long,
            device="cuda:0",
        )
        targets = {
            "tr": torch.tensor(
                [sidecar_by_index[int(row["index"])]["hiva_theta_tr_raw"] for row in chunk],
                dtype=torch.float32,
                device="cuda:0",
            ),
            "rot": torch.tensor(
                [sidecar_by_index[int(row["index"])]["hiva_theta_rot_raw"] for row in chunk],
                dtype=torch.float32,
                device="cuda:0",
            ),
            "grip": torch.tensor(
                [sidecar_by_index[int(row["index"])]["hiva_theta_grip_raw"] for row in chunk],
                dtype=torch.float32,
                device="cuda:0",
            ),
        }
        batch = {
            "observation.images.agentview": agent,
            "observation.images.wrist": wrist,
            "observation.state": state,
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
        }

        images, img_masks = policy.prepare_images(batch)
        state_pad = policy.prepare_state(batch)
        targets_norm = model.normalize_coeffs(targets)
        noise = model._sample_coeff_noise(targets_norm)
        time_t = model.sample_time(current_batch_size, state_pad.device)
        x_t, _ = model._mix_coeffs(targets_norm, noise, time_t)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state_pad
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        _, duration_logits = model._forward_hiva_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=x_t,
            timestep=time_t,
            use_cache=True,
        )

        duration_logits = duration_logits.float()
        pred = duration_logits.argmax(dim=-1)
        correct = (pred == duration_target).float()
        weights = model.duration_noisy_weights(time_t).to(device=correct.device, dtype=torch.float32)
        ce = torch.nn.functional.cross_entropy(duration_logits, duration_target, reduction="none")

        stats["n"] += int(correct.numel())
        stats["correct"] += float(correct.sum().item())
        stats["weighted_correct"] += float((correct * weights).sum().item())
        stats["weight_sum"] += float(weights.sum().item())
        stats["ce_sum"] += float(ce.sum().item())
        stats["weighted_loss_sum"] += float((ce * weights).sum().item())

        time_float = time_t.float()
        add_bucket(buckets, "t_lt_010", time_float < 0.10, correct)
        add_bucket(buckets, "t_lt_020", time_float < 0.20, correct)
        add_bucket(buckets, "t_lt_025", time_float < 0.25, correct)
        add_bucket(buckets, "t_025_050", (time_float >= 0.25) & (time_float < 0.50), correct)
        add_bucket(buckets, "t_gt_050", time_float >= 0.50, correct)

        acc_all = stats["correct"] / max(stats["n"], 1)
        acc_weighted = stats["weighted_correct"] / max(stats["weight_sum"], 1e-6)
        print(
            f"batch={batch_index:03d} samples={stats['n']} "
            f"acc_all={acc_all:.4f} acc_weighted={acc_weighted:.4f} "
            f"weight_mean={stats['weight_sum'] / stats['n']:.4f}",
            flush=True,
        )

result = {
    "checkpoint": str(checkpoint),
    "data_file": str(data_file),
    "sidecar": str(sidecar_path),
    "seed": seed,
    "samples": stats["n"],
    "batch_size": batch_size,
    "hiva_duration_noisy_acc_all": stats["correct"] / max(stats["n"], 1),
    "hiva_duration_noisy_acc_weighted": stats["weighted_correct"] / max(stats["weight_sum"], 1e-6),
    "hiva_duration_noisy_ce": stats["ce_sum"] / max(stats["n"], 1),
    "hiva_duration_noisy_loss_weighted_mean_over_samples": stats["weighted_loss_sum"] / max(stats["n"], 1),
    "hiva_duration_noisy_weight_mean": stats["weight_sum"] / max(stats["n"], 1),
}
for name, (correct_sum, count) in buckets.items():
    result[f"hiva_duration_acc_{name}"] = None if count == 0 else correct_sum / count
    result[f"hiva_duration_count_{name}"] = count

result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
print("RESULT_JSON=" + json.dumps(result, indent=2, sort_keys=True), flush=True)
print(f"wrote_result={result_path}", flush=True)
PY
