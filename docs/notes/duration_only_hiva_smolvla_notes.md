# Duration-only HiVA-lite for SmolVLA 0.45B

This patch implements the duration-only HiVA-lite variant for SmolVLA:

- keep SmolVLA's original raw action-chunk path,
- add a 3-way duration head over classes `{1,3,8}`,
- train with `L = L_action + lambda_dur * CE(duration_logits, duration_class)`,
- at inference, execute only the predicted prefix and then re-observe.

## Dataset assumptions now baked into the helper scripts

Your local LeRobot v3 dataset has the best-case frame-id layout:

- raw frame rows expose `index`, `episode_index`, `frame_index`, `task_index`, and `timestamp`,
- `index` is the global frame id for the full dataset,
- `episode_index + frame_index` is a clean secondary identifier,
- episode metadata exposes `dataset_from_index` and `dataset_to_index` for each episode.

Because of that, the updated helper code now does the following:

- uses the real v3 `index` column as `dataset_index`,
- stores `episode_index` and `frame_index` in every sidecar row,
- removes the old “missing frame_index” fallback assumption,
- supports **subset sidecar builds by episode** for quick review/debugging.

## Files changed

Existing files to patch:

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/datasets/factory.py`

New files:

- `src/lerobot/datasets/hiva_duration_sidecar.py`
- `src/lerobot/scripts/build_hiva_duration_sidecar.py`

## What each change does

### 1. `configuration_smolvla.py`
Adds four knobs:

- `use_duration_head`
- `duration_classes=(1,3,8)`
- `duration_loss_weight`
- `duration_sidecar_path`

### 2. `modeling_smolvla.py`
Adds the duration-only HiVA-lite behavior.

Training:
- SmolVLA still predicts the full raw action chunk.
- A new linear `duration_head` reads pooled action-expert suffix features.
- The total training loss becomes raw-action flow matching plus duration cross entropy.

Inference:
- SmolVLA still denoises the full chunk.
- An extra suffix pass estimates duration logits from the final denoised chunk.
- `select_action()` enqueues only the predicted prefix instead of always enqueuing `n_action_steps`.

### 3. `factory.py`
Wraps the base LeRobot dataset with a duration sidecar loader whenever:

- `--policy.use_duration_head=true`
- `--policy.duration_sidecar_path=/path/to/sidecar.parquet`

### 4. `hiva_duration_sidecar.py`
Loads duration labels from a sidecar table and injects:

- `duration_class`
- `duration_label`

into each training sample.

The updated wrapper now assumes the local v3 dataset exposes the canonical raw frame ids:

- primary key: `dataset_index` (copied from raw `index`)
- secondary consistency check: `(episode_index, frame_index)`

It also emits an early warning if you try to wrap a full dataset with a subset-only sidecar.

### 5. `build_hiva_duration_sidecar.py`
Builds the sidecar from the raw local LeRobot v3 LIBERO dataset and now:

- reads the real v3 `index`, `episode_index`, and `frame_index` columns,
- uses episode metadata to select **subsets by episode**,
- writes an optional JSON summary for quick inspection,
- preserves full-episode context while building subset sidecars.

Subset sidecars are for **review/debugging** only.
For full-dataset training, build a full sidecar with no subset restriction.

## Environment and local paths

Use your conda environment and your exact local dataset paths:

```bash
conda activate smolvla_libero

cd ~/lerobot
export MUJOCO_GL=egl
export DATA_ROOT=/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys
export DATA_REPO_ID=local/libero_lerobot_v3_lerobotkeys
export TASKS=libero_spatial,libero_object,libero_goal,libero_10
```

## Apply the patch

```bash
cd ~/lerobot
git apply /path/to/duration_only_hiva_smolvla_patch.diff
```

Then copy the two helper files into place if you are applying them manually:

```bash
cp /path/to/hiva_duration_sidecar.py ~/lerobot/src/lerobot/datasets/hiva_duration_sidecar.py
cp /path/to/build_hiva_duration_sidecar.py ~/lerobot/src/lerobot/scripts/build_hiva_duration_sidecar.py
```

## Quick subset sidecar for review

Build a sidecar only for the first 8 episodes so you can inspect labels quickly:

```bash
python -m lerobot.scripts.build_hiva_duration_sidecar \
  --dataset.repo-id ${DATA_REPO_ID} \
  --dataset.root ${DATA_ROOT} \
  --output /nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_subset_ep0_7.parquet \
  --summary-json /nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_subset_ep0_7.summary.json \
  --episode-start 0 \
  --max-episodes 8 \
  --labeler-version hiva_duration_v2_subset
```

You can also choose explicit episodes:

```bash
python -m lerobot.scripts.build_hiva_duration_sidecar \
  --dataset.repo-id ${DATA_REPO_ID} \
  --dataset.root ${DATA_ROOT} \
  --output /nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_subset_custom.parquet \
  --summary-json /nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_subset_custom.summary.json \
  --episode-indices 0-3,10,12 \
  --labeler-version hiva_duration_v2_subset
```

Quickly inspect the subset output:

```bash
python - <<'PY'
import pyarrow.parquet as pq

sidecar = "/nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_subset_ep0_7.parquet"
cols = ["dataset_index", "episode_index", "frame_index", "duration_label", "duration_class", "phase"]
print(pq.read_table(sidecar, columns=cols).to_pandas().head(30))
PY
```

## Full sidecar for training

Build the full sidecar with the real v3 global frame ids:

```bash
python -m lerobot.scripts.build_hiva_duration_sidecar \
  --dataset.repo-id ${DATA_REPO_ID} \
  --dataset.root ${DATA_ROOT} \
  --output /nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_full.parquet \
  --summary-json /nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_full.summary.json \
  --labeler-version hiva_duration_v2_full
```

## Train duration-only HiVA-lite SmolVLA

Use the **full** sidecar for full-dataset training:

```bash
export SIDECAR=/nfs/bigbrain/add_disk0/jongwoopark/libero_duration_sidecar_full.parquet

lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.use_duration_head=true \
  --policy.duration_loss_weight=1.0 \
  --policy.duration_sidecar_path=${SIDECAR} \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=${DATA_REPO_ID} \
  --dataset.root=${DATA_ROOT} \
  --env.type=libero \
  --env.control_mode=relative \
  --env.task=${TASKS} \
  --output_dir=outputs/train/smolvla_hiva_duration_only_libero \
  --job_name=smolvla_hiva_duration_only_libero \
  --steps=100000 \
  --batch_size=64 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=1000 \
  --wandb.enable=false
```

## Evaluate

```bash
export CKPT=~/lerobot/outputs/train/smolvla_hiva_duration_only_libero/checkpoints/last/pretrained_model

lerobot-eval \
  --output_dir=outputs/eval/smolvla_hiva_duration_only_libero \
  --policy.path=${CKPT} \
  --policy.device=cuda \
  --env.type=libero \
  --env.control_mode=relative \
  --env.task=${TASKS} \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1
```

For the LIBERO-Long number only:

```bash
lerobot-eval \
  --output_dir=outputs/eval/smolvla_hiva_duration_only_libero_long \
  --policy.path=${CKPT} \
  --policy.device=cuda \
  --env.type=libero \
  --env.control_mode=relative \
  --env.task=libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1
```
