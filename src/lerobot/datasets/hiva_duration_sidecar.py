from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import torch


class DurationSidecarDataset(torch.utils.data.Dataset):
    """Attach duration labels from a sidecar table without modifying the base LeRobot dataset.

    This version is tailored to a LeRobot v3 dataset that exposes stable frame identifiers in the
    raw parquet rows:

    - ``index``          : global frame id across the whole dataset
    - ``episode_index``  : episode id
    - ``frame_index``    : frame id within the episode

    Expected sidecar columns:
        - dataset_index  (required; copied from raw v3 ``index``)
        - episode_index  (required)
        - frame_index    (required)
        - duration_class (required)
        - duration_label (optional)

    Matching strategy:
        1. Look up the sidecar row by the raw dataset ``index`` (primary key).
        2. Verify that ``episode_index`` and ``frame_index`` also match.

    A subset sidecar is useful for quick review/debugging, but it will not cover the full training
    dataset. In that case this wrapper emits an early warning and later raises once an uncovered
    sample is requested.
    """

    REQUIRED_KEYS = ("dataset_index", "episode_index", "frame_index", "duration_class")
    RAW_ID_KEYS = ("index", "episode_index", "frame_index")

    def __init__(self, dataset, sidecar_path: str | Path):
        self.dataset = dataset
        self.sidecar_path = Path(sidecar_path)
        rows = self._load_rows(self.sidecar_path)
        if not rows:
            raise ValueError(f"Sidecar at {self.sidecar_path} is empty.")

        self._by_dataset_index: dict[int, dict[str, Any]] = {}
        self._by_episode_frame: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            missing = [k for k in self.REQUIRED_KEYS if row.get(k) in (None, "")]
            if missing:
                raise ValueError(
                    f"Sidecar row is missing required keys {missing}. Row excerpt: "
                    f"{ {k: row.get(k) for k in ('dataset_index', 'episode_index', 'frame_index', 'duration_class')} }"
                )

            dataset_index = int(row["dataset_index"])
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            key = (episode_index, frame_index)

            if dataset_index in self._by_dataset_index:
                raise ValueError(f"Duplicate `dataset_index`={dataset_index} found in {self.sidecar_path}.")
            if key in self._by_episode_frame:
                raise ValueError(
                    f"Duplicate (`episode_index`, `frame_index`)={key} found in {self.sidecar_path}."
                )

            self._by_dataset_index[dataset_index] = row
            self._by_episode_frame[key] = row

        if len(self._by_dataset_index) != len(rows):
            raise ValueError("Internal error: sidecar indexing by `dataset_index` lost rows.")

        if len(rows) != len(dataset):
            warnings.warn(
                "The duration sidecar covers a different number of rows than the wrapped dataset "
                f"({len(rows)} vs {len(dataset)}). This is expected for a quick subset-review sidecar, "
                "but full-dataset training requires full coverage.",
                stacklevel=2,
            )

        self.meta = dataset.meta

    def __len__(self):
        return len(self.dataset)

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def _load_rows(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()
        if suffix in {".jsonl", ".json"}:
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        raise ValueError(f"Unsupported sidecar format for {path}. Use Parquet, JSONL, or CSV.")

    def _get_raw_row(self, idx: int) -> dict[str, Any]:
        if hasattr(self.dataset, "get_raw_item"):
            raw = self.dataset.get_raw_item(idx)
        elif hasattr(self.dataset, "hf_dataset"):
            raw = self.dataset.hf_dataset[idx]
        else:
            raise AttributeError(
                "The wrapped dataset does not expose `get_raw_item(idx)` or `hf_dataset[idx]`, "
                "so the duration sidecar cannot recover raw v3 frame identifiers."
            )

        missing = [k for k in self.RAW_ID_KEYS if k not in raw]
        if missing:
            raise KeyError(
                "The wrapped dataset raw row does not expose the required v3 identifiers. "
                f"Missing keys: {missing}. Available keys: {sorted(raw.keys())}"
            )
        return raw

    def _row_for_index(self, idx: int) -> tuple[dict[str, Any], int, int, int]:
        raw = self._get_raw_row(idx)
        dataset_index = int(raw["index"])
        episode_index = int(raw["episode_index"])
        frame_index = int(raw["frame_index"])

        row = self._by_dataset_index.get(dataset_index)
        if row is None:
            row = self._by_episode_frame.get((episode_index, frame_index))

        if row is None:
            raise KeyError(
                "Could not match a duration sidecar row for sample "
                f"(dataset_index={dataset_index}, episode_index={episode_index}, frame_index={frame_index}). "
                "If you built only a subset sidecar, rebuild a full sidecar before full-dataset training."
            )

        row_episode_index = int(row["episode_index"])
        row_frame_index = int(row["frame_index"])
        if row_episode_index != episode_index or row_frame_index != frame_index:
            raise ValueError(
                "Sidecar key mismatch: the row found by `dataset_index` does not agree with the "
                "dataset's (`episode_index`, `frame_index`). "
                f"dataset_index={dataset_index}, dataset says ({episode_index}, {frame_index}), "
                f"sidecar says ({row_episode_index}, {row_frame_index})."
            )

        return row, dataset_index, episode_index, frame_index

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        row, dataset_index, _episode_index, _frame_index = self._row_for_index(idx)

        sample["duration_class"] = torch.tensor(int(row["duration_class"]), dtype=torch.long)
        if row.get("duration_label") not in (None, ""):
            sample["duration_label"] = torch.tensor(int(row["duration_label"]), dtype=torch.long)
        return sample
