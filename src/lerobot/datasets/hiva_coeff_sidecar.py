from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import torch


class HiVACoeffSidecarDataset(torch.utils.data.Dataset):
    """Attach HiVA duration and B-spline coefficient targets from a sidecar table.

    The wrapper indexes sidecar rows by the stable LeRobot v3 frame identifiers:
    ``dataset_index``, ``episode_index``, and ``frame_index``.
    """

    REQUIRED_KEYS = (
        "dataset_index",
        "episode_index",
        "frame_index",
        "duration_class",
        "duration_label",
        "hiva_theta_tr_raw",
        "hiva_theta_rot_raw",
        "hiva_theta_grip_raw",
    )
    RAW_ID_KEYS = ("index", "episode_index", "frame_index")
    OPTIONAL_KEYS = (
        "hiva_real_steps",
        "hiva_fit_real_steps",
        "hiva_has_synthetic_tail",
        "hiva_tail_mode",
        "hiva_fit_steps",
        "hiva_basis_mode",
        "hiva_target_mode",
        "hiva_basis_dmax",
        "hiva_exec_dmax",
        "hiva_fit_horizon",
        "hiva_basis_k",
        "hiva_basis_degree",
        "hiva_basis_degree_tr",
        "hiva_basis_degree_rot",
        "hiva_basis_degree_grip",
        "hiva_prefix_tr_rmse",
        "hiva_prefix_rot_rmse",
        "hiva_prefix_grip_rmse",
        "hiva_prefix_action_rmse",
        "hiva_real_prefix_action_rmse",
        "hiva_full_action_rmse",
        "hiva_full_action_max_abs_diff",
        "hiva_wrong_long_tr_energy",
        "hiva_wrong_long_rot_energy",
        "hiva_wrong_long_grip_energy",
        "hiva_wrong_long_action_energy",
        "hiva_tail_motion_energy",
    )

    def __init__(
        self,
        dataset,
        sidecar_path: str | Path,
        *,
        n_ctrl: int,
        duration_classes: tuple[int, ...],
        require_full_coverage: bool = True,
        expected_basis_mode: str | None = None,
        expected_basis_dmax: int | None = None,
        expected_fit_horizon: int | None = None,
        expected_basis_degree: int | None = None,
        expected_basis_degree_tr: int | None = None,
        expected_basis_degree_rot: int | None = None,
        expected_basis_degree_grip: int | None = None,
    ):
        self.dataset = dataset
        self.sidecar_path = Path(sidecar_path)
        self.n_ctrl = int(n_ctrl)
        self.duration_classes = tuple(int(d) for d in duration_classes)
        self.require_full_coverage = bool(require_full_coverage)
        self.expected_basis_mode = expected_basis_mode
        self.expected_basis_dmax = None if expected_basis_dmax is None else int(expected_basis_dmax)
        self.expected_fit_horizon = None if expected_fit_horizon is None else int(expected_fit_horizon)
        self.expected_basis_degree = None if expected_basis_degree is None else int(expected_basis_degree)
        self.expected_basis_degree_tr = None if expected_basis_degree_tr is None else int(expected_basis_degree_tr)
        self.expected_basis_degree_rot = None if expected_basis_degree_rot is None else int(expected_basis_degree_rot)
        self.expected_basis_degree_grip = None if expected_basis_degree_grip is None else int(expected_basis_degree_grip)

        rows = self._load_rows(self.sidecar_path)
        if not rows:
            raise ValueError(f"HiVA coefficient sidecar at {self.sidecar_path} is empty.")

        self._validate_coeff_shape(rows[0]["hiva_theta_tr_raw"], (self.n_ctrl, 3), "hiva_theta_tr_raw")
        self._validate_coeff_shape(rows[0]["hiva_theta_rot_raw"], (self.n_ctrl, 3), "hiva_theta_rot_raw")
        self._validate_coeff_shape(rows[0]["hiva_theta_grip_raw"], (self.n_ctrl, 1), "hiva_theta_grip_raw")

        self._by_dataset_index: dict[int, dict[str, Any]] = {}
        self._by_episode_frame: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            missing = [key for key in self.REQUIRED_KEYS if row.get(key) in (None, "")]
            if missing:
                raise ValueError(f"HiVA coefficient sidecar row is missing keys {missing}.")

            dataset_index = int(row["dataset_index"])
            episode_index = int(row["episode_index"])
            frame_index = int(row["frame_index"])
            key = (episode_index, frame_index)
            if dataset_index in self._by_dataset_index:
                raise ValueError(f"Duplicate dataset_index={dataset_index} in {self.sidecar_path}.")
            if key in self._by_episode_frame:
                raise ValueError(f"Duplicate (episode_index, frame_index)={key} in {self.sidecar_path}.")

            if self.expected_basis_mode:
                row_basis_mode_value = row.get("hiva_basis_mode")
                if row_basis_mode_value in (None, ""):
                    if self.expected_basis_mode != "duration_specific":
                        raise ValueError(
                            "Configured canonical HiVA coefficient mode requires a sidecar with "
                            f"hiva_basis_mode={self.expected_basis_mode!r}, but that value is missing."
                        )
                else:
                    row_basis_mode = str(row_basis_mode_value)
                    if row_basis_mode != self.expected_basis_mode:
                        raise ValueError(
                            f"Sidecar hiva_basis_mode={row_basis_mode!r} does not match configured "
                            f"expected_basis_mode={self.expected_basis_mode!r}."
                        )
            self._validate_optional_int(row, "hiva_basis_k", self.n_ctrl)
            self._validate_optional_int(row, "hiva_basis_dmax", self.expected_fit_horizon or self.expected_basis_dmax)
            self._validate_optional_int(row, "hiva_fit_horizon", self.expected_fit_horizon)
            self._validate_optional_int(row, "hiva_exec_dmax", self.expected_basis_dmax)
            self._validate_optional_int(row, "hiva_basis_degree", self.expected_basis_degree)
            self._validate_optional_int(row, "hiva_basis_degree_tr", self.expected_basis_degree_tr)
            self._validate_optional_int(row, "hiva_basis_degree_rot", self.expected_basis_degree_rot)
            self._validate_optional_int(row, "hiva_basis_degree_grip", self.expected_basis_degree_grip)

            duration_label = int(row["duration_label"])
            if duration_label not in self.duration_classes:
                raise ValueError(
                    f"Sidecar duration_label={duration_label} is not in configured classes "
                    f"{self.duration_classes}."
                )

            self._by_dataset_index[dataset_index] = row
            self._by_episode_frame[key] = row

        if len(rows) < len(dataset) and self.require_full_coverage:
            msg = (
                "The HiVA coefficient sidecar has fewer rows than the wrapped dataset "
                f"({len(rows)} vs {len(dataset)})."
            )
            raise ValueError(msg + " Full coefficient training requires full sidecar coverage.")
        if len(rows) != len(dataset):
            warnings.warn(
                "The HiVA coefficient sidecar covers a different number of rows than the wrapped dataset "
                f"({len(rows)} vs {len(dataset)}). This is okay for episode subsets if every requested "
                "sample is present in the sidecar.",
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

            parquet_file = pq.ParquetFile(path)
            available = set(parquet_file.schema_arrow.names)
            missing_required = [key for key in self.REQUIRED_KEYS if key not in available]
            if missing_required:
                raise ValueError(f"HiVA coefficient sidecar {path} is missing required columns {missing_required}.")
            columns = [
                key for key in dict.fromkeys((*self.REQUIRED_KEYS, *self.OPTIONAL_KEYS))
                if key in available
            ]
            return pq.read_table(path, columns=columns).to_pylist()
        if suffix in {".jsonl", ".json"}:
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        raise ValueError(f"Unsupported sidecar format for {path}. Use Parquet, JSONL, or CSV.")

    def _validate_optional_int(self, row: dict[str, Any], key: str, expected: int | None) -> None:
        if expected is None:
            return
        value = row.get(key)
        if value in (None, ""):
            return
        if int(value) != int(expected):
            raise ValueError(
                f"Sidecar {key}={int(value)} does not match configured expected value {int(expected)} "
                f"for dataset_index={row.get('dataset_index')}."
            )

    def _validate_coeff_shape(self, value: Any, expected: tuple[int, int], name: str) -> None:
        tensor = torch.as_tensor(value)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}.")

    def _get_raw_row(self, idx: int) -> dict[str, Any]:
        if hasattr(self.dataset, "get_raw_item"):
            raw = self.dataset.get_raw_item(idx)
        elif hasattr(self.dataset, "hf_dataset"):
            raw = self.dataset.hf_dataset[idx]
        else:
            raise AttributeError(
                "The wrapped dataset does not expose `get_raw_item(idx)` or `hf_dataset[idx]`, "
                "so the HiVA coefficient sidecar cannot recover raw frame identifiers."
            )

        missing = [key for key in self.RAW_ID_KEYS if key not in raw]
        if missing:
            raise KeyError(f"The wrapped dataset raw row is missing required identifiers: {missing}.")
        return raw

    def _row_for_index(self, idx: int) -> dict[str, Any]:
        raw = self._get_raw_row(idx)
        dataset_index = int(raw["index"])
        episode_index = int(raw["episode_index"])
        frame_index = int(raw["frame_index"])

        row = self._by_dataset_index.get(dataset_index)
        if row is None:
            row = self._by_episode_frame.get((episode_index, frame_index))
        if row is None:
            raise KeyError(
                "Could not match a HiVA coefficient sidecar row for sample "
                f"(dataset_index={dataset_index}, episode_index={episode_index}, frame_index={frame_index})."
            )

        if int(row["episode_index"]) != episode_index or int(row["frame_index"]) != frame_index:
            raise ValueError(
                "HiVA coefficient sidecar key mismatch: "
                f"dataset says ({episode_index}, {frame_index}), "
                f"sidecar says ({row['episode_index']}, {row['frame_index']})."
            )
        return row

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        row = self._row_for_index(idx)

        sample["duration_class"] = torch.tensor(int(row["duration_class"]), dtype=torch.long)
        sample["duration_label"] = torch.tensor(int(row["duration_label"]), dtype=torch.long)
        sample["hiva_theta_tr_raw"] = torch.as_tensor(row["hiva_theta_tr_raw"], dtype=torch.float32)
        sample["hiva_theta_rot_raw"] = torch.as_tensor(row["hiva_theta_rot_raw"], dtype=torch.float32)
        sample["hiva_theta_grip_raw"] = torch.as_tensor(row["hiva_theta_grip_raw"], dtype=torch.float32)
        sample["hiva_real_steps"] = torch.tensor(int(row.get("hiva_real_steps", -1)), dtype=torch.long)
        sample["hiva_fit_real_steps"] = torch.tensor(
            int(row.get("hiva_fit_real_steps", row.get("hiva_fit_steps", -1))), dtype=torch.long
        )
        sample["hiva_fit_steps"] = torch.tensor(int(row.get("hiva_fit_steps", -1)), dtype=torch.long)
        sample["hiva_has_synthetic_tail"] = torch.tensor(
            int(row.get("hiva_has_synthetic_tail", 0)), dtype=torch.bool
        )
        sample["hiva_tail_mode_id"] = torch.tensor(
            1 if str(row.get("hiva_tail_mode", "real")) == "terminal_hold" else 0,
            dtype=torch.long,
        )
        for key in (
            "hiva_prefix_tr_rmse",
            "hiva_prefix_rot_rmse",
            "hiva_prefix_grip_rmse",
            "hiva_prefix_action_rmse",
            "hiva_real_prefix_action_rmse",
            "hiva_full_action_rmse",
            "hiva_full_action_max_abs_diff",
            "hiva_wrong_long_tr_energy",
            "hiva_wrong_long_rot_energy",
            "hiva_wrong_long_grip_energy",
            "hiva_wrong_long_action_energy",
            "hiva_tail_motion_energy",
        ):
            if row.get(key) not in (None, ""):
                sample[key] = torch.tensor(float(row[key]), dtype=torch.float32)
        return sample

    def __getitems__(self, indices):
        return [self[idx] for idx in indices]
