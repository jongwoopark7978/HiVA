from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path("/nfs/bigbrain/add_disk0/jongwoopark/libero_lerobot_v3_lerobotkeys")
DEFAULT_LIBERO_HDF5_ROOT = Path("/nfs/bigbrain/add_disk0/jongwoopark/LIBERO-datasets")
DEFAULT_LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_FUZZY_MEAN_ABS_DIFF_THRESHOLD = 0.02


def parse_suite_list(spec: str) -> tuple[str, ...]:
    suites = tuple(token.strip() for token in spec.split(",") if token.strip())
    if not suites:
        raise ValueError("At least one LIBERO suite must be provided.")
    return suites


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def action_digest(actions: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest()


def task_text_from_hdf5_name(path: Path) -> str:
    stem = path.name.removesuffix("_demo.hdf5")
    parts = stem.split("_")
    if parts[:2] == ["LIVING", "ROOM"]:
        return "_".join(parts[3:]).replace("_", " ")
    if len(parts) > 2 and parts[0] in {"KITCHEN", "STUDY"} and parts[1].startswith("SCENE"):
        return "_".join(parts[2:]).replace("_", " ")
    return stem.replace("_", " ")


def hdf5_path_by_task(libero_hdf5_root: Path, suites: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    paths: dict[str, dict[str, Any]] = {}
    for suite in suites:
        suite_root = libero_hdf5_root / suite
        if not suite_root.is_dir():
            raise FileNotFoundError(f"Missing LIBERO HDF5 suite directory: {suite_root}")
        for path in sorted(suite_root.glob("*.hdf5")):
            task_slug = slug(task_text_from_hdf5_name(path))
            if task_slug in paths:
                raise ValueError(f"Duplicate HDF5 task slug {task_slug!r}: {paths[task_slug]['path']} and {path}")
            paths[task_slug] = {"path": path, "suite": suite}
    return paths


def load_benchmark_task_metadata(suites: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    try:
        from lerobot.envs.libero import _get_suite
    except Exception:
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    for suite in suites:
        suite_obj = _get_suite(suite)
        for task_id, task in enumerate(suite_obj.tasks):
            metadata[str(task.language)] = {"suite": suite, "task_id": int(task_id)}
    return metadata


def load_benchmark_task_ids(suite: str) -> dict[str, int]:
    """Backward-compatible helper for callers that still import this script."""
    metadata = load_benchmark_task_metadata((suite,))
    return {task: int(values["task_id"]) for task, values in metadata.items()}


def load_hdf5_action_index(
    tasks: pa.Table,
    hdf5_by_task_slug: dict[str, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    index: dict[int, list[dict[str, Any]]] = {}
    for row in tasks.to_pylist():
        task_index = int(row["task_index"])
        task = str(row["task"])
        task_slug = slug(task)
        task_hdf5 = hdf5_by_task_slug.get(task_slug)
        if task_hdf5 is None:
            continue
        hdf5_path = Path(task_hdf5["path"])
        suite = str(task_hdf5["suite"])

        task_demos: list[dict[str, Any]] = []
        with h5py.File(hdf5_path, "r") as h5_file:
            for demo_key in h5_file["data"].keys():
                actions = h5_file["data"][demo_key]["actions"][()].astype(np.float32)
                demo_id = int(demo_key.split("_", 1)[1])
                task_demos.append(
                    {
                        "actions": actions,
                        "length": int(actions.shape[0]),
                        "digest": action_digest(actions),
                        "libero_suite": suite,
                        "libero_init_state_id": demo_id,
                        "libero_demo_key": demo_key,
                        "libero_hdf5_path": str(hdf5_path),
                        "libero_actions_hdf5_path": str(hdf5_path),
                    }
                )
        index[task_index] = task_demos
    return index


def match_demo_actions(
    episode_actions: np.ndarray,
    task_demos: list[dict[str, Any]],
    *,
    fuzzy_mean_abs_diff_threshold: float,
) -> dict[str, Any] | None:
    episode_len = int(episode_actions.shape[0])
    episode_digest = action_digest(episode_actions)

    for demo in task_demos:
        if demo["length"] == episode_len and demo["digest"] == episode_digest:
            return {
                **{key: value for key, value in demo.items() if key not in {"actions", "length", "digest"}},
                "libero_action_match_start": 0,
                "libero_action_match_mean_abs_diff": 0.0,
                "libero_action_match_max_abs_diff": 0.0,
                "libero_init_state_source": "downloaded_libero_hdf5_action_hash",
            }

    best: tuple[float, float, dict[str, Any], int] | None = None
    second: tuple[float, float, dict[str, Any], int] | None = None
    for demo in task_demos:
        raw_actions = demo["actions"]
        if len(raw_actions) < episode_len:
            continue
        for start in range(0, len(raw_actions) - episode_len + 1):
            diff = np.abs(raw_actions[start : start + episode_len] - episode_actions)
            score = (float(np.mean(diff)), float(np.max(diff)), demo, int(start))
            if best is None or score[:2] < best[:2]:
                second = best
                best = score
            elif second is None or score[:2] < second[:2]:
                second = score

    if best is None or best[0] > fuzzy_mean_abs_diff_threshold:
        return None
    if second is not None and second[:2] == best[:2]:
        return None

    mean_abs_diff, max_abs_diff, demo, start = best
    return {
        **{key: value for key, value in demo.items() if key not in {"actions", "length", "digest"}},
        "libero_action_match_start": start,
        "libero_action_match_mean_abs_diff": mean_abs_diff,
        "libero_action_match_max_abs_diff": max_abs_diff,
        "libero_init_state_source": "downloaded_libero_hdf5_action_window",
    }


def add_or_replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, values)
    return table.append_column(name, values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill LIBERO source demo/init-state metadata into a LeRobot v3 episodes parquet."
    )
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--libero-hdf5-root", type=Path, default=DEFAULT_LIBERO_HDF5_ROOT)
    parser.add_argument("--suites", type=parse_suite_list, default=DEFAULT_LIBERO_SUITES)
    parser.add_argument(
        "--fuzzy-mean-abs-diff-threshold",
        type=float,
        default=DEFAULT_FUZZY_MEAN_ABS_DIFF_THRESHOLD,
        help="Mean absolute action-difference threshold for matching trimmed downloaded HDF5 demos.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = root / "meta" / "tasks.parquet"
    data_root = root / "data"

    episodes = pq.read_table(episodes_path)
    tasks = pq.read_table(tasks_path)
    data = pads.dataset(str(data_root), format="parquet").to_table(
        columns=["episode_index", "frame_index", "task_index", "action"]
    )

    hdf5_index = load_hdf5_action_index(tasks, hdf5_path_by_task(args.libero_hdf5_root, args.suites))
    benchmark_metadata = load_benchmark_task_metadata(args.suites)
    episode_tasks = {
        int(row["episode_index"]): str(row["tasks"][0]) if row["tasks"] else ""
        for row in episodes.select(["episode_index", "tasks"]).to_pylist()
    }

    matches: dict[int, dict[str, Any]] = {}
    misses: list[dict[str, int]] = []
    for episode_index in sorted(set(data.column("episode_index").to_pylist())):
        episode_table = data.filter(pc.equal(data["episode_index"], pa.scalar(episode_index, pa.int64())))
        rows = sorted(episode_table.to_pylist(), key=lambda row: int(row["frame_index"]))
        if not rows:
            continue
        task_index = int(rows[0]["task_index"])
        task_hdf5_index = hdf5_index.get(task_index)
        if task_hdf5_index is None:
            continue
        actions = np.stack([np.asarray(row["action"], dtype=np.float32) for row in rows])
        match = match_demo_actions(
            actions,
            task_hdf5_index,
            fuzzy_mean_abs_diff_threshold=float(args.fuzzy_mean_abs_diff_threshold),
        )
        if match is None:
            misses.append(
                {
                    "episode_index": int(episode_index),
                    "task_index": int(task_index),
                    "length": int(actions.shape[0]),
                }
            )
            continue
        task_text = episode_tasks.get(int(episode_index), "")
        task_metadata = benchmark_metadata.get(task_text, {})
        matches[int(episode_index)] = {
            **match,
            "libero_suite": task_metadata.get("suite", match["libero_suite"]),
            "libero_benchmark_task_id": task_metadata.get("task_id"),
        }

    init_ids: list[int | None] = []
    demo_keys: list[str | None] = []
    suites: list[str | None] = []
    hdf5_paths: list[str | None] = []
    action_hdf5_paths: list[str | None] = []
    sources: list[str | None] = []
    benchmark_task_ids_col: list[int | None] = []
    action_match_starts: list[int | None] = []
    action_match_mean_diffs: list[float | None] = []
    action_match_max_diffs: list[float | None] = []
    for row in episodes.to_pylist():
        episode_index = int(row["episode_index"])
        match = matches.get(episode_index)
        init_ids.append(None if match is None else int(match["libero_init_state_id"]))
        demo_keys.append(None if match is None else str(match["libero_demo_key"]))
        suites.append(None if match is None else str(match["libero_suite"]))
        hdf5_paths.append(None if match is None else str(match["libero_hdf5_path"]))
        action_hdf5_paths.append(None if match is None else str(match["libero_actions_hdf5_path"]))
        sources.append(None if match is None else str(match["libero_init_state_source"]))
        benchmark_task_ids_col.append(
            None if match is None or match["libero_benchmark_task_id"] is None else int(match["libero_benchmark_task_id"])
        )
        action_match_starts.append(None if match is None else int(match["libero_action_match_start"]))
        action_match_mean_diffs.append(None if match is None else float(match["libero_action_match_mean_abs_diff"]))
        action_match_max_diffs.append(None if match is None else float(match["libero_action_match_max_abs_diff"]))

    updated = episodes
    updated = add_or_replace_column(updated, "libero_init_state_id", pa.array(init_ids, type=pa.int64()))
    updated = add_or_replace_column(updated, "libero_demo_key", pa.array(demo_keys, type=pa.string()))
    updated = add_or_replace_column(updated, "libero_suite", pa.array(suites, type=pa.string()))
    updated = add_or_replace_column(updated, "libero_hdf5_path", pa.array(hdf5_paths, type=pa.string()))
    updated = add_or_replace_column(
        updated, "libero_actions_hdf5_path", pa.array(action_hdf5_paths, type=pa.string())
    )
    updated = add_or_replace_column(updated, "libero_init_state_source", pa.array(sources, type=pa.string()))
    updated = add_or_replace_column(updated, "libero_benchmark_task_id", pa.array(benchmark_task_ids_col, type=pa.int64()))
    updated = add_or_replace_column(updated, "libero_action_match_start", pa.array(action_match_starts, type=pa.int64()))
    updated = add_or_replace_column(
        updated,
        "libero_action_match_mean_abs_diff",
        pa.array(action_match_mean_diffs, type=pa.float64()),
    )
    updated = add_or_replace_column(
        updated,
        "libero_action_match_max_abs_diff",
        pa.array(action_match_max_diffs, type=pa.float64()),
    )

    matched_by_source = Counter(str(match["libero_init_state_source"]) for match in matches.values())
    matched_by_suite = Counter(str(match["libero_suite"]) for match in matches.values())
    report = {
        "dataset_root": str(root),
        "libero_hdf5_root": str(args.libero_hdf5_root),
        "suites": list(args.suites),
        "episodes_total": int(episodes.num_rows),
        "matched_episodes": len(matches),
        "unmatched_episodes": len(misses),
        "matched_by_source": dict(sorted(matched_by_source.items())),
        "matched_by_suite": dict(sorted(matched_by_suite.items())),
        "fuzzy_mean_abs_diff_threshold": float(args.fuzzy_mean_abs_diff_threshold),
        "unmatched_examples": misses[:25],
        "columns": [
            "libero_init_state_id",
            "libero_demo_key",
            "libero_suite",
            "libero_hdf5_path",
            "libero_actions_hdf5_path",
            "libero_init_state_source",
            "libero_benchmark_task_id",
            "libero_action_match_start",
            "libero_action_match_mean_abs_diff",
            "libero_action_match_max_abs_diff",
        ],
    }
    print(json.dumps(report, indent=2))

    if args.dry_run:
        return

    if not args.no_backup:
        backup_dir = root / "meta" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / (
            f"episodes_file-000.backup_before_libero_init_state_id_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        shutil.copy2(episodes_path, backup_path)
        print(f"Backed up {episodes_path} to {backup_path}")

    pq.write_table(updated, episodes_path)
    print(f"Wrote updated metadata to {episodes_path}")


if __name__ == "__main__":
    main()
