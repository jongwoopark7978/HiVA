from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from lerobot.utils.constants import ACTION, OBS_STATE


DURATIONS = (1, 3, 8)
REQUIRED_FEATURES = (ACTION, OBS_STATE, "index", "episode_index", "frame_index")


@dataclass(frozen=True)
class EpisodeRange:
    episode_index: int
    dataset_from_index: int
    dataset_to_index: int


def parse_int_ranges(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    values: list[int] = []
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid descending range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def switch_times(binary_state):
    return np.where(binary_state[1:] != binary_state[:-1])[0] + 1


def command_state_from_libero_action(actions, delta=0.1):
    """
    actions: [T, 7] raw LIBERO actions.
    Returns close_state: 1=close command, 0=open command.
    Native assumption: action[-1] < 0 opens, action[-1] > 0 closes.
    """
    u = actions[:, -1]
    close_state = np.zeros(len(u), dtype=np.int64)

    active = np.where(np.abs(u) > delta)[0]
    if len(active) == 0:
        return close_state

    first = active[0]
    close_state[:first] = int(u[first] > delta)
    for t in range(first, len(u)):
        if u[t] > delta:
            close_state[t] = 1
        elif u[t] < -delta:
            close_state[t] = 0
        else:
            close_state[t] = close_state[t - 1]
    return close_state


def gripper_width_from_state(state):
    q_pair = state[:, -2:]
    return np.mean(np.abs(q_pair), axis=-1)


def open_state_from_qpos_hysteresis(state, low_pct=10, high_pct=90, alpha_closed=0.35, alpha_open=0.65):
    q_width = gripper_width_from_state(state)
    q_low = np.percentile(q_width, low_pct)
    q_high = np.percentile(q_width, high_pct)

    if q_high - q_low < 1e-6:
        open_state = np.zeros(len(q_width), dtype=np.int64)
        return open_state, q_width, (q_low, q_high, q_low, q_high)

    q_closed_thr = q_low + alpha_closed * (q_high - q_low)
    q_open_thr = q_low + alpha_open * (q_high - q_low)

    open_state = np.zeros(len(q_width), dtype=np.int64)
    midpoint = 0.5 * (q_low + q_high)
    open_state[0] = int(q_width[0] > midpoint)

    for t in range(1, len(q_width)):
        if q_width[t] > q_open_thr:
            open_state[t] = 1
        elif q_width[t] < q_closed_thr:
            open_state[t] = 0
        else:
            open_state[t] = open_state[t - 1]

    thresholds = (q_low, q_high, q_closed_thr, q_open_thr)
    return open_state, q_width, thresholds


def merge_close_events(events, merge_window=3):
    events = np.array(sorted(set(map(int, events))), dtype=np.int64)
    if len(events) == 0:
        return events
    merged = [events[0]]
    for e in events[1:]:
        if e - merged[-1] > merge_window:
            merged.append(e)
    return np.array(merged, dtype=np.int64)


def find_neighbor_events(s, events, T):
    prev_events = events[events <= s]
    next_events = events[events > s]
    prev_e = int(prev_events[-1]) if len(prev_events) else 0
    next_e = int(next_events[0]) if len(next_events) else T
    return prev_e, next_e


def horizon_safe_duration_label(s, events, T, W1=4, W3=6):
    prev_e, next_e = find_neighbor_events(s, events, T)
    if s - prev_e < W1:
        return 1
    if next_e - s <= W1:
        return 1
    if (s - prev_e >= W1 + W3) and (s + 8 <= next_e - W1):
        return 8
    if s + 3 <= next_e - W1:
        return 3
    return 1


def classify_phase(s, d, events, T, W1=4, W3=6):
    near_gripper = any((s < int(e) + W1) and (s + d > int(e) - W1) for e in events)
    if near_gripper:
        return "near_gripper", True
    prev_e, next_e = find_neighbor_events(s, events, T)
    if (s >= prev_e + W1 + W3) and (s + d <= next_e - W1 - W3):
        return "free_motion", False
    return "other", False


def detect_events_and_labels(actions, state, W1=4, W3=6, merge_window=3):
    T = len(actions)
    close_cmd = command_state_from_libero_action(actions)
    cmd_switches = switch_times(close_cmd)

    open_qpos, q_width, q_thresholds = open_state_from_qpos_hysteresis(state)
    qpos_switches = switch_times(open_qpos)

    events = merge_close_events(np.concatenate([cmd_switches, qpos_switches]), merge_window=merge_window)

    duration_labels = np.array(
        [horizon_safe_duration_label(s, events, T, W1=W1, W3=W3) for s in range(T)],
        dtype=np.int64,
    )

    duration_to_class = {1: 0, 3: 1, 8: 2}
    duration_classes = np.array([duration_to_class[int(d)] for d in duration_labels], dtype=np.int64)

    return {
        "events": events,
        "cmd_switches": cmd_switches,
        "qpos_switches": qpos_switches,
        "q_width": q_width,
        "q_thresholds": q_thresholds,
        "duration_labels": duration_labels,
        "duration_classes": duration_classes,
    }


def load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = set(info.get("features", {}).keys())
    missing = [k for k in REQUIRED_FEATURES if k not in features]
    if missing:
        raise KeyError(
            f"Dataset at {root} is missing required v3 features {missing}. Found: {sorted(features)}"
        )
    return info


def load_episode_ranges(root: Path) -> list[EpisodeRange]:
    episodes_dir = root / "meta" / "episodes"
    ep_ds = pads.dataset(str(episodes_dir), format="parquet")
    required = ("episode_index", "dataset_from_index", "dataset_to_index")
    missing = [k for k in required if k not in ep_ds.schema.names]
    if missing:
        raise KeyError(
            f"Episode metadata under {episodes_dir} is missing required columns {missing}. "
            f"Found: {ep_ds.schema.names}"
        )

    table = ep_ds.to_table(columns=list(required))
    ranges = [
        EpisodeRange(
            episode_index=int(row["episode_index"]),
            dataset_from_index=int(row["dataset_from_index"]),
            dataset_to_index=int(row["dataset_to_index"]),
        )
        for row in table.to_pylist()
    ]
    ranges.sort(key=lambda r: r.episode_index)
    return ranges


def select_episode_ranges(
    episode_ranges: list[EpisodeRange],
    episode_indices: list[int] | None,
    episode_start: int | None,
    max_episodes: int | None,
) -> list[EpisodeRange]:
    if episode_indices:
        wanted = set(episode_indices)
        selected = [r for r in episode_ranges if r.episode_index in wanted]
        found = {r.episode_index for r in selected}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Requested episode indices were not found: {missing}")
        selected.sort(key=lambda r: r.episode_index)
        return selected

    selected = episode_ranges
    if episode_start is not None:
        selected = [r for r in selected if r.episode_index >= episode_start]
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected




def merge_contiguous_spans(selected_ranges: list[EpisodeRange]) -> list[tuple[int, int]]:
    if not selected_ranges:
        return []

    ordered = sorted(selected_ranges, key=lambda r: (r.dataset_from_index, r.dataset_to_index))
    spans: list[tuple[int, int]] = []
    cur_from = ordered[0].dataset_from_index
    cur_to = ordered[0].dataset_to_index
    for r in ordered[1:]:
        if r.dataset_from_index == cur_to:
            cur_to = r.dataset_to_index
        else:
            spans.append((cur_from, cur_to))
            cur_from = r.dataset_from_index
            cur_to = r.dataset_to_index
    spans.append((cur_from, cur_to))
    return spans

def load_frame_rows(root: Path, selected_ranges: list[EpisodeRange]) -> list[dict]:
    data_dir = root / "data"
    data_ds = pads.dataset(str(data_dir), format="parquet")
    needed = [ACTION, OBS_STATE, "index", "episode_index", "frame_index"]
    if "task_index" in data_ds.schema.names:
        needed.append("task_index")

    if not selected_ranges:
        return []

    spans = merge_contiguous_spans(selected_ranges)
    if len(spans) == 1:
        start, end = spans[0]
        filt = (pads.field("index") >= start) & (pads.field("index") < end)
        return data_ds.to_table(columns=needed, filter=filt).to_pylist()

    tables = []
    idx_field = pads.field("index")
    for start, end in spans:
        filt = (idx_field >= start) & (idx_field < end)
        tables.append(data_ds.to_table(columns=needed, filter=filt))

    if not tables:
        return []
    return pa.concat_tables(tables).to_pylist()


def write_rows(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, output_path)
        return
    if suffix in {".jsonl", ".json"}:
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return
    raise ValueError("Use a `.parquet` or `.jsonl` output file.")


def maybe_write_summary(summary: dict, summary_json: Path | None):
    if summary_json is None:
        return
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def build_sidecar_rows_from_frame_rows(frame_rows, W1=4, W3=6, merge_window=3, labeler_version="hiva_duration_v2"):
    grouped = defaultdict(list)
    for row in frame_rows:
        grouped[int(row["episode_index"])].append(row)

    rows_out = []
    for episode_index in sorted(grouped):
        rows = sorted(grouped[episode_index], key=lambda r: (int(r["frame_index"]), int(r["index"])))

        actual_frame_indices = [int(r["frame_index"]) for r in rows]
        expected = list(range(len(rows)))
        if actual_frame_indices != expected:
            print(
                "[warn] episode_index={} has non-canonical frame_index ordering; using sorted order for labeling. "
                "First frame indices: {}".format(episode_index, actual_frame_indices[:10])
            )

        actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in rows], axis=0)
        state = np.stack([np.asarray(row[OBS_STATE], dtype=np.float32) for row in rows], axis=0)
        meta = detect_events_and_labels(actions, state, W1=W1, W3=W3, merge_window=merge_window)

        cmd_switches = set(map(int, meta["cmd_switches"]))
        qpos_switches = set(map(int, meta["qpos_switches"]))
        events = np.asarray(meta["events"], dtype=np.int64)
        T = len(rows)

        episode_task_index = int(rows[0].get("task_index", -1)) if rows[0].get("task_index") is not None else -1
        for local_pos, row in enumerate(rows):
            dataset_index = int(row["index"])
            frame_index = int(row["frame_index"])
            duration_label = int(meta["duration_labels"][local_pos])
            duration_class = int(meta["duration_classes"][local_pos])
            phase, near_gripper = classify_phase(local_pos, duration_label, events, T, W1=W1, W3=W3)
            dist_to_switch = int(min([abs(local_pos - int(e)) for e in events], default=T))

            sidecar_row = {
                "dataset_index": dataset_index,
                "episode_index": int(episode_index),
                "frame_index": frame_index,
                "task_index": episode_task_index,
                "episode_length": T,
                "duration_label": duration_label,
                "duration_class": duration_class,
                "phase": phase,
                "near_gripper": int(near_gripper),
                "dist_to_switch": dist_to_switch,
                "cmd_switch": int(local_pos in cmd_switches),
                "qpos_switch": int(local_pos in qpos_switches),
                "labeler_version": labeler_version,
            }
            rows_out.append(sidecar_row)

    return rows_out


def build_summary(rows: list[dict], dataset_root: str, dataset_repo_id: str, selected_ranges: list[EpisodeRange]) -> dict:
    label_counts = Counter(int(row["duration_label"]) for row in rows)
    class_counts = Counter(int(row["duration_class"]) for row in rows)
    per_episode = Counter(int(row["episode_index"]) for row in rows)
    summary = {
        "dataset_root": dataset_root,
        "dataset_repo_id": dataset_repo_id,
        "num_rows": len(rows),
        "num_episodes": len(per_episode),
        "selected_episode_indices": [int(r.episode_index) for r in selected_ranges],
        "selected_episode_ranges": [
            {
                "episode_index": int(r.episode_index),
                "dataset_from_index": int(r.dataset_from_index),
                "dataset_to_index": int(r.dataset_to_index),
            }
            for r in selected_ranges
        ],
        "duration_label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "duration_class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "rows_per_episode": {str(k): int(v) for k, v in sorted(per_episode.items())},
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a duration-only HiVA-lite sidecar for a local LeRobot v3 dataset. "
            "Subset selection is episode-based so each labeled row still sees its full episode context."
        )
    )
    parser.add_argument("--dataset.repo-id", dest="repo_id", required=True)
    parser.add_argument("--dataset.root", dest="root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w1", type=int, default=4)
    parser.add_argument("--w3", type=int, default=6)
    parser.add_argument("--merge-window", type=int, default=3)
    parser.add_argument("--labeler-version", type=str, default="hiva_duration_v2")
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Comma/range list of episode ids to build, e.g. '0-7,10,12'.",
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=None,
        help="First episode index to include when `--episode-indices` is not given.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to include after `--episode-start` filtering.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary for quick label review.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    load_info(root)
    all_episode_ranges = load_episode_ranges(root)
    requested_episode_indices = parse_int_ranges(args.episode_indices)
    selected_ranges = select_episode_ranges(
        all_episode_ranges,
        episode_indices=requested_episode_indices,
        episode_start=args.episode_start,
        max_episodes=args.max_episodes,
    )
    if not selected_ranges:
        raise ValueError("No episodes were selected. Check `--episode-indices`, `--episode-start`, and `--max-episodes`.")

    frame_rows = load_frame_rows(root, selected_ranges)
    rows = build_sidecar_rows_from_frame_rows(
        frame_rows,
        W1=args.w1,
        W3=args.w3,
        merge_window=args.merge_window,
        labeler_version=args.labeler_version,
    )
    write_rows(rows, args.output)

    summary = build_summary(rows, str(root), args.repo_id, selected_ranges)
    maybe_write_summary(summary, args.summary_json)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
