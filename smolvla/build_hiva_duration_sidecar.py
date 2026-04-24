from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE


DURATIONS = (1, 3, 8)


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


def write_rows(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, output_path)
        return
    if suffix in {".jsonl", ".json"}:
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return
    raise ValueError("Use a `.parquet` or `.jsonl` output file.")


def build_sidecar_rows(dataset, W1=4, W3=6, merge_window=3, labeler_version="hiva_duration_v1"):
    hf_dataset = dataset.select_columns([ACTION, OBS_STATE, "episode_index", "task_index"])
    grouped = defaultdict(list)
    for dataset_index, row in enumerate(hf_dataset):
        episode_index = int(row["episode_index"])
        grouped[episode_index].append((dataset_index, row))

    rows_out = []
    for episode_index, rows in grouped.items():
        actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for _, row in rows], axis=0)
        state = np.stack([np.asarray(row[OBS_STATE], dtype=np.float32) for _, row in rows], axis=0)
        meta = detect_events_and_labels(actions, state, W1=W1, W3=W3, merge_window=merge_window)

        cmd_switches = set(map(int, meta["cmd_switches"]))
        qpos_switches = set(map(int, meta["qpos_switches"]))
        events = np.asarray(meta["events"], dtype=np.int64)
        T = len(rows)

        for frame_index, (dataset_index, row) in enumerate(rows):
            duration_label = int(meta["duration_labels"][frame_index])
            duration_class = int(meta["duration_classes"][frame_index])
            phase, near_gripper = classify_phase(frame_index, duration_label, events, T, W1=W1, W3=W3)
            dist_to_switch = int(min([abs(frame_index - int(e)) for e in events], default=T))

            sidecar_row = {
                "dataset_index": int(dataset_index),
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "task_index": int(row.get("task_index", -1)) if row.get("task_index") is not None else -1,
                "duration_label": duration_label,
                "duration_class": duration_class,
                "phase": phase,
                "near_gripper": int(near_gripper),
                "dist_to_switch": dist_to_switch,
                "cmd_switch": int(frame_index in cmd_switches),
                "qpos_switch": int(frame_index in qpos_switches),
                "labeler_version": labeler_version,
            }
            rows_out.append(sidecar_row)

    return rows_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset.repo-id", dest="repo_id", required=True)
    parser.add_argument("--dataset.root", dest="root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w1", type=int, default=4)
    parser.add_argument("--w3", type=int, default=6)
    parser.add_argument("--merge-window", type=int, default=3)
    parser.add_argument("--labeler-version", type=str, default="hiva_duration_v1")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    rows = build_sidecar_rows(
        dataset,
        W1=args.w1,
        W3=args.w3,
        merge_window=args.merge_window,
        labeler_version=args.labeler_version,
    )
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
