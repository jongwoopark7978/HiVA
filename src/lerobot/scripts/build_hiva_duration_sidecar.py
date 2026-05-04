from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from lerobot.utils.constants import ACTION, OBS_STATE


DURATIONS = (1, 3, 8)
DURATION_TO_CLASS = {1: 0, 3: 1, 8: 2}
DEFAULT_W1 = 15
DEFAULT_W3 = 15
DEFAULT_MERGE_WINDOW = 3
DEFAULT_MATCH_WINDOW = 20
DEFAULT_PURECONTACT_PCF_META = "meta/purecontact_pcf.json"


def parse_int_ranges(spec: str | None) -> list[int]:
    """Parse strings like '0-3,8,10-12' into sorted unique ints."""
    if spec is None:
        return []
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


def parse_int_tuple(spec: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    if not values:
        raise ValueError("Expected at least one integer duration")
    if tuple(sorted(values)) != values:
        raise ValueError(f"Durations must be sorted ascending, got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"Durations must be unique, got {values}")
    return values


@dataclass(frozen=True)
class SwitchZone:
    pair_index: int
    cmd_frame: int
    qpos_frame: int
    start: int
    end: int


@dataclass(frozen=True)
class FrameContext:
    inside_zone: bool
    active_zone_index: int | None
    active_zone_start: int | None
    active_zone_end: int | None
    prev_switch_end: int | None
    next_switch_start: int | None
    has_prev_zone: bool
    has_next_zone: bool


@dataclass(frozen=True)
class EpisodeLabels:
    cmd_switches_raw: np.ndarray
    qpos_switches_raw: np.ndarray
    cmd_switches_merged: np.ndarray
    qpos_switches_merged: np.ndarray
    unmatched_cmd_switches: np.ndarray
    unmatched_qpos_switches: np.ndarray
    switch_zones: list[SwitchZone]
    duration_labels: np.ndarray
    duration_classes: np.ndarray
    phases: list[str]
    dist_to_switch: np.ndarray
    frame_contexts: list[FrameContext]
    q_width: np.ndarray
    q_thresholds: tuple[float, float, float, float]


def _read_parquet_dataset(path: Path, columns: list[str] | None = None, filter_expr=None):
    dataset = pads.dataset(str(path), format="parquet")
    return dataset.to_table(columns=columns, filter=filter_expr)


def load_episode_indices(root: Path) -> list[int]:
    table = _read_parquet_dataset(root / "meta" / "episodes", columns=["episode_index"])
    eps = sorted(int(row["episode_index"]) for row in table.to_pylist())
    return eps


def select_episode_indices(
    root: Path,
    *,
    explicit_episode_indices: list[int] | None,
    episode_start: int | None,
    max_episodes: int | None,
) -> list[int]:
    available = load_episode_indices(root)
    available_set = set(available)

    if explicit_episode_indices:
        missing = [ep for ep in explicit_episode_indices if ep not in available_set]
        if missing:
            raise KeyError(
                f"Requested episode indices not found in dataset: {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )
        selected = sorted(explicit_episode_indices)
    else:
        selected = available
        if episode_start is not None:
            selected = [ep for ep in selected if ep >= episode_start]
        if max_episodes is not None:
            selected = selected[:max_episodes]

    return selected


def load_episode_rows(root: Path, episode_index: int) -> list[dict[str, Any]]:
    columns = ["index", "episode_index", "frame_index", "task_index", ACTION, OBS_STATE, "timestamp"]
    table = _read_parquet_dataset(
        root / "data",
        columns=columns,
        filter_expr=(pads.field("episode_index") == int(episode_index)),
    )
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row["frame_index"]))
    return rows


def switch_times(binary_state: np.ndarray) -> np.ndarray:
    return np.where(binary_state[1:] != binary_state[:-1])[0] + 1


def command_state_from_libero_action(actions: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """
    actions: [T, 7] raw LIBERO actions.
    Returns close_state: 1=close command, 0=open command.
    Assumes action[-1] < 0 opens, action[-1] > 0 closes.
    """
    u = actions[:, -1]
    close_state = np.zeros(len(u), dtype=np.int64)

    active = np.where(np.abs(u) > delta)[0]
    if len(active) == 0:
        return close_state

    first = int(active[0])
    close_state[:first] = int(u[first] > delta)
    for t in range(first, len(u)):
        if u[t] > delta:
            close_state[t] = 1
        elif u[t] < -delta:
            close_state[t] = 0
        else:
            close_state[t] = close_state[t - 1]
    return close_state


def gripper_width_from_state(state: np.ndarray) -> np.ndarray:
    q_pair = state[:, -2:]
    return np.mean(np.abs(q_pair), axis=-1)


def open_state_from_qpos_hysteresis(
    state: np.ndarray,
    low_pct: float = 10,
    high_pct: float = 90,
    alpha_closed: float = 0.35,
    alpha_open: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    q_width = gripper_width_from_state(state)
    q_low = float(np.percentile(q_width, low_pct))
    q_high = float(np.percentile(q_width, high_pct))

    if q_high - q_low < 1e-6:
        open_state = np.zeros(len(q_width), dtype=np.int64)
        return open_state, q_width, (q_low, q_high, q_low, q_high)

    # Use a more robust threshold calculation:
    # Start with percentile-based thresholds, but clamp them to reasonable ranges
    # based on the observed gripper widths in typical LIBERO episodes
    q_closed_thr = q_low + alpha_closed * (q_high - q_low)
    q_open_thr = q_low + alpha_open * (q_high - q_low)

    # Clamp thresholds to reasonable ranges for LIBERO gripper
    # Typical gripper range is ~0.031 to ~0.040
    q_closed_thr = max(0.030, min(0.035, q_closed_thr))
    q_open_thr = max(0.035, min(0.040, q_open_thr))

    # Additionally, use a fixed threshold fallback for episodes where
    # the percentile-based thresholds don't work well (e.g., episode 69)
    # The fixed threshold of 0.032 works well across episodes
    fixed_q_closed_thr = 0.032
    fixed_q_open_thr = 0.036

    # Use the more sensitive of the two approaches (higher threshold = more sensitive for opening)
    q_closed_thr = max(q_closed_thr, fixed_q_closed_thr)
    q_open_thr = max(q_open_thr, fixed_q_open_thr)

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


def merge_same_type_switches(events: np.ndarray, merge_window: int) -> np.ndarray:
    """
    Merge same-type switch detections only.

    Example with merge_window=5:
      [50, 52] -> [50]
      [50, 57] -> [50, 57]

    We keep the earliest frame in each cluster because it is the safest estimate of
    the onset/completion time for that signal type.
    """
    events = np.array(sorted(set(int(e) for e in events)), dtype=np.int64)
    if len(events) == 0:
        return events

    merged = [int(events[0])]
    cluster_last = int(events[0])
    for e in events[1:]:
        e = int(e)
        if e - cluster_last <= merge_window:
            cluster_last = e
            continue
        merged.append(e)
        cluster_last = e
    return np.array(merged, dtype=np.int64)


def normalize_switch_zones(zones: list[SwitchZone]) -> list[SwitchZone]:
    zones.sort(key=lambda z: (z.start, z.end, z.cmd_frame, z.qpos_frame))
    return [
        SwitchZone(
            pair_index=k,
            cmd_frame=z.cmd_frame,
            qpos_frame=z.qpos_frame,
            start=z.start,
            end=z.end,
        )
        for k, z in enumerate(zones)
    ]


def _better_match(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    """Return True if lhs is a better DP score than rhs.

    Score format: (num_pairs, -total_abs_diff). Maximize pairs first, then minimize
    total absolute frame difference.
    """
    return lhs > rhs


def pair_switches_monotonic(
    cmd_switches: np.ndarray,
    qpos_switches: np.ndarray,
    *,
    match_window: int,
) -> tuple[list[SwitchZone], np.ndarray, np.ndarray]:
    """
    Monotonic one-to-one pairing between merged cmd and merged qpos switches.

    Objective:
      1) maximize number of valid pairs
      2) among those, minimize total abs(cmd - qpos)

    Only pairs with abs(cmd - qpos) <= match_window are allowed.
    Unpaired switches are treated as unmatched singletons and ignored downstream.
    """
    cmd = [int(x) for x in cmd_switches]
    qpos = [int(x) for x in qpos_switches]
    m = len(cmd)
    n = len(qpos)

    dp: list[list[tuple[int, int]]] = [[(0, 0) for _ in range(n + 1)] for _ in range(m + 1)]
    choice: list[list[tuple[str, int, int] | None]] = [[None for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            best = dp[i + 1][j]
            best_choice: tuple[str, int, int] | None = ("skip_cmd", i + 1, j)

            if _better_match(dp[i][j + 1], best):
                best = dp[i][j + 1]
                best_choice = ("skip_qpos", i, j + 1)

            diff = abs(cmd[i] - qpos[j])
            if cmd[i] <= qpos[j] and diff <= match_window:
                cand_suffix = dp[i + 1][j + 1]
                cand = (cand_suffix[0] + 1, cand_suffix[1] - diff)
                if _better_match(cand, best):
                    best = cand
                    best_choice = ("pair", i + 1, j + 1)

            dp[i][j] = best
            choice[i][j] = best_choice

    used_cmd: set[int] = set()
    used_qpos: set[int] = set()
    pairs: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < m and j < n:
        step = choice[i][j]
        if step is None:
            break
        action, ni, nj = step
        if action == "pair":
            pairs.append((cmd[i], qpos[j]))
            used_cmd.add(i)
            used_qpos.add(j)
        i, j = ni, nj

    unmatched_cmd = np.array([cmd[idx] for idx in range(m) if idx not in used_cmd], dtype=np.int64)
    unmatched_qpos = np.array([qpos[idx] for idx in range(n) if idx not in used_qpos], dtype=np.int64)

    zones = normalize_switch_zones(
        [
            SwitchZone(
                pair_index=k,
                cmd_frame=int(c),
                qpos_frame=int(q),
                start=min(int(c), int(q)),
                end=max(int(c), int(q)),
            )
            for k, (c, q) in enumerate(pairs)
        ]
    )

    return zones, unmatched_cmd, unmatched_qpos


def recover_singleton_switch_zones(
    zones: list[SwitchZone],
    unmatched_cmd: np.ndarray,
    unmatched_qpos: np.ndarray,
    *,
    close_cmd: np.ndarray,
    q_width: np.ndarray,
    q_thresholds: tuple[float, float, float, float],
    T: int,
) -> tuple[list[SwitchZone], np.ndarray, np.ndarray]:
    """
    Convert unmatched one-sided detections into valid switch zones.

    If a command switch has no qpos switch, choose the later frame whose measured
    gripper width is closest to the appropriate hysteresis threshold. If a qpos
    switch has no command switch, synthesize a command switch three frames earlier.
    """
    recovered: list[SwitchZone] = []
    remaining_cmd: list[int] = []
    remaining_qpos: list[int] = []
    q_closed_thr = q_thresholds[2]
    q_open_thr = q_thresholds[3]

    known_boundaries = sorted(
        set(
            [int(x) for x in unmatched_cmd.tolist()]
            + [int(x) for x in unmatched_qpos.tolist()]
            + [int(z.start) for z in zones]
        )
    )

    for cmd_frame_raw in unmatched_cmd.tolist():
        cmd_frame = int(cmd_frame_raw)
        start = cmd_frame + 1
        if start >= T:
            remaining_cmd.append(cmd_frame)
            continue

        later_boundaries = [x for x in known_boundaries if x > cmd_frame]
        end = later_boundaries[0] if later_boundaries else T
        if start >= end:
            end = T

        candidate_frames = np.arange(start, end, dtype=np.int64)
        if len(candidate_frames) == 0:
            remaining_cmd.append(cmd_frame)
            continue

        target = q_closed_thr if int(close_cmd[cmd_frame]) == 1 else q_open_thr
        qpos_frame = int(candidate_frames[np.argmin(np.abs(q_width[candidate_frames] - target))])
        recovered.append(
            SwitchZone(
                pair_index=-1,
                cmd_frame=cmd_frame,
                qpos_frame=qpos_frame,
                start=cmd_frame,
                end=qpos_frame,
            )
        )

    for qpos_frame_raw in unmatched_qpos.tolist():
        qpos_frame = int(qpos_frame_raw)
        if qpos_frame <= 0:
            remaining_qpos.append(qpos_frame)
            continue

        cmd_frame = max(0, qpos_frame - 3)
        recovered.append(
            SwitchZone(
                pair_index=-1,
                cmd_frame=cmd_frame,
                qpos_frame=qpos_frame,
                start=cmd_frame,
                end=qpos_frame,
            )
        )

    return (
        normalize_switch_zones([*zones, *recovered]),
        np.asarray(remaining_cmd, dtype=np.int64),
        np.asarray(remaining_qpos, dtype=np.int64),
    )


def frame_context_for_time(s: int, switch_zones: list[SwitchZone], T: int) -> FrameContext:
    if not switch_zones:
        return FrameContext(
            inside_zone=False,
            active_zone_index=None,
            active_zone_start=None,
            active_zone_end=None,
            prev_switch_end=None,
            next_switch_start=None,
            has_prev_zone=False,
            has_next_zone=False,
        )

    for idx, zone in enumerate(switch_zones):
        if zone.start <= s <= zone.end:
            return FrameContext(
                inside_zone=True,
                active_zone_index=idx,
                active_zone_start=zone.start,
                active_zone_end=zone.end,
                prev_switch_end=switch_zones[idx - 1].end if idx > 0 else None,
                next_switch_start=switch_zones[idx + 1].start if idx + 1 < len(switch_zones) else None,
                has_prev_zone=(idx > 0),
                has_next_zone=(idx + 1 < len(switch_zones)),
            )
        if s < zone.start:
            return FrameContext(
                inside_zone=False,
                active_zone_index=None,
                active_zone_start=None,
                active_zone_end=None,
                prev_switch_end=switch_zones[idx - 1].end if idx > 0 else None,
                next_switch_start=zone.start,
                has_prev_zone=(idx > 0),
                has_next_zone=True,
            )

    return FrameContext(
        inside_zone=False,
        active_zone_index=None,
        active_zone_start=None,
        active_zone_end=None,
        prev_switch_end=switch_zones[-1].end,
        next_switch_start=None,
        has_prev_zone=True,
        has_next_zone=False,
    )


def duration_is_horizon_safe(
    s: int,
    d: int,
    *,
    durations: tuple[int, ...],
    prev_e: int | None,
    next_e: int | None,
    has_prev_zone: bool,
    has_next_zone: bool,
    W1: int,
    W3: int,
) -> bool:
    """
    Horizon-safe rule aligned with the corrected TeX logic.

    We use the end of the previous valid interaction zone as e_i and the start of the
    next valid interaction zone as e_{i+1}. For the start-of-trajectory case where no
    previous zone exists, we keep the left side open so early free-space frames can
    still receive the longest duration when safe.
    """
    shortest = min(durations)
    longest = max(durations)
    near_left = has_prev_zone and prev_e is not None and (prev_e <= s < prev_e + W1)
    has_right_limit = has_next_zone and next_e is not None
    near_right = has_right_limit and (next_e - W1 <= s < next_e)
    near_zone = near_left or near_right

    center_start = (prev_e + W1 + W3) if (has_prev_zone and prev_e is not None) else 0
    center_end = (next_e - W1 - W3) if has_right_limit else None
    in_center = center_start <= s and (center_end is None or s < center_end)
    right_safe = (not has_right_limit) or (s + d <= next_e - W1)

    if d == shortest:
        return True

    if d == longest:
        return in_center and right_safe

    return (not near_zone) and right_safe

def violates_phase_rule(
    s: int,
    d: int,
    *,
    prev_e: int | None,
    next_e: int | None,
    has_prev_zone: bool,
    has_next_zone: bool,
    W1: int,
    W3: int,
) -> bool:
    return not duration_is_horizon_safe(
        s,
        d,
        durations=DURATIONS,
        prev_e=prev_e,
        next_e=next_e,
        has_prev_zone=has_prev_zone,
        has_next_zone=has_next_zone,
        W1=W1,
        W3=W3,
    )


def horizon_safe_duration_label(
    s: int,
    *,
    context: FrameContext,
    T: int,
    W1: int,
    W3: int,
    durations: tuple[int, ...] = DURATIONS,
) -> int:
    if context.inside_zone:
        return min(durations)

    prev_e = context.prev_switch_end
    next_e = context.next_switch_start

    valid = [
        d
        for d in durations
        if duration_is_horizon_safe(
            s,
            d,
            durations=durations,
            prev_e=prev_e,
            next_e=next_e,
            has_prev_zone=context.has_prev_zone,
            has_next_zone=context.has_next_zone,
            W1=W1,
            W3=W3,
        )
    ]
    return max(valid) if valid else min(durations)


def classify_phase(
    s: int,
    *,
    duration_label: int,
    context: FrameContext,
    T: int,
    W1: int,
    W3: int,
    durations: tuple[int, ...] = DURATIONS,
) -> str:
    if context.inside_zone:
        return "near_target"

    near_left_target = (
        context.has_prev_zone
        and context.prev_switch_end is not None
        and (context.prev_switch_end <= s < context.prev_switch_end + W1)
    )
    near_right_target = (
        context.has_next_zone
        and context.next_switch_start is not None
        and (context.next_switch_start - W1 <= s < context.next_switch_start)
    )
    if near_left_target or near_right_target:
        return "near_target"

    center_start = (
        (context.prev_switch_end + W1 + W3)
        if (context.has_prev_zone and context.prev_switch_end is not None)
        else 0
    )
    has_right_limit = context.has_next_zone and context.next_switch_start is not None
    center_end = (context.next_switch_start - W1 - W3) if has_right_limit else None
    longest = max(durations)
    right_safe = (not has_right_limit) or (s + longest <= context.next_switch_start - W1)
    if center_start <= s and (center_end is None or s < center_end) and right_safe and duration_label == longest:
        return "free_motion"

    return "approach"


def min_dist_to_valid_switch(s: int, switch_zones: list[SwitchZone], *, inside_zone: bool, T: int) -> int:
    if inside_zone:
        return 0
    if not switch_zones:
        return T
    return int(
        min(
            min(abs(s - zone.cmd_frame), abs(s - zone.qpos_frame))
            for zone in switch_zones
        )
    )


def load_purecontact_pcf_annotations(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    payload = json.loads(path.read_text())
    entries: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("episodes"), dict):
            for ep, item in payload["episodes"].items():
                if isinstance(item, dict):
                    entries.append({"episode_index": int(ep), **item})
        elif isinstance(payload.get("episodes"), list):
            entries.extend([item for item in payload["episodes"] if isinstance(item, dict)])
        elif isinstance(payload.get("videos"), list):
            entries.extend([item for item in payload["videos"] if isinstance(item, dict)])
    elif isinstance(payload, list):
        entries.extend([item for item in payload if isinstance(item, dict)])

    out: dict[int, dict[str, Any]] = {}
    for item in entries:
        if "episode_index" not in item:
            continue
        pcf = item.get("pcf", item.get("contact_onset_frame"))
        if pcf is None:
            continue
        ep = int(item["episode_index"])
        combo = item.get("combo") if isinstance(item.get("combo"), dict) else {}
        out[ep] = {
            "pcf": int(pcf),
            "contact_interval": item.get("contact_interval"),
            "case_category": item.get("case_category"),
            "purecontact_keyword": item.get("purecontact_keyword", combo.get("keyword")),
            "purecontact_verb": item.get("purecontact_verb", combo.get("verb")),
            "method": item.get("method", "sim_contact_plus_target_joint_verification"),
            "reset_mode": item.get("reset_mode"),
            "seed": item.get("seed"),
        }
    return out


def pcf_switch_zones(pcf_frames: list[int]) -> list[SwitchZone]:
    return normalize_switch_zones(
        [
            SwitchZone(pair_index=idx, cmd_frame=int(frame), qpos_frame=int(frame), start=int(frame), end=int(frame))
            for idx, frame in enumerate(sorted(set(int(x) for x in pcf_frames)))
        ]
    )


def pcf_duration_labels(
    T: int,
    *,
    pcf_frames: list[int],
    W1: int,
    W3: int,
    durations: tuple[int, ...],
    duration_to_class: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, list[FrameContext]]:
    if not pcf_frames:
        duration_labels = np.full(T, max(durations), dtype=np.int64)
        duration_classes = np.array([duration_to_class[int(d)] for d in duration_labels], dtype=np.int64)
        contexts = [frame_context_for_time(s, [], T) for s in range(T)]
        return duration_labels, duration_classes, contexts

    zones = pcf_switch_zones(pcf_frames)
    contexts = [frame_context_for_time(s, zones, T) for s in range(T)]
    duration_labels = np.array(
        [
            horizon_safe_duration_label(
                s,
                context=contexts[s],
                T=T,
                W1=W1,
                W3=W3,
                durations=durations,
            )
            for s in range(T)
        ],
        dtype=np.int64,
    )
    duration_classes = np.array([duration_to_class[int(d)] for d in duration_labels], dtype=np.int64)
    return duration_labels, duration_classes, contexts


def dist_to_pure_contact(s: int, pcf_frames: list[int], T: int) -> int:
    if not pcf_frames:
        return T
    return int(min(abs(int(frame) - int(s)) for frame in pcf_frames))


def detect_switch_zones_and_labels(
    actions: np.ndarray,
    state: np.ndarray,
    *,
    W1: int,
    W3: int,
    merge_window: int,
    match_window: int,
    durations: tuple[int, ...],
    duration_to_class: dict[int, int],
) -> EpisodeLabels:
    T = len(actions)
    close_cmd = command_state_from_libero_action(actions)
    cmd_switches_raw = switch_times(close_cmd)

    open_qpos, q_width, q_thresholds = open_state_from_qpos_hysteresis(state)
    qpos_switches_raw = switch_times(open_qpos)

    # Important: same-type merge only. We do not merge cmd with qpos here.
    cmd_switches_merged = merge_same_type_switches(cmd_switches_raw, merge_window=merge_window)
    qpos_switches_merged = merge_same_type_switches(qpos_switches_raw, merge_window=merge_window)

    switch_zones, unmatched_cmd, unmatched_qpos = pair_switches_monotonic(
        cmd_switches_merged,
        qpos_switches_merged,
        match_window=match_window,
    )
    switch_zones, unmatched_cmd, unmatched_qpos = recover_singleton_switch_zones(
        switch_zones,
        unmatched_cmd,
        unmatched_qpos,
        close_cmd=close_cmd,
        q_width=q_width,
        q_thresholds=q_thresholds,
        T=T,
    )

    frame_contexts = [frame_context_for_time(s, switch_zones, T) for s in range(T)]
    duration_labels = np.array(
        [
            horizon_safe_duration_label(
                s,
                context=frame_contexts[s],
                T=T,
                W1=W1,
                W3=W3,
                durations=durations,
            )
            for s in range(T)
        ],
        dtype=np.int64,
    )
    duration_classes = np.array([duration_to_class[int(d)] for d in duration_labels], dtype=np.int64)

    phases = [
        classify_phase(
            s,
            duration_label=int(duration_labels[s]),
            context=frame_contexts[s],
            T=T,
            W1=W1,
            W3=W3,
            durations=durations,
        )
        for s in range(T)
    ]
    dist_to_switch = np.array(
        [min_dist_to_valid_switch(s, switch_zones, inside_zone=frame_contexts[s].inside_zone, T=T) for s in range(T)],
        dtype=np.int64,
    )

    return EpisodeLabels(
        cmd_switches_raw=np.asarray(cmd_switches_raw, dtype=np.int64),
        qpos_switches_raw=np.asarray(qpos_switches_raw, dtype=np.int64),
        cmd_switches_merged=np.asarray(cmd_switches_merged, dtype=np.int64),
        qpos_switches_merged=np.asarray(qpos_switches_merged, dtype=np.int64),
        unmatched_cmd_switches=np.asarray(unmatched_cmd, dtype=np.int64),
        unmatched_qpos_switches=np.asarray(unmatched_qpos, dtype=np.int64),
        switch_zones=switch_zones,
        duration_labels=duration_labels,
        duration_classes=duration_classes,
        phases=phases,
        dist_to_switch=dist_to_switch,
        frame_contexts=frame_contexts,
        q_width=q_width,
        q_thresholds=q_thresholds,
    )


def build_sidecar_rows(
    root: Path,
    *,
    episode_indices: list[int],
    W1: int,
    W3: int,
    merge_window: int,
    match_window: int,
    labeler_version: str,
    durations: tuple[int, ...] = DURATIONS,
    purecontact_annotations: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    duration_to_class = {int(d): i for i, d in enumerate(durations)}

    overall_duration_hist: Counter[str] = Counter()
    overall_duration_s_hist: Counter[str] = Counter()
    overall_duration_p_hist: Counter[str] = Counter()
    overall_phase_hist: Counter[str] = Counter()
    purecontact_annotations = purecontact_annotations or {}

    for episode_index in episode_indices:
        rows = load_episode_rows(root, episode_index)
        if not rows:
            continue

        actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in rows], axis=0)
        state = np.stack([np.asarray(row[OBS_STATE], dtype=np.float32) for row in rows], axis=0)
        task_index = int(rows[0].get("task_index", -1)) if rows[0].get("task_index") is not None else -1

        labels = detect_switch_zones_and_labels(
            actions,
            state,
            W1=W1,
            W3=W3,
            merge_window=merge_window,
            match_window=match_window,
            durations=durations,
            duration_to_class=duration_to_class,
        )

        matched_cmd_frames = {zone.cmd_frame for zone in labels.switch_zones}
        matched_qpos_frames = {zone.qpos_frame for zone in labels.switch_zones}

        pcf_annotation = purecontact_annotations.get(int(episode_index), {})
        pcf_frames = []
        if pcf_annotation.get("pcf") is not None:
            pcf_frames = [int(pcf_annotation["pcf"])]
        duration_labels_p, duration_classes_p, pcf_contexts = pcf_duration_labels(
            len(rows),
            pcf_frames=pcf_frames,
            W1=W1,
            W3=W3,
            durations=durations,
            duration_to_class=duration_to_class,
        )
        duration_labels_final = np.minimum(labels.duration_labels, duration_labels_p)
        duration_classes_final = np.array([duration_to_class[int(d)] for d in duration_labels_final], dtype=np.int64)

        duration_hist = Counter(int(x) for x in duration_labels_final.tolist())
        duration_s_hist = Counter(int(x) for x in labels.duration_labels.tolist())
        duration_p_hist = Counter(int(x) for x in duration_labels_p.tolist())
        phase_hist = Counter(labels.phases)
        overall_duration_hist.update({str(k): int(v) for k, v in duration_hist.items()})
        overall_duration_s_hist.update({str(k): int(v) for k, v in duration_s_hist.items()})
        overall_duration_p_hist.update({str(k): int(v) for k, v in duration_p_hist.items()})
        overall_phase_hist.update(phase_hist)

        for idx_in_episode, row in enumerate(rows):
            dataset_index = int(row["index"])
            frame_index = int(row["frame_index"])
            context = labels.frame_contexts[idx_in_episode]
            pcf_context = pcf_contexts[idx_in_episode]
            phase = labels.phases[idx_in_episode]
            near_target = int(phase == "near_target")
            pcf_frame = int(pcf_frames[0]) if pcf_frames else -1
            contact_interval = pcf_annotation.get("contact_interval") if pcf_annotation else None
            contact_start = int(contact_interval[0]) if isinstance(contact_interval, list) and len(contact_interval) >= 2 else -1
            contact_end = int(contact_interval[1]) if isinstance(contact_interval, list) and len(contact_interval) >= 2 else -1

            sidecar_row = {
                "dataset_index": dataset_index,
                "episode_index": int(episode_index),
                "frame_index": frame_index,
                "task_index": task_index,
                "duration_label": int(duration_labels_final[idx_in_episode]),
                "duration_class": int(duration_classes_final[idx_in_episode]),
                "duration_label_s": int(labels.duration_labels[idx_in_episode]),
                "duration_class_s": int(labels.duration_classes[idx_in_episode]),
                "duration_label_p": int(duration_labels_p[idx_in_episode]),
                "duration_class_p": int(duration_classes_p[idx_in_episode]),
                "phase": phase,
                "near_target": near_target,
                # Backward compatibility for older viewers that still expect near_gripper.
                "near_gripper": near_target,
                "dist_to_switch": int(labels.dist_to_switch[idx_in_episode]),
                # Mark only valid matched switch endpoints. Unmatched singletons are ignored.
                "cmd_switch": int(frame_index in matched_cmd_frames),
                "qpos_switch": int(frame_index in matched_qpos_frames),
                "inside_switch_zone": int(context.inside_zone),
                "prev_switch_end": int(context.prev_switch_end) if context.prev_switch_end is not None else -1,
                "next_switch_start": int(context.next_switch_start) if context.next_switch_start is not None else -1,
                "active_switch_zone_index": int(context.active_zone_index) if context.active_zone_index is not None else -1,
                "active_switch_zone_start": int(context.active_zone_start) if context.active_zone_start is not None else -1,
                "active_switch_zone_end": int(context.active_zone_end) if context.active_zone_end is not None else -1,
                "pcf_frame": pcf_frame,
                "is_pcf": int(frame_index in pcf_frames),
                "dist_to_pure_contact": int(dist_to_pure_contact(frame_index, pcf_frames, len(rows))),
                "inside_purecontact_zone": int(pcf_context.inside_zone),
                "purecontact_case_category": pcf_annotation.get("case_category") if pcf_annotation else None,
                "purecontact_keyword": pcf_annotation.get("purecontact_keyword") if pcf_annotation else None,
                "purecontact_verb": pcf_annotation.get("purecontact_verb") if pcf_annotation else None,
                "purecontact_contact_start": contact_start,
                "purecontact_contact_end": contact_end,
                "purecontact_reset_mode": pcf_annotation.get("reset_mode") if pcf_annotation else None,
                "purecontact_reset_seed": (
                    int(pcf_annotation["seed"])
                    if pcf_annotation and pcf_annotation.get("seed") is not None
                    else -1
                ),
                "labeler_version": labeler_version,
            }
            rows_out.append(sidecar_row)

        episode_summaries.append(
            {
                "episode_index": int(episode_index),
                "task_index": int(task_index),
                "num_frames": int(len(rows)),
                "raw_cmd_switches": [int(x) for x in labels.cmd_switches_raw.tolist()],
                "raw_qpos_switches": [int(x) for x in labels.qpos_switches_raw.tolist()],
                "merged_cmd_switches": [int(x) for x in labels.cmd_switches_merged.tolist()],
                "merged_qpos_switches": [int(x) for x in labels.qpos_switches_merged.tolist()],
                "unmatched_cmd_switches": [int(x) for x in labels.unmatched_cmd_switches.tolist()],
                "unmatched_qpos_switches": [int(x) for x in labels.unmatched_qpos_switches.tolist()],
                "switch_zones": [
                    {
                        "pair_index": int(zone.pair_index),
                        "cmd_frame": int(zone.cmd_frame),
                        "qpos_frame": int(zone.qpos_frame),
                        "start": int(zone.start),
                        "end": int(zone.end),
                    }
                    for zone in labels.switch_zones
                ],
                "q_width_thresholds": {
                    "q_low": float(labels.q_thresholds[0]),
                    "q_high": float(labels.q_thresholds[1]),
                    "q_closed_thr": float(labels.q_thresholds[2]),
                    "q_open_thr": float(labels.q_thresholds[3]),
                },
                "duration_hist": {str(k): int(v) for k, v in sorted(duration_hist.items())},
                "duration_s_hist": {str(k): int(v) for k, v in sorted(duration_s_hist.items())},
                "duration_p_hist": {str(k): int(v) for k, v in sorted(duration_p_hist.items())},
                "phase_hist": {str(k): int(v) for k, v in sorted(phase_hist.items())},
                "purecontact": {
                    "pcf": int(pcf_frames[0]) if pcf_frames else None,
                    "contact_interval": pcf_annotation.get("contact_interval") if pcf_annotation else None,
                    "case_category": pcf_annotation.get("case_category") if pcf_annotation else None,
                    "keyword": pcf_annotation.get("purecontact_keyword") if pcf_annotation else None,
                    "verb": pcf_annotation.get("purecontact_verb") if pcf_annotation else None,
                    "reset_mode": pcf_annotation.get("reset_mode") if pcf_annotation else None,
                    "seed": pcf_annotation.get("seed") if pcf_annotation else None,
                },
            }
        )

    summary = {
        "root": str(root),
        "num_episodes": int(len(episode_summaries)),
        "num_rows": int(len(rows_out)),
        "episode_indices": [int(ep) for ep in episode_indices],
        "W1": int(W1),
        "W3": int(W3),
        "durations": [int(d) for d in durations],
        "duration_to_class": {str(k): int(v) for k, v in duration_to_class.items()},
        "merge_window": int(merge_window),
        "match_window": int(match_window),
        "purecontact_annotation_count": int(len(purecontact_annotations)),
        "duration_hist": {str(k): int(v) for k, v in sorted(overall_duration_hist.items())},
        "duration_s_hist": {str(k): int(v) for k, v in sorted(overall_duration_s_hist.items())},
        "duration_p_hist": {str(k): int(v) for k, v in sorted(overall_duration_p_hist.items())},
        "phase_hist": {str(k): int(v) for k, v in sorted(overall_phase_hist.items())},
        "episodes": episode_summaries,
    }
    return rows_out, summary


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow as pa

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, output_path)
        return
    if suffix in {".jsonl", ".json"}:
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return
    raise ValueError("Use a `.parquet`, `.pq`, `.jsonl`, or `.json` output file.")


def write_summary(summary: dict[str, Any], summary_path: Path | None) -> None:
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset.repo-id", dest="repo_id", required=True)
    parser.add_argument("--dataset.root", dest="root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--episode-indices", type=str, default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--w1", type=int, default=DEFAULT_W1)
    parser.add_argument("--w3", type=int, default=DEFAULT_W3)
    parser.add_argument("--durations", type=str, default=",".join(str(d) for d in DURATIONS))
    parser.add_argument("--merge-window", type=int, default=DEFAULT_MERGE_WINDOW)
    parser.add_argument("--match-window", type=int, default=DEFAULT_MATCH_WINDOW)
    parser.add_argument("--labeler-version", type=str, default="hiva_duration_v9")
    parser.add_argument(
        "--purecontact-pcf-json",
        type=Path,
        default=None,
        help="Optional PCF metadata JSON. Defaults to dataset.root/meta/purecontact_pcf.json when it exists.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    explicit_episode_indices = parse_int_ranges(args.episode_indices)
    selected_episodes = select_episode_indices(
        root,
        explicit_episode_indices=explicit_episode_indices,
        episode_start=args.episode_start,
        max_episodes=args.max_episodes,
    )
    pcf_json = args.purecontact_pcf_json
    if pcf_json is None:
        default_pcf_json = root / DEFAULT_PURECONTACT_PCF_META
        pcf_json = default_pcf_json if default_pcf_json.exists() else None
    purecontact_annotations = load_purecontact_pcf_annotations(pcf_json)

    rows, summary = build_sidecar_rows(
        root,
        episode_indices=selected_episodes,
        W1=args.w1,
        W3=args.w3,
        merge_window=args.merge_window,
        match_window=args.match_window,
        labeler_version=args.labeler_version,
        durations=parse_int_tuple(args.durations),
        purecontact_annotations=purecontact_annotations,
    )
    write_rows(rows, args.output)
    write_summary(summary, args.summary_json)

    print(
        f"Wrote {len(rows)} rows from {len(selected_episodes)} episodes to {args.output}. "
        f"W1={args.w1}, W3={args.w3}, durations={args.durations}, "
        f"merge_window={args.merge_window}, match_window={args.match_window}"
    )
    if args.summary_json is not None:
        print(f"Wrote summary JSON to {args.summary_json}")


if __name__ == "__main__":
    main()
