from __future__ import annotations

"""Build priority-based contact-commit HiVA duration sidecars.

This labeler is intended for coefficient HiVA / LP-MT runs where very short
execution horizons near a contact or gripper switch can create hesitation and
misalignment between motion and gripper commands.  Instead of assigning the
shortest duration inside interaction zones, it assigns a medium "commit"
duration around the landmark so one model call can complete the local contact
sequence smoothly.

Profiles:
  v1: D={4,6,10}, commit=6, transition=4, free=10
  v2: D={4,10}, commit=4, transition=4, free=10
  v3: D={2,4,6,10}, pre-near=2, pre-far=4, commit=6, post=4, free=10
  v4: D={2,4,10}, pre-near=2, pre-far=4, commit=4, post=4, free=10
  v5: D={4,6,10}, wide commit=6, transition=4, free=10
  v6: D={4,10}, wide commit=4, transition=4, free=10
  v7: D={2,4,6,10}, wide pre-near=2, pre-far=4, commit=6, post=4, free=10
  v8: D={2,4,10}, wide pre-near=2, pre-far=4, commit=4, post=4, free=10
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.scripts.build_hiva_duration_sidecar import (
    DEFAULT_MATCH_WINDOW,
    DEFAULT_MERGE_WINDOW,
    DEFAULT_PURECONTACT_PCF_META,
    SwitchZone,
    detect_switch_zones_and_labels,
    dist_to_pure_contact,
    load_episode_rows,
    load_purecontact_pcf_annotations,
    parse_int_ranges,
    parse_int_tuple,
    pcf_switch_zones,
    select_episode_indices,
    write_rows,
    write_summary,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass(frozen=True)
class ContactCommitProfile:
    name: str
    durations: tuple[int, ...]
    free_duration: int
    commit_duration: int
    transition_duration: int
    pre_near_duration: int | None = None
    commit_pre: int = 4
    commit_post: int = 4
    pre_far_start: int = 24
    pre_far_end: int = 15
    pre_near_start: int = 14
    pre_near_end: int = 5
    post_start: int = 5
    post_end: int = 14


PROFILES: dict[str, ContactCommitProfile] = {
    "v1": ContactCommitProfile(
        name="v1_commit6_d4_6_10",
        durations=(4, 6, 10),
        free_duration=10,
        commit_duration=6,
        transition_duration=4,
        pre_near_duration=None,
    ),
    "v2": ContactCommitProfile(
        name="v2_commit4_d4_10",
        durations=(4, 10),
        free_duration=10,
        commit_duration=4,
        transition_duration=4,
        pre_near_duration=None,
    ),
    "v3": ContactCommitProfile(
        name="v3_prenear2_commit6_d2_4_6_10",
        durations=(2, 4, 6, 10),
        free_duration=10,
        commit_duration=6,
        transition_duration=4,
        pre_near_duration=2,
    ),
    "v4": ContactCommitProfile(
        name="v4_prenear2_commit4_d2_4_10",
        durations=(2, 4, 10),
        free_duration=10,
        commit_duration=4,
        transition_duration=4,
        pre_near_duration=2,
    ),
    "v5": ContactCommitProfile(
        name="v5_wide_commit6_d4_6_10",
        durations=(4, 6, 10),
        free_duration=10,
        commit_duration=6,
        transition_duration=4,
        pre_near_duration=None,
        commit_pre=10,
        commit_post=10,
        pre_far_start=40,
        pre_far_end=11,
        post_start=11,
        post_end=30,
    ),
    "v6": ContactCommitProfile(
        name="v6_wide_commit4_d4_10",
        durations=(4, 10),
        free_duration=10,
        commit_duration=4,
        transition_duration=4,
        pre_near_duration=None,
        commit_pre=10,
        commit_post=10,
        pre_far_start=40,
        pre_far_end=11,
        post_start=11,
        post_end=30,
    ),
    "v7": ContactCommitProfile(
        name="v7_wide_prenear2_commit6_d2_4_6_10",
        durations=(2, 4, 6, 10),
        free_duration=10,
        commit_duration=6,
        transition_duration=4,
        pre_near_duration=2,
        commit_pre=10,
        commit_post=10,
        pre_far_start=40,
        pre_far_end=26,
        pre_near_start=25,
        pre_near_end=11,
        post_start=11,
        post_end=30,
    ),
    "v8": ContactCommitProfile(
        name="v8_wide_prenear2_commit4_d2_4_10",
        durations=(2, 4, 10),
        free_duration=10,
        commit_duration=4,
        transition_duration=4,
        pre_near_duration=2,
        commit_pre=10,
        commit_post=10,
        pre_far_start=40,
        pre_far_end=26,
        pre_near_start=25,
        pre_near_end=11,
        post_start=11,
        post_end=30,
    ),
}


@dataclass(frozen=True)
class LabelDecision:
    duration: int
    class_id: int
    phase: str
    priority: int
    source: str
    zone_start: int
    zone_end: int


def _validate_profile(profile: ContactCommitProfile) -> None:
    durations = tuple(sorted(set(int(d) for d in profile.durations)))
    if durations != profile.durations:
        raise ValueError(f"Profile durations must be sorted and unique, got {profile.durations}.")
    for value_name in ("free_duration", "commit_duration", "transition_duration"):
        value = int(getattr(profile, value_name))
        if value not in durations:
            raise ValueError(f"{value_name}={value} must be in durations={durations}.")
    if profile.pre_near_duration is not None and int(profile.pre_near_duration) not in durations:
        raise ValueError(
            f"pre_near_duration={profile.pre_near_duration} must be in durations={durations}."
        )
    if min(profile.commit_pre, profile.commit_post, profile.pre_far_start, profile.pre_far_end) < 0:
        raise ValueError(f"Profile window sizes must be non-negative: {profile}.")


def _candidate_decisions_for_zone(
    s: int,
    zone: SwitchZone,
    *,
    profile: ContactCommitProfile,
    duration_to_class: dict[int, int],
    source: str,
) -> list[LabelDecision]:
    """Return all duration candidates proposed by one interaction interval.

    The priority order intentionally uses a medium commit duration over the
    actual contact/switch interval rather than the shortest duration.  The
    pre-near rule has higher priority than generic transition zones, so a
    future contact can override a previous post-contact transition zone.
    """

    b = int(zone.start)
    e = int(zone.end)
    candidates: list[LabelDecision] = []

    def add_if(cond: bool, *, duration: int, phase: str, priority: int, z0: int, z1: int) -> None:
        if not cond:
            return
        candidates.append(
            LabelDecision(
                duration=int(duration),
                class_id=int(duration_to_class[int(duration)]),
                phase=phase,
                priority=int(priority),
                source=source,
                zone_start=int(z0),
                zone_end=int(z1),
            )
        )

    # Highest priority: commit through the local interaction interval.
    commit_start = b - profile.commit_pre
    commit_end = e + profile.commit_post
    add_if(
        commit_start <= s <= commit_end,
        duration=profile.commit_duration,
        phase="contact_commit",
        priority=30,
        z0=commit_start,
        z1=commit_end,
    )

    # For v3/v4, a narrow immediately-pre-contact zone can use duration 2.
    # This has higher priority than generic transition zones but lower priority
    # than the contact-commit zone itself.
    if profile.pre_near_duration is not None:
        pre_near_start = b - profile.pre_near_start
        pre_near_end = b - profile.pre_near_end
        add_if(
            pre_near_start <= s <= pre_near_end,
            duration=profile.pre_near_duration,
            phase="pre_near_contact",
            priority=20,
            z0=pre_near_start,
            z1=pre_near_end,
        )
        pre_far_start = b - profile.pre_far_start
        pre_far_end = b - profile.pre_far_end
    else:
        # For v1/v2, the full pre-contact transition window is duration 4.
        pre_far_start = b - profile.pre_far_start
        pre_far_end = b - profile.pre_near_end

    add_if(
        pre_far_start <= s <= pre_far_end,
        duration=profile.transition_duration,
        phase="pre_contact_transition",
        priority=10,
        z0=pre_far_start,
        z1=pre_far_end,
    )

    post_start = e + profile.post_start
    post_end = e + profile.post_end
    add_if(
        post_start <= s <= post_end,
        duration=profile.transition_duration,
        phase="post_contact_transition",
        priority=10,
        z0=post_start,
        z1=post_end,
    )

    return candidates


def contact_commit_label(
    s: int,
    zones: list[SwitchZone],
    *,
    profile: ContactCommitProfile,
    duration_to_class: dict[int, int],
    source: str,
) -> LabelDecision:
    candidates: list[LabelDecision] = []
    for zone in zones:
        candidates.extend(
            _candidate_decisions_for_zone(
                s,
                zone,
                profile=profile,
                duration_to_class=duration_to_class,
                source=source,
            )
        )

    if not candidates:
        return LabelDecision(
            duration=int(profile.free_duration),
            class_id=int(duration_to_class[int(profile.free_duration)]),
            phase="free_motion",
            priority=0,
            source=source,
            zone_start=-1,
            zone_end=-1,
        )

    # Highest priority wins.  If two candidates have the same priority, use the shorter duration.
    return sorted(candidates, key=lambda x: (-x.priority, x.duration, x.zone_start, x.zone_end))[0]


def combined_contact_commit_label(
    s: int,
    switch_zones: list[SwitchZone],
    pcf_zones: list[SwitchZone],
    *,
    profile: ContactCommitProfile,
    duration_to_class: dict[int, int],
) -> LabelDecision:
    candidates: list[LabelDecision] = []
    for zone in switch_zones:
        candidates.extend(
            _candidate_decisions_for_zone(
                s,
                zone,
                profile=profile,
                duration_to_class=duration_to_class,
                source="switch",
            )
        )
    for zone in pcf_zones:
        candidates.extend(
            _candidate_decisions_for_zone(
                s,
                zone,
                profile=profile,
                duration_to_class=duration_to_class,
                source="pcf",
            )
        )

    if not candidates:
        return LabelDecision(
            duration=int(profile.free_duration),
            class_id=int(duration_to_class[int(profile.free_duration)]),
            phase="free_motion",
            priority=0,
            source="none",
            zone_start=-1,
            zone_end=-1,
        )

    # Source priority only breaks exact ties.  PCF and switch are both interaction landmarks, but
    # PCF is usually more contact-specific when all other fields match.
    source_rank = {"pcf": 0, "switch": 1, "none": 2}
    return sorted(
        candidates,
        key=lambda x: (-x.priority, x.duration, source_rank.get(x.source, 3), x.zone_start, x.zone_end),
    )[0]


def _nearest_zone_distance(s: int, zones: list[SwitchZone], T: int) -> int:
    if not zones:
        return int(T)
    best = T
    for zone in zones:
        if zone.start <= s <= zone.end:
            return 0
        best = min(best, abs(s - int(zone.start)), abs(s - int(zone.end)))
    return int(best)


def _maybe_json_load(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _as_list(value: Any) -> list[Any]:
    value = _maybe_json_load(value)
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, float) and np.isnan(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _switch_zones_from_cached_value(value: Any) -> list[SwitchZone]:
    zones: list[SwitchZone] = []
    for i, item in enumerate(_as_list(value)):
        item = _maybe_json_load(item)
        if not isinstance(item, dict):
            continue
        start = _as_int(item.get("start"))
        end = _as_int(item.get("end"))
        cmd_frame = _as_int(item.get("cmd_frame"), start)
        qpos_frame = _as_int(item.get("qpos_frame"), end)
        if start < 0 or end < start:
            continue
        zones.append(
            SwitchZone(
                pair_index=_as_int(item.get("pair_index"), i),
                cmd_frame=cmd_frame,
                qpos_frame=qpos_frame,
                start=start,
                end=end,
            )
        )
    return zones


def _pcf_frames_from_cached_row(row: dict[str, Any]) -> list[int]:
    frames = [_as_int(x) for x in _as_list(row.get("pcf_frames"))]
    return [frame for frame in frames if frame >= 0]


def _pcf_annotation_from_cached_row(row: dict[str, Any]) -> dict[str, Any]:
    contact_start = _as_int(row.get("purecontact_contact_start"))
    contact_end = _as_int(row.get("purecontact_contact_end"))
    pcf_frames = _pcf_frames_from_cached_row(row)
    annotation: dict[str, Any] = {
        "pcf": int(pcf_frames[0]) if pcf_frames else None,
        "case_category": _none_if_nan(row.get("purecontact_case_category")),
        "purecontact_keyword": _none_if_nan(row.get("purecontact_keyword")),
        "purecontact_verb": _none_if_nan(row.get("purecontact_verb")),
        "reset_mode": _none_if_nan(row.get("purecontact_reset_mode")),
        "seed": None
        if _as_int(row.get("purecontact_reset_seed")) < 0
        else _as_int(row.get("purecontact_reset_seed")),
    }
    if contact_start >= 0 and contact_end >= contact_start:
        annotation["contact_interval"] = [int(contact_start), int(contact_end)]
    else:
        annotation["contact_interval"] = None
    return annotation


def _pcf_zones_from_cached_row(row: dict[str, Any]) -> list[SwitchZone]:
    contact_start = _as_int(row.get("purecontact_contact_start"))
    contact_end = _as_int(row.get("purecontact_contact_end"))
    if contact_start >= 0 and contact_end >= contact_start:
        return [
            SwitchZone(
                pair_index=0,
                cmd_frame=contact_start,
                qpos_frame=contact_end,
                start=contact_start,
                end=contact_end,
            )
        ]
    pcf_frames = _pcf_frames_from_cached_row(row)
    return pcf_switch_zones(pcf_frames) if pcf_frames else []


def load_interaction_zone_sidecar(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    df = pd.read_parquet(path)
    if "episode_index" not in df.columns:
        raise ValueError(f"Interaction-zone sidecar is missing episode_index: {path}")
    out: dict[int, dict[str, Any]] = {}
    for record in df.to_dict(orient="records"):
        out[int(record["episode_index"])] = record
    return out


def build_sidecar_rows(
    root: Path,
    *,
    episode_indices: list[int],
    profile: ContactCommitProfile,
    merge_window: int,
    match_window: int,
    labeler_version: str,
    purecontact_annotations: dict[int, dict[str, Any]] | None = None,
    interaction_zones_by_episode: dict[int, dict[str, Any]] | None = None,
    interaction_zone_sidecar: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_profile(profile)
    duration_to_class = {int(d): i for i, d in enumerate(profile.durations)}
    rows_out: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    purecontact_annotations = purecontact_annotations or {}
    interaction_zones_by_episode = interaction_zones_by_episode or {}

    overall_duration_hist: Counter[str] = Counter()
    overall_duration_s_hist: Counter[str] = Counter()
    overall_duration_p_hist: Counter[str] = Counter()
    overall_phase_hist: Counter[str] = Counter()
    overall_source_hist: Counter[str] = Counter()
    overall_priority_hist: Counter[str] = Counter()

    for episode_index in episode_indices:
        rows = load_episode_rows(root, episode_index)
        if not rows:
            continue

        task_index = int(rows[0].get("task_index", -1)) if rows[0].get("task_index") is not None else -1

        cached_zones = interaction_zones_by_episode.get(int(episode_index))
        if cached_zones is not None:
            switch_zones = _switch_zones_from_cached_value(cached_zones.get("switch_zones"))
            matched_cmd_frames = {zone.cmd_frame for zone in switch_zones}
            matched_qpos_frames = {zone.qpos_frame for zone in switch_zones}
            pcf_annotation = _pcf_annotation_from_cached_row(cached_zones)
            pcf_frames = _pcf_frames_from_cached_row(cached_zones)
            pcf_zones = _pcf_zones_from_cached_row(cached_zones)
        else:
            actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in rows], axis=0)
            state = np.stack([np.asarray(row[OBS_STATE], dtype=np.float32) for row in rows], axis=0)
            switch_labels = detect_switch_zones_and_labels(
                actions,
                state,
                W1=1,
                W3=0,
                merge_window=merge_window,
                match_window=match_window,
                durations=profile.durations,
                duration_to_class=duration_to_class,
            )
            switch_zones = switch_labels.switch_zones
            matched_cmd_frames = {zone.cmd_frame for zone in switch_zones}
            matched_qpos_frames = {zone.qpos_frame for zone in switch_zones}

            pcf_annotation = purecontact_annotations.get(int(episode_index), {})
            pcf_frames = []
            if pcf_annotation.get("pcf") is not None:
                pcf_frames = [int(pcf_annotation["pcf"])]
            contact_interval = pcf_annotation.get("contact_interval") if pcf_annotation else None
            contact_start = (
                int(contact_interval[0])
                if isinstance(contact_interval, list) and len(contact_interval) >= 2
                else -1
            )
            contact_end = (
                int(contact_interval[1])
                if isinstance(contact_interval, list) and len(contact_interval) >= 2
                else -1
            )
            if contact_start >= 0 and contact_end >= contact_start:
                pcf_zones = [
                    SwitchZone(
                        pair_index=0,
                        cmd_frame=contact_start,
                        qpos_frame=contact_end,
                        start=contact_start,
                        end=contact_end,
                    )
                ]
            else:
                pcf_zones = pcf_switch_zones(pcf_frames) if pcf_frames else []

        contact_interval = pcf_annotation.get("contact_interval") if pcf_annotation else None
        contact_start = (
            int(contact_interval[0])
            if isinstance(contact_interval, list) and len(contact_interval) >= 2
            else -1
        )
        contact_end = (
            int(contact_interval[1])
            if isinstance(contact_interval, list) and len(contact_interval) >= 2
            else -1
        )

        duration_hist: Counter[int] = Counter()
        duration_s_hist: Counter[int] = Counter()
        duration_p_hist: Counter[int] = Counter()
        phase_hist: Counter[str] = Counter()
        source_hist: Counter[str] = Counter()
        priority_hist: Counter[str] = Counter()

        switch_decisions = [
            contact_commit_label(
                s,
                switch_zones,
                profile=profile,
                duration_to_class=duration_to_class,
                source="switch",
            )
            for s in range(len(rows))
        ]
        pcf_decisions = [
            contact_commit_label(
                s,
                pcf_zones,
                profile=profile,
                duration_to_class=duration_to_class,
                source="pcf",
            )
            for s in range(len(rows))
        ]
        final_decisions = [
            combined_contact_commit_label(
                s,
                switch_zones,
                pcf_zones,
                profile=profile,
                duration_to_class=duration_to_class,
            )
            for s in range(len(rows))
        ]

        for idx_in_episode, row in enumerate(rows):
            dataset_index = int(row["index"])
            frame_index = int(row["frame_index"])
            final = final_decisions[idx_in_episode]
            switch_decision = switch_decisions[idx_in_episode]
            pcf_decision = pcf_decisions[idx_in_episode]
            pcf_frame = int(pcf_frames[0]) if pcf_frames else -1
            inside_switch_zone = any(zone.start <= idx_in_episode <= zone.end for zone in switch_zones)
            inside_pcf_zone = any(zone.start <= idx_in_episode <= zone.end for zone in pcf_zones)

            duration_hist[final.duration] += 1
            duration_s_hist[switch_decision.duration] += 1
            duration_p_hist[pcf_decision.duration] += 1
            phase_hist[final.phase] += 1
            source_hist[final.source] += 1
            priority_hist[str(final.priority)] += 1

            sidecar_row = {
                "dataset_index": dataset_index,
                "episode_index": int(episode_index),
                "frame_index": frame_index,
                "task_index": task_index,
                "duration_label": int(final.duration),
                "duration_class": int(final.class_id),
                "duration_label_s": int(switch_decision.duration),
                "duration_class_s": int(switch_decision.class_id),
                "duration_label_p": int(pcf_decision.duration),
                "duration_class_p": int(pcf_decision.class_id),
                "phase": final.phase,
                "near_target": int(final.priority > 0),
                "near_gripper": int(final.priority > 0),
                "contact_commit_source": final.source,
                "contact_commit_priority": int(final.priority),
                "contact_commit_zone_start": int(final.zone_start),
                "contact_commit_zone_end": int(final.zone_end),
                "contact_commit_window_pre": int(profile.commit_pre),
                "contact_commit_window_post": int(profile.commit_post),
                "contact_pre_far_window_start": int(profile.pre_far_start),
                "contact_pre_far_window_end": int(profile.pre_far_end),
                "contact_pre_near_window_start": int(profile.pre_near_start),
                "contact_pre_near_window_end": int(profile.pre_near_end),
                "contact_post_window_start": int(profile.post_start),
                "contact_post_window_end": int(profile.post_end),
                "contact_free_duration": int(profile.free_duration),
                "contact_commit_duration": int(profile.commit_duration),
                "contact_transition_duration": int(profile.transition_duration),
                "contact_pre_near_duration": (
                    -1 if profile.pre_near_duration is None else int(profile.pre_near_duration)
                ),
                "dist_to_switch": _nearest_zone_distance(idx_in_episode, switch_zones, len(rows)),
                "cmd_switch": int(frame_index in matched_cmd_frames),
                "qpos_switch": int(frame_index in matched_qpos_frames),
                "inside_switch_zone": int(inside_switch_zone),
                "prev_switch_end": -1,
                "next_switch_start": -1,
                "active_switch_zone_index": -1,
                "active_switch_zone_start": -1,
                "active_switch_zone_end": -1,
                "pcf_frame": pcf_frame,
                "is_pcf": int(frame_index in pcf_frames),
                "dist_to_pure_contact": int(dist_to_pure_contact(frame_index, pcf_frames, len(rows))),
                "inside_purecontact_zone": int(inside_pcf_zone),
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
                "labeler_profile": profile.name,
            }
            rows_out.append(sidecar_row)

        overall_duration_hist.update({str(k): int(v) for k, v in duration_hist.items()})
        overall_duration_s_hist.update({str(k): int(v) for k, v in duration_s_hist.items()})
        overall_duration_p_hist.update({str(k): int(v) for k, v in duration_p_hist.items()})
        overall_phase_hist.update(phase_hist)
        overall_source_hist.update(source_hist)
        overall_priority_hist.update(priority_hist)

        episode_summaries.append(
            {
                "episode_index": int(episode_index),
                "task_index": int(task_index),
                "num_frames": int(len(rows)),
                "switch_zones": [
                    {
                        "pair_index": int(zone.pair_index),
                        "cmd_frame": int(zone.cmd_frame),
                        "qpos_frame": int(zone.qpos_frame),
                        "start": int(zone.start),
                        "end": int(zone.end),
                    }
                    for zone in switch_zones
                ],
                "duration_hist": {str(k): int(v) for k, v in sorted(duration_hist.items())},
                "duration_s_hist": {str(k): int(v) for k, v in sorted(duration_s_hist.items())},
                "duration_p_hist": {str(k): int(v) for k, v in sorted(duration_p_hist.items())},
                "phase_hist": {str(k): int(v) for k, v in sorted(phase_hist.items())},
                "source_hist": {str(k): int(v) for k, v in sorted(source_hist.items())},
                "priority_hist": {str(k): int(v) for k, v in sorted(priority_hist.items())},
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
        "durations": [int(d) for d in profile.durations],
        "duration_to_class": {str(k): int(v) for k, v in duration_to_class.items()},
        "merge_window": int(merge_window),
        "match_window": int(match_window),
        "purecontact_annotation_count": int(len(purecontact_annotations)),
        "interaction_zone_sidecar": str(interaction_zone_sidecar) if interaction_zone_sidecar else None,
        "interaction_zone_episode_count": int(len(interaction_zones_by_episode)),
        "labeler_version": labeler_version,
        "labeler_profile": profile.name,
        "profile": {
            "name": profile.name,
            "durations": [int(d) for d in profile.durations],
            "free_duration": int(profile.free_duration),
            "commit_duration": int(profile.commit_duration),
            "transition_duration": int(profile.transition_duration),
            "pre_near_duration": (
                None if profile.pre_near_duration is None else int(profile.pre_near_duration)
            ),
            "commit_window": [int(-profile.commit_pre), int(profile.commit_post)],
            "pre_far_window": [int(-profile.pre_far_start), int(-profile.pre_far_end)],
            "pre_near_window": (
                None
                if profile.pre_near_duration is None
                else [int(-profile.pre_near_start), int(-profile.pre_near_end)]
            ),
            "post_window": [int(profile.post_start), int(profile.post_end)],
        },
        "duration_hist": {str(k): int(v) for k, v in sorted(overall_duration_hist.items())},
        "duration_s_hist": {str(k): int(v) for k, v in sorted(overall_duration_s_hist.items())},
        "duration_p_hist": {str(k): int(v) for k, v in sorted(overall_duration_p_hist.items())},
        "phase_hist": {str(k): int(v) for k, v in sorted(overall_phase_hist.items())},
        "source_hist": {str(k): int(v) for k, v in sorted(overall_source_hist.items())},
        "priority_hist": {str(k): int(v) for k, v in sorted(overall_priority_hist.items())},
        "episodes": episode_summaries,
    }
    return rows_out, summary


def _profile_from_args(args: argparse.Namespace) -> ContactCommitProfile:
    if args.profile in PROFILES:
        profile = PROFILES[args.profile]
    else:
        if args.durations is None:
            raise ValueError("Custom profile requires --durations.")
        durations = parse_int_tuple(args.durations)
        free_duration = args.free_duration if args.free_duration is not None else max(durations)
        transition_duration = args.transition_duration if args.transition_duration is not None else min(durations)
        commit_duration = args.commit_duration if args.commit_duration is not None else transition_duration
        profile = ContactCommitProfile(
            name=args.profile,
            durations=durations,
            free_duration=int(free_duration),
            commit_duration=int(commit_duration),
            transition_duration=int(transition_duration),
            pre_near_duration=args.pre_near_duration,
            commit_pre=args.commit_pre,
            commit_post=args.commit_post,
            pre_far_start=args.pre_far_start,
            pre_far_end=args.pre_far_end,
            pre_near_start=args.pre_near_start,
            pre_near_end=args.pre_near_end,
            post_start=args.post_start,
            post_end=args.post_end,
        )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset.repo-id", dest="repo_id", required=True)
    parser.add_argument("--dataset.root", dest="root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--episode-indices", type=str, default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--profile", type=str, default="v1", help="One of v1,v2,v3,v4, or a custom name.")
    parser.add_argument("--durations", type=str, default=None, help="Custom sorted durations, e.g. 4,6,10.")
    parser.add_argument("--free-duration", type=int, default=None)
    parser.add_argument("--commit-duration", type=int, default=None)
    parser.add_argument("--transition-duration", type=int, default=None)
    parser.add_argument("--pre-near-duration", type=int, default=None)
    parser.add_argument("--commit-pre", type=int, default=4)
    parser.add_argument("--commit-post", type=int, default=4)
    parser.add_argument("--pre-far-start", type=int, default=24)
    parser.add_argument("--pre-far-end", type=int, default=15)
    parser.add_argument("--pre-near-start", type=int, default=14)
    parser.add_argument("--pre-near-end", type=int, default=5)
    parser.add_argument("--post-start", type=int, default=5)
    parser.add_argument("--post-end", type=int, default=14)
    parser.add_argument("--merge-window", type=int, default=DEFAULT_MERGE_WINDOW)
    parser.add_argument("--match-window", type=int, default=DEFAULT_MATCH_WINDOW)
    parser.add_argument("--labeler-version", type=str, default="hiva_contact_commit_v1")
    parser.add_argument(
        "--purecontact-pcf-json",
        type=Path,
        default=None,
        help="Optional PCF metadata JSON. Defaults to dataset.root/meta/purecontact_pcf.json when it exists.",
    )
    parser.add_argument(
        "--interaction-zone-sidecar",
        type=Path,
        default=None,
        help=(
            "Optional cached interaction-zone parquet with switch_zones and PCF/contact columns. "
            "When provided, switch/PCF decisions are reused instead of recomputed."
        ),
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
    interaction_zones_by_episode = load_interaction_zone_sidecar(args.interaction_zone_sidecar)
    profile = _profile_from_args(args)
    _validate_profile(profile)

    rows, summary = build_sidecar_rows(
        root,
        episode_indices=selected_episodes,
        profile=profile,
        merge_window=args.merge_window,
        match_window=args.match_window,
        labeler_version=args.labeler_version,
        purecontact_annotations=purecontact_annotations,
        interaction_zones_by_episode=interaction_zones_by_episode,
        interaction_zone_sidecar=args.interaction_zone_sidecar,
    )
    write_rows(rows, args.output)
    write_summary(summary, args.summary_json)

    print(
        f"Wrote {len(rows)} rows from {len(selected_episodes)} episodes to {args.output}. "
        f"profile={profile.name}, durations={profile.durations}, "
        f"merge_window={args.merge_window}, match_window={args.match_window}"
    )
    print(json.dumps({"duration_hist": summary["duration_hist"], "phase_hist": summary["phase_hist"]}, indent=2))
    if args.summary_json is not None:
        print(f"Wrote summary JSON to {args.summary_json}")


if __name__ == "__main__":
    main()
