#!/usr/bin/env python
from __future__ import annotations

"""Recompute HiVA dense duration labels from a reusable interaction-zone sidecar."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.scripts.build_hiva_duration_sidecar import (
    FrameContext,
    SwitchZone,
    classify_phase,
    dist_to_pure_contact,
    duration_is_horizon_safe,
    frame_context_for_time,
    horizon_safe_duration_label,
    min_dist_to_valid_switch,
    parse_int_tuple,
    pcf_duration_labels,
)


def _switch_zones(items: list[dict[str, Any]] | None) -> list[SwitchZone]:
    return [
        SwitchZone(
            pair_index=int(item.get("pair_index", idx)),
            cmd_frame=int(item.get("cmd_frame", item.get("start", -1))),
            qpos_frame=int(item.get("qpos_frame", item.get("end", -1))),
            start=int(item["start"]),
            end=int(item["end"]),
        )
        for idx, item in enumerate(items or [])
    ]


def relabel_episode(
    ep_rows: list[dict[str, Any]],
    zone_row: dict[str, Any],
    *,
    durations: tuple[int, ...],
    duration_to_class: dict[int, int],
    W1: int,
    W3: int,
    labeler_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    T = len(ep_rows)
    switch_zones = _switch_zones(zone_row.get("switch_zones"))
    switch_contexts = [frame_context_for_time(s, switch_zones, T) for s in range(T)]
    duration_s = np.asarray(
        [
            horizon_safe_duration_label(
                s,
                context=switch_contexts[s],
                T=T,
                W1=W1,
                W3=W3,
                durations=durations,
            )
            for s in range(T)
        ],
        dtype=np.int64,
    )
    duration_class_s = np.asarray([duration_to_class[int(x)] for x in duration_s], dtype=np.int64)

    pcf_frames = [int(x) for x in (zone_row.get("pcf_frames") or [])]
    duration_p, duration_class_p, pcf_contexts = pcf_duration_labels(
        T,
        pcf_frames=pcf_frames,
        W1=W1,
        W3=W3,
        durations=durations,
        duration_to_class=duration_to_class,
    )

    duration_final = np.minimum(duration_s, duration_p)
    duration_class_final = np.asarray([duration_to_class[int(x)] for x in duration_final], dtype=np.int64)
    phases = [
        classify_phase(
            s,
            duration_label=int(duration_s[s]),
            context=switch_contexts[s],
            T=T,
            W1=W1,
            W3=W3,
            durations=durations,
        )
        for s in range(T)
    ]

    matched_cmd_frames = {int(z.cmd_frame) for z in switch_zones}
    matched_qpos_frames = {int(z.qpos_frame) for z in switch_zones}
    pcf_frame = int(pcf_frames[0]) if pcf_frames else -1
    out_rows = []
    for i, row in enumerate(ep_rows):
        frame_index = int(row["frame_index"])
        context: FrameContext = switch_contexts[i]
        pcf_context: FrameContext = pcf_contexts[i]
        phase = phases[i]
        out = dict(row)
        out.update(
            {
                "duration_label": int(duration_final[i]),
                "duration_class": int(duration_class_final[i]),
                "duration_label_s": int(duration_s[i]),
                "duration_class_s": int(duration_class_s[i]),
                "duration_label_p": int(duration_p[i]),
                "duration_class_p": int(duration_class_p[i]),
                "phase": phase,
                "near_target": int(phase == "near_target"),
                "near_gripper": int(phase == "near_target"),
                "dist_to_switch": int(
                    min_dist_to_valid_switch(frame_index, switch_zones, inside_zone=context.inside_zone, T=T)
                ),
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
                "dist_to_pure_contact": int(dist_to_pure_contact(frame_index, pcf_frames, T)),
                "inside_purecontact_zone": int(pcf_context.inside_zone),
                "purecontact_case_category": zone_row.get("purecontact_case_category"),
                "purecontact_keyword": zone_row.get("purecontact_keyword"),
                "purecontact_verb": zone_row.get("purecontact_verb"),
                "purecontact_contact_start": int(zone_row.get("purecontact_contact_start", -1)),
                "purecontact_contact_end": int(zone_row.get("purecontact_contact_end", -1)),
                "purecontact_reset_mode": zone_row.get("purecontact_reset_mode"),
                "purecontact_reset_seed": int(zone_row.get("purecontact_reset_seed", -1)),
                "labeler_version": labeler_version,
            }
        )
        out_rows.append(out)

    duration_hist = Counter(int(x) for x in duration_final.tolist())
    duration_s_hist = Counter(int(x) for x in duration_s.tolist())
    duration_p_hist = Counter(int(x) for x in duration_p.tolist())
    phase_hist = Counter(phases)
    episode_summary = {
        "episode_index": int(zone_row["episode_index"]),
        "task_index": int(zone_row["task_index"]),
        "num_frames": int(T),
        "switch_zones": zone_row.get("switch_zones") or [],
        "pcf_frames": pcf_frames,
        "duration_hist": {str(k): int(v) for k, v in sorted(duration_hist.items())},
        "duration_s_hist": {str(k): int(v) for k, v in sorted(duration_s_hist.items())},
        "duration_p_hist": {str(k): int(v) for k, v in sorted(duration_p_hist.items())},
        "phase_hist": {str(k): int(v) for k, v in sorted(phase_hist.items())},
    }
    return out_rows, episode_summary


def build(args: argparse.Namespace) -> None:
    source_sidecar = Path(args.source_sidecar)
    zone_sidecar = Path(args.zone_sidecar)
    output = Path(args.output)
    summary_json = Path(args.summary_json) if args.summary_json else output.with_suffix(".summary.json")
    durations = parse_int_tuple(args.durations)
    duration_to_class = {int(d): i for i, d in enumerate(durations)}

    source_rows = pq.read_table(source_sidecar).to_pylist()
    zone_rows = pq.read_table(zone_sidecar).to_pylist()
    zones_by_episode = {int(r["episode_index"]): r for r in zone_rows}

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_episode[int(row["episode_index"])].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: int(r["frame_index"]))

    out_rows = []
    episode_summaries = []
    overall_duration_hist: Counter[str] = Counter()
    overall_duration_s_hist: Counter[str] = Counter()
    overall_duration_p_hist: Counter[str] = Counter()
    overall_phase_hist: Counter[str] = Counter()

    for ep, ep_rows in sorted(by_episode.items()):
        if ep not in zones_by_episode:
            raise KeyError(f"episode_index={ep} is missing from {zone_sidecar}")
        rows, episode_summary = relabel_episode(
            ep_rows,
            zones_by_episode[ep],
            durations=durations,
            duration_to_class=duration_to_class,
            W1=args.W1,
            W3=args.W3,
            labeler_version=args.labeler_version,
        )
        out_rows.extend(rows)
        episode_summaries.append(episode_summary)
        overall_duration_hist.update(episode_summary["duration_hist"])
        overall_duration_s_hist.update(episode_summary["duration_s_hist"])
        overall_duration_p_hist.update(episode_summary["duration_p_hist"])
        overall_phase_hist.update(episode_summary["phase_hist"])

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows), output)

    summary = {
        "labeler_version": args.labeler_version,
        "source_sidecar": str(source_sidecar),
        "zone_sidecar": str(zone_sidecar),
        "duration_classes": list(durations),
        "W1": int(args.W1),
        "W3": int(args.W3),
        "rows": len(out_rows),
        "num_episodes": len(episode_summaries),
        "duration_hist": {str(k): int(v) for k, v in sorted(overall_duration_hist.items(), key=lambda kv: int(kv[0]))},
        "duration_s_hist": {str(k): int(v) for k, v in sorted(overall_duration_s_hist.items(), key=lambda kv: int(kv[0]))},
        "duration_p_hist": {str(k): int(v) for k, v in sorted(overall_duration_p_hist.items(), key=lambda kv: int(kv[0]))},
        "phase_hist": {str(k): int(v) for k, v in sorted(overall_phase_hist.items())},
        "episode_summaries": episode_summaries,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows to {output}")
    print(f"Wrote summary to {summary_json}")
    print(
        json.dumps(
            {
                "duration_hist": summary["duration_hist"],
                "duration_s_hist": summary["duration_s_hist"],
                "duration_p_hist": summary["duration_p_hist"],
                "phase_hist": summary["phase_hist"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sidecar", required=True, help="Existing duration sidecar used only for frame ids and metadata.")
    parser.add_argument("--zone-sidecar", required=True, help="Reusable interaction-zone sidecar parquet.")
    parser.add_argument("--output", required=True, help="Output relabeled duration sidecar parquet.")
    parser.add_argument("--summary-json", default=None, help="Output summary JSON.")
    parser.add_argument("--durations", default="4,15", help="Sorted unique durations, e.g. '4,15'.")
    parser.add_argument("--W1", type=int, default=10)
    parser.add_argument("--W3", type=int, default=0)
    parser.add_argument("--labeler-version", default="hiva_duration_d4_15_w1_10_w3_0_from_zones_v1")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
