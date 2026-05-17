#!/usr/bin/env python
from __future__ import annotations

"""Extract duration-independent HiVA interaction zones from a duration sidecar.

The output is intentionally independent of duration candidates and W1/W3. It stores
the gripper switch zones and PCF metadata once, so new duration-label ablations can
reuse these zones without re-detecting gripper command/qpos switches from raw data.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _as_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    return int(value)


def extract_switch_zones(rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        zone_idx = _as_int(row.get("active_switch_zone_index"))
        if zone_idx >= 0:
            grouped[zone_idx].append(row)

    zones = []
    for zone_idx, zone_rows in sorted(grouped.items()):
        starts = [_as_int(r.get("active_switch_zone_start")) for r in zone_rows]
        ends = [_as_int(r.get("active_switch_zone_end")) for r in zone_rows]
        frame_indices = [_as_int(r.get("frame_index")) for r in zone_rows]
        cmd_frames = [_as_int(r.get("frame_index")) for r in zone_rows if _as_int(r.get("cmd_switch"), 0) == 1]
        qpos_frames = [_as_int(r.get("frame_index")) for r in zone_rows if _as_int(r.get("qpos_switch"), 0) == 1]
        start = min([x for x in starts if x >= 0] or frame_indices)
        end = max([x for x in ends if x >= 0] or frame_indices)
        zones.append(
            {
                "pair_index": int(zone_idx),
                "cmd_frame": int(cmd_frames[0] if cmd_frames else start),
                "qpos_frame": int(qpos_frames[0] if qpos_frames else end),
                "start": int(start),
                "end": int(end),
            }
        )
    return zones


def extract_pcf(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pcf_frames = sorted({int(r["pcf_frame"]) for r in rows if _as_int(r.get("pcf_frame")) >= 0})
    first = next((r for r in rows if _as_int(r.get("pcf_frame")) >= 0), None)
    if first is None:
        return {
            "pcf_frames": [],
            "case_category": None,
            "purecontact_keyword": None,
            "purecontact_verb": None,
            "contact_start": -1,
            "contact_end": -1,
            "reset_mode": None,
            "reset_seed": -1,
        }
    return {
        "pcf_frames": pcf_frames,
        "case_category": first.get("purecontact_case_category"),
        "purecontact_keyword": first.get("purecontact_keyword"),
        "purecontact_verb": first.get("purecontact_verb"),
        "contact_start": _as_int(first.get("purecontact_contact_start")),
        "contact_end": _as_int(first.get("purecontact_contact_end")),
        "reset_mode": first.get("purecontact_reset_mode"),
        "reset_seed": _as_int(first.get("purecontact_reset_seed")),
    }


def build(args: argparse.Namespace) -> None:
    source_sidecar = Path(args.source_sidecar)
    output = Path(args.output)
    summary_json = Path(args.summary_json) if args.summary_json else output.with_suffix(".summary.json")

    table = pq.read_table(source_sidecar)
    rows = table.to_pylist()
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)
    for ep_rows in by_episode.values():
        ep_rows.sort(key=lambda r: int(r["frame_index"]))

    out_rows = []
    switch_zone_count = 0
    pcf_episode_count = 0
    task_hist: Counter[str] = Counter()
    for ep, ep_rows in sorted(by_episode.items()):
        switch_zones = extract_switch_zones(ep_rows)
        pcf = extract_pcf(ep_rows)
        switch_zone_count += len(switch_zones)
        pcf_episode_count += int(bool(pcf["pcf_frames"]))
        task_index = int(ep_rows[0].get("task_index", -1))
        task_hist[str(task_index)] += 1
        out_rows.append(
            {
                "episode_index": int(ep),
                "task_index": task_index,
                "num_frames": int(len(ep_rows)),
                "switch_zones": switch_zones,
                "pcf_frames": pcf["pcf_frames"],
                "purecontact_case_category": pcf["case_category"],
                "purecontact_keyword": pcf["purecontact_keyword"],
                "purecontact_verb": pcf["purecontact_verb"],
                "purecontact_contact_start": int(pcf["contact_start"]),
                "purecontact_contact_end": int(pcf["contact_end"]),
                "purecontact_reset_mode": pcf["reset_mode"],
                "purecontact_reset_seed": int(pcf["reset_seed"]),
                "source_duration_labeler_version": ep_rows[0].get("labeler_version"),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows), output)
    summary = {
        "rows": len(out_rows),
        "source_sidecar": str(source_sidecar),
        "switch_zone_count": int(switch_zone_count),
        "pcf_episode_count": int(pcf_episode_count),
        "task_episode_hist": dict(sorted(task_hist.items(), key=lambda kv: int(kv[0]))),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} episode interaction-zone rows to {output}")
    print(f"Wrote summary to {summary_json}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sidecar", required=True, help="Existing duration sidecar parquet with zone metadata.")
    parser.add_argument("--output", required=True, help="Output interaction-zone sidecar parquet.")
    parser.add_argument("--summary-json", default=None, help="Output summary JSON.")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
