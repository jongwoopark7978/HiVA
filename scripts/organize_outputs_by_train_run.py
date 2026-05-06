#!/usr/bin/env python3
"""Organize completed train/eval artifacts under their model train directory.

Default mode is a dry run. It writes a manifest of proposed moves and skipped /
ambiguous artifacts, but does not move anything unless --apply is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict


DEFAULT_OUTPUTS_ROOT = Path("/home/jongwoopark/lerobot/outputs")
DEFAULT_SKIP_RUNS = {
    "smolvla_hiva_coeff_hp_k10_bigflow_job1_duration_noisy_weights_full_w1p0_sigma0p25_b160_s4_20260505_201244"
}
TEXT_SUFFIXES = {".log", ".outer", ".script_path", ".session", ".pid", ".txt", ".json", ".csv"}


@dataclass
class Move:
    kind: str
    source: str
    destination: str
    train_run: str
    reason: str
    eval_run: str | None = None


@dataclass
class Skipped:
    kind: str
    source: str
    reason: str
    candidates: list[str] | None = None


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def collect_live_text() -> str:
    user = os.environ.get("USER") or "jongwoopark"
    chunks = [
        run_cmd(["ps", "-u", user, "-o", "pid=,etime=,cmd="]),
        run_cmd(["tmux", "ls"]),
    ]
    return "\n".join(chunks)


def safe_read(path: Path, max_bytes: int = 2_000_000) -> str:
    if path.is_dir():
        return ""
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def path_is_live(path: Path, live_text: str) -> bool:
    text = str(path)
    return text in live_text or path.name in live_text


def get_train_runs(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        return []
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def mentioned_train_runs(text: str, train_runs: list[str]) -> list[str]:
    found = {run for run in train_runs if run in text}
    patterns = [
        r"(?:POLICY_PATH|PRETRAINED|PRETRAINED_PATH|CHECKPOINT|INIT_SMOLVLA|OUTPUT_DIR|BASE_OUTPUT_DIR)=(\S+)",
        r"--output_dir(?:=|\s+)(\S+)",
        r"--policy\.pretrained_path(?:=|\s+)(\S+)",
        r"--policy\.init_smolvla_checkpoint_path(?:=|\s+)(\S+)",
        r"Output dir:\s*(\S+)",
    ]
    for pattern in patterns:
        for value in re.findall(pattern, text):
            for run in train_runs:
                marker = f"/outputs/train/{run}"
                if marker in value or f"/outputs/train/{run}/" in value:
                    found.add(run)
    return sorted(found)


def infer_from_filename(name: str, train_runs: list[str]) -> tuple[str | None, list[str]]:
    exact = [run for run in train_runs if run in name]
    if len(exact) == 1:
        return exact[0], exact
    if len(exact) > 1:
        return None, exact
    return None, []


def infer_train_run(path: Path, train_runs: list[str]) -> tuple[str | None, str, list[str]]:
    text = safe_read(path)
    mentioned = mentioned_train_runs(text, train_runs)
    if len(mentioned) == 1:
        return mentioned[0], "metadata", mentioned
    if len(mentioned) > 1:
        return None, "ambiguous_metadata", mentioned

    inferred, candidates = infer_from_filename(path.name, train_runs)
    if inferred:
        return inferred, "filename", candidates
    if candidates:
        return None, "ambiguous_filename", candidates
    return None, "unmatched", []


def infer_eval_run_from_log(path: Path, eval_runs: list[str]) -> tuple[str | None, str, list[str]]:
    text = safe_read(path)
    found = {run for run in eval_runs if run in text or run in path.name}
    patterns = [
        r"BASE_OUTPUT_DIR=(\S+)",
        r"Output dir:\s*(\S+)",
        r"Videos are under:\s*(\S+)",
        r"Wrote combined overlay summary:\s*(\S+)",
    ]
    for pattern in patterns:
        for value in re.findall(pattern, text):
            for run in eval_runs:
                marker = f"/outputs/eval/{run}"
                if marker in value or f"/outputs/eval/{run}/" in value:
                    found.add(run)
    if len(found) == 1:
        return next(iter(found)), "eval_metadata", sorted(found)
    if len(found) > 1:
        return None, "ambiguous_eval_metadata", sorted(found)

    inferred, candidates = infer_from_filename(path.name, eval_runs)
    if inferred:
        return inferred, "eval_filename", candidates
    if candidates:
        return None, "ambiguous_eval_filename", candidates
    return None, "no_eval_match", []


def unique_destination(dest: Path, source: Path) -> Path:
    if not dest.exists():
        return dest
    if dest.is_dir() and source.is_dir():
        # Existing eval directory for the same run: merge into the same folder.
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}.dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def add_move(moves: list[Move], kind: str, source: Path, dest: Path, train_run: str, reason: str, eval_run: str | None = None):
    moves.append(
        Move(
            kind=kind,
            source=str(source),
            destination=str(dest),
            train_run=train_run,
            reason=reason,
            eval_run=eval_run,
        )
    )


def apply_move(move: Move):
    source = Path(move.source)
    dest = Path(move.destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and source.is_dir() and dest.is_dir():
        for child in source.iterdir():
            child_dest = unique_destination(dest / child.name, child)
            shutil.move(str(child), str(child_dest))
        source.rmdir()
    else:
        shutil.move(str(source), str(dest))


def build_plan(outputs_root: Path, explicit_skip_runs: set[str]) -> dict:
    train_dir = outputs_root / "train"
    train_logs_dir = outputs_root / "train_logs"
    eval_dir = outputs_root / "eval"
    eval_logs_dir = outputs_root / "eval_logs"
    live_text = collect_live_text()
    train_runs = get_train_runs(train_dir)
    eval_runs = sorted(p.name for p in eval_dir.iterdir() if p.is_dir()) if eval_dir.exists() else []
    active_runs = sorted({run for run in train_runs if run in live_text} | explicit_skip_runs)
    known_eval_candidates: dict[str, set[str]] = defaultdict(set)
    for run in train_runs:
        if run in active_runs:
            continue
        run_eval_dir = train_dir / run / "eval"
        if not run_eval_dir.exists():
            continue
        for existing_eval in run_eval_dir.iterdir():
            if existing_eval.is_dir():
                known_eval_candidates[existing_eval.name].add(run)

    for path in sorted(eval_logs_dir.iterdir()) if eval_logs_dir.exists() else []:
        if not path.is_file() or path_is_live(path, live_text):
            continue
        eval_run, _eval_reason, _eval_candidates = infer_eval_run_from_log(path, eval_runs)
        run, _reason, _train_candidates = infer_train_run(path, train_runs)
        if eval_run and run and run not in active_runs:
            known_eval_candidates[eval_run].add(run)

    known_eval_to_train = {
        eval_run: next(iter(runs)) for eval_run, runs in known_eval_candidates.items() if len(runs) == 1
    }
    ambiguous_known_eval = {
        eval_run: sorted(runs) for eval_run, runs in known_eval_candidates.items() if len(runs) > 1
    }

    moves: list[Move] = []
    skipped: list[Skipped] = []

    for path in sorted(train_logs_dir.iterdir()) if train_logs_dir.exists() else []:
        if not path.is_file():
            continue
        run, reason, candidates = infer_train_run(path, train_runs)
        if path_is_live(path, live_text):
            skipped.append(Skipped("train_log", str(path), "live_file"))
        elif run in active_runs:
            skipped.append(Skipped("train_log", str(path), f"active_train_run:{run}", [run]))
        elif run:
            dest = unique_destination(train_dir / run / "logs" / "train" / path.name, path)
            add_move(moves, "train_log", path, dest, run, reason)
        else:
            skipped.append(Skipped("train_log", str(path), reason, candidates))

    eval_run_to_train: dict[str, str] = {}
    eval_run_reason: dict[str, str] = {}
    for path in sorted(eval_dir.iterdir()) if eval_dir.exists() else []:
        if not path.is_dir():
            continue
        if path.name in ambiguous_known_eval:
            skipped.append(Skipped("eval_dir", str(path), "ambiguous_known_eval_mapping", ambiguous_known_eval[path.name]))
            continue
        if path.name in known_eval_to_train:
            run, reason, candidates = known_eval_to_train[path.name], "known_eval_log_mapping", [known_eval_to_train[path.name]]
        else:
            run, reason, candidates = infer_train_run(path, train_runs)
        if path_is_live(path, live_text):
            skipped.append(Skipped("eval_dir", str(path), "live_dir"))
        elif run in active_runs:
            skipped.append(Skipped("eval_dir", str(path), f"active_train_run:{run}", [run]))
        elif run:
            eval_run_to_train[path.name] = run
            eval_run_reason[path.name] = reason
            dest = unique_destination(train_dir / run / "eval" / path.name, path)
            add_move(moves, "eval_dir", path, dest, run, reason, path.name)
        else:
            skipped.append(Skipped("eval_dir", str(path), reason, candidates))

    for path in sorted(eval_logs_dir.iterdir()) if eval_logs_dir.exists() else []:
        if not path.is_file():
            continue
        eval_run, eval_reason, eval_candidates = infer_eval_run_from_log(path, eval_runs)
        if path_is_live(path, live_text):
            skipped.append(Skipped("eval_log", str(path), "live_file"))
            continue
        if eval_run and (eval_run in eval_run_to_train or eval_run in known_eval_to_train):
            run = eval_run_to_train.get(eval_run) or known_eval_to_train[eval_run]
            if run in active_runs:
                skipped.append(Skipped("eval_log", str(path), f"active_train_run:{run}", [run]))
                continue
            dest = unique_destination(train_dir / run / "eval" / eval_run / "logs" / path.name, path)
            add_move(moves, "eval_log", path, dest, run, eval_reason, eval_run)
            continue

        run, reason, train_candidates = infer_train_run(path, train_runs)
        if run and run not in active_runs:
            eval_name = eval_run if eval_run else "_unassigned_eval_logs"
            dest = unique_destination(train_dir / run / "eval" / eval_name / "logs" / path.name, path)
            add_move(moves, "eval_log", path, dest, run, reason, eval_run)
        elif run in active_runs:
            skipped.append(Skipped("eval_log", str(path), f"active_train_run:{run}", [run]))
        else:
            candidates = eval_candidates or train_candidates
            skipped.append(Skipped("eval_log", str(path), eval_reason if eval_candidates else reason, candidates))

    return {
        "outputs_root": str(outputs_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "active_train_runs": active_runs,
        "counts": {
            "train_runs": len(train_runs),
            "eval_runs": len(eval_runs),
            "proposed_moves": len(moves),
            "skipped": len(skipped),
        },
        "proposed_moves": [asdict(m) for m in moves],
        "skipped": [asdict(s) for s in skipped],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Actually move files/directories. Default is dry-run.")
    parser.add_argument(
        "--skip-run",
        action="append",
        default=[],
        help="Train run name to skip. Can be passed multiple times.",
    )
    args = parser.parse_args()

    explicit_skip_runs = set(DEFAULT_SKIP_RUNS) | set(args.skip_run)
    manifest = build_plan(args.outputs_root, explicit_skip_runs)
    manifest["mode"] = "apply" if args.apply else "dry-run"

    if args.apply:
        for move in manifest["proposed_moves"]:
            apply_move(Move(**move))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = args.manifest or args.outputs_root / f"artifact_organization_manifest_{ts}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"mode={manifest['mode']}")
    print(f"manifest={manifest_path}")
    print(f"active_train_runs={len(manifest['active_train_runs'])}")
    for run in manifest["active_train_runs"]:
        print(f"  active: {run}")
    print(f"proposed_moves={manifest['counts']['proposed_moves']}")
    print(f"skipped={manifest['counts']['skipped']}")
    if not args.apply:
        print("dry-run only; no files were moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
