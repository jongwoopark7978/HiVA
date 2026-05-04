from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation as R

from lerobot.utils.constants import ACTION


DEFAULT_DATASET_ROOT = "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_lerobot_v3_lerobotkeys"
DEFAULT_DURATION_SIDECAR = (
    "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_duration_sidecar_all_episodes.parquet"
)
DEFAULT_OUTPUT = "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d4_8_12_k4.parquet"
DEFAULT_SUMMARY = (
    "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/libero_hiva_coeff_sidecar_d4_8_12_k4.summary.json"
)


SOURCE_DURATIONS = (1, 3, 8)
TARGET_DURATIONS = (4, 8, 12)
TARGET_DURATION_TO_CLASS = {4: 0, 8: 1, 12: 2}
MAX_DURATION = 12
K = 4
DEGREE = 3
OSC_ROT_SCALE = 0.5


def parse_int_tuple(spec: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def as_float32_nested(x: np.ndarray) -> list:
    return np.asarray(x, dtype=np.float32).tolist()


def clamped_bspline_basis(d: int, *, n_ctrl: int, degree: int) -> np.ndarray:
    """Open-uniform clamped B-spline basis evaluated at integer timesteps."""
    if d < 2:
        raise ValueError(f"Duration must be >= 2 for B-spline basis, got {d}")
    if n_ctrl < degree + 1:
        raise ValueError(f"n_ctrl={n_ctrl} must be at least degree+1={degree + 1}")

    x = np.arange(d, dtype=np.float64)
    t_internal = np.linspace(x[0], x[-1], n_ctrl - degree + 1)
    knots = np.concatenate(([x[0]] * degree, t_internal, [x[-1]] * degree)).astype(np.float64)
    phi = np.zeros((d, n_ctrl), dtype=np.float64)
    for j in range(n_ctrl):
        coeff = np.zeros(n_ctrl, dtype=np.float64)
        coeff[j] = 1.0
        phi[:, j] = BSpline(knots, coeff, degree, extrapolate=False)(x)
    return phi


def ridge_lstsq(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0:
        coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
        return coef
    lhs = phi.T @ phi + ridge * np.eye(phi.shape[1], dtype=np.float64)
    rhs = phi.T @ y
    return np.linalg.solve(lhs, rhs)


def cumulative_rotation_from_raw_actions(raw_rot: np.ndarray, eta: float) -> np.ndarray:
    """Compose raw LIBERO rotation-vector commands into cumulative rotvecs.

    This follows the Experiment 3 convention in HiVA_v1_pilot.tex:
      R_tau = R_{tau-1} Exp(eta * action[3:6]).
    """
    current = R.identity()
    out = []
    for raw in raw_rot:
        delta = R.from_rotvec(float(eta) * np.clip(np.asarray(raw, dtype=np.float64), -1.0, 1.0))
        current = current * delta
        out.append(current.as_rotvec())
    return np.asarray(out, dtype=np.float64)


def decode_rotation_to_raw_actions(rho_hat: np.ndarray, eta: float) -> np.ndarray:
    rotations = R.from_rotvec(np.asarray(rho_hat, dtype=np.float64))
    raw = []
    prev = R.identity()
    for cur in rotations:
        delta = prev.inv() * cur
        raw.append(delta.as_rotvec() / float(eta))
        prev = cur
    return np.asarray(raw, dtype=np.float64)


def decode_translation_to_raw_actions(p_hat: np.ndarray) -> np.ndarray:
    p0 = np.zeros((1, p_hat.shape[1]), dtype=np.float64)
    return np.diff(np.concatenate([p0, p_hat], axis=0), axis=0)


def fit_one_window(
    actions: np.ndarray,
    *,
    d: int,
    n_ctrl: int,
    degree: int,
    ridge: float,
    eta: float,
) -> dict[str, Any]:
    action_chunk = np.asarray(actions[:d], dtype=np.float64)
    phi = clamped_bspline_basis(d, n_ctrl=n_ctrl, degree=degree)

    p_tr = np.cumsum(action_chunk[:, 0:3], axis=0)
    rho_rot = cumulative_rotation_from_raw_actions(action_chunk[:, 3:6], eta=eta)
    g_grip = action_chunk[:, 6:7]

    theta_tr = ridge_lstsq(phi, p_tr, ridge)
    theta_rot = ridge_lstsq(phi, rho_rot, ridge)
    theta_grip = ridge_lstsq(phi, g_grip, ridge)

    p_hat = phi @ theta_tr
    rho_hat = phi @ theta_rot
    g_hat = phi @ theta_grip

    decoded = np.concatenate(
        [
            decode_translation_to_raw_actions(p_hat),
            decode_rotation_to_raw_actions(rho_hat, eta=eta),
            g_hat,
        ],
        axis=1,
    )

    diff = decoded - action_chunk
    tr_diff = diff[:, 0:3]
    rot_diff = diff[:, 3:6]
    grip_diff = diff[:, 6:7]

    return {
        "theta_tr": theta_tr,
        "theta_rot": theta_rot,
        "theta_grip": theta_grip,
        "p_tr": p_tr,
        "rho_rot": rho_rot,
        "g_grip": g_grip,
        "decoded": decoded,
        "action_rmse": float(np.sqrt(np.mean(diff**2))),
        "action_max_abs_diff": float(np.max(np.abs(diff))),
        "tr_rmse": float(np.sqrt(np.mean(tr_diff**2))),
        "tr_max_abs_diff": float(np.max(np.abs(tr_diff))),
        "rot_rmse": float(np.sqrt(np.mean(rot_diff**2))),
        "rot_max_abs_diff": float(np.max(np.abs(rot_diff))),
        "grip_rmse": float(np.sqrt(np.mean(grip_diff**2))),
        "grip_max_abs_diff": float(np.max(np.abs(grip_diff))),
    }


def terminal_hold_action_chunk(actions: np.ndarray, *, start: int, d: int) -> tuple[np.ndarray, int, bool]:
    """Return a d-step chunk, terminal-holding when fewer real actions remain.

    Translation and rotation should stop during the synthetic tail, so their
    raw deltas are zero. The gripper is an absolute command, so its final real
    command is held.
    """
    real = np.asarray(actions[start : min(start + d, len(actions))], dtype=np.float64)
    real_steps = int(len(real))
    if real_steps <= 0:
        raise ValueError(f"No real actions available at start={start}")
    if real_steps == d:
        return real, real_steps, False

    pad_count = d - real_steps
    pad = np.zeros((pad_count, real.shape[1]), dtype=np.float64)
    pad[:, 6] = real[-1, 6]
    return np.concatenate([real, pad], axis=0), real_steps, True


def hold_pad(x: np.ndarray, d: int, dmax: int) -> np.ndarray:
    out = np.zeros((dmax, x.shape[1]), dtype=np.float64)
    out[:d] = x
    if d < dmax:
        out[d:] = x[d - 1]
    return out


def load_pylist(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    table = pads.dataset(str(path), format="parquet").to_table(columns=columns)
    return table.to_pylist()


def group_rows_by_episode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_index"])].append(row)
    for ep_rows in grouped.values():
        ep_rows.sort(key=lambda x: int(x["frame_index"]))
    return grouped


def top_records(records: list[dict[str, Any]], key: str, n: int) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: float(row.get(key, 0.0)), reverse=True)[:n]


def build_coeff_sidecar(
    *,
    dataset_root: Path,
    duration_sidecar: Path,
    output_path: Path,
    summary_path: Path,
    source_durations: tuple[int, ...],
    target_durations: tuple[int, ...],
    n_ctrl: int,
    degree: int,
    ridge: float,
    eta: float,
    hiva_labeler_version: str,
    max_episodes: int | None,
    bad_rmse_threshold: float,
    bad_max_abs_threshold: float,
    max_bad_samples: int,
) -> dict[str, Any]:
    if len(source_durations) != len(target_durations):
        raise ValueError(f"source_durations and target_durations must have the same length: {source_durations}, {target_durations}")
    if len(set(target_durations)) != len(target_durations):
        raise ValueError(f"target_durations must be unique, got {target_durations}")
    if n_ctrl < degree + 1:
        raise ValueError(f"k={n_ctrl} must be at least degree+1={degree + 1}")

    duration_map = dict(zip(source_durations, target_durations, strict=True))
    target_duration_to_class = {int(d): i for i, d in enumerate(target_durations)}
    dmax = int(max(target_durations))

    duration_columns = [
        "dataset_index",
        "episode_index",
        "frame_index",
        "task_index",
        "duration_label",
        "duration_class",
        "duration_label_s",
        "duration_class_s",
        "duration_label_p",
        "duration_class_p",
        "phase",
        "near_target",
        "near_gripper",
        "dist_to_switch",
        "cmd_switch",
        "qpos_switch",
        "inside_switch_zone",
        "prev_switch_end",
        "next_switch_start",
        "active_switch_zone_index",
        "active_switch_zone_start",
        "active_switch_zone_end",
        "pcf_frame",
        "is_pcf",
        "dist_to_pure_contact",
        "inside_purecontact_zone",
        "labeler_version",
    ]
    action_columns = ["index", "episode_index", "frame_index", "task_index", ACTION]

    print(f"Loading duration sidecar: {duration_sidecar}")
    duration_rows = load_pylist(duration_sidecar, duration_columns)
    duration_by_ep = group_rows_by_episode(duration_rows)

    print(f"Loading dataset action columns: {dataset_root / 'data'}")
    action_rows = load_pylist(dataset_root / "data", action_columns)
    actions_by_ep = group_rows_by_episode(action_rows)

    episode_indices = sorted(set(duration_by_ep) & set(actions_by_ep))
    if max_episodes is not None:
        episode_indices = episode_indices[: int(max_episodes)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None

    hist = Counter()
    hist_source = Counter()
    valid_count = 0
    invalid_count = 0
    terminal_hold_count = 0
    bad_samples: list[dict[str, Any]] = []
    top_samples: list[dict[str, Any]] = []
    all_coef_tr = []
    all_coef_rot = []
    all_coef_grip = []
    episode_summaries = []

    try:
        for ep_i, episode_index in enumerate(episode_indices):
            duration_ep = duration_by_ep[episode_index]
            action_ep_rows = actions_by_ep[episode_index]
            action_ep_rows.sort(key=lambda x: int(x["frame_index"]))

            actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in action_ep_rows], axis=0)
            frame_to_pos = {int(row["frame_index"]): i for i, row in enumerate(action_ep_rows)}
            T = len(actions)
            rows_out = []
            ep_hist = Counter()
            ep_valid = 0
            ep_invalid = 0

            for row in duration_ep:
                frame_index = int(row["frame_index"])
                pos = frame_to_pos.get(frame_index)
                source_label = int(row["duration_label"])
                d = int(duration_map[source_label])
                duration_class = int(target_duration_to_class[d])
                hist_source[source_label] += 1
                hist[d] += 1
                ep_hist[d] += 1

                base = dict(row)
                base["source_duration_label"] = source_label
                base["source_duration_class"] = int(row["duration_class"])
                base["duration_label"] = d
                base["duration_class"] = duration_class
                for suffix in ("s", "p"):
                    src = int(row[f"duration_label_{suffix}"])
                    mapped = int(duration_map[src])
                    base[f"source_duration_label_{suffix}"] = src
                    base[f"source_duration_class_{suffix}"] = int(row[f"duration_class_{suffix}"])
                    base[f"duration_label_{suffix}"] = mapped
                    base[f"duration_class_{suffix}"] = int(target_duration_to_class[mapped])

                real_steps = int(max(0, T - pos)) if pos is not None else 0
                valid = pos is not None and real_steps > 0
                has_synthetic_tail = bool(valid and real_steps < d)
                tail_mode = "terminal_hold" if has_synthetic_tail else "real"
                base["hiva_valid"] = int(valid)
                base["hiva_tail_mode"] = tail_mode if valid else "invalid"
                base["hiva_real_steps"] = int(min(real_steps, d)) if valid else 0
                base["hiva_requested_duration"] = d
                base["hiva_has_synthetic_tail"] = int(has_synthetic_tail)
                base["hiva_available_steps"] = real_steps
                base["hiva_k"] = int(n_ctrl)
                base["hiva_degree"] = int(degree)
                base["hiva_dmax"] = int(dmax)
                base["hiva_rotation_convention"] = "R_tau=R_tau_minus_1@Exp(eta*raw_rot)"
                base["hiva_rot_scale_eta"] = float(eta)
                base["hiva_labeler_version"] = hiva_labeler_version

                if valid:
                    action_chunk, chunk_real_steps, chunk_has_synthetic_tail = terminal_hold_action_chunk(
                        actions, start=pos, d=d
                    )
                    fit = fit_one_window(action_chunk, d=d, n_ctrl=n_ctrl, degree=degree, ridge=ridge, eta=eta)
                    p_hold = hold_pad(fit["p_tr"], d, dmax)
                    rho_hold = hold_pad(fit["rho_rot"], d, dmax)
                    g_hold = hold_pad(fit["g_grip"], d, dmax)
                    if chunk_has_synthetic_tail:
                        terminal_hold_count += 1

                    base.update(
                        {
                            "hiva_theta_tr_raw": as_float32_nested(fit["theta_tr"]),
                            "hiva_theta_rot_raw": as_float32_nested(fit["theta_rot"]),
                            "hiva_theta_grip_raw": as_float32_nested(fit["theta_grip"]),
                            "hiva_P_tr_hold_raw": as_float32_nested(p_hold),
                            "hiva_rho_rot_hold_raw": as_float32_nested(rho_hold),
                            "hiva_G_grip_hold_raw": as_float32_nested(g_hold),
                            "hiva_action_rmse": fit["action_rmse"],
                            "hiva_action_max_abs_diff": fit["action_max_abs_diff"],
                            "hiva_tr_rmse": fit["tr_rmse"],
                            "hiva_tr_max_abs_diff": fit["tr_max_abs_diff"],
                            "hiva_rot_rmse": fit["rot_rmse"],
                            "hiva_rot_max_abs_diff": fit["rot_max_abs_diff"],
                            "hiva_grip_rmse": fit["grip_rmse"],
                            "hiva_grip_max_abs_diff": fit["grip_max_abs_diff"],
                            "hiva_tail_real_steps": int(chunk_real_steps),
                            "hiva_tail_synthetic_steps": int(d - chunk_real_steps),
                        }
                    )
                    valid_count += 1
                    ep_valid += 1
                    all_coef_tr.append(fit["theta_tr"])
                    all_coef_rot.append(fit["theta_rot"])
                    all_coef_grip.append(fit["theta_grip"])

                    score_record = {
                        "episode_index": int(episode_index),
                        "frame_index": frame_index,
                        "task_index": int(base.get("task_index", -1)),
                        "duration_label": d,
                        "phase": base.get("phase"),
                        "hiva_tail_mode": base["hiva_tail_mode"],
                        "hiva_real_steps": int(base["hiva_real_steps"]),
                        "hiva_has_synthetic_tail": int(base["hiva_has_synthetic_tail"]),
                        "hiva_action_rmse": fit["action_rmse"],
                        "hiva_action_max_abs_diff": fit["action_max_abs_diff"],
                        "hiva_tr_rmse": fit["tr_rmse"],
                        "hiva_tr_max_abs_diff": fit["tr_max_abs_diff"],
                        "hiva_rot_rmse": fit["rot_rmse"],
                        "hiva_rot_max_abs_diff": fit["rot_max_abs_diff"],
                        "hiva_grip_rmse": fit["grip_rmse"],
                        "hiva_grip_max_abs_diff": fit["grip_max_abs_diff"],
                    }
                    top_samples.append(score_record)
                    if (
                        fit["action_rmse"] >= bad_rmse_threshold
                        or fit["action_max_abs_diff"] >= bad_max_abs_threshold
                    ):
                        bad_samples.append(score_record)
                else:
                    zeros_tr = np.zeros((n_ctrl, 3), dtype=np.float32)
                    zeros_rot = np.zeros((n_ctrl, 3), dtype=np.float32)
                    zeros_grip = np.zeros((n_ctrl, 1), dtype=np.float32)
                    zeros_tr_hold = np.zeros((dmax, 3), dtype=np.float32)
                    zeros_rot_hold = np.zeros((dmax, 3), dtype=np.float32)
                    zeros_grip_hold = np.zeros((dmax, 1), dtype=np.float32)
                    base.update(
                        {
                            "hiva_theta_tr_raw": as_float32_nested(zeros_tr),
                            "hiva_theta_rot_raw": as_float32_nested(zeros_rot),
                            "hiva_theta_grip_raw": as_float32_nested(zeros_grip),
                            "hiva_P_tr_hold_raw": as_float32_nested(zeros_tr_hold),
                            "hiva_rho_rot_hold_raw": as_float32_nested(zeros_rot_hold),
                            "hiva_G_grip_hold_raw": as_float32_nested(zeros_grip_hold),
                            "hiva_action_rmse": -1.0,
                            "hiva_action_max_abs_diff": -1.0,
                            "hiva_tr_rmse": -1.0,
                            "hiva_tr_max_abs_diff": -1.0,
                            "hiva_rot_rmse": -1.0,
                            "hiva_rot_max_abs_diff": -1.0,
                            "hiva_grip_rmse": -1.0,
                            "hiva_grip_max_abs_diff": -1.0,
                            "hiva_tail_real_steps": 0,
                            "hiva_tail_synthetic_steps": 0,
                        }
                    )
                    invalid_count += 1
                    ep_invalid += 1

                rows_out.append(base)

            table = pa.Table.from_pylist(rows_out)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)

            episode_summaries.append(
                {
                    "episode_index": int(episode_index),
                    "num_frames": int(T),
                    "valid_rows": int(ep_valid),
                    "invalid_rows": int(ep_invalid),
                    "duration_hist": {str(k): int(v) for k, v in sorted(ep_hist.items())},
                }
            )
            if (ep_i + 1) % 100 == 0 or ep_i + 1 == len(episode_indices):
                print(f"Processed {ep_i + 1}/{len(episode_indices)} episodes; valid={valid_count}, invalid={invalid_count}")
    finally:
        if writer is not None:
            writer.close()

    coef_stats = {}
    if all_coef_tr:
        for name, values in (
            ("tr", np.stack(all_coef_tr, axis=0)),
            ("rot", np.stack(all_coef_rot, axis=0)),
            ("grip", np.stack(all_coef_grip, axis=0)),
        ):
            coef_stats[name] = {
                "mean": np.mean(values, axis=(0, 1)).astype(float).tolist(),
                "std": np.std(values, axis=(0, 1)).astype(float).tolist(),
                "min": np.min(values, axis=(0, 1)).astype(float).tolist(),
                "max": np.max(values, axis=(0, 1)).astype(float).tolist(),
            }

    top_action_rmse = top_records(top_samples, "hiva_action_rmse", max_bad_samples)
    top_action_max_abs = top_records(top_samples, "hiva_action_max_abs_diff", max_bad_samples)
    bad_samples_sorted = sorted(
        bad_samples,
        key=lambda row: (float(row["hiva_action_rmse"]), float(row["hiva_action_max_abs_diff"])),
        reverse=True,
    )[:max_bad_samples]

    summary = {
        "dataset_root": str(dataset_root),
        "duration_sidecar": str(duration_sidecar),
        "output_path": str(output_path),
        "num_episodes": int(len(episode_indices)),
        "num_rows": int(valid_count + invalid_count),
        "valid_rows": int(valid_count),
        "invalid_rows": int(invalid_count),
        "terminal_hold_rows": int(terminal_hold_count),
        "real_tail_rows": int(valid_count - terminal_hold_count),
        "source_durations": list(source_durations),
        "target_durations": list(target_durations),
        "source_to_target_duration_map": {str(k): int(v) for k, v in duration_map.items()},
        "target_duration_to_class": {str(k): int(v) for k, v in target_duration_to_class.items()},
        "duration_hist": {str(k): int(v) for k, v in sorted(hist.items())},
        "source_duration_hist": {str(k): int(v) for k, v in sorted(hist_source.items())},
        "k": int(n_ctrl),
        "degree": int(degree),
        "dmax": int(dmax),
        "ridge": float(ridge),
        "rot_scale_eta": float(eta),
        "basis": f"open-uniform clamped degree={degree} B-spline with K={n_ctrl} controls, evaluated at integer timesteps 0..d-1",
        "hiva_labeler_version": hiva_labeler_version,
        "rotation_convention": "R_tau=R_tau_minus_1@Exp(eta*raw_rot); decode raw_rot=Log(inv(R_tau_minus_1)@R_tau)/eta",
        "coef_stats": coef_stats,
        "bad_thresholds": {
            "action_rmse": float(bad_rmse_threshold),
            "action_max_abs_diff": float(bad_max_abs_threshold),
        },
        "bad_sample_count": int(len(bad_samples)),
        "bad_samples_top": bad_samples_sorted,
        "top_action_rmse": top_action_rmse,
        "top_action_max_abs_diff": top_action_max_abs,
        "episodes": episode_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build HiVA Experiment 3 coefficient sidecar with cubic B-spline macro-action coefficients."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--duration-sidecar", type=Path, default=Path(DEFAULT_DURATION_SIDECAR))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", type=Path, default=Path(DEFAULT_SUMMARY))
    parser.add_argument("--source-durations", default="1,3,8")
    parser.add_argument("--target-durations", default="4,8,12")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--rot-scale-eta", type=float, default=OSC_ROT_SCALE)
    parser.add_argument("--hiva-labeler-version", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--bad-rmse-threshold", type=float, default=0.05)
    parser.add_argument("--bad-max-abs-threshold", type=float, default=0.20)
    parser.add_argument("--max-bad-samples", type=int, default=25)
    args = parser.parse_args()
    target_durations = parse_int_tuple(args.target_durations)
    labeler_version = args.hiva_labeler_version
    if labeler_version is None:
        dur_tag = "_".join(str(d) for d in target_durations)
        labeler_version = f"hiva_coeff_d{dur_tag}_k{args.k}_v1"

    summary = build_coeff_sidecar(
        dataset_root=args.dataset_root,
        duration_sidecar=args.duration_sidecar,
        output_path=args.output,
        summary_path=args.summary_output,
        source_durations=parse_int_tuple(args.source_durations),
        target_durations=target_durations,
        n_ctrl=args.k,
        degree=args.degree,
        ridge=args.ridge,
        eta=args.rot_scale_eta,
        hiva_labeler_version=labeler_version,
        max_episodes=args.max_episodes,
        bad_rmse_threshold=args.bad_rmse_threshold,
        bad_max_abs_threshold=args.bad_max_abs_threshold,
        max_bad_samples=args.max_bad_samples,
    )
    print(json.dumps({k: summary[k] for k in ("output_path", "num_rows", "valid_rows", "invalid_rows", "duration_hist", "bad_sample_count")}, indent=2))
    print("Top action RMSE samples:")
    print(json.dumps(summary["top_action_rmse"][:5], indent=2))
    print("Top action max-abs-diff samples:")
    print(json.dumps(summary["top_action_max_abs_diff"][:5], indent=2))


if __name__ == "__main__":
    main()
