#!/usr/bin/env python
from __future__ import annotations

"""Build canonical hold-pad HiVA coefficient sidecar.

This builder consumes an existing frame-aligned duration sidecar whose duration labels were already
recomputed for the target duration set, e.g. {6, 10, 15}, and creates canonical-Dmax B-spline
coefficient labels. For every sample, the GT target is hold-padded after its GT duration d*: translation and
rotation cumulative targets hold their final value, and the absolute gripper command holds its final
value. Coefficients are fit against one shared Dmax x K cubic B-spline basis for all durations.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def clamped_bspline_basis(d: int, *, n_ctrl: int, degree: int) -> np.ndarray:
    if d < 2:
        raise ValueError(f"Duration must be >= 2, got {d}.")
    if n_ctrl < degree + 1:
        raise ValueError(f"n_ctrl={n_ctrl} must be at least degree+1={degree + 1}.")

    x_values = [float(x) for x in range(d)]
    internal_count = n_ctrl - degree + 1
    if internal_count <= 1:
        t_internal = [0.0]
    else:
        t_internal = [(d - 1) * i / (internal_count - 1) for i in range(internal_count)]
    knots = [0.0] * degree + t_internal + [float(d - 1)] * degree

    def basis_one(i: int, p: int, x: float) -> float:
        if p == 0:
            left, right = knots[i], knots[i + 1]
            if left <= x < right:
                return 1.0
            if x == knots[-1] and left <= x <= right:
                return 1.0
            return 0.0
        left_den = knots[i + p] - knots[i]
        right_den = knots[i + p + 1] - knots[i + 1]
        left_term = 0.0
        right_term = 0.0
        if left_den > 0:
            left_term = (x - knots[i]) / left_den * basis_one(i, p - 1, x)
        if right_den > 0:
            right_term = (knots[i + p + 1] - x) / right_den * basis_one(i + 1, p - 1, x)
        return left_term + right_term

    return np.asarray([[basis_one(i, degree, x) for i in range(n_ctrl)] for x in x_values], dtype=np.float32)


def solve_coeff(phi: np.ndarray, target: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    # target: [Dmax, C], phi: [Dmax, K]
    lhs = phi.T @ phi + ridge * np.eye(phi.shape[1], dtype=np.float32)
    rhs = phi.T @ target.astype(np.float32)
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    k = skew(rotvec.astype(np.float64))
    eye = np.eye(3, dtype=np.float64)
    if theta < 1e-8:
        return eye + k + 0.5 * (k @ k)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return eye + a * k + b * (k @ k)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    cos_theta = max(-1.0 + 1e-7, min(1.0 - 1e-7, 0.5 * (trace - 1.0)))
    theta = math.acos(cos_theta)
    omega = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-6:
        return (0.5 * omega).astype(np.float32)
    return ((theta / (2.0 * math.sin(theta))) * omega).astype(np.float32)


def cumulative_command_rotation(rot_actions: np.ndarray, *, eta: float) -> np.ndarray:
    out = []
    cur = np.eye(3, dtype=np.float64)
    for r in rot_actions:
        cur = cur @ rotvec_to_matrix(float(eta) * r.astype(np.float64))
        out.append(matrix_to_rotvec(cur))
    return np.asarray(out, dtype=np.float32)


def decode_actions(phi: np.ndarray, theta_tr: np.ndarray, theta_rot: np.ndarray, theta_grip: np.ndarray, *, eta: float):
    dmax = phi.shape[0]
    p_hat = phi @ theta_tr
    tr_delta = np.diff(np.concatenate([np.zeros((1, 3), dtype=np.float32), p_hat], axis=0), axis=0)

    rho_hat = phi @ theta_rot
    rot_mats = np.stack([rotvec_to_matrix(r) for r in rho_hat], axis=0)
    raw_rot = []
    prev = np.eye(3, dtype=np.float64)
    for mat in rot_mats:
        delta = prev.T @ mat
        raw_rot.append(matrix_to_rotvec(delta) / float(eta))
        prev = mat
    raw_rot = np.asarray(raw_rot, dtype=np.float32).reshape(dmax, 3)

    grip = np.clip(phi @ theta_grip, -1.0, 1.0).astype(np.float32)
    return np.concatenate([tr_delta.astype(np.float32), raw_rot, grip], axis=-1)


def parse_duration_map(items: list[str] | None) -> dict[int, int]:
    if not items:
        return {}
    out = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad duration map item {item!r}; expected old:new.")
        old, new = item.split(":", 1)
        out[int(old)] = int(new)
    return out


def load_base_actions(data_root: Path, columns: tuple[str, ...]) -> pa.Table:
    data_dir = data_root / "data"
    if data_dir.exists():
        dataset = ds.dataset(str(data_dir), format="parquet", partitioning="hive")
        return dataset.to_table(columns=list(columns))
    # Fallback: caller may pass a parquet file or directory directly.
    if data_root.suffix.lower() in {".parquet", ".pq"}:
        return pq.read_table(data_root, columns=list(columns))
    dataset = ds.dataset(str(data_root), format="parquet")
    return dataset.to_table(columns=list(columns))


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0


def max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x.astype(np.float64)))) if x.size else 0.0


def mean_norm(x: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(x.astype(np.float64), axis=-1))) if x.size else 0.0


def build(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    duration_sidecar = Path(args.duration_sidecar)
    output = Path(args.output)
    summary_json = Path(args.summary_json) if args.summary_json else output.with_suffix(".summary.json")
    duration_map = parse_duration_map(args.duration_map)
    duration_classes = tuple(int(x) for x in args.duration_classes)
    duration_to_class = {d: i for i, d in enumerate(duration_classes)}

    phi_can = clamped_bspline_basis(args.dmax, n_ctrl=args.n_ctrl, degree=args.degree)
    duration_rows = pq.read_table(duration_sidecar).to_pylist()
    base_table = load_base_actions(data_root, columns=("index", "episode_index", "frame_index", args.action_column))
    base_rows = base_table.to_pylist()

    by_dataset_index: dict[int, dict[str, Any]] = {int(r["index"]): r for r in base_rows}
    episode_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in base_rows:
        episode_rows[int(r["episode_index"])].append(r)
    for rows in episode_rows.values():
        rows.sort(key=lambda x: int(x["frame_index"]))

    episode_actions: dict[int, np.ndarray] = {}
    episode_frame_to_pos: dict[int, dict[int, int]] = {}
    for ep, rows in episode_rows.items():
        episode_actions[ep] = np.asarray([r[args.action_column] for r in rows], dtype=np.float32)
        episode_frame_to_pos[ep] = {int(r["frame_index"]): i for i, r in enumerate(rows)}

    out_rows = []
    stats_tr = []
    stats_rot = []
    stats_grip = []
    duration_hist = defaultdict(int)
    tail_mode_hist = defaultdict(int)
    metric_by_duration: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in duration_rows:
        dataset_index = int(row["dataset_index"])
        base = by_dataset_index.get(dataset_index)
        if base is None:
            raise KeyError(f"dataset_index={dataset_index} from duration sidecar is missing in base dataset.")
        ep = int(base["episode_index"])
        frame = int(base["frame_index"])
        pos = episode_frame_to_pos[ep][frame]
        actions_ep = episode_actions[ep]
        remaining = int(actions_ep.shape[0] - pos)
        if remaining <= 0:
            raise ValueError(f"No action remains for episode={ep}, frame={frame}.")

        raw_label = int(row["duration_label"])
        d_star = duration_map.get(raw_label, raw_label)
        if d_star not in duration_to_class:
            raise ValueError(f"Mapped duration {d_star} is not in {duration_classes}; raw label was {raw_label}.")
        real_steps = min(d_star, remaining)
        real_actions = actions_ep[pos : pos + real_steps]

        # HP targets over Dmax. For unavailable or post-duration steps, translation/rotation deltas are zero,
        # and gripper holds its final absolute command.
        tr_delta_target = np.zeros((args.dmax, 3), dtype=np.float32)
        tr_delta_target[:real_steps] = real_actions[:, 0:3]
        p_hold = np.cumsum(tr_delta_target, axis=0).astype(np.float32)

        rho_hold = np.zeros((args.dmax, 3), dtype=np.float32)
        rho_real = cumulative_command_rotation(real_actions[:, 3:6], eta=args.rot_scale_eta)
        rho_hold[:real_steps] = rho_real
        if real_steps < args.dmax:
            rho_hold[real_steps:] = rho_hold[real_steps - 1]

        g_hold = np.zeros((args.dmax, 1), dtype=np.float32)
        g_hold[:real_steps, 0] = real_actions[:, 6]
        if real_steps < args.dmax:
            g_hold[real_steps:, 0] = g_hold[real_steps - 1, 0]

        theta_tr = solve_coeff(phi_can, p_hold, ridge=args.ridge)
        theta_rot = solve_coeff(phi_can, rho_hold, ridge=args.ridge)
        theta_grip = solve_coeff(phi_can, g_hold, ridge=args.ridge)

        pred_actions = decode_actions(phi_can, theta_tr, theta_rot, theta_grip, eta=args.rot_scale_eta)
        target_actions = np.concatenate([tr_delta_target, np.zeros((args.dmax, 3), dtype=np.float32), g_hold], axis=-1)
        target_actions[:real_steps, 3:6] = real_actions[:, 3:6]

        prefix_slice = slice(0, d_star)
        real_slice = slice(0, real_steps)
        prefix_err = pred_actions[prefix_slice] - target_actions[prefix_slice]
        real_prefix_err = pred_actions[real_slice] - target_actions[real_slice]
        if d_star < args.dmax:
            tail = pred_actions[d_star:]
            grip_ref = pred_actions[d_star - 1, 6]
            wrong_long_tr = mean_norm(tail[:, 0:3])
            wrong_long_rot = mean_norm(tail[:, 3:6])
            wrong_long_grip = float(np.mean(np.abs(tail[:, 6] - grip_ref))) if tail.size else 0.0
        else:
            wrong_long_tr = wrong_long_rot = wrong_long_grip = 0.0
        wrong_long_action = wrong_long_tr + wrong_long_rot + wrong_long_grip

        # Preserve duration-sidecar metadata such as Ds/Dp labels, switch zones, and pure-contact
        # PCF fields so replay/debug overlays stay aligned with the coefficient targets.
        out_row = dict(row)
        out_row.update({
            "dataset_index": dataset_index,
            "episode_index": ep,
            "frame_index": frame,
            "duration_label": int(d_star),
            "duration_class": int(duration_to_class[d_star]),
            "source_duration_label": raw_label,
            "hiva_theta_tr_raw": theta_tr.tolist(),
            "hiva_theta_rot_raw": theta_rot.tolist(),
            "hiva_theta_grip_raw": theta_grip.tolist(),
            "hiva_P_tr_hold_raw": p_hold.tolist(),
            "hiva_rho_rot_hold_raw": rho_hold.tolist(),
            "hiva_G_grip_hold_raw": g_hold.tolist(),
            "hiva_basis_mode": "canonical_hp",
            "hiva_target_mode": "hold_pad",
            "hiva_basis_dmax": int(args.dmax),
            "hiva_basis_k": int(args.n_ctrl),
            "hiva_basis_degree": int(args.degree),
            "hiva_real_steps": int(real_steps),
            "hiva_has_synthetic_tail": bool(real_steps < d_star),
            "hiva_tail_mode": "terminal_hold" if real_steps < d_star else "real",
            "hiva_prefix_tr_rmse": rmse(prefix_err[:, 0:3]),
            "hiva_prefix_rot_rmse": rmse(prefix_err[:, 3:6]),
            "hiva_prefix_grip_rmse": rmse(prefix_err[:, 6:7]),
            "hiva_prefix_action_rmse": rmse(prefix_err),
            "hiva_prefix_tr_max_abs_diff": max_abs(prefix_err[:, 0:3]),
            "hiva_prefix_rot_max_abs_diff": max_abs(prefix_err[:, 3:6]),
            "hiva_prefix_grip_max_abs_diff": max_abs(prefix_err[:, 6:7]),
            "hiva_prefix_action_max_abs_diff": max_abs(prefix_err),
            "hiva_real_prefix_action_rmse": rmse(real_prefix_err),
            "hiva_real_prefix_action_max_abs_diff": max_abs(real_prefix_err),
            "hiva_wrong_long_tr_energy": wrong_long_tr,
            "hiva_wrong_long_rot_energy": wrong_long_rot,
            "hiva_wrong_long_grip_energy": wrong_long_grip,
            "hiva_wrong_long_action_energy": wrong_long_action,
            "hiva_tail_motion_energy": wrong_long_action,
        })
        out_rows.append(out_row)
        stats_tr.append(theta_tr)
        stats_rot.append(theta_rot)
        stats_grip.append(theta_grip)
        duration_hist[int(d_star)] += 1
        tail_mode_hist[out_row["hiva_tail_mode"]] += 1
        for k in (
            "hiva_prefix_tr_rmse",
            "hiva_prefix_rot_rmse",
            "hiva_prefix_grip_rmse",
            "hiva_prefix_action_rmse",
            "hiva_prefix_tr_max_abs_diff",
            "hiva_prefix_rot_max_abs_diff",
            "hiva_prefix_grip_max_abs_diff",
            "hiva_prefix_action_max_abs_diff",
            "hiva_real_prefix_action_rmse",
            "hiva_real_prefix_action_max_abs_diff",
            "hiva_wrong_long_action_energy",
            "hiva_tail_motion_energy",
        ):
            metric_by_duration[int(d_star)][k].append(float(out_row[k]))

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows), output)

    tr = np.stack(stats_tr, axis=0).reshape(-1, 3)
    rot = np.stack(stats_rot, axis=0).reshape(-1, 3)
    grip = np.stack(stats_grip, axis=0).reshape(-1, 1)

    def stat_block(arr: np.ndarray) -> dict[str, list[float]]:
        std = arr.std(axis=0).astype(np.float64)
        std = np.maximum(std, args.norm_eps)
        return {"mean": arr.mean(axis=0).astype(np.float64).tolist(), "std": std.tolist()}

    metrics = {}
    for d, groups in metric_by_duration.items():
        metrics[str(d)] = {key: float(np.mean(vals)) for key, vals in groups.items() if vals}

    summary = {
        "basis_mode": "canonical_hp",
        "target_mode": "hold_pad",
        "rows": len(out_rows),
        "duration_classes": list(duration_classes),
        "duration_hist": {str(k): int(v) for k, v in sorted(duration_hist.items())},
        "tail_mode_hist": {str(k): int(v) for k, v in sorted(tail_mode_hist.items())},
        "dmax": int(args.dmax),
        "n_ctrl": int(args.n_ctrl),
        "degree": int(args.degree),
        "rot_scale_eta": float(args.rot_scale_eta),
        "coef_stats": {"tr": stat_block(tr), "rot": stat_block(rot), "grip": stat_block(grip)},
        "diagnostics_by_duration": metrics,
        "basis_canonical": phi_can.tolist(),
        "source_duration_sidecar": str(duration_sidecar),
        "source_data_root": str(data_root),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows to {output}")
    print(f"Wrote summary to {summary_json}")
    print(json.dumps({"duration_hist": summary["duration_hist"], "tail_mode_hist": summary["tail_mode_hist"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="LeRobot dataset root or parquet dataset path.")
    parser.add_argument("--duration-sidecar", required=True, help="Existing frame-aligned duration sidecar parquet.")
    parser.add_argument("--output", required=True, help="Output HP coefficient sidecar parquet.")
    parser.add_argument("--summary-json", default=None, help="Output summary JSON. Defaults to output with .summary.json.")
    parser.add_argument("--duration-classes", nargs="+", type=int, default=[6, 10, 15])
    parser.add_argument(
        "--duration-map",
        nargs="*",
        default=None,
        help="Optional debugging-only old:new mapping. Production sidecars should use recomputed duration labels.",
    )
    parser.add_argument("--dmax", type=int, default=15)
    parser.add_argument("--n-ctrl", type=int, default=6)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--rot-scale-eta", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--action-column", default="action")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
