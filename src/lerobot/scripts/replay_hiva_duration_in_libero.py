from __future__ import annotations

import argparse
import json
from io import BytesIO
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation as R

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'imageio'. Install it with `pip install imageio`." ) from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'Pillow'. Install it with `pip install pillow`." ) from exc

try:
    from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action
    from lerobot.utils.constants import ACTION
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import LeRobot LIBERO modules. Run this inside your patched `smolvla_libero` env."
    ) from exc


DEFAULT_CAMERA = "image"
DEFAULT_DATASET_MATCH_IMAGE_KEY = "observation.images.agentview"
DEFAULT_METADATA_INIT_STATE_COLUMN = "libero_init_state_id"
DEFAULT_VIDEO_FPS = 20  # LIBERO currently steps at robosuite's default control rate in practice.
REQUIRED_SIDECAR_COLS = (
    "dataset_index",
    "episode_index",
    "frame_index",
    "duration_label",
    "duration_class",
)
OPTIONAL_SIDECAR_COLS = (
    "phase",
    "near_gripper",
    "dist_to_switch",
    "cmd_switch",
    "qpos_switch",
    "duration_label_s",
    "duration_class_s",
    "duration_label_p",
    "duration_class_p",
    "pcf_frame",
    "is_pcf",
    "dist_to_pure_contact",
    "inside_purecontact_zone",
    "purecontact_case_category",
    "purecontact_keyword",
    "purecontact_verb",
    "purecontact_contact_start",
    "purecontact_contact_end",
    "purecontact_reset_mode",
    "purecontact_reset_seed",
    "labeler_version",
    "hiva_theta_tr_raw",
    "hiva_theta_rot_raw",
    "hiva_theta_grip_raw",
    "hiva_k",
    "hiva_degree",
    "hiva_tail_mode",
    "hiva_real_steps",
    "hiva_requested_duration",
    "hiva_has_synthetic_tail",
    "hiva_rot_scale_eta",
    "hiva_action_rmse",
    "hiva_action_max_abs_diff",
    "hiva_basis_mode",
    "hiva_target_mode",
    "hiva_basis_dmax",
    "hiva_basis_k",
    "hiva_basis_degree",
    "hiva_basis_degree_tr",
    "hiva_basis_degree_rot",
    "hiva_basis_degree_grip",
    "hiva_prefix_action_rmse",
    "hiva_prefix_action_max_abs_diff",
    "hiva_real_prefix_action_rmse",
    "hiva_real_prefix_action_max_abs_diff",
)


def parse_int_ranges(spec: str | None) -> list[int]:
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


def parse_episode_init_state_map(spec: str | Path | None) -> dict[int, int]:
    if spec is None:
        return {}

    text_or_path = str(spec).strip()
    if not text_or_path:
        return {}

    path = Path(text_or_path)
    if path.exists():
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Expected {path} to contain a JSON object mapping episode ids to init-state ids")
        return {int(k): int(v) for k, v in payload.items()}

    out: dict[int, int] = {}
    for chunk in text_or_path.split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" not in token and "=" not in token:
            raise ValueError(
                "Episode init-state maps must be JSON files or inline pairs like '26:17,39:9'. "
                f"Could not parse token: {token!r}"
            )
        left, right = token.replace("=", ":", 1).split(":", 1)
        out[int(left.strip())] = int(right.strip())
    return out


def _read_parquet_dataset(path: Path, columns: list[str] | None = None, filter_expr=None):
    if path.is_file():
        return pq.read_table(path, columns=columns)
    return pads.dataset(str(path), format="parquet").to_table(columns=columns, filter=filter_expr)


def load_tasks_map(root: Path) -> dict[int, str]:
    task_path = root / "meta" / "tasks.parquet"
    table = _read_parquet_dataset(task_path)
    cols = set(table.column_names)
    if "task_index" not in cols or "task" not in cols:
        raise KeyError(
            f"Expected `task_index` and `task` columns in {task_path}. Found: {sorted(cols)}"
        )
    out: dict[int, str] = {}
    for row in table.to_pylist():
        out[int(row["task_index"])] = str(row["task"])
    return out


def load_episode_ranges(root: Path) -> dict[int, tuple[int, int]]:
    ep_dir = root / "meta" / "episodes"
    table = _read_parquet_dataset(ep_dir, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
    out: dict[int, tuple[int, int]] = {}
    for row in table.to_pylist():
        out[int(row["episode_index"])] = (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
    return out


def load_episode_init_state_ids_from_metadata(root: Path, column_name: str) -> dict[int, int]:
    ep_dir = root / "meta" / "episodes"
    schema_names = set(pads.dataset(str(ep_dir), format="parquet").schema.names)
    if column_name not in schema_names:
        return {}

    table = _read_parquet_dataset(ep_dir, columns=["episode_index", column_name])
    out: dict[int, int] = {}
    for row in table.to_pylist():
        value = row.get(column_name)
        if value is not None:
            out[int(row["episode_index"])] = int(value)
    return out


def load_episode_init_state_metadata(root: Path, column_name: str) -> dict[int, dict[str, Any]]:
    ep_dir = root / "meta" / "episodes"
    schema_names = set(pads.dataset(str(ep_dir), format="parquet").schema.names)
    if column_name not in schema_names:
        return {}

    optional_columns = [
        "libero_demo_key",
        "libero_hdf5_path",
        "libero_init_state_source",
        "libero_benchmark_task_id",
    ]
    columns = ["episode_index", column_name] + [col for col in optional_columns if col in schema_names]
    table = _read_parquet_dataset(ep_dir, columns=columns)
    out: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        value = row.get(column_name)
        if value is None:
            continue
        resolution = {"init_state_id": int(value)}
        for col in optional_columns:
            if col in row and row[col] is not None:
                resolution[col] = row[col]
        out[int(row["episode_index"])] = resolution
    return out


def load_hdf5_init_state(hdf5_path: str | Path, demo_key: str) -> np.ndarray:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'h5py'. Install it or omit raw-HDF5 metadata replay.") from exc

    resolved_path = resolve_existing_hdf5_path(hdf5_path)
    with h5py.File(resolved_path, "r") as h5_file:
        return np.asarray(h5_file["data"][str(demo_key)].attrs["init_state"])


def resolve_existing_hdf5_path(hdf5_path: str | Path) -> Path:
    path = Path(hdf5_path)
    if path.exists():
        return path

    path_text = str(path)
    prefix_map = (
        (
            "/nfs/bigbrain/add_disk0/jongwoopark/LIBERO-datasets/",
            "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/LIBERO-datasets/",
        ),
        (
            "/nfs/bigbrain.cs.stonybrook.edu/add_disk0/jongwoopark/LIBERO-datasets/",
            "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/LIBERO-datasets/",
        ),
        ("/nfs/bigbrain/add_disk0/", "/nfs/bigbrain.cs.stonybrook.edu/add_disk0/"),
        ("/nfs/bigflow/add_disk0/", "/nfs/bigflow.cs.stonybrook.edu/add_disk0/"),
        ("/nfs/bigcornea/add_disk2/", "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/"),
    )
    for old_prefix, new_prefix in prefix_map:
        if path_text.startswith(old_prefix):
            candidate = Path(new_prefix + path_text[len(old_prefix):])
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Could not find HDF5 init-state file: {hdf5_path}")


def reset_to_raw_init_state(env: LiberoEnv, raw_init_state: np.ndarray, seed: int | None = None):
    env._env.seed(seed)
    env._env.reset()
    raw_obs = env._env.set_init_state(raw_init_state)
    for _ in range(env.num_steps_wait):
        raw_obs, _, _, _ = env._env.step(get_libero_dummy_action())
    if env.control_mode == "absolute":
        for robot in env._env.robots:
            robot.controller.use_delta = False
    elif env.control_mode == "relative":
        for robot in env._env.robots:
            robot.controller.use_delta = True
    return env._format_raw_obs(raw_obs), {"is_success": False}


def load_first_frame_task_indices(root: Path) -> dict[int, int]:
    data_dir = root / "data"
    table = _read_parquet_dataset(
        data_dir,
        columns=["episode_index", "task_index", "frame_index"],
        filter_expr=(pads.field("frame_index") == 0),
    )
    out: dict[int, int] = {}
    for row in table.to_pylist():
        out[int(row["episode_index"])] = int(row["task_index"])
    return out


def load_episode_rows(root: Path, episode_index: int) -> list[dict[str, Any]]:
    data_dir = root / "data"
    table = _read_parquet_dataset(
        data_dir,
        columns=["index", "episode_index", "frame_index", "task_index", ACTION, "timestamp"],
        filter_expr=(pads.field("episode_index") == int(episode_index)),
    )
    rows = table.to_pylist()
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def load_first_frame_image(root: Path, episode_index: int, image_key: str) -> np.ndarray:
    data_dir = root / "data"
    table = _read_parquet_dataset(
        data_dir,
        columns=["frame_index", image_key],
        filter_expr=(pads.field("episode_index") == int(episode_index)),
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"No image rows found for episode_index={episode_index}")
    rows.sort(key=lambda row: int(row["frame_index"]))

    image_value = rows[0][image_key]
    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        return np.asarray(Image.open(BytesIO(image_value["bytes"])).convert("RGB"))
    if isinstance(image_value, dict) and image_value.get("path"):
        image_path = root / str(image_value["path"])
        return np.asarray(Image.open(image_path).convert("RGB"))
    raise TypeError(
        f"Unsupported image value for {image_key} in episode {episode_index}: {type(image_value).__name__}"
    )


def load_sidecar_map(sidecar_path: Path) -> dict[int, dict[str, Any]]:
    table = _read_parquet_dataset(sidecar_path)
    cols = set(table.column_names)
    missing = [c for c in REQUIRED_SIDECAR_COLS if c not in cols]
    if missing:
        raise KeyError(f"Sidecar {sidecar_path} missing required columns {missing}. Found: {sorted(cols)}")

    wanted = [c for c in REQUIRED_SIDECAR_COLS if c in cols] + [c for c in OPTIONAL_SIDECAR_COLS if c in cols]
    table = table.select(wanted)

    out: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        out[int(row["dataset_index"])] = row
    return out


def greedy_segments_for_episode(rows: list[dict[str, Any]], sidecar_by_dataset_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    t = 0
    T = len(rows)
    while t < T:
        dataset_index = int(rows[t]["index"])
        side = sidecar_by_dataset_index[dataset_index]
        d = int(side["duration_label"])
        end_exclusive = min(T, t + d)
        segments.append(
            {
                "segment_id": len(segments),
                "start_frame": t,
                "end_frame_exclusive": end_exclusive,
                "duration": d,
                "phase": side.get("phase"),
                "cmd_switch": int(side.get("cmd_switch", 0)),
                "qpos_switch": int(side.get("qpos_switch", 0)),
                "dist_to_switch": int(side.get("dist_to_switch", -1)),
            }
        )
        t = end_exclusive
    return segments


def clamped_bspline_basis(d: int, *, n_ctrl: int, degree: int) -> np.ndarray:
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


def decode_rotation_to_raw_actions(rho_hat: np.ndarray, eta: float) -> np.ndarray:
    rotations = R.from_rotvec(np.asarray(rho_hat, dtype=np.float64))
    raw_rot = []
    prev = R.identity()
    for cur in rotations:
        delta = prev.inv() * cur
        raw_rot.append(delta.as_rotvec() / float(eta))
        prev = cur
    return np.asarray(raw_rot, dtype=np.float64)


def decode_hiva_segment_actions(side_row: dict[str, Any]) -> np.ndarray:
    duration = int(side_row["duration_label"])
    theta_tr = np.asarray(side_row["hiva_theta_tr_raw"], dtype=np.float64)
    theta_rot = np.asarray(side_row["hiva_theta_rot_raw"], dtype=np.float64)
    theta_grip = np.asarray(side_row["hiva_theta_grip_raw"], dtype=np.float64)
    n_ctrl = int(side_row.get("hiva_basis_k", side_row.get("hiva_k", theta_tr.shape[0])))
    degree = int(side_row.get("hiva_basis_degree", side_row.get("hiva_degree", 3)))
    degree_tr = int(side_row.get("hiva_basis_degree_tr", degree))
    degree_rot = int(side_row.get("hiva_basis_degree_rot", degree))
    degree_grip = int(side_row.get("hiva_basis_degree_grip", degree))
    eta = float(side_row.get("hiva_rot_scale_eta", 0.5))
    basis_mode = str(side_row.get("hiva_basis_mode", "duration_specific"))
    basis_horizon = (
        int(side_row.get("hiva_basis_dmax", duration))
        if basis_mode in {"canonical_hp", "canonical_mt", "canonical_lp_mt"}
        else duration
    )

    phi_tr = clamped_bspline_basis(basis_horizon, n_ctrl=n_ctrl, degree=degree_tr)
    phi_rot = clamped_bspline_basis(basis_horizon, n_ctrl=n_ctrl, degree=degree_rot)
    phi_grip = clamped_bspline_basis(basis_horizon, n_ctrl=n_ctrl, degree=degree_grip)
    p_tr = phi_tr @ theta_tr
    rho_rot = phi_rot @ theta_rot
    g_grip = phi_grip @ theta_grip

    tr = np.diff(np.concatenate([np.zeros((1, 3), dtype=np.float64), p_tr], axis=0), axis=0)
    rot = decode_rotation_to_raw_actions(rho_rot, eta=eta)
    grip = np.clip(g_grip, -1.0, 1.0)
    return np.concatenate([tr, rot, grip], axis=1).astype(np.float32)


def decoded_actions_for_episode(
    rows: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    sidecar_by_dataset_index: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    decoded = np.zeros((len(rows), 7), dtype=np.float32)
    decode_summary: list[dict[str, Any]] = []

    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        side_row = sidecar_by_dataset_index[int(rows[start]["index"])]
        required = ["hiva_theta_tr_raw", "hiva_theta_rot_raw", "hiva_theta_grip_raw"]
        missing = [key for key in required if key not in side_row]
        if missing:
            raise KeyError(f"Cannot decode HiVA actions. Sidecar row is missing {missing}.")

        decoded_segment = decode_hiva_segment_actions(side_row)
        segment_len = end - start
        decoded[start:end] = decoded_segment[:segment_len]
        original = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in rows[start:end]], axis=0)
        diff = decoded[start:end].astype(np.float64) - original.astype(np.float64)
        decode_summary.append(
            {
                "segment_id": int(segment["segment_id"]),
                "start_frame": start,
                "end_frame_exclusive": end,
                "duration": int(segment["duration"]),
                "tail_mode": side_row.get("hiva_tail_mode", "unknown"),
                "hiva_real_steps": int(side_row.get("hiva_real_steps", segment["duration"])),
                "hiva_has_synthetic_tail": int(side_row.get("hiva_has_synthetic_tail", 0)),
                "basis_mode": str(side_row.get("hiva_basis_mode", "duration_specific")),
                "basis_degree": int(side_row.get("hiva_basis_degree", side_row.get("hiva_degree", 3))),
                "basis_degree_tr": int(
                    side_row.get(
                        "hiva_basis_degree_tr",
                        side_row.get("hiva_basis_degree", side_row.get("hiva_degree", 3)),
                    )
                ),
                "basis_degree_rot": int(
                    side_row.get(
                        "hiva_basis_degree_rot",
                        side_row.get("hiva_basis_degree", side_row.get("hiva_degree", 3)),
                    )
                ),
                "basis_degree_grip": int(
                    side_row.get(
                        "hiva_basis_degree_grip",
                        side_row.get("hiva_basis_degree", side_row.get("hiva_degree", 3)),
                    )
                ),
                "stored_action_rmse": float(
                    side_row.get("hiva_prefix_action_rmse", side_row.get("hiva_action_rmse", -1.0))
                ),
                "stored_action_max_abs_diff": float(
                    side_row.get(
                        "hiva_prefix_action_max_abs_diff",
                        side_row.get("hiva_action_max_abs_diff", -1.0),
                    )
                ),
                "episode_replay_action_rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
                "episode_replay_action_max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            }
        )

    return decoded, decode_summary


def write_decoded_actions(
    path: Path,
    *,
    episode_index: int,
    action_source: str,
    rows: list[dict[str, Any]],
    decoded_actions: np.ndarray,
    decode_summary: list[dict[str, Any]],
) -> None:
    original_actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in rows], axis=0)
    diff = decoded_actions.astype(np.float64) - original_actions.astype(np.float64)
    payload = {
        "episode_index": int(episode_index),
        "action_source": action_source,
        "num_frames": int(len(rows)),
        "decoded_shape": list(decoded_actions.shape),
        "overall_action_rmse_vs_dataset": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "overall_action_max_abs_diff_vs_dataset": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "segments": decode_summary,
        "frames": [
            {
                "frame_index": int(row["frame_index"]),
                "dataset_index": int(row["index"]),
                "decoded_action": decoded_actions[i].astype(float).tolist(),
                "dataset_action": original_actions[i].astype(float).tolist(),
            }
            for i, row in enumerate(rows)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def draw_overlay(
    image_np: np.ndarray,
    *,
    task_language: str,
    dataset_task_index: int,
    episode_index: int,
    frame_index: int,
    total_frames: int,
    side_row: dict[str, Any],
    status_text: str,
) -> np.ndarray:
    image = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    duration = int(side_row["duration_label"])
    duration_s = int(side_row.get("duration_label_s", duration))
    duration_p = int(side_row.get("duration_label_p", 8))
    cmd_switch = int(side_row.get("cmd_switch", 0))
    qpos_switch = int(side_row.get("qpos_switch", 0))
    dist_to_switch = int(side_row.get("dist_to_switch", -1))
    pcf_frame = int(side_row.get("pcf_frame", -1))
    pcf_text = "None" if pcf_frame < 0 else str(pcf_frame)
    dist_to_pure_contact = int(side_row.get("dist_to_pure_contact", -1))

    lines = [
        f"TK='{task_language}'",
        f"{status_text} D={duration} EPS={episode_index} FR={frame_index}/{max(total_frames - 1, 0)}",
        f"Ds={duration_s} DTS={dist_to_switch} CMS={cmd_switch} QPS={qpos_switch}",
        f"Dp={duration_p} PCF={pcf_text} DTPC={dist_to_pure_contact}",
    ]

    margin = 8
    line_h = 14
    box_h = margin * 2 + line_h * len(lines)
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0, 180))
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    return np.asarray(image)


def build_task_episode_lists(episode_to_task_index: dict[int, int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for ep, task_idx in sorted(episode_to_task_index.items()):
        grouped[int(task_idx)].append(int(ep))
    return dict(grouped)


def transform_match_image(image: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return image
    if transform == "h":
        return image[::-1, :, :]
    if transform == "w":
        return image[:, ::-1, :]
    if transform == "hw":
        return image[::-1, ::-1, :]
    raise ValueError(f"Unsupported image transform: {transform}")


def image_mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[:2] != right.shape[:2]:
        right = np.asarray(Image.fromarray(right.astype(np.uint8)).resize((left.shape[1], left.shape[0])))
    return float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))


def infer_init_state_ids_by_first_frame(
    *,
    root: Path,
    suite: str,
    task_id: int,
    episode_indices: list[int],
    image_key: str,
    control_mode: str,
    match_transform: str,
    max_init_states: int | None,
    top_k: int,
) -> dict[int, dict[str, Any]]:
    target_images = {
        int(ep): load_first_frame_image(root, int(ep), image_key)
        for ep in episode_indices
    }
    first_target = next(iter(target_images.values()))
    match_height, match_width = first_target.shape[:2]

    suite_obj = _get_suite(suite)
    env = LiberoEnv(
        task_suite=suite_obj,
        task_id=int(task_id),
        task_suite_name=str(suite),
        episode_index=0,
        n_envs=1,
        camera_name=("agentview_image", "robot0_eye_in_hand_image"),
        obs_type="pixels",
        render_mode="rgb_array",
        observation_height=int(match_height),
        observation_width=int(match_width),
        control_mode=str(control_mode),
    )
    env.reset()

    try:
        num_init_states = len(env._init_states)
        if max_init_states is not None:
            num_init_states = min(num_init_states, int(max_init_states))

        scores: dict[int, list[dict[str, Any]]] = {int(ep): [] for ep in episode_indices}
        transforms = ("none", "h", "w", "hw") if match_transform == "auto" else (match_transform,)

        for init_state_id in range(num_init_states):
            raw_obs = env._env.set_init_state(env._init_states[init_state_id])
            for _ in range(env.num_steps_wait):
                raw_obs, _, _, _ = env._env.step(get_libero_dummy_action())
            candidate = np.asarray(env._format_raw_obs(raw_obs)["pixels"][DEFAULT_CAMERA]).astype(np.uint8)

            for episode_index, target in target_images.items():
                best_score = float("inf")
                best_transform = transforms[0]
                for transform in transforms:
                    transformed = transform_match_image(candidate, transform)
                    score = image_mse(transformed, target)
                    if score < best_score:
                        best_score = score
                        best_transform = transform
                scores[episode_index].append(
                    {
                        "init_state_id": int(init_state_id),
                        "mse": float(best_score),
                        "transform": str(best_transform),
                    }
                )

        resolved: dict[int, dict[str, Any]] = {}
        for episode_index, episode_scores in scores.items():
            ranked = sorted(episode_scores, key=lambda item: item["mse"])
            resolved[int(episode_index)] = {
                "init_state_id": int(ranked[0]["init_state_id"]),
                "mse": float(ranked[0]["mse"]),
                "transform": str(ranked[0]["transform"]),
                "top_matches": ranked[: int(top_k)],
            }
        return resolved
    finally:
        env.close()


def infer_dataset_task_index(
    *,
    suite: str,
    task_id: int,
    task_map: dict[int, str],
    explicit_dataset_task_index: int | None,
) -> tuple[int, str, str]:
    suite_obj = _get_suite(suite)
    task_obj = suite_obj.get_task(task_id)
    target_language = str(task_obj.language)

    if explicit_dataset_task_index is not None:
        if explicit_dataset_task_index not in task_map:
            raise KeyError(
                f"dataset_task_index={explicit_dataset_task_index} not found in tasks.parquet. "
                f"Available task indices: {sorted(task_map)}"
            )
        return explicit_dataset_task_index, task_map[explicit_dataset_task_index], target_language

    matches = [idx for idx, text in task_map.items() if str(text) == target_language]
    if len(matches) == 1:
        idx = matches[0]
        return idx, task_map[idx], target_language

    if not matches:
        raise ValueError(
            "Could not auto-map the requested LIBERO task to tasks.parquet. "
            f"No dataset task string exactly matched LIBERO language: {target_language!r}. "
            "Pass --dataset-task-index explicitly after inspecting meta/tasks.parquet."
        )

    raise ValueError(
        "Multiple dataset task indices matched the requested LIBERO task language. "
        f"Matches: {matches}. Pass --dataset-task-index explicitly."
    )


def make_env_for_episode(
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    observation_height: int,
    observation_width: int,
    control_mode: str,
) -> LiberoEnv:
    suite_obj = _get_suite(suite)
    env = LiberoEnv(
        task_suite=suite_obj,
        task_id=int(task_id),
        task_suite_name=str(suite),
        episode_index=int(init_state_id),
        n_envs=1,
        camera_name=("agentview_image", "robot0_eye_in_hand_image"),
        obs_type="pixels",
        render_mode="rgb_array",
        observation_height=int(observation_height),
        observation_width=int(observation_width),
        control_mode=str(control_mode),
        auto_reset_on_done=False,
    )
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay relabeled LIBERO expert episodes with duration overlays.")
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset.repo-id", dest="dataset_repo_id", type=str, default="local/libero_lerobot_v3_lerobotkeys")
    parser.add_argument("--sidecar", type=Path, required=False)
    parser.add_argument("--suite", type=str, required=True, help="e.g. libero_10")
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO task id within the suite")
    parser.add_argument("--episode-indices", type=str, required=False, help="Global dataset episode ids, e.g. 0-2 or 4,9,12")
    parser.add_argument("--dataset-task-index", type=int, default=None, help="Override auto-mapping from LIBERO task language -> tasks.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument("--control-mode", type=str, default="relative", choices=["relative", "absolute"])
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--action-source",
        choices=["dataset", "hiva-decoded"],
        default="dataset",
        help=(
            "Action stream used for replay. 'dataset' replays saved expert actions. "
            "'hiva-decoded' decodes actions from hiva_theta_* coefficients in the sidecar."
        ),
    )
    parser.add_argument(
        "--save-decoded-actions",
        action="store_true",
        help="Write per-frame decoded actions and reconstruction errors to JSON.",
    )
    parser.add_argument(
        "--init-state-source",
        choices=["auto", "metadata", "map", "first-frame"],
        default="auto",
        help=(
            "How to choose the LIBERO init-state id for each global dataset episode. "
            "'auto' uses --episode-init-state-map first, then metadata, then first-frame matching."
        ),
    )
    parser.add_argument(
        "--episode-init-state-map",
        type=str,
        default=None,
        help="JSON file or inline mapping such as '26:17,39:9,69:21'.",
    )
    parser.add_argument(
        "--metadata-init-state-column",
        type=str,
        default=DEFAULT_METADATA_INIT_STATE_COLUMN,
        help="meta/episodes column containing the original LIBERO init-state id, if present.",
    )
    parser.add_argument(
        "--first-frame-image-key",
        type=str,
        default=DEFAULT_DATASET_MATCH_IMAGE_KEY,
        help="Dataset image feature used for first-frame init-state matching.",
    )
    parser.add_argument(
        "--first-frame-match-transform",
        choices=["none", "h", "w", "hw", "auto"],
        default="hw",
        help="Transform applied to rendered LIBERO frames before matching dataset first frames.",
    )
    parser.add_argument(
        "--max-init-states",
        type=int,
        default=None,
        help="Optional cap for first-frame matching candidates. Defaults to all LIBERO init states.",
    )
    parser.add_argument("--list-task-episodes", action="store_true", help="Print dataset episode ids for the resolved task and exit")
    args = parser.parse_args()

    root = args.dataset_root
    sidecar_path = args.sidecar
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_episodes = parse_int_ranges(args.episode_indices)
    if not args.list_task_episodes and not selected_episodes:
        raise ValueError("No episode indices parsed from --episode-indices")

    task_map = load_tasks_map(root)
    episode_ranges = load_episode_ranges(root)
    episode_to_task_index = load_first_frame_task_indices(root)

    sidecar_by_dataset_index = {}
    if not args.list_task_episodes:
        if sidecar_path is None:
            raise ValueError("--sidecar is required unless --list-task-episodes is used")
        sidecar_by_dataset_index = load_sidecar_map(sidecar_path)

    dataset_task_index, dataset_task_text, libero_task_language = infer_dataset_task_index(
        suite=args.suite,
        task_id=args.task_id,
        task_map=task_map,
        explicit_dataset_task_index=args.dataset_task_index,
    )

    task_episode_lists = build_task_episode_lists(episode_to_task_index)
    if dataset_task_index not in task_episode_lists:
        raise ValueError(f"No episodes found for dataset_task_index={dataset_task_index}")
    episodes_for_task = task_episode_lists[dataset_task_index]
    episodes_for_task_set = set(episodes_for_task)

    if args.list_task_episodes:
        payload = {
            "suite": args.suite,
            "task_id": int(args.task_id),
            "libero_task_language": libero_task_language,
            "dataset_task_index": int(dataset_task_index),
            "dataset_task_text": dataset_task_text,
            "num_task_episodes": len(episodes_for_task),
            "task_episodes": episodes_for_task,
        }
        print(json.dumps(payload, indent=2))
        return

    summary = {
        "suite": args.suite,
        "task_id": int(args.task_id),
        "libero_task_language": libero_task_language,
        "dataset_task_index": int(dataset_task_index),
        "dataset_task_text": dataset_task_text,
        "selected_episodes": selected_episodes,
        "action_source": args.action_source,
        "init_state_source": args.init_state_source,
        "replays": [],
    }

    episode_init_state_map = parse_episode_init_state_map(args.episode_init_state_map)
    metadata_init_state_map = load_episode_init_state_metadata(root, args.metadata_init_state_column)
    resolved_init_states: dict[int, dict[str, Any]] = {}

    if args.init_state_source == "map":
        missing = [ep for ep in selected_episodes if ep not in episode_init_state_map]
        if missing:
            raise KeyError(f"--episode-init-state-map is missing selected episodes: {missing}")
        resolved_init_states = {
            ep: {"init_state_id": int(episode_init_state_map[ep]), "source": "map"}
            for ep in selected_episodes
        }
    elif args.init_state_source == "metadata":
        missing = [ep for ep in selected_episodes if ep not in metadata_init_state_map]
        if missing:
            raise KeyError(
                f"Metadata column {args.metadata_init_state_column!r} is missing selected episodes: {missing}"
            )
        resolved_init_states = {
            ep: {**metadata_init_state_map[ep], "source": "metadata"}
            for ep in selected_episodes
        }
    elif args.init_state_source == "auto":
        for ep in selected_episodes:
            if ep in episode_init_state_map:
                resolved_init_states[ep] = {"init_state_id": int(episode_init_state_map[ep]), "source": "map"}
            elif ep in metadata_init_state_map:
                resolved_init_states[ep] = {**metadata_init_state_map[ep], "source": "metadata"}

    remaining_for_first_frame = [
        ep for ep in selected_episodes
        if args.init_state_source == "first-frame" or ep not in resolved_init_states
    ]
    if remaining_for_first_frame:
        first_frame_matches = infer_init_state_ids_by_first_frame(
            root=root,
            suite=args.suite,
            task_id=args.task_id,
            episode_indices=remaining_for_first_frame,
            image_key=args.first_frame_image_key,
            control_mode=args.control_mode,
            match_transform=args.first_frame_match_transform,
            max_init_states=args.max_init_states,
            top_k=5,
        )
        for ep, match in first_frame_matches.items():
            resolved_init_states[ep] = {**match, "source": "first-frame"}

    summary["resolved_init_states"] = resolved_init_states

    for episode_index in selected_episodes:
        if episode_index not in episode_ranges:
            raise KeyError(f"episode_index={episode_index} not found in episode metadata")
        if episode_index not in episodes_for_task_set:
            task_idx = episode_to_task_index.get(episode_index, None)
            task_txt = task_map.get(task_idx, "<unknown>")
            raise ValueError(
                f"episode_index={episode_index} belongs to dataset_task_index={task_idx} ({task_txt!r}), "
                f"not the requested dataset_task_index={dataset_task_index} ({dataset_task_text!r})."
            )

        init_state_resolution = resolved_init_states[int(episode_index)]
        init_state_id = int(init_state_resolution["init_state_id"])
        rows = load_episode_rows(root, episode_index)
        if not rows:
            raise ValueError(f"No frame rows found for episode_index={episode_index}")

        first_task_index = int(rows[0]["task_index"])
        if first_task_index != dataset_task_index:
            raise ValueError(
                f"episode_index={episode_index} has task_index={first_task_index}, expected {dataset_task_index}"
            )

        for row in rows:
            dataset_index = int(row["index"])
            if dataset_index not in sidecar_by_dataset_index:
                raise KeyError(
                    f"dataset_index={dataset_index} from episode {episode_index} is missing in sidecar {sidecar_path}. "
                    "Build the sidecar for these episodes first."
                )

        segments = greedy_segments_for_episode(rows, sidecar_by_dataset_index)
        decoded_actions_path = output_dir / f"episode_{episode_index:04d}_decoded_actions.json"
        decoded_action_summary: list[dict[str, Any]] = []
        if args.action_source == "hiva-decoded":
            episode_actions, decoded_action_summary = decoded_actions_for_episode(rows, segments, sidecar_by_dataset_index)
            write_decoded_actions(
                decoded_actions_path,
                episode_index=episode_index,
                action_source=args.action_source,
                rows=rows,
                decoded_actions=episode_actions,
                decode_summary=decoded_action_summary,
            )
        else:
            episode_actions = np.stack([np.asarray(row[ACTION], dtype=np.float32) for row in rows], axis=0)
            if args.save_decoded_actions:
                decoded_actions, decoded_action_summary = decoded_actions_for_episode(rows, segments, sidecar_by_dataset_index)
                write_decoded_actions(
                    decoded_actions_path,
                    episode_index=episode_index,
                    action_source="hiva-decoded-reference",
                    rows=rows,
                    decoded_actions=decoded_actions,
                    decode_summary=decoded_action_summary,
                )

        env = make_env_for_episode(
            suite=args.suite,
            task_id=args.task_id,
            init_state_id=init_state_id,
            observation_height=args.observation_height,
            observation_width=args.observation_width,
            control_mode=args.control_mode,
        )
        first_side_row = sidecar_by_dataset_index[int(rows[0]["index"])]
        purecontact_reset_mode = first_side_row.get("purecontact_reset_mode")
        purecontact_reset_seed = int(first_side_row.get("purecontact_reset_seed", -1))
        reset_seed = purecontact_reset_seed if purecontact_reset_seed >= 0 else args.seed
        use_libero_init_state_reset = purecontact_reset_mode == "libero-init-state"

        if (
            not use_libero_init_state_reset
            and init_state_resolution.get("libero_hdf5_path")
            and init_state_resolution.get("libero_demo_key")
        ):
            raw_init_state = load_hdf5_init_state(
                resolve_existing_hdf5_path(init_state_resolution["libero_hdf5_path"]),
                init_state_resolution["libero_demo_key"],
            )
            obs, _info = reset_to_raw_init_state(env, raw_init_state, seed=reset_seed)
        else:
            obs, _info = env.reset(seed=reset_seed)

        video_path = output_dir / f"episode_{episode_index:04d}_task{args.task_id}_replay.mp4"
        segments_path = output_dir / f"episode_{episode_index:04d}_segments.json"
        meta_path = output_dir / f"episode_{episode_index:04d}_summary.json"

        frames_written = 0
        env_done_seen = False
        first_done_frame: int | None = None
        success = False
        frame_records: list[tuple[np.ndarray, dict[str, Any], int]] = []

        for t, row in enumerate(rows):
            side_row = sidecar_by_dataset_index[int(row["index"])]
            raw_img = obs["pixels"][DEFAULT_CAMERA]
            frame_records.append((np.asarray(raw_img)[::-1, ::-1].copy(), side_row, int(t)))
            frames_written += 1

            action = np.asarray(episode_actions[t], dtype=np.float32)
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if not env_done_seen:
                    first_done_frame = int(t)
                env_done_seen = True
                success = success or bool(
                    info.get("is_success") or info.get("final_info", {}).get("is_success", False)
                )

        result_text = "SC" if success else "FA"
        with imageio.get_writer(video_path, fps=args.video_fps, codec="libx264") as writer:
            for viz_img, side_row, t in frame_records:
                viz_img = draw_overlay(
                    viz_img,
                    task_language=libero_task_language,
                    dataset_task_index=dataset_task_index,
                    episode_index=episode_index,
                    frame_index=t,
                    total_frames=len(rows),
                    side_row=side_row,
                    status_text=result_text,
                )
                writer.append_data(viz_img)

        env.close()

        segments_path.write_text(json.dumps(segments, indent=2))
        episode_summary = {
            "episode_index": int(episode_index),
            "libero_init_state_id": int(init_state_id),
            "init_state_resolution": init_state_resolution,
            "action_source": args.action_source,
            "decoded_actions_path": str(decoded_actions_path) if decoded_actions_path.exists() else None,
            "decoded_action_summary": {
                "num_segments": len(decoded_action_summary),
                "mean_segment_rmse_vs_dataset": float(np.mean([s["episode_replay_action_rmse"] for s in decoded_action_summary]))
                if decoded_action_summary
                else None,
                "max_segment_abs_diff_vs_dataset": float(max(s["episode_replay_action_max_abs_diff"] for s in decoded_action_summary))
                if decoded_action_summary
                else None,
            },
            "task_index": int(dataset_task_index),
            "task_text": dataset_task_text,
            "libero_task_language": libero_task_language,
            "video_path": str(video_path),
            "segments_path": str(segments_path),
            "num_frames_replayed": int(frames_written),
            "requested_frames": int(len(rows)),
            "env_done_seen": bool(env_done_seen),
            "first_done_frame": first_done_frame,
            "terminated_early": False,
            "success": bool(success),
            "result_text": result_text,
            "dataset_frame_span": {
                "dataset_from_index": int(episode_ranges[episode_index][0]),
                "dataset_to_index": int(episode_ranges[episode_index][1]),
            },
        }
        meta_path.write_text(json.dumps(episode_summary, indent=2))
        summary["replays"].append(episode_summary)
        print(json.dumps(episode_summary, indent=2))

    (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote replay artifacts to {output_dir}")


if __name__ == "__main__":
    main()
