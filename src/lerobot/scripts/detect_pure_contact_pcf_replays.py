from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'imageio'. Install it with `pip install imageio`.") from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'Pillow'. Install it with `pip install pillow`.") from exc

from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.scripts.build_hiva_duration_sidecar import (
    DEFAULT_PURECONTACT_PCF_META,
    build_sidecar_rows,
    load_purecontact_pcf_annotations,
)
from lerobot.scripts.replay_hiva_duration_in_libero import (
    DEFAULT_CAMERA,
    load_episode_init_state_metadata,
    load_episode_rows,
    load_hdf5_init_state,
    reset_to_raw_init_state,
)
from lerobot.utils.constants import ACTION


PURECONTACT_KEYWORDS = ("drawer", "microwave")
PURECONTACT_VERBS = ("open", "close", "pull")
DRAWER_LEVELS = ("top", "middle", "bottom")


def default_purecontact_json(root: Path) -> Path:
    return root / DEFAULT_PURECONTACT_PCF_META


def write_purecontact_metadata(
    path: Path,
    *,
    source_summary: Path | None,
    episode_purecontact_labels: list[dict[str, Any]],
    annotations: dict[int, dict[str, Any]],
) -> None:
    episodes = {}
    for ep, ann in sorted(annotations.items()):
        record = {
            "pcf": int(ann["pcf"]),
            "contact_interval": ann.get("contact_interval"),
            "case_category": ann.get("case_category"),
            "purecontact_keyword": ann.get("purecontact_keyword"),
            "purecontact_verb": ann.get("purecontact_verb"),
            "method": ann.get("method", "sim_contact_plus_target_joint_verification"),
            "source_summary": None if source_summary is None else str(source_summary),
        }
        for optional_key in (
            "reset_mode",
            "seed",
            "movement_window",
            "movement_threshold",
            "same_sign_k",
            "require_horizontal_motion",
            "movement_after_onset",
            "same_sign_count_onset",
        ):
            if optional_key in ann:
                record[optional_key] = ann[optional_key]
        episodes[str(int(ep))] = record
    payload = {
        "schema_version": 1,
        "description": (
            "Saved pure-contact frame annotations. These are consumed by duration sidecar and replay generation "
            "so PCF does not need to be recomputed from MuJoCo contacts unless this file is being refreshed."
        ),
        "source_summary": None if source_summary is None else str(source_summary),
        "num_episodes_with_pcf": len(episodes),
        "episode_purecontact_labels": episode_purecontact_labels,
        "episodes": episodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _read_parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    table = pads.dataset(str(path), format="parquet").to_table(columns=columns)
    return table.to_pylist()


def load_task_map(root: Path) -> dict[int, str]:
    rows = _read_parquet_rows(root / "meta" / "tasks.parquet")
    out: dict[int, str] = {}
    for row in rows:
        # In this dataset, pyarrow preserves the task text index as a "task" column.
        out[int(row["task_index"])] = str(row.get("task", ""))
    return out


def load_episode_metadata(root: Path) -> list[dict[str, Any]]:
    columns = [
        "episode_index",
        "dataset_from_index",
        "dataset_to_index",
        "stats/task_index/mean",
        "libero_suite",
        "libero_benchmark_task_id",
        "libero_init_state_id",
        "libero_demo_key",
        "libero_hdf5_path",
    ]
    rows = _read_parquet_rows(root / "meta" / "episodes", columns=columns)
    for row in rows:
        task_index_value = row["stats/task_index/mean"]
        if isinstance(task_index_value, list):
            task_index_value = task_index_value[0]
        row["task_index"] = int(round(float(task_index_value)))
    return rows


def contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None


def combo_for_language(language: str) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for keyword in PURECONTACT_KEYWORDS:
        if not contains_word(language, keyword):
            continue
        for verb in PURECONTACT_VERBS:
            if contains_word(language, verb):
                combos.append((keyword, verb))
    return combos


def purecontact_case(language: str, keyword: str, verb: str) -> str:
    """Case1 means object interaction first, then open/close/pull target. Case2 means target interaction first."""
    text = language.lower()
    verb_match = re.search(rf"\b{re.escape(verb)}\b", text)
    if verb_match is None:
        return "none"
    prefix = text[: verb_match.start()]
    if any(contains_word(prefix, action) for action in ("put", "pick", "place")) and " and " in prefix:
        return "case1"
    if re.search(rf"\b(put|pick|place)\b.*\b{re.escape(keyword)}\b.*\band\b.*\b{re.escape(verb)}\b", text):
        return "case1"
    return "case2"


def target_spec(language: str, keyword: str) -> dict[str, Any]:
    text = language.lower()
    if keyword == "drawer":
        levels = [level for level in DRAWER_LEVELS if contains_word(text, level)] or list(DRAWER_LEVELS)
        return {
            "kind": "drawer",
            "target_body_substrings": [f"cabinet_{level}" for level in levels],
            "target_joint_substrings": [f"{level}_level" for level in levels],
            "levels": levels,
        }
    if keyword == "microwave":
        return {
            "kind": "microwave",
            "target_body_substrings": ["microdoorroot"],
            "target_joint_substrings": ["microjoint"],
            "levels": [],
        }
    raise ValueError(f"Unsupported purecontact keyword: {keyword}")


def make_env(suite: str, task_id: int, init_state_id: int, height: int, width: int) -> LiberoEnv:
    suite_obj = _get_suite(suite)
    return LiberoEnv(
        task_suite=suite_obj,
        task_id=int(task_id),
        task_suite_name=str(suite),
        episode_index=int(init_state_id),
        n_envs=1,
        camera_name=("agentview_image", "robot0_eye_in_hand_image"),
        obs_type="pixels",
        render_mode="rgb_array",
        observation_height=int(height),
        observation_width=int(width),
        control_mode="relative",
        auto_reset_on_done=False,
    )


def reset_env_from_metadata(
    env: LiberoEnv,
    init_resolution: dict[str, Any],
    seed: int | None,
    *,
    reset_mode: str = "raw-hdf5",
):
    if reset_mode == "libero-init-state":
        return env.reset(seed=seed)
    if reset_mode != "raw-hdf5":
        raise ValueError(f"Unsupported reset_mode: {reset_mode}")
    if init_resolution.get("libero_hdf5_path") and init_resolution.get("libero_demo_key"):
        raw_init_state = load_hdf5_init_state(
            init_resolution["libero_hdf5_path"],
            init_resolution["libero_demo_key"],
        )
        return reset_to_raw_init_state(env, raw_init_state, seed=seed)
    return env.reset(seed=seed)


def parse_fallback_seeds(spec: str | None) -> list[int | None]:
    if spec is None:
        return []
    seeds: list[int | None] = []
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if token.lower() in {"none", "null"}:
            seeds.append(None)
        else:
            seeds.append(int(token))
    return seeds


def slim_detection(detection: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in detection.items()
        if k
        not in {
            "contacts",
            "movement_after_by_frame",
            "same_sign_count_by_frame",
            "gate_debug_by_frame",
        }
    }


def geom_groups(env: LiberoEnv, spec: dict[str, Any]) -> tuple[set[int], set[int], list[int]]:
    model = env._env.sim.model
    gripper_geoms: set[int] = set()
    target_geoms: set[int] = set()
    joint_ids: list[int] = []

    body_substrings = [s.lower() for s in spec["target_body_substrings"]]
    joint_substrings = [s.lower() for s in spec["target_joint_substrings"]]

    for gid in range(model.ngeom):
        geom_name = (model.geom_id2name(gid) or "").lower()
        body_name = (model.body_id2name(model.geom_bodyid[gid]) or "").lower()
        if body_name.startswith("gripper0") or geom_name.startswith("gripper0"):
            gripper_geoms.add(gid)
        if any(token in body_name for token in body_substrings):
            target_geoms.add(gid)

    for jid in range(model.njnt):
        joint_name = (model.joint_id2name(jid) or "").lower()
        if any(token in joint_name for token in joint_substrings):
            joint_ids.append(jid)

    return gripper_geoms, target_geoms, joint_ids


def target_joint_qpos(env: LiberoEnv, joint_ids: list[int]) -> np.ndarray:
    model = env._env.sim.model
    data = env._env.sim.data
    values = []
    for jid in joint_ids:
        addr = int(model.jnt_qposadr[jid])
        values.append(float(data.qpos[addr]))
    return np.asarray(values, dtype=np.float64)


def contact_details(env: LiberoEnv, gripper_geoms: set[int], target_geoms: set[int]) -> list[dict[str, str]]:
    model = env._env.sim.model
    data = env._env.sim.data
    details: list[dict[str, str]] = []
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if not ((g1 in gripper_geoms and g2 in target_geoms) or (g2 in gripper_geoms and g1 in target_geoms)):
            continue
        details.append(
            {
                "geom1": model.geom_id2name(g1) or str(g1),
                "geom2": model.geom_id2name(g2) or str(g2),
                "body1": model.body_id2name(model.geom_bodyid[g1]) or "",
                "body2": model.body_id2name(model.geom_bodyid[g2]) or "",
            }
        )
    return details


def _detect_episode_pcf_once(
    *,
    root: Path,
    episode_meta: dict[str, Any],
    init_state_meta: dict[int, dict[str, Any]],
    task_language: str,
    keyword: str,
    seed: int | None,
    height: int,
    width: int,
    movement_window: int,
    movement_threshold: float,
    same_sign_k: int,
    case_category: str,
    verb: str,
    require_horizontal_motion: bool,
    reset_mode: str,
) -> dict[str, Any]:
    episode_index = int(episode_meta["episode_index"])
    suite = str(episode_meta["libero_suite"])
    task_id = int(episode_meta["libero_benchmark_task_id"])
    init_resolution = init_state_meta[episode_index]
    init_state_id = int(init_resolution["init_state_id"])
    rows = load_episode_rows(root, episode_index)
    spec = target_spec(task_language, keyword)

    env = make_env(suite, task_id, init_state_id, height, width)
    try:
        reset_env_from_metadata(env, init_resolution, seed, reset_mode=reset_mode)
        gripper_geoms, target_geoms, joint_ids = geom_groups(env, spec)
        if not gripper_geoms or not target_geoms or not joint_ids:
            return {
                "episode_index": episode_index,
                "detected": False,
                "reason": "missing_geom_or_joint_group",
                "num_gripper_geoms": len(gripper_geoms),
                "num_target_geoms": len(target_geoms),
                "num_target_joints": len(joint_ids),
                "reset_mode": reset_mode,
                "seed": seed,
            }

        contacts: list[bool] = []
        contact_detail_by_frame: list[list[dict[str, str]]] = []
        qpos_before: list[np.ndarray] = []

        for row in rows:
            qpos_before.append(target_joint_qpos(env, joint_ids))
            details = contact_details(env, gripper_geoms, target_geoms)
            contacts.append(bool(details))
            contact_detail_by_frame.append(details)
            action = np.asarray(row[ACTION], dtype=np.float32)
            env.step(action)

        qpos_after_final = target_joint_qpos(env, joint_ids)
        qpos_series = qpos_before + [qpos_after_final]

        pcf: int | None = None
        movement_after_by_frame: list[float] = []
        same_sign_count_by_frame: list[int] = []
        gate_debug_by_frame: list[dict[str, Any]] = []
        for frame_idx in range(len(rows)):
            current = qpos_series[frame_idx]
            end = min(len(qpos_series), frame_idx + movement_window + 1)
            if end <= frame_idx + 1:
                movement = 0.0
            else:
                future = np.stack(qpos_series[frame_idx + 1 : end], axis=0)
                movement = float(np.max(np.abs(future - current))) if future.size else 0.0
            q_values = np.asarray([q[0] for q in qpos_series], dtype=np.float64)
            joint_vel = np.diff(q_values)
            vel_window = joint_vel[frame_idx : min(len(joint_vel), frame_idx + movement_window)]
            signs = np.sign(vel_window[np.abs(vel_window) > 1e-9])
            same_sign_count = 0
            if signs.size:
                same_sign_count = max(int(np.sum(signs > 0)), int(np.sum(signs < 0)))

            action = np.asarray(rows[frame_idx][ACTION], dtype=np.float32)
            horizontal_motion_ok = True
            if require_horizontal_motion:
                horizontal_motion_ok = float(max(abs(action[0]), abs(action[1]))) > float(abs(action[2]))

            case1_release_ok = True
            if case_category == "case1" and verb in {"close", "open", "pull"}:
                case1_release_ok = float(action[6]) <= 0.0

            movement_ok = movement >= movement_threshold
            same_sign_ok = same_sign_count >= same_sign_k
            movement_after_by_frame.append(movement)
            same_sign_count_by_frame.append(int(same_sign_count))
            gate_debug_by_frame.append(
                {
                    "contact": bool(contacts[frame_idx]),
                    "movement_ok": bool(movement_ok),
                    "same_sign_ok": bool(same_sign_ok),
                    "case1_release_ok": bool(case1_release_ok),
                    "horizontal_motion_ok": bool(horizontal_motion_ok),
                    "gripper_action": float(action[6]),
                    "max_abs_xy_action": float(max(abs(action[0]), abs(action[1]))),
                    "z_action": float(action[2]),
                }
            )

            if (
                contacts[frame_idx]
                and movement_ok
                and same_sign_ok
                and case1_release_ok
                and horizontal_motion_ok
                and pcf is None
            ):
                pcf = int(frame_idx)

        if pcf is None:
            return {
                "episode_index": episode_index,
                "detected": False,
                "reason": "no_contact_followed_by_target_joint_motion",
                "num_contact_frames": int(sum(contacts)),
                "max_movement_after": float(max(movement_after_by_frame) if movement_after_by_frame else 0.0),
                "target_spec": spec,
                "case_category": case_category,
                "verb": verb,
                "reset_mode": reset_mode,
                "seed": seed,
            }

        start = pcf
        end = pcf
        for idx in range(pcf + 1, len(contacts)):
            if not contacts[idx]:
                break
            end = idx

        return {
            "episode_index": episode_index,
            "detected": True,
            "contact_onset_frame": int(pcf),
            "contact_interval": [int(start), int(end)],
            "method": "sim_contact_plus_target_joint_verification",
            "case_category": case_category,
            "verb": verb,
            "movement_window": int(movement_window),
            "movement_threshold": float(movement_threshold),
            "same_sign_k": int(same_sign_k),
            "require_horizontal_motion": bool(require_horizontal_motion),
            "reset_mode": reset_mode,
            "seed": seed,
            "num_contact_frames": int(sum(contacts)),
            "movement_after_onset": float(movement_after_by_frame[pcf]),
            "same_sign_count_onset": int(same_sign_count_by_frame[pcf]),
            "gate_debug_onset": gate_debug_by_frame[pcf],
            "target_spec": spec,
            "first_contact_details": contact_detail_by_frame[pcf],
            "contacts": contacts,
            "movement_after_by_frame": movement_after_by_frame,
            "same_sign_count_by_frame": same_sign_count_by_frame,
            "gate_debug_by_frame": gate_debug_by_frame,
        }
    finally:
        env.close()


def detect_episode_pcf(
    *,
    root: Path,
    episode_meta: dict[str, Any],
    init_state_meta: dict[int, dict[str, Any]],
    task_language: str,
    keyword: str,
    seed: int | None,
    fallback_seeds: list[int | None] | None,
    height: int,
    width: int,
    movement_window: int,
    movement_threshold: float,
    same_sign_k: int,
    case_category: str,
    verb: str,
    require_horizontal_motion: bool,
    init_reset_mode: str,
) -> dict[str, Any]:
    if init_reset_mode == "auto":
        modes = ["raw-hdf5", "libero-init-state"]
    elif init_reset_mode in {"raw-hdf5", "libero-init-state"}:
        modes = [init_reset_mode]
    else:
        raise ValueError(f"Unsupported init_reset_mode: {init_reset_mode}")

    fallback_seeds = fallback_seeds or []
    attempts: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    last_detection: dict[str, Any] | None = None
    for reset_mode in modes:
        mode_seeds: list[int | None] = [seed]
        if reset_mode == "libero-init-state":
            mode_seeds.extend(fallback_seeds)
        for candidate_seed in mode_seeds:
            key = (reset_mode, candidate_seed)
            if key in seen:
                continue
            seen.add(key)
            detection = _detect_episode_pcf_once(
                root=root,
                episode_meta=episode_meta,
                init_state_meta=init_state_meta,
                task_language=task_language,
                keyword=keyword,
                seed=candidate_seed,
                height=height,
                width=width,
                movement_window=movement_window,
                movement_threshold=movement_threshold,
                same_sign_k=same_sign_k,
                case_category=case_category,
                verb=verb,
                require_horizontal_motion=require_horizontal_motion,
                reset_mode=reset_mode,
            )
            attempts.append(slim_detection(detection))
            last_detection = detection
            if detection.get("detected"):
                detection["reset_attempts"] = attempts
                return detection

    if last_detection is None:
        raise RuntimeError("No PCF detection attempts were run")
    last_detection["reset_attempts"] = attempts
    return last_detection


def draw_overlay(
    image_np: np.ndarray,
    *,
    task_language: str,
    combo: tuple[str, str],
    episode_index: int,
    frame_index: int,
    total_frames: int,
    pcf: int | None,
    contact: bool,
    movement_after: float,
    side_row: dict[str, Any],
) -> np.ndarray:
    image = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    duration = int(side_row.get("duration_label", -1))
    cmd_switch = int(side_row.get("cmd_switch", 0))
    qpos_switch = int(side_row.get("qpos_switch", 0))
    dist_to_switch = int(side_row.get("dist_to_switch", -1))
    duration_s = int(side_row.get("duration_label_s", duration))
    duration_p = int(side_row.get("duration_label_p", 8))
    pcf_frame = int(side_row.get("pcf_frame", -1))
    pcf_text = "None" if pcf_frame < 0 else str(pcf_frame)
    dist_to_pure_contact = int(side_row.get("dist_to_pure_contact", -1))

    lines = [
        f"TK='{task_language}'",
        f"D={duration} EPS={episode_index} FR={frame_index}/{max(total_frames - 1, 0)}",
        f"Ds={duration_s} DTS={dist_to_switch} CMS={cmd_switch} QPS={qpos_switch}",
        f"Dp={duration_p} PCF={pcf_text} DTPC={dist_to_pure_contact}",
    ]

    margin = 8
    line_h = 14
    box_h = margin * 2 + line_h * len(lines)
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0))
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    return np.asarray(image)


def write_replay_video(
    *,
    root: Path,
    output_dir: Path,
    episode_meta: dict[str, Any],
    init_state_meta: dict[int, dict[str, Any]],
    task_language: str,
    combo: tuple[str, str],
    detection: dict[str, Any],
    sidecar_by_dataset_index: dict[int, dict[str, Any]],
    seed: int | None,
    height: int,
    width: int,
    fps: int,
) -> Path:
    episode_index = int(episode_meta["episode_index"])
    suite = str(episode_meta["libero_suite"])
    task_id = int(episode_meta["libero_benchmark_task_id"])
    init_resolution = init_state_meta[episode_index]
    init_state_id = int(init_resolution["init_state_id"])
    rows = load_episode_rows(root, episode_index)
    contacts = detection.get("contacts", [False] * len(rows))
    movement = detection.get("movement_after_by_frame", [0.0] * len(rows))
    pcf = detection.get("contact_onset_frame")

    env = make_env(suite, task_id, init_state_id, height, width)
    safe_language = re.sub(r"[^a-zA-Z0-9]+", "_", task_language).strip("_")[:80]
    video_path = output_dir / f"{combo[0]}_{combo[1]}_episode_{episode_index:04d}_{safe_language}_pcf.mp4"
    try:
        reset_mode = str(detection.get("reset_mode", "raw-hdf5"))
        reset_seed = detection.get("seed", seed)
        obs, _ = reset_env_from_metadata(env, init_resolution, reset_seed, reset_mode=reset_mode)
        with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
            for idx, row in enumerate(rows):
                raw_img = obs["pixels"][DEFAULT_CAMERA]
                viz_img = np.asarray(raw_img)[::-1, ::-1].copy()
                side_row = sidecar_by_dataset_index.get(int(row["index"]), {})
                viz_img = draw_overlay(
                    viz_img,
                    task_language=task_language,
                    combo=combo,
                    episode_index=episode_index,
                    frame_index=idx,
                    total_frames=len(rows),
                    pcf=pcf,
                    contact=bool(contacts[idx]) if idx < len(contacts) else False,
                    movement_after=float(movement[idx]) if idx < len(movement) else 0.0,
                    side_row=side_row,
                )
                writer.append_data(viz_img)
                action = np.asarray(row[ACTION], dtype=np.float32)
                obs, _, _, _, _ = env.step(action)
    finally:
        env.close()
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save representative pure-contact PCF replays from saved PCF metadata or fresh MuJoCo detection."
    )
    parser.add_argument("--dataset.root", dest="root", type=Path, required=True)
    parser.add_argument("--dataset.repo-id", dest="repo_id", type=str, default="local/libero_lerobot_v3_lerobotkeys")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--movement-window", type=int, default=15)
    parser.add_argument("--movement-threshold", type=float, default=0.03)
    parser.add_argument("--same-sign-k", type=int, default=5)
    parser.add_argument("--require-horizontal-motion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--init-reset-mode",
        choices=("auto", "raw-hdf5", "libero-init-state"),
        default="auto",
        help=(
            "How to initialize LIBERO before replaying expert actions for PCF detection. "
            "auto tries raw HDF5 first, then falls back to the LIBERO init-state id."
        ),
    )
    parser.add_argument(
        "--fallback-seeds",
        type=str,
        default="2,3,4,5,10,42,100",
        help=(
            "Comma-separated seeds tried only for the libero-init-state fallback when --init-reset-mode=auto. "
            "Use 'none' for an unseeded reset."
        ),
    )
    parser.add_argument("--max-episodes-per-combo", type=int, default=20)
    parser.add_argument("--videos-per-combo", type=int, default=1)
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Detect or load PCF metadata and write summaries/metadata without rendering replay videos.",
    )
    parser.add_argument("--episode-indices", type=str, default=None)
    parser.add_argument(
        "--pcf-source",
        choices=("saved", "detect"),
        default="saved",
        help=(
            "saved: use dataset.root/meta/purecontact_pcf.json or --purecontact-pcf-json. "
            "detect: replay candidates and recompute PCF from MuJoCo contacts."
        ),
    )
    parser.add_argument(
        "--purecontact-pcf-json",
        type=Path,
        default=None,
        help="Saved PCF metadata JSON. Defaults to dataset.root/meta/purecontact_pcf.json.",
    )
    parser.add_argument(
        "--write-purecontact-pcf-json",
        type=Path,
        default=None,
        help="Only useful with --pcf-source detect: write recalculated PCF metadata to this JSON.",
    )
    args = parser.parse_args()
    if args.write_purecontact_pcf_json is not None and args.pcf_source != "detect":
        raise ValueError("--write-purecontact-pcf-json is only allowed with --pcf-source detect")

    root = args.root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_seeds = parse_fallback_seeds(args.fallback_seeds)

    task_map = load_task_map(root)
    episode_rows = load_episode_metadata(root)
    init_state_meta = load_episode_init_state_metadata(root, "libero_init_state_id")
    pcf_json = args.purecontact_pcf_json or default_purecontact_json(root)

    candidates_by_combo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    explicit_episode_indices = None
    if args.episode_indices:
        explicit_episode_indices = {
            int(token.strip())
            for token in args.episode_indices.split(",")
            if token.strip()
        }

    reviewed_tasks: list[dict[str, Any]] = []
    episode_purecontact_labels: list[dict[str, Any]] = []
    for task_index, language in sorted(task_map.items()):
        combos = combo_for_language(language)
        combo_records = [
            {
                "keyword": keyword,
                "verb": verb,
                "case_category": purecontact_case(language, keyword, verb),
            }
            for keyword, verb in combos
        ]
        reviewed_tasks.append({"task_index": int(task_index), "language": language, "purecontact_combos": combo_records})
        for combo in combos:
            eps = [
                row
                for row in episode_rows
                if int(row["task_index"]) == int(task_index) and int(row["episode_index"]) in init_state_meta
            ]
            if explicit_episode_indices is not None:
                eps = [row for row in eps if int(row["episode_index"]) in explicit_episode_indices]
            candidates_by_combo.setdefault(combo, []).extend(sorted(eps, key=lambda r: int(r["episode_index"])))

    for row in sorted(episode_rows, key=lambda item: int(item["episode_index"])):
        language = task_map[int(row["task_index"])]
        combos = combo_for_language(language)
        combo_records = [
            {
                "keyword": keyword,
                "verb": verb,
                "case_category": purecontact_case(language, keyword, verb),
            }
            for keyword, verb in combos
        ]
        episode_purecontact_labels.append(
            {
                "episode_index": int(row["episode_index"]),
                "libero_suite": row.get("libero_suite"),
                "libero_benchmark_task_id": (
                    None if row.get("libero_benchmark_task_id") is None else int(row["libero_benchmark_task_id"])
                ),
                "task_index": int(row["task_index"]),
                "libero_task_language": language,
                "has_purecontact_language": bool(combo_records),
                "gripper_target_contact_without_openclose_candidate": bool(combo_records),
                "purecontact_combos": combo_records,
            }
        )

    detections: list[dict[str, Any]] = []
    selected_episode_indices: list[int] = []
    if args.pcf_source == "saved":
        saved_annotations = load_purecontact_pcf_annotations(pcf_json)
        if not saved_annotations:
            raise FileNotFoundError(
                f"No saved PCF annotations found at {pcf_json}. "
                "Run with --pcf-source detect --write-purecontact-pcf-json to create/update it."
            )
        for combo, candidates in sorted(candidates_by_combo.items()):
            selected_items: list[dict[str, Any]] = []
            for episode_meta in candidates:
                if len(selected_items) >= int(args.videos_per_combo):
                    break
                episode_index = int(episode_meta["episode_index"])
                ann = saved_annotations.get(episode_index)
                if not ann:
                    continue
                if ann.get("purecontact_keyword") != combo[0] or ann.get("purecontact_verb") != combo[1]:
                    continue
                task_language = task_map[int(episode_meta["task_index"])]
                case_category = ann.get("case_category") or purecontact_case(task_language, combo[0], combo[1])
                detection = {
                    "detected": True,
                    "episode_index": episode_index,
                    "contact_onset_frame": int(ann["pcf"]),
                    "contact_interval": ann.get("contact_interval"),
                    "method": "saved_purecontact_pcf_json",
                    "annotation_method": ann.get("method", "sim_contact_plus_target_joint_verification"),
                    "source_purecontact_pcf_json": str(pcf_json),
                    "reset_mode": ann.get("reset_mode", "raw-hdf5"),
                    "seed": ann.get("seed", args.seed),
                }
                selected_items.append(
                    {
                        "combo": {"keyword": combo[0], "verb": combo[1]},
                        "case_category": case_category,
                        "episode_meta": episode_meta,
                        "task_language": task_language,
                        "detection": detection,
                        "attempts": [],
                    }
                )
                selected_episode_indices.append(episode_index)
            if selected_items:
                detections.extend(selected_items)
            else:
                detections.append(
                    {
                        "combo": {"keyword": combo[0], "verb": combo[1]},
                        "case_category": None,
                        "selected": None,
                        "attempts": [],
                    }
                )
    else:
        for combo, candidates in sorted(candidates_by_combo.items()):
            selected_items: list[dict[str, Any]] = []
            attempts: list[dict[str, Any]] = []
            for episode_meta in candidates[: int(args.max_episodes_per_combo)]:
                task_language = task_map[int(episode_meta["task_index"])]
                case_category = purecontact_case(task_language, combo[0], combo[1])
                detection = detect_episode_pcf(
                    root=root,
                    episode_meta=episode_meta,
                    init_state_meta=init_state_meta,
                    task_language=task_language,
                    keyword=combo[0],
                    verb=combo[1],
                    case_category=case_category,
                    seed=args.seed,
                    height=args.observation_height,
                    width=args.observation_width,
                    movement_window=args.movement_window,
                    movement_threshold=args.movement_threshold,
                    same_sign_k=args.same_sign_k,
                    require_horizontal_motion=args.require_horizontal_motion,
                    init_reset_mode=args.init_reset_mode,
                    fallback_seeds=fallback_seeds,
                )
                attempts.append(
                    {
                        k: v
                        for k, v in detection.items()
                        if k
                        not in {
                            "contacts",
                            "movement_after_by_frame",
                            "same_sign_count_by_frame",
                            "gate_debug_by_frame",
                        }
                    }
                )
                if detection.get("detected"):
                    selected_items.append(
                        {
                            "combo": {"keyword": combo[0], "verb": combo[1]},
                            "case_category": case_category,
                            "episode_meta": episode_meta,
                            "task_language": task_language,
                            "detection": detection,
                            "attempts": attempts,
                        }
                    )
                    selected_episode_indices.append(int(episode_meta["episode_index"]))
                    if len(selected_items) >= int(args.videos_per_combo):
                        break
            if not selected_items:
                detections.append(
                    {
                        "combo": {"keyword": combo[0], "verb": combo[1]},
                        "case_category": None,
                        "selected": None,
                        "attempts": attempts,
                    }
                )
            else:
                detections.extend(selected_items)

    selected_episode_indices = sorted(set(selected_episode_indices))
    purecontact_annotations = {
        int(item["episode_meta"]["episode_index"]): {
            "pcf": int(item["detection"]["contact_onset_frame"]),
            "contact_interval": item["detection"]["contact_interval"],
            "case_category": item["case_category"],
            "purecontact_keyword": item["combo"]["keyword"],
            "purecontact_verb": item["combo"]["verb"],
            "method": item["detection"]["method"],
            "reset_mode": item["detection"].get("reset_mode"),
            "seed": item["detection"].get("seed"),
            "movement_window": item["detection"].get("movement_window"),
            "movement_threshold": item["detection"].get("movement_threshold"),
            "same_sign_k": item["detection"].get("same_sign_k"),
            "require_horizontal_motion": item["detection"].get("require_horizontal_motion"),
            "movement_after_onset": item["detection"].get("movement_after_onset"),
            "same_sign_count_onset": item["detection"].get("same_sign_count_onset"),
        }
        for item in detections
        if item.get("detection", {}).get("detected")
    }
    sidecar_rows, sidecar_summary = build_sidecar_rows(
        root,
        episode_indices=selected_episode_indices,
        W1=15,
        W3=15,
        merge_window=3,
        match_window=10,
        labeler_version="hiva_duration_v9_pcf_overlay",
        purecontact_annotations=purecontact_annotations,
    )
    sidecar_by_dataset_index = {int(row["dataset_index"]): row for row in sidecar_rows}

    videos: list[dict[str, Any]] = []
    for item in detections:
        if not item.get("detection", {}).get("detected"):
            continue
        combo = (item["combo"]["keyword"], item["combo"]["verb"])
        video_path = None
        if not args.skip_videos:
            video_path = write_replay_video(
                root=root,
                output_dir=output_dir,
                episode_meta=item["episode_meta"],
                init_state_meta=init_state_meta,
                task_language=item["task_language"],
                combo=combo,
                detection=item["detection"],
                sidecar_by_dataset_index=sidecar_by_dataset_index,
                seed=args.seed,
                height=args.observation_height,
                width=args.observation_width,
                fps=args.video_fps,
            )
        videos.append(
            {
                "combo": item["combo"],
                "case_category": item["case_category"],
                "episode_index": int(item["episode_meta"]["episode_index"]),
                "task_language": item["task_language"],
                "pcf": int(item["detection"]["contact_onset_frame"]),
                "contact_interval": item["detection"]["contact_interval"],
                "video_path": None if video_path is None else str(video_path),
            }
        )
        item["video_path"] = None if video_path is None else str(video_path)

    serializable_detections = []
    for item in detections:
        serial = dict(item)
        if "detection" in serial:
            serial["detection"] = {
                k: v
                for k, v in serial["detection"].items()
                if k
                not in {
                    "contacts",
                    "movement_after_by_frame",
                    "same_sign_count_by_frame",
                    "gate_debug_by_frame",
                }
            }
        serializable_detections.append(serial)

    summary = {
        "dataset_root": str(root),
        "purecontact_keywords": list(PURECONTACT_KEYWORDS),
        "purecontact_verbs": list(PURECONTACT_VERBS),
        "reviewed_tasks": reviewed_tasks,
        "episode_purecontact_labels": episode_purecontact_labels,
        "selected_episode_indices": selected_episode_indices,
        "pcf_source": args.pcf_source,
        "purecontact_pcf_json": str(pcf_json) if args.pcf_source == "saved" else None,
        "movement_window": int(args.movement_window),
        "movement_threshold": float(args.movement_threshold),
        "same_sign_k": int(args.same_sign_k),
        "require_horizontal_motion": bool(args.require_horizontal_motion),
        "init_reset_mode": args.init_reset_mode,
        "fallback_seeds": fallback_seeds,
        "skip_videos": bool(args.skip_videos),
        "method": (
            "saved_purecontact_pcf_json_plus_expert_action_replay"
            if args.pcf_source == "saved"
            else "language_filter_then_sim_contact_plus_target_joint_verification"
        ),
        "detections": serializable_detections,
        "videos": videos,
        "sidecar_summary": sidecar_summary,
    }
    summary_path = output_dir / "purecontact_pcf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.write_purecontact_pcf_json is not None:
        write_purecontact_metadata(
            args.write_purecontact_pcf_json,
            source_summary=summary_path,
            episode_purecontact_labels=episode_purecontact_labels,
            annotations=purecontact_annotations,
        )
    print(json.dumps({"summary_path": str(summary_path), "videos": videos}, indent=2))


if __name__ == "__main__":
    main()
