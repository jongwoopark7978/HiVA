#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import os
import threading
import time
import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)


def _draw_duration_eval_overlay(
    image_np: np.ndarray,
    *,
    duration: int | None,
    inference_call: int,
    mean_duration: float | None = None,
    total_time_s: float | None = None,
    success: bool | None = None,
    task_prompt: str | None = None,
    frame_index: int | None = None,
    total_frames: int | None = None,
) -> np.ndarray:
    image = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    duration_text = "-" if duration is None else str(duration)
    mean_duration_text = "-" if mean_duration is None else f"{mean_duration:.1f}"
    total_time_text = "-" if total_time_s is None else f"{total_time_s:.1f}s"
    success_text = "SC=?" if success is None else ("SC" if success else "FA")
    prompt_lines = textwrap.wrap(task_prompt, width=42) if task_prompt else []
    frame_text = None
    if frame_index is not None and total_frames is not None:
        frame_text = f"FR={int(frame_index)}/{max(int(total_frames) - 1, 0)}"
    lines = [
        *prompt_lines,
        f"D={duration_text} INF={inference_call}" + (f" {frame_text}" if frame_text else ""),
        f"MD={mean_duration_text} TT={total_time_text}",
        success_text,
    ]

    margin = 8
    line_h = 14
    box_h = margin * 2 + line_h * len(lines)
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0, 180))
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h

    return np.asarray(image)


def _policy_duration_overlay_info(policy: PreTrainedPolicy, env_ix: int | None = None) -> dict[str, Any]:
    if env_ix is not None:
        histories = getattr(policy, "_execution_horizon_histories", None)
        counts = getattr(policy, "_duration_inference_counts_by_env", None)
        sums = getattr(policy, "_execution_horizon_sums_by_env", None)
        last_horizons = getattr(policy, "_last_execution_horizons", None)
        latency_histories = getattr(policy, "_model_forward_latency_histories_s", None)
        latency_sums = getattr(policy, "_model_forward_latency_sums_by_env", None)
        if histories is not None and 0 <= env_ix < len(histories):
            duration_sequence = [int(value) for value in histories[env_ix]]
            inference_call = int(counts[env_ix]) if counts is not None else len(duration_sequence)
            horizon_sum = float(sums[env_ix]) if sums is not None else float(sum(duration_sequence))
            latency_sequence_s = (
                [float(value) for value in latency_histories[env_ix]]
                if latency_histories is not None and 0 <= env_ix < len(latency_histories)
                else []
            )
            latency_sum_s = (
                float(latency_sums[env_ix])
                if latency_sums is not None and 0 <= env_ix < len(latency_sums)
                else float(sum(latency_sequence_s))
            )
            duration = (
                int(last_horizons[env_ix])
                if last_horizons is not None and last_horizons[env_ix] is not None
                else None
            )
            mean_duration = horizon_sum / inference_call if inference_call > 0 else None
            mean_model_forward_latency_s = latency_sum_s / inference_call if inference_call > 0 else None
            return {
                "duration": duration,
                "duration_sequence": duration_sequence,
                "inference_call": inference_call,
                "mean_duration": mean_duration,
                "model_forward_latency_s": latency_sum_s,
                "model_forward_latency_sequence_s": latency_sequence_s,
                "mean_model_forward_latency_s": mean_model_forward_latency_s,
            }

    duration = getattr(policy, "_last_execution_horizon", None)
    inference_call = int(getattr(policy, "_duration_inference_count", 0))
    horizon_sum = float(getattr(policy, "_execution_horizon_sum", 0))
    duration_sequence = [int(value) for value in getattr(policy, "_execution_horizon_history", [])]
    latency_sum_s = float(getattr(policy, "_model_forward_latency_sum_s", 0.0))
    latency_sequence_s = [
        float(value) for value in getattr(policy, "_model_forward_latency_history_s", [])
    ]
    mean_duration = horizon_sum / inference_call if inference_call > 0 else None
    mean_model_forward_latency_s = latency_sum_s / inference_call if inference_call > 0 else None
    return {
        "duration": duration,
        "duration_sequence": duration_sequence,
        "inference_call": inference_call,
        "mean_duration": mean_duration,
        "model_forward_latency_s": latency_sum_s,
        "model_forward_latency_sequence_s": latency_sequence_s,
        "mean_model_forward_latency_s": mean_model_forward_latency_s,
    }


def _draw_policy_duration_overlay(
    image_np: np.ndarray,
    overlay_info: dict[str, int | float | None],
    *,
    total_time_s: float | None = None,
    success: bool | None = None,
    task_prompt: str | None = None,
    frame_index: int | None = None,
    total_frames: int | None = None,
) -> np.ndarray:
    return _draw_duration_eval_overlay(
        image_np,
        duration=overlay_info.get("duration"),
        inference_call=int(overlay_info.get("inference_call") or 0),
        mean_duration=overlay_info.get("mean_duration"),
        total_time_s=total_time_s,
        success=success,
        task_prompt=task_prompt,
        frame_index=frame_index,
        total_frames=total_frames,
    )


def _json_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _mean_optional_numbers(values: list[int | float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return float(np.nanmean(numeric_values))


ACTION_JITTER_GROUPS: dict[str, tuple[int, ...]] = {
    "translation": (0, 1, 2),
    "rotation": (3, 4, 5),
    "gripper": (6,),
    "translation_rotation": (0, 1, 2, 3, 4, 5),
    "all_action_dims": (0, 1, 2, 3, 4, 5, 6),
}


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _percentile_or_none(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64), q))


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values.astype(np.float64)))


def _max_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values.astype(np.float64)))


def _duration_sequence_to_boundaries(
    duration_sequence: list[int] | None,
    action_len: int,
    fallback_horizon: int | None = None,
) -> list[int]:
    """Return zero-based action indices where a new model-call chunk starts."""
    boundaries: list[int] = []
    if duration_sequence:
        cursor = 0
        for duration in duration_sequence[:-1]:
            try:
                step = int(duration)
            except (TypeError, ValueError):
                continue
            if step <= 0:
                continue
            cursor += step
            if 0 < cursor < action_len:
                boundaries.append(cursor)
        return boundaries

    if fallback_horizon is not None and int(fallback_horizon) > 0:
        horizon = int(fallback_horizon)
        return list(range(horizon, action_len, horizon))
    return []


def _boundary_values(
    series: np.ndarray,
    boundaries: list[int],
    *,
    kind: str,
    window: int = 1,
) -> np.ndarray:
    """Collect velocity or acceleration magnitudes around model-call boundaries."""
    if series.size == 0 or not boundaries:
        return np.asarray([], dtype=np.float64)
    values = []
    for boundary in boundaries:
        for action_t in range(boundary - window, boundary + window + 1):
            if kind == "velocity":
                candidate_indices = (action_t - 1, action_t)
            elif kind == "acceleration":
                candidate_indices = (action_t - 1,)
            else:
                raise ValueError(f"Unknown boundary series kind: {kind}")
            for idx in candidate_indices:
                if 0 <= idx < len(series):
                    values.append(float(series[idx]))
    return np.asarray(values, dtype=np.float64)


def _nonboundary_acc_values(acc_norm: np.ndarray, boundaries: list[int], window: int = 1) -> np.ndarray:
    if acc_norm.size == 0:
        return np.asarray([], dtype=np.float64)
    boundary_acc_indices = set()
    for boundary in boundaries:
        for action_t in range(boundary - window, boundary + window + 1):
            idx = action_t - 1
            if 0 <= idx < len(acc_norm):
                boundary_acc_indices.add(idx)
    values = [float(value) for idx, value in enumerate(acc_norm) if idx not in boundary_acc_indices]
    return np.asarray(values, dtype=np.float64)


def _velocity_zero_crossing_rate(vel: np.ndarray, eps: float = 1e-8) -> tuple[float | None, float | None]:
    """Average sign-change rate over action dimensions."""
    if vel.shape[0] < 2:
        return None, None
    dim_rates = []
    for dim in range(vel.shape[1]):
        v0 = vel[:-1, dim]
        v1 = vel[1:, dim]
        valid = (np.abs(v0) > eps) & (np.abs(v1) > eps)
        if not np.any(valid):
            dim_rates.append(0.0)
            continue
        sign_changes = np.sign(v0[valid]) * np.sign(v1[valid]) < 0
        dim_rates.append(float(np.mean(sign_changes)))
    rates = np.asarray(dim_rates, dtype=np.float64)
    return float(np.mean(rates)), float(np.max(rates))


def _action_jitter_metrics_for_group(
    actions: np.ndarray,
    dims: tuple[int, ...],
    boundaries: list[int],
) -> dict[str, float | int | None]:
    valid_dims = [dim for dim in dims if dim < actions.shape[1]]
    if len(valid_dims) == 0:
        return {
            "n_dims": 0,
            "vel_abs_mean": None,
            "vel_abs_p95": None,
            "vel_abs_max": None,
            "acc_abs_mean": None,
            "acc_abs_p95": None,
            "acc_abs_max": None,
            "boundary_vel_abs_mean": None,
            "boundary_vel_abs_p95": None,
            "boundary_vel_abs_max": None,
            "boundary_acc_abs_mean": None,
            "boundary_acc_abs_p95": None,
            "boundary_acc_abs_max": None,
            "nonboundary_acc_abs_p95": None,
            "boundary_count": int(len(boundaries)),
            "boundary_acc_sample_count": 0,
            "vel_zcr_mean": None,
            "vel_zcr_max_dim": None,
            "jerk_abs_p95": None,
        }

    x = actions[:, valid_dims].astype(np.float64)
    vel = np.diff(x, axis=0)
    acc = np.diff(x, n=2, axis=0)
    jerk = np.diff(x, n=3, axis=0)

    vel_norm = np.linalg.norm(vel, axis=-1) if vel.size else np.asarray([], dtype=np.float64)
    acc_norm = np.linalg.norm(acc, axis=-1) if acc.size else np.asarray([], dtype=np.float64)
    jerk_norm = np.linalg.norm(jerk, axis=-1) if jerk.size else np.asarray([], dtype=np.float64)
    boundary_vel = _boundary_values(vel_norm, boundaries, kind="velocity", window=1)
    boundary_acc = _boundary_values(acc_norm, boundaries, kind="acceleration", window=1)
    nonboundary_acc = _nonboundary_acc_values(acc_norm, boundaries, window=1)
    zcr_mean, zcr_max = _velocity_zero_crossing_rate(vel)

    return {
        "n_dims": len(valid_dims),
        "vel_abs_mean": _mean_or_none(vel_norm),
        "vel_abs_p95": _percentile_or_none(vel_norm, 95),
        "vel_abs_max": _max_or_none(vel_norm),
        "acc_abs_mean": _mean_or_none(acc_norm),
        "acc_abs_p95": _percentile_or_none(acc_norm, 95),
        "acc_abs_max": _max_or_none(acc_norm),
        "boundary_vel_abs_mean": _mean_or_none(boundary_vel),
        "boundary_vel_abs_p95": _percentile_or_none(boundary_vel, 95),
        "boundary_vel_abs_max": _max_or_none(boundary_vel),
        "boundary_acc_abs_mean": _mean_or_none(boundary_acc),
        "boundary_acc_abs_p95": _percentile_or_none(boundary_acc, 95),
        "boundary_acc_abs_max": _max_or_none(boundary_acc),
        "nonboundary_acc_abs_p95": _percentile_or_none(nonboundary_acc, 95),
        "boundary_count": int(len(boundaries)),
        "boundary_acc_sample_count": int(boundary_acc.size),
        "vel_zcr_mean": zcr_mean,
        "vel_zcr_max_dim": zcr_max,
        "jerk_abs_p95": _percentile_or_none(jerk_norm, 95),
    }


def _compute_action_jitter_metrics(
    actions: np.ndarray,
    *,
    duration_sequence: list[int] | None = None,
    fallback_horizon: int | None = None,
) -> dict[str, Any]:
    """Compute executed-action smoothness metrics for one rollout episode."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected executed actions with shape [T, D], got {actions.shape}.")
    boundaries = _duration_sequence_to_boundaries(duration_sequence, len(actions), fallback_horizon)
    metrics = {
        "n_actions": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "boundaries": [int(value) for value in boundaries],
        "boundary_count": int(len(boundaries)),
        "groups": {},
    }
    for group_name, dims in ACTION_JITTER_GROUPS.items():
        metrics["groups"][group_name] = _action_jitter_metrics_for_group(actions, dims, boundaries)
    return metrics


def _normalize_actions_for_jitter(
    actions: np.ndarray,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map executed actions back to policy-normalized space when stats are available."""
    actions = np.asarray(actions, dtype=np.float64)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    for step in getattr(postprocessor, "steps", []):
        normalize_action = getattr(step, "_normalize_action", None)
        if normalize_action is None:
            continue
        try:
            normalized = normalize_action(action_tensor, inverse=False)
        except Exception as exc:  # noqa: BLE001 - fallback should not break evaluation.
            logging.warning("Could not compute normalized action jitter metrics: %s", exc)
            break
        return normalized.detach().cpu().numpy(), {
            "quantitative_action_space": "normalized_action",
            "normalization_available": True,
            "normalization_source": f"{type(step).__name__}._normalize_action",
        }
    return actions, {
        "quantitative_action_space": "raw_env_action_fallback",
        "normalization_available": False,
        "normalization_source": None,
    }


def _mean_action_jitter_metrics(action_jitter_metrics: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    grouped_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    top_level_values: dict[str, list[float]] = defaultdict(list)

    for metrics in action_jitter_metrics:
        if not isinstance(metrics, dict):
            continue
        for key in ("n_actions", "action_dim", "boundary_count"):
            value = _finite_float_or_none(metrics.get(key))
            if value is not None:
                top_level_values[key].append(value)
        for group_name, group_metrics in (metrics.get("groups") or {}).items():
            if not isinstance(group_metrics, dict):
                continue
            for metric_name, value in group_metrics.items():
                numeric = _finite_float_or_none(value)
                if numeric is not None:
                    grouped_values[str(group_name)][str(metric_name)].append(numeric)

    if not grouped_values and not top_level_values:
        return None

    return {
        "n_episodes": int(sum(1 for metrics in action_jitter_metrics if isinstance(metrics, dict))),
        "mean_n_actions": _mean_optional_numbers(top_level_values.get("n_actions", [])),
        "mean_action_dim": _mean_optional_numbers(top_level_values.get("action_dim", [])),
        "mean_boundary_count": _mean_optional_numbers(top_level_values.get("boundary_count", [])),
        "groups": {
            group_name: {
                metric_name: _mean_optional_numbers(values)
                for metric_name, values in sorted(metric_values.items())
            }
            for group_name, metric_values in sorted(grouped_values.items())
        },
    }


def _mean_action_jitter_metrics_from_display_metrics(
    display_metrics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return _mean_action_jitter_metrics(
        [metrics.get("action_jitter_metrics") for metrics in display_metrics if isinstance(metrics, dict)]
    )


def _last_optional_int(values: Any) -> int | None:
    if not values:
        return None
    return int(values[-1])


def _task_prompt_from_env(env: gym.vector.VectorEnv) -> str | None:
    try:
        result = env.call("task_description")
    except Exception as exc:
        logging.warning("Could not read task_description from env: %s", exc)
        return None
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        return first if isinstance(first, str) else str(first)
    return None


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)
    episode_start_times = np.array([time.perf_counter()] * env.num_envs)
    episode_total_times: list[float | None] = [None] * env.num_envs

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        next_done = terminated | truncated | done
        if step + 1 == max_steps:
            next_done = np.ones_like(next_done, dtype=bool)
        episode_done_time = time.perf_counter()
        newly_done = next_done & ~done
        for env_ix, is_new_done in enumerate(newly_done):
            if is_new_done and episode_total_times[env_ix] is None:
                episode_total_times[env_ix] = episode_done_time - episode_start_times[env_ix]
        done = next_done

        if render_callback is not None:
            render_callback(env)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        "episode_total_time_s": torch.tensor(
            [t if t is not None else time.perf_counter() - episode_start_times[i] for i, t in enumerate(episode_total_times)],
            dtype=torch.float32,
        ),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()
    task_prompt = _task_prompt_from_env(env)

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos
    episode_display_metrics: list[dict[str, Any]] = []

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            frames = [env.envs[i].render() for i in range(n_to_render_now)]  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            frames = env.call("render")[:n_to_render_now]
        else:
            raise ValueError(f"Unsupported vector env type for rendering: {type(env)}.")
        frames = frames[:n_to_render_now]
        ep_frames.append(np.stack(frames))
        ep_overlay_infos.append(
            [_policy_duration_overlay_info(policy, env_ix=env_ix).copy() for env_ix in range(len(frames))]
        )

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []
            ep_overlay_infos: list[list[dict[str, Any]]] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        n_metrics_to_take = min(env.num_envs, n_episodes - len(episode_display_metrics))
        for env_ix in range(n_metrics_to_take):
            done_index = int(done_indices[env_ix].item())
            final_overlay_info = None
            if max_episodes_rendered > 0 and len(ep_overlay_infos) > 0:
                final_step_ix = min(done_index, len(ep_overlay_infos) - 1)
                if env_ix < len(ep_overlay_infos[final_step_ix]):
                    final_overlay_info = ep_overlay_infos[final_step_ix][env_ix]
            if final_overlay_info is None:
                final_overlay_info = _policy_duration_overlay_info(policy, env_ix=env_ix)

            total_time_s = float(rollout_data["episode_total_time_s"][env_ix].item())
            duration_sequence = [
                int(duration) for duration in final_overlay_info.get("duration_sequence", [])
            ]
            action_len = min(done_index + 1, rollout_data[ACTION].shape[1])
            episode_actions = rollout_data[ACTION][env_ix, :action_len].detach().cpu().numpy()
            fallback_horizon = int(getattr(policy.config, "n_action_steps", 0) or 0)
            raw_action_jitter_metrics = _compute_action_jitter_metrics(
                episode_actions,
                duration_sequence=duration_sequence,
                fallback_horizon=fallback_horizon,
            )
            metric_episode_actions, action_norm_metadata = _normalize_actions_for_jitter(
                episode_actions, postprocessor
            )
            action_jitter_metrics = _compute_action_jitter_metrics(
                metric_episode_actions,
                duration_sequence=duration_sequence,
                fallback_horizon=fallback_horizon,
            )
            action_jitter_metrics.update(action_norm_metadata)
            raw_action_jitter_metrics["quantitative_action_space"] = "raw_env_action"
            display_record = {
                "duration": duration_sequence,
                "inference_calls": int(final_overlay_info.get("inference_call") or 0),
                "mean_duration": _json_optional_float(final_overlay_info.get("mean_duration")),
                "model_forward_latency_s": _json_optional_float(
                    final_overlay_info.get("model_forward_latency_s")
                ),
                "model_forward_latency_sequence_s": [
                    float(value)
                    for value in final_overlay_info.get("model_forward_latency_sequence_s", [])
                ],
                "mean_model_forward_latency_s": _json_optional_float(
                    final_overlay_info.get("mean_model_forward_latency_s")
                ),
                "total_time_s": total_time_s,
                "task_prompt": task_prompt,
                "action_jitter_metrics": action_jitter_metrics,
                "action_jitter_metric_space": action_norm_metadata["quantitative_action_space"],
                "action_jitter_metrics_raw": raw_action_jitter_metrics,
            }
            if action_norm_metadata["normalization_available"]:
                display_record["action_jitter_metrics_normalized"] = action_jitter_metrics
            if os.environ.get("LEROBOT_SAVE_ACTION_TRAJECTORIES", "1").lower() not in {
                "0",
                "false",
                "no",
            }:
                display_record["action_trajectory"] = episode_actions.tolist()
                display_record["action_trajectory_raw"] = episode_actions.tolist()
                if action_norm_metadata["normalization_available"]:
                    display_record["action_trajectory_normalized"] = metric_episode_actions.tolist()
            episode_display_metrics.append(display_record)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for env_ix, (stacked_frames, done_index, success, total_time_s) in enumerate(
                zip(
                    batch_stacked_frames,
                    done_indices.flatten().tolist(),
                    batch_successes.tolist(),
                    rollout_data["episode_total_time_s"].flatten().tolist(),
                    strict=False,
                )
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                frame_infos = [step_infos[env_ix] for step_infos in ep_overlay_infos[: done_index + 1]]
                total_video_frames = done_index + 1
                frames_with_overlay = np.stack(
                    [
                        _draw_policy_duration_overlay(
                            frame,
                            info,
                            total_time_s=float(total_time_s),
                            success=bool(success),
                            task_prompt=task_prompt,
                            frame_index=frame_ix,
                            total_frames=total_video_frames,
                        )
                        for frame_ix, (frame, info) in enumerate(
                            zip(stacked_frames[:total_video_frames], frame_infos, strict=False)
                        )
                    ]
                )
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        frames_with_overlay,  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    display_metrics_for_episodes = episode_display_metrics[:n_episodes]
    successes_for_episodes = [bool(success) for success in all_successes[:n_episodes]]
    success_display_metrics = [
        metrics
        for metrics, success in zip(display_metrics_for_episodes, successes_for_episodes, strict=False)
        if success
    ]
    failure_display_metrics = [
        metrics
        for metrics, success in zip(display_metrics_for_episodes, successes_for_episodes, strict=False)
        if not success
    ]
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
                "display_metrics": display_metrics,
            }
            for i, (sum_reward, max_reward, success, seed, display_metrics) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    episode_display_metrics[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "mean_final_duration": _mean_optional_numbers(
                [_last_optional_int(metrics["duration"]) for metrics in episode_display_metrics[:n_episodes]]
            ),
            "mean_inference_calls": _mean_optional_numbers(
                [metrics["inference_calls"] for metrics in episode_display_metrics[:n_episodes]]
            ),
            "mean_model_forward_latency_s": _mean_optional_numbers(
                [
                    metrics["mean_model_forward_latency_s"]
                    for metrics in episode_display_metrics[:n_episodes]
                ]
            ),
            "mean_duration": _mean_optional_numbers(
                [metrics["mean_duration"] for metrics in episode_display_metrics[:n_episodes]]
            ),
            "mean_total_time_s": _mean_optional_numbers(
                [metrics["total_time_s"] for metrics in display_metrics_for_episodes]
            ),
            "avg_action_jitter_metrics_all_episodes": _mean_action_jitter_metrics_from_display_metrics(
                display_metrics_for_episodes
            ),
            "avg_action_jitter_metrics_success_episodes": _mean_action_jitter_metrics_from_display_metrics(
                success_display_metrics
            ),
            "avg_action_jitter_metrics_failure_episodes": _mean_action_jitter_metrics_from_display_metrics(
                failure_display_metrics
            ),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.max_episodes_rendered,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    durations: list[int | None]
    duration_sequences: list[list[int]]
    inference_calls: list[int]
    mean_model_forward_latencies_s: list[float | None]
    model_forward_latencies_s: list[float | None]
    model_forward_latency_sequences_s: list[list[float]]
    mean_durations: list[float | None]
    total_times_s: list[float]
    episode_metrics: list[dict[str, Any]]
    task_prompt: str | None


ACC_KEYS = (
    "sum_rewards",
    "max_rewards",
    "successes",
    "video_paths",
    "durations",
    "duration_sequences",
    "inference_calls",
    "mean_model_forward_latencies_s",
    "model_forward_latencies_s",
    "model_forward_latency_sequences_s",
    "mean_durations",
    "total_times_s",
    "episode_metrics",
    "task_prompt",
)


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    display_metrics = [ep["display_metrics"] for ep in per_episode]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        durations=[_last_optional_int(metrics["duration"]) for metrics in display_metrics],
        duration_sequences=[metrics["duration"] for metrics in display_metrics],
        inference_calls=[metrics["inference_calls"] for metrics in display_metrics],
        mean_model_forward_latencies_s=[
            metrics["mean_model_forward_latency_s"] for metrics in display_metrics
        ],
        model_forward_latencies_s=[metrics["model_forward_latency_s"] for metrics in display_metrics],
        model_forward_latency_sequences_s=[
            metrics["model_forward_latency_sequence_s"] for metrics in display_metrics
        ],
        mean_durations=[metrics["mean_duration"] for metrics in display_metrics],
        total_times_s=[metrics["total_time_s"] for metrics in display_metrics],
        episode_metrics=per_episode,
        task_prompt=display_metrics[0].get("task_prompt") if display_metrics else None,
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        _append("durations", metrics.get("durations"))
        _append("duration_sequences", metrics.get("duration_sequences"))
        _append("inference_calls", metrics.get("inference_calls"))
        _append("mean_model_forward_latencies_s", metrics.get("mean_model_forward_latencies_s"))
        _append("model_forward_latencies_s", metrics.get("model_forward_latencies_s"))
        _append("model_forward_latency_sequences_s", metrics.get("model_forward_latency_sequences_s"))
        _append("mean_durations", metrics.get("mean_durations"))
        _append("total_times_s", metrics.get("total_times_s"))
        _append("episode_metrics", metrics.get("episode_metrics"))
        group_acc[group]["task_prompt"].append(metrics.get("task_prompt"))
        overall["task_prompt"].append(metrics.get("task_prompt"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append(
                {"task_group": tg, "task_id": tid, "task_prompt": metrics.get("task_prompt"), "metrics": metrics}
            )
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append(
                    {"task_group": tg, "task_id": tid, "task_prompt": metrics.get("task_prompt"), "metrics": metrics}
                )

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        numeric_values = [x for x in xs if x is not None]
        if not numeric_values:
            return float("nan")
        arr = np.array(numeric_values, dtype=float)
        return float(np.nanmean(arr))

    def _display_metrics_from_episode_metrics(episodes):
        return [
            episode.get("display_metrics", {}) or {}
            for episode in episodes
            if isinstance(episode, dict)
        ]

    def _display_metrics_by_success(episodes, desired_success: bool):
        return [
            episode.get("display_metrics", {}) or {}
            for episode in episodes
            if isinstance(episode, dict) and bool(episode.get("success")) is desired_success
        ]

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        group_episodes = list(acc["episode_metrics"])
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "mean_final_duration": _agg_from_list(acc["durations"]),
            "mean_inference_calls": _agg_from_list(acc["inference_calls"]),
            "mean_model_forward_latency_s": _agg_from_list(acc["mean_model_forward_latencies_s"]),
            "mean_total_model_forward_latency_s": _agg_from_list(acc["model_forward_latencies_s"]),
            "mean_duration": _agg_from_list(acc["mean_durations"]),
            "mean_total_time_s": _agg_from_list(acc["total_times_s"]),
            "avg_action_jitter_metrics_all_episodes": _mean_action_jitter_metrics_from_display_metrics(
                _display_metrics_from_episode_metrics(group_episodes)
            ),
            "avg_action_jitter_metrics_success_episodes": _mean_action_jitter_metrics_from_display_metrics(
                _display_metrics_by_success(group_episodes, True)
            ),
            "avg_action_jitter_metrics_failure_episodes": _mean_action_jitter_metrics_from_display_metrics(
                _display_metrics_by_success(group_episodes, False)
            ),
            "task_prompts": sorted({prompt for prompt in acc["task_prompt"] if prompt}),
            "episode_metrics": group_episodes,
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_episodes = list(overall["episode_metrics"])
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "mean_final_duration": _agg_from_list(overall["durations"]),
        "mean_inference_calls": _agg_from_list(overall["inference_calls"]),
        "mean_model_forward_latency_s": _agg_from_list(overall["mean_model_forward_latencies_s"]),
        "mean_total_model_forward_latency_s": _agg_from_list(overall["model_forward_latencies_s"]),
        "mean_duration": _agg_from_list(overall["mean_durations"]),
        "mean_total_time_s": _agg_from_list(overall["total_times_s"]),
        "avg_action_jitter_metrics_all_episodes": _mean_action_jitter_metrics_from_display_metrics(
            _display_metrics_from_episode_metrics(overall_episodes)
        ),
        "avg_action_jitter_metrics_success_episodes": _mean_action_jitter_metrics_from_display_metrics(
            _display_metrics_by_success(overall_episodes, True)
        ),
        "avg_action_jitter_metrics_failure_episodes": _mean_action_jitter_metrics_from_display_metrics(
            _display_metrics_by_success(overall_episodes, False)
        ),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "task_prompts": sorted({prompt for prompt in overall["task_prompt"] if prompt}),
        "episode_metrics": overall_episodes,
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
