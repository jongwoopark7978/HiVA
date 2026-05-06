from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
    VLAFlowMatching,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_smolvla_hiva_coeff import HiVACoeffSmolVLAConfig


class HiVACoeffTargets(TypedDict):
    tr: Tensor
    rot: Tensor
    grip: Tensor


def _skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _rotvec_to_matrix(rotvec: Tensor) -> Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    theta2 = theta * theta
    small = theta < 1e-4
    a = torch.where(small, 1 - theta2 / 6 + theta2 * theta2 / 120, torch.sin(theta) / theta)
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - torch.cos(theta)) / torch.clamp(theta2, min=1e-12),
    )
    k = _skew(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(*rotvec.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def _matrix_to_rotvec(matrix: Tensor) -> Tensor:
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_theta = ((trace - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    omega = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = theta / torch.clamp(2 * sin_theta, min=1e-7)
    small = theta < 1e-4
    scale = torch.where(small, torch.full_like(scale, 0.5), scale)
    return scale.unsqueeze(-1) * omega


def _clamped_bspline_basis(d: int, *, n_ctrl: int, degree: int) -> Tensor:
    if d < 2:
        raise ValueError(f"Duration must be >= 2 for B-spline basis, got {d}.")
    if n_ctrl < degree + 1:
        raise ValueError(f"n_ctrl={n_ctrl} must be at least degree+1={degree + 1}.")

    x_values = [float(x) for x in range(d)]
    internal_count = n_ctrl - degree + 1
    if internal_count <= 1:
        t_internal = [0.0]
    else:
        t_internal = [
            (d - 1) * i / (internal_count - 1)
            for i in range(internal_count)
        ]
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

    rows = [[basis_one(i, degree, x) for i in range(n_ctrl)] for x in x_values]
    return torch.tensor(rows, dtype=torch.float32)


class HiVACoeffSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLA policy that predicts duration and B-spline coefficient macro-actions."""

    config_class = HiVACoeffSmolVLAConfig
    name = "smolvla_hiva_coeff"

    def __init__(
        self,
        config: HiVACoeffSmolVLAConfig,
        **kwargs,
    ):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = HiVACoeffVLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self._init_action_normalization(kwargs.get("dataset_stats"))
        self._maybe_load_init_smolvla_checkpoint()
        self.reset()

    def _maybe_load_init_smolvla_checkpoint(self) -> None:
        if not self.config.init_smolvla_checkpoint_path:
            return

        checkpoint_dir = Path(self.config.init_smolvla_checkpoint_path)
        model_file = checkpoint_dir / SAFETENSORS_SINGLE_FILE
        if not model_file.exists():
            logging.warning("Skipping SmolVLA init: %s does not exist.", model_file)
            return

        logging.info("Initializing HiVA coefficient policy from SmolVLA checkpoint: %s", checkpoint_dir)
        type(self)._load_as_safetensor(
            self,
            str(model_file),
            self.config.device,
            strict=self.config.init_smolvla_strict,
        )

    def _init_action_normalization(self, dataset_stats) -> None:
        action_dim = self.config.action_feature.shape[0] if self.config.action_feature is not None else 7
        mean = torch.zeros(action_dim, dtype=torch.float32)
        std = torch.ones(action_dim, dtype=torch.float32)
        if dataset_stats is not None and ACTION in dataset_stats:
            stats = dataset_stats[ACTION]
            if "mean" in stats and "std" in stats:
                mean = torch.as_tensor(stats["mean"], dtype=torch.float32).flatten()[:action_dim]
                std = torch.as_tensor(stats["std"], dtype=torch.float32).flatten()[:action_dim]
        self.register_buffer("_hiva_action_mean", mean.view(1, 1, -1), persistent=True)
        self.register_buffer("_hiva_action_std", std.view(1, 1, -1), persistent=True)

    def _normalize_raw_actions(self, raw_actions: Tensor) -> Tensor:
        mean = self._hiva_action_mean.to(device=raw_actions.device, dtype=raw_actions.dtype)
        std = self._hiva_action_std.to(device=raw_actions.device, dtype=raw_actions.dtype)
        return (raw_actions - mean) / (std + 1e-8)

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._last_duration_steps = None
        self._last_execution_horizon = None
        self._duration_inference_count = 0
        self._execution_horizon_sum = 0
        self._execution_horizon_history = []

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: dict[str, Tensor] | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> tuple[Tensor, Tensor]:
        for key in batch:
            if key in self._queues and key != ACTION:
                batch[key] = torch.stack(list(self._queues[key]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        raw_actions, duration_steps = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )
        self._last_duration_steps = duration_steps
        actions = self._normalize_raw_actions(raw_actions)
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions, duration_steps

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: dict[str, Tensor] | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        actions, _duration_steps = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: dict[str, Tensor] | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        assert not self._rtc_enabled(), "RTC is not supported for select_action."

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions, duration_steps = self._get_action_chunk(batch, noise)
            execution_horizon = self._execution_horizon_from_duration(duration_steps, actions)
            self._last_execution_horizon = execution_horizon
            self._duration_inference_count += 1
            self._execution_horizon_sum += execution_horizon
            self._execution_horizon_history.append(execution_horizon)
            self._queues[ACTION].extend(actions.transpose(0, 1)[:execution_horizon])

        return self._queues[ACTION].popleft()

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        if self.config.adapt_to_pi_aloha:
            raise NotImplementedError("HiVA coefficient SmolVLA is currently implemented for LIBERO actions.")

        for key in ("hiva_theta_tr_raw", "hiva_theta_rot_raw", "hiva_theta_grip_raw", "duration_class"):
            if key not in batch:
                raise KeyError(
                    f"`{key}` is missing from the training batch. "
                    "Enable the HiVA coefficient sidecar wrapper before training."
                )

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        targets: HiVACoeffTargets = {
            "tr": batch["hiva_theta_tr_raw"].to(device=state.device, dtype=torch.float32),
            "rot": batch["hiva_theta_rot_raw"].to(device=state.device, dtype=torch.float32),
            "grip": batch["hiva_theta_grip_raw"].to(device=state.device, dtype=torch.float32),
        }
        expected_shapes = {
            "tr": (self.config.hiva_k, 3),
            "rot": (self.config.hiva_k, 3),
            "grip": (self.config.hiva_k, 1),
        }
        for key, expected_shape in expected_shapes.items():
            if tuple(targets[key].shape[-2:]) != expected_shape:
                raise ValueError(
                    f"`hiva_theta_{key}_raw` must have trailing shape {expected_shape} for "
                    f"hiva_k={self.config.hiva_k}; got {tuple(targets[key].shape[-2:])}."
                )
        duration_target = batch["duration_class"].to(device=state.device, dtype=torch.long)

        per_sample_loss, loss_dict = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            targets,
            duration_target,
            noise=noise,
            time=time,
        )

        if reduction == "none":
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        loss = per_sample_loss.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        common = (
            "state_proj|hiva_tr_in_proj|hiva_rot_in_proj|hiva_grip_in_proj|"
            "hiva_tr_out_proj|hiva_rot_out_proj|hiva_grip_out_proj|"
            "hiva_duration_token|hiva_duration_head|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }


class HiVACoeffVLAFlowMatching(VLAFlowMatching):
    """Flow matching over HiVA B-spline coefficient tokens."""

    def __init__(self, config: HiVACoeffSmolVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config = config
        hidden = self.vlm_with_expert.expert_hidden_size
        for module in (self.action_in_proj, self.action_out_proj):
            for param in module.parameters():
                param.requires_grad = False

        self.hiva_duration_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.hiva_duration_token, mean=0.0, std=0.02)
        self.hiva_tr_in_proj = nn.Linear(3, hidden)
        self.hiva_rot_in_proj = nn.Linear(3, hidden)
        self.hiva_grip_in_proj = nn.Linear(1, hidden)
        self.hiva_tr_out_proj = nn.Linear(hidden, 3)
        self.hiva_rot_out_proj = nn.Linear(hidden, 3)
        self.hiva_grip_out_proj = nn.Linear(hidden, 1)
        self.hiva_duration_head = nn.Linear(hidden, len(config.hiva_duration_classes))
        self._register_coeff_stats()
        self._register_basis_buffers()

    @property
    def hiva_suffix_len(self) -> int:
        return 1 + 3 * self.config.hiva_k

    def _register_coeff_stats(self) -> None:
        stats = None
        if self.config.hiva_coeff_sidecar_summary_path:
            summary_path = Path(self.config.hiva_coeff_sidecar_summary_path)
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    stats = json.load(f).get("coef_stats")
            else:
                logging.warning("HiVA coefficient summary does not exist: %s", summary_path)

        def stat_tensor(modality: str, name: str, dim: int, default: float) -> Tensor:
            value = None if stats is None else stats.get(modality, {}).get(name)
            if value is None:
                value = [default] * dim
            tensor = torch.as_tensor(value, dtype=torch.float32).view(1, 1, dim)
            if name == "std":
                tensor = torch.clamp(tensor, min=self.config.hiva_coeff_norm_eps)
            return tensor

        self.register_buffer("hiva_tr_mean", stat_tensor("tr", "mean", 3, 0.0), persistent=True)
        self.register_buffer("hiva_tr_std", stat_tensor("tr", "std", 3, 1.0), persistent=True)
        self.register_buffer("hiva_rot_mean", stat_tensor("rot", "mean", 3, 0.0), persistent=True)
        self.register_buffer("hiva_rot_std", stat_tensor("rot", "std", 3, 1.0), persistent=True)
        self.register_buffer("hiva_grip_mean", stat_tensor("grip", "mean", 1, 0.0), persistent=True)
        self.register_buffer("hiva_grip_std", stat_tensor("grip", "std", 1, 1.0), persistent=True)

    def _register_basis_buffers(self) -> None:
        # Keep legacy duration-specific bases so older sidecars/checkpoints continue to work.
        for duration in self.config.hiva_duration_classes:
            basis = _clamped_bspline_basis(
                int(duration),
                n_ctrl=self.config.hiva_k,
                degree=self.config.hiva_degree,
            )
            self.register_buffer(f"hiva_basis_{duration}", basis, persistent=False)

        canonical_basis = _clamped_bspline_basis(
            int(self.config.hiva_dmax),
            n_ctrl=self.config.hiva_k,
            degree=self.config.hiva_degree,
        )
        self.register_buffer("hiva_basis_canonical", canonical_basis, persistent=False)

    def _uses_canonical_basis(self) -> bool:
        return self.config.hiva_basis_mode in ("canonical_hp", "canonical_mt")

    def _basis(self, duration: int, *, device, dtype) -> Tensor:
        return getattr(self, f"hiva_basis_{int(duration)}").to(device=device, dtype=dtype)

    def _canonical_basis(self, *, device, dtype) -> Tensor:
        return self.hiva_basis_canonical.to(device=device, dtype=dtype)

    def normalize_coeffs(self, targets: HiVACoeffTargets) -> HiVACoeffTargets:
        if not self.config.hiva_normalize_coefficients:
            return targets
        return {
            "tr": (targets["tr"] - self.hiva_tr_mean) / self.hiva_tr_std,
            "rot": (targets["rot"] - self.hiva_rot_mean) / self.hiva_rot_std,
            "grip": (targets["grip"] - self.hiva_grip_mean) / self.hiva_grip_std,
        }

    def unnormalize_coeffs(self, coeffs: HiVACoeffTargets) -> HiVACoeffTargets:
        if not self.config.hiva_normalize_coefficients:
            return coeffs
        return {
            "tr": coeffs["tr"] * self.hiva_tr_std + self.hiva_tr_mean,
            "rot": coeffs["rot"] * self.hiva_rot_std + self.hiva_rot_mean,
            "grip": coeffs["grip"] * self.hiva_grip_std + self.hiva_grip_mean,
        }

    def _sample_coeff_noise(self, coeffs: HiVACoeffTargets) -> HiVACoeffTargets:
        return {key: self.sample_noise(value.shape, value.device) for key, value in coeffs.items()}

    def _mix_coeffs(self, targets: HiVACoeffTargets, noise: HiVACoeffTargets, time: Tensor):
        time_expanded = time[:, None, None]
        x_t = {key: time_expanded * noise[key] + (1 - time_expanded) * targets[key] for key in targets}
        u_t = {key: noise[key] - targets[key] for key in targets}
        return x_t, u_t

    def duration_noisy_weights(self, time: Tensor) -> Tensor:
        sigma = self.config.hiva_duration_noisy_sigma
        return torch.exp(-(time.to(dtype=torch.float32) ** 2) / (sigma**2))

    def _hiva_suffix_att_list(self, suffix_len: int) -> list[int]:
        """Return the block-attention mask for the HiVA coefficient suffix.

        Suffix order is always [duration][translation coeffs][rotation coeffs][gripper coeffs].
        SmolVLA's make_att_2d_masks treats 1 as a new cumulative attention block and
        0 as sharing the previous block.
        """
        if suffix_len < 2:
            raise ValueError(f"HiVA suffix must contain duration plus coefficient tokens, got {suffix_len}.")

        mode = self.config.hiva_suffix_attention
        if mode == "duration_prefix":
            # Duration attends prefix + itself; coefficient tokens attend prefix + duration + all coeffs.
            return [1, 1] + [0] * (suffix_len - 2)
        if mode == "full":
            # One bidirectional suffix block: duration and coefficients all attend each other.
            return [1] + [0] * (suffix_len - 1)
        if mode == "causal":
            # Original cumulative/causal suffix ordering.
            return [1] * suffix_len
        raise ValueError(f"Unknown hiva_suffix_attention={mode!r}.")

    def embed_hiva_suffix(self, coeffs: HiVACoeffTargets, timestep: Tensor):
        tr_emb = self.hiva_tr_in_proj(coeffs["tr"])
        rot_emb = self.hiva_rot_in_proj(coeffs["rot"])
        grip_emb = self.hiva_grip_in_proj(coeffs["grip"])
        coeff_emb = torch.cat([tr_emb, rot_emb, grip_emb], dim=1)

        device = coeff_emb.device
        bsize = coeff_emb.shape[0]
        dtype = coeff_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(coeff_emb)
        coeff_time_emb = torch.cat([coeff_emb, time_emb], dim=-1)
        coeff_time_emb = self.action_time_mlp_in(coeff_time_emb)
        coeff_time_emb = F.silu(coeff_time_emb)
        coeff_time_emb = self.action_time_mlp_out(coeff_time_emb)

        duration_emb = self.hiva_duration_token.to(device=device, dtype=dtype).expand(bsize, -1, -1)
        embs = torch.cat([duration_emb, coeff_time_emb], dim=1)
        pad_masks = torch.ones(bsize, embs.shape[1], dtype=torch.bool, device=device)
        att = self._hiva_suffix_att_list(embs.shape[1])
        att_masks = torch.tensor(att, dtype=torch.bool, device=device)[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    def _split_hiva_suffix_out(self, suffix_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        suffix_out = suffix_out[:, -self.hiva_suffix_len :].to(dtype=torch.float32)
        duration_out = suffix_out[:, 0]
        coeff_out = suffix_out[:, 1:]
        k = self.config.hiva_k
        tr_out = coeff_out[:, :k]
        rot_out = coeff_out[:, k : 2 * k]
        grip_out = coeff_out[:, 2 * k : 3 * k]
        return duration_out, tr_out, rot_out, grip_out

    def _heads_from_suffix_out(self, suffix_out: Tensor) -> tuple[HiVACoeffTargets, Tensor]:
        duration_out, tr_out, rot_out, grip_out = self._split_hiva_suffix_out(suffix_out)
        pred = {
            "tr": self.hiva_tr_out_proj(tr_out),
            "rot": self.hiva_rot_out_proj(rot_out),
            "grip": self.hiva_grip_out_proj(grip_out),
        }
        duration_logits = self.hiva_duration_head(duration_out)
        return pred, duration_logits

    def _forward_hiva_suffix_with_prefix_cache(
        self,
        prefix_pad_masks,
        past_key_values,
        coeffs: HiVACoeffTargets,
        timestep: Tensor,
        use_cache=None,
    ) -> tuple[HiVACoeffTargets, Tensor]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_hiva_suffix(coeffs, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=use_cache,
            fill_kv_cache=False,
        )
        return self._heads_from_suffix_out(outputs_embeds[1])

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        targets: HiVACoeffTargets,
        duration_target: Tensor,
        noise: HiVACoeffTargets | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        targets_norm = self.normalize_coeffs(targets)
        if noise is None:
            noise = self._sample_coeff_noise(targets_norm)
        if time is None:
            time = self.sample_time(state.shape[0], state.device)

        x_t, u_t = self._mix_coeffs(targets_norm, noise, time)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        pred, duration_logits = self._forward_hiva_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=x_t,
            timestep=time,
            use_cache=True,
        )

        tr_loss = F.mse_loss(pred["tr"], u_t["tr"], reduction="none").mean(dim=(1, 2))
        rot_loss = F.mse_loss(pred["rot"], u_t["rot"], reduction="none").mean(dim=(1, 2))
        grip_loss = F.mse_loss(pred["grip"], u_t["grip"], reduction="none").mean(dim=(1, 2))
        duration_ce = F.cross_entropy(duration_logits.to(dtype=torch.float32), duration_target, reduction="none")
        duration_noisy_weights = self.duration_noisy_weights(time).to(
            device=duration_ce.device,
            dtype=duration_ce.dtype,
        )
        duration_noisy_weight_sum = duration_noisy_weights.sum().clamp_min(1e-6)
        duration_weighted_loss_per_sample = duration_ce * duration_noisy_weights
        # Keep normalized weighted CE as a per-sample tensor because the training wrapper later calls mean().
        # Multiplying by batch_size / sum(weight) makes that later mean equal sum(CE * weight) / sum(weight).
        duration_weighted_loss_normalized_per_sample = (
            duration_weighted_loss_per_sample * (duration_ce.numel() / duration_noisy_weight_sum)
        )

        if self.config.hiva_duration_loss == "ce_mean":
            duration_noisy_loss = duration_ce
            duration_noisy_loss_reduced = duration_ce.mean()
        elif self.config.hiva_duration_loss == "duration_noisy_weights":
            duration_noisy_loss = duration_weighted_loss_normalized_per_sample
            duration_noisy_loss_reduced = duration_weighted_loss_per_sample.sum() / duration_noisy_weight_sum
        elif self.config.hiva_duration_loss == "mean":
            duration_noisy_loss = duration_weighted_loss_per_sample
            duration_noisy_loss_reduced = duration_weighted_loss_per_sample.mean()
        else:
            raise ValueError(f"Unknown hiva_duration_loss={self.config.hiva_duration_loss!r}.")

        duration_noisy_loss_scaled_reduced = (
            self.config.hiva_duration_noisy_loss_weight * duration_noisy_loss_reduced
        )
        per_sample_loss = (
            self.config.hiva_tr_loss_weight * tr_loss
            + self.config.hiva_rot_loss_weight * rot_loss
            + self.config.hiva_grip_loss_weight * grip_loss
            + self.config.hiva_duration_noisy_loss_weight * duration_noisy_loss
        )

        duration_values = torch.as_tensor(
            self.config.hiva_duration_classes,
            device=duration_logits.device,
            dtype=torch.long,
        )
        duration_pred = duration_logits.argmax(dim=-1)
        duration_correct = (duration_pred == duration_target).float()
        duration_weighted_acc = (
            (duration_correct * duration_noisy_weights).sum() / duration_noisy_weight_sum
        )
        duration_noisy_weight_sq_sum = (duration_noisy_weights * duration_noisy_weights).sum().clamp_min(1e-6)
        duration_noisy_weight_ess = (
            duration_noisy_weight_sum * duration_noisy_weight_sum / duration_noisy_weight_sq_sum
        )
        fm_total_reduced = (
            self.config.hiva_tr_loss_weight * tr_loss.mean()
            + self.config.hiva_rot_loss_weight * rot_loss.mean()
            + self.config.hiva_grip_loss_weight * grip_loss.mean()
        )
        duration_to_fm_ratio = (
            duration_noisy_loss_scaled_reduced / fm_total_reduced.detach().clamp_min(1e-6)
        )

        def duration_acc_for_time_mask(mask: Tensor) -> Tensor:
            count = mask.sum()
            if count == 0:
                return torch.zeros((), device=duration_correct.device, dtype=duration_correct.dtype)
            return duration_correct[mask].mean()

        time_float = time.to(device=duration_correct.device, dtype=torch.float32)
        loss_dict = {
            "hiva_tr_fm_loss": tr_loss.mean().item(),
            "hiva_rot_fm_loss": rot_loss.mean().item(),
            "hiva_grip_fm_loss": grip_loss.mean().item(),
            "hiva_duration_noisy_loss": duration_noisy_loss_reduced.item(),
            "hiva_duration_noisy_loss_scaled": duration_noisy_loss_scaled_reduced.item(),
            "hiva_duration_noisy_ce": duration_ce.mean().item(),
            "hiva_duration_noisy_loss_unweighted_ce": duration_ce.mean().item(),
            "hiva_duration_noisy_loss_weighted_mean_legacy": duration_weighted_loss_per_sample.mean().item(),
            "hiva_duration_noisy_loss_weighted_normalized": (
                duration_weighted_loss_per_sample.sum() / duration_noisy_weight_sum
            ).item(),
            "hiva_fm_total": fm_total_reduced.item(),
            "hiva_duration_to_fm_ratio": duration_to_fm_ratio.item(),
            "hiva_duration_acc": duration_correct.mean().item(),
            "hiva_duration_noisy_acc": duration_correct.mean().item(),
            "hiva_duration_noisy_acc_all": duration_correct.mean().item(),
            "hiva_duration_noisy_acc_weighted": duration_weighted_acc.item(),
            "hiva_duration_noisy_correct_sum": duration_correct.sum().item(),
            "hiva_duration_noisy_weighted_correct_sum": (duration_correct * duration_noisy_weights).sum().item(),
            "hiva_duration_noisy_weight_sum": duration_noisy_weight_sum.item(),
            "hiva_duration_noisy_weight_sq_sum": duration_noisy_weight_sq_sum.item(),
            "hiva_duration_noisy_weight_ess": duration_noisy_weight_ess.item(),
            "hiva_duration_noisy_weight_ess_frac": (duration_noisy_weight_ess / duration_ce.numel()).item(),
            "hiva_duration_acc_t_lt_010": duration_acc_for_time_mask(time_float < 0.10).item(),
            "hiva_duration_acc_t_lt_020": duration_acc_for_time_mask(time_float < 0.20).item(),
            "hiva_duration_acc_t_lt_025": duration_acc_for_time_mask(time_float < 0.25).item(),
            "hiva_duration_acc_t_025_050": duration_acc_for_time_mask(
                (time_float >= 0.25) & (time_float < 0.50)
            ).item(),
            "hiva_duration_acc_t_gt_050": duration_acc_for_time_mask(time_float >= 0.50).item(),
            "hiva_duration_count_t_lt_010": (time_float < 0.10).sum().item(),
            "hiva_duration_count_t_lt_020": (time_float < 0.20).sum().item(),
            "hiva_duration_count_t_lt_025": (time_float < 0.25).sum().item(),
            "hiva_duration_count_t_025_050": ((time_float >= 0.25) & (time_float < 0.50)).sum().item(),
            "hiva_duration_count_t_gt_050": (time_float >= 0.50).sum().item(),
            "hiva_duration_pred_mean": duration_values[duration_pred].float().mean().item(),
            "hiva_duration_target_mean": duration_values[duration_target].float().mean().item(),
            "hiva_duration_noisy_weight_mean": duration_noisy_weights.mean().item(),
            "hiva_duration_noisy_weight_min": duration_noisy_weights.min().item(),
            "hiva_duration_noisy_weight_max": duration_noisy_weights.max().item(),
            "hiva_tr_loss_weight": self.config.hiva_tr_loss_weight,
            "hiva_rot_loss_weight": self.config.hiva_rot_loss_weight,
            "hiva_grip_loss_weight": self.config.hiva_grip_loss_weight,
            "hiva_duration_noisy_loss_weight": self.config.hiva_duration_noisy_loss_weight,
            "hiva_duration_noisy_sigma": self.config.hiva_duration_noisy_sigma,
            "hiva_duration_loss": self.config.hiva_duration_loss,
            "hiva_suffix_attention": self.config.hiva_suffix_attention,
            "hiva_basis_mode": self.config.hiva_basis_mode,
            "hiva_k": self.config.hiva_k,
            "hiva_suffix_len": self.hiva_suffix_len,
        }
        return per_sample_loss, loss_dict

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t: HiVACoeffTargets,
        timestep: Tensor,
        use_cache=None,
    ) -> HiVACoeffTargets:
        pred, _duration_logits = self._forward_hiva_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=x_t,
            timestep=timestep,
            use_cache=use_cache,
        )
        return pred

    def predict_duration_logits(self, prefix_pad_masks, past_key_values, coeffs: HiVACoeffTargets, timestep):
        _pred, duration_logits = self._forward_hiva_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=coeffs,
            timestep=timestep,
        )
        return duration_logits

    def _duration_steps_from_logits(self, duration_logits: Tensor) -> Tensor:
        duration_classes = torch.as_tensor(
            self.config.hiva_duration_classes,
            device=duration_logits.device,
            dtype=torch.long,
        )
        return duration_classes[duration_logits.argmax(dim=-1)]

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise: HiVACoeffTargets | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> tuple[Tensor, Tensor]:
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = {
                "tr": self.sample_noise((bsize, self.config.hiva_k, 3), device),
                "rot": self.sample_noise((bsize, self.config.hiva_k, 3), device),
                "grip": self.sample_noise((bsize, self.config.hiva_k, 1), device),
            }
        x_t = noise

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        for step in range(self.config.num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.full((bsize,), time, dtype=torch.float32, device=device)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = {key: x_t[key] + dt * v_t[key] for key in x_t}

        final_timestep = torch.zeros(bsize, dtype=torch.float32, device=device)
        duration_logits = self.predict_duration_logits(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=x_t,
            timestep=final_timestep,
        )
        duration_steps = self._duration_steps_from_logits(duration_logits)
        coeffs_raw = self.unnormalize_coeffs(x_t)
        raw_actions = self.decode_coefficients_to_raw_actions(coeffs_raw, duration_steps)
        return raw_actions, duration_steps

    def _decode_basis_to_raw_actions(self, coeffs_raw: HiVACoeffTargets, phi: Tensor) -> Tensor:
        """Decode raw LIBERO actions from a single B-spline basis shared by a batch.

        Translation and rotation coefficients decode cumulative labels; gripper coefficients decode
        absolute continuous commands directly.
        """
        bsize = coeffs_raw["tr"].shape[0]
        device = coeffs_raw["tr"].device
        dtype = coeffs_raw["tr"].dtype
        horizon = int(phi.shape[0])
        raw_actions = torch.zeros(
            bsize,
            self.config.hiva_dmax,
            self.config.action_feature.shape[0],
            dtype=dtype,
            device=device,
        )
        eye = torch.eye(3, dtype=dtype, device=device)

        p_hat = torch.einsum("dk,bkc->bdc", phi, coeffs_raw["tr"])
        p0 = torch.zeros(bsize, 1, 3, dtype=dtype, device=device)
        tr_delta = torch.diff(torch.cat([p0, p_hat], dim=1), dim=1)

        rho_hat = torch.einsum("dk,bkc->bdc", phi, coeffs_raw["rot"])
        rot_mats = _rotvec_to_matrix(rho_hat.reshape(-1, 3)).reshape(bsize, horizon, 3, 3)
        prev = torch.cat([eye.expand(bsize, 1, 3, 3), rot_mats[:, :-1]], dim=1)
        delta = prev.transpose(-1, -2) @ rot_mats
        raw_rot = _matrix_to_rotvec(delta.reshape(-1, 3, 3)).reshape(bsize, horizon, 3)
        raw_rot = raw_rot / float(self.config.hiva_rot_scale_eta)

        grip = torch.einsum("dk,bkc->bdc", phi, coeffs_raw["grip"]).clamp(-1.0, 1.0)
        decoded = torch.cat([tr_delta, raw_rot, grip], dim=-1)
        raw_actions[:, :horizon, : decoded.shape[-1]] = decoded
        return raw_actions

    def decode_coefficients_to_raw_actions(self, coeffs_raw: HiVACoeffTargets, duration_steps: Tensor) -> Tensor:
        """Decode coefficients to raw LIBERO action chunks.

        Legacy duration_specific mode decodes with the predicted-duration basis.
        Canonical HP/MT modes always decode a full Dmax chunk with the canonical basis; duration
        only controls how many prefix actions are queued for execution.
        """
        device = duration_steps.device
        dtype = coeffs_raw["tr"].dtype
        if self._uses_canonical_basis():
            phi = self._canonical_basis(device=device, dtype=dtype)
            return self._decode_basis_to_raw_actions(coeffs_raw, phi)

        bsize = duration_steps.shape[0]
        raw_actions = torch.zeros(
            bsize,
            self.config.hiva_dmax,
            self.config.action_feature.shape[0],
            dtype=dtype,
            device=device,
        )

        for duration in self.config.hiva_duration_classes:
            mask = duration_steps == int(duration)
            if not mask.any():
                continue
            phi = self._basis(int(duration), device=device, dtype=dtype)
            sub_coeffs = {
                "tr": coeffs_raw["tr"][mask],
                "rot": coeffs_raw["rot"][mask],
                "grip": coeffs_raw["grip"][mask],
            }
            decoded = self._decode_basis_to_raw_actions(sub_coeffs, phi)
            raw_actions[mask, : int(duration), : decoded.shape[-1]] = decoded[:, : int(duration)]

        return raw_actions
