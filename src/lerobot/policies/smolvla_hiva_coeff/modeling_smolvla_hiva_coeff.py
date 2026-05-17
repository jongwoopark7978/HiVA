from __future__ import annotations

import copy
import json
import logging
from collections import deque
from pathlib import Path
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import safe_open
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


class HiVAResidualFFNDurationHead(nn.Module):
    """Pre-norm residual FFN adapter for the HiVA duration token."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        *,
        hidden_mult: float = 4.0,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        ffn_hidden = max(1, int(round(hidden_size * hidden_mult)))
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, hidden_size),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden.to(dtype=torch.float32)
        adapted = hidden + self.alpha.to(dtype=hidden.dtype) * self.ffn(self.norm(hidden))
        return self.classifier(adapted)


class HiVACoeffModalityPoolDurationHead(nn.Module):
    """Categorical duration readout from pooled coefficient-token hidden states."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        *,
        hidden_mult: float = 4.0,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        pooled_size = 3 * hidden_size
        ffn_hidden = max(1, int(round(pooled_size * hidden_mult)))
        self.norm = nn.LayerNorm(pooled_size)
        self.ffn = nn.Sequential(
            nn.Linear(pooled_size, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, pooled_size),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Linear(pooled_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, tr_hidden: Tensor, rot_hidden: Tensor, grip_hidden: Tensor) -> Tensor:
        tr_pool = tr_hidden.to(dtype=torch.float32).mean(dim=1)
        rot_pool = rot_hidden.to(dtype=torch.float32).mean(dim=1)
        grip_pool = grip_hidden.to(dtype=torch.float32).mean(dim=1)
        pooled = torch.cat([tr_pool, rot_pool, grip_pool], dim=-1)
        adapted = pooled + self.alpha.to(dtype=pooled.dtype) * self.ffn(self.norm(pooled))
        return self.classifier(adapted)


class HiVAFullHorizonResidualHead(nn.Module):
    """Predict bounded raw-action residuals from coefficient-token hidden states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_coeff_tokens: int,
        fit_horizon: int,
        action_dim: int,
        ffn_hidden_mult: float = 4.0,
        token_time_hidden_mult: float = 2.0,
        alpha_init: float = 0.1,
        zero_init: bool = True,
    ):
        super().__init__()
        coeff_hidden = max(1, int(round(hidden_size * ffn_hidden_mult)))
        token_hidden = max(1, int(round(fit_horizon * token_time_hidden_mult)))
        time_hidden = max(1, int(round(hidden_size * ffn_hidden_mult)))
        self.fit_horizon = int(fit_horizon)
        self.action_dim = int(action_dim)
        self.coeff_norm = nn.LayerNorm(hidden_size)
        self.coeff_ffn = nn.Sequential(
            nn.Linear(hidden_size, coeff_hidden),
            nn.SiLU(),
            nn.Linear(coeff_hidden, hidden_size),
        )
        self.coeff_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.token_to_time = nn.Sequential(
            nn.Linear(num_coeff_tokens, token_hidden),
            nn.SiLU(),
            nn.Linear(token_hidden, fit_horizon),
        )
        self.action_time_embed = nn.Parameter(torch.zeros(1, fit_horizon, hidden_size))
        nn.init.normal_(self.action_time_embed, mean=0.0, std=0.02)
        self.time_norm = nn.LayerNorm(hidden_size)
        self.time_ffn = nn.Sequential(
            nn.Linear(hidden_size, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, hidden_size),
        )
        self.time_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_hidden = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, action_dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, coeff_hidden: Tensor, **_: Tensor) -> Tensor:
        coeff_hidden = coeff_hidden.to(dtype=torch.float32)
        h = coeff_hidden + self.coeff_alpha.to(dtype=coeff_hidden.dtype) * self.coeff_ffn(
            self.coeff_norm(coeff_hidden)
        )
        # Mix the coefficient-token axis into decoded action time, then refine per action step.
        z = self.token_to_time(h.transpose(1, 2)).transpose(1, 2)
        z = z + self.action_time_embed[:, : z.shape[1], :].to(device=z.device, dtype=z.dtype)
        z = z + self.time_alpha.to(dtype=z.dtype) * self.time_ffn(self.time_norm(z))
        z = self.out_norm(z)
        z = F.silu(self.out_hidden(z))
        return self.out_proj(z)


def _delta_basis(phi: Tensor) -> Tensor:
    """Convert cumulative B-spline basis rows into finite-difference basis rows."""
    zero = torch.zeros(1, phi.shape[1], dtype=phi.dtype, device=phi.device)
    return torch.diff(torch.cat([zero, phi], dim=0), dim=0)


def _pad_last_dim(tensor: Tensor, target_dim: int) -> Tensor:
    """Pad a [..., D] tensor to [..., target_dim] without changing existing values."""
    if tensor.shape[-1] == target_dim:
        return tensor
    if tensor.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad last dimension {tensor.shape[-1]} down to {target_dim}.")
    return F.pad(tensor, (0, target_dim - tensor.shape[-1]))


class HiVABasisHiddenActionResidualHead(nn.Module):
    """Basis-aware residual head using coefficient hidden states and decoded base actions."""

    def __init__(
        self,
        *,
        hidden_size: int,
        fit_horizon: int,
        action_dim: int,
        ffn_hidden_mult: float = 4.0,
        alpha_init: float = 0.1,
        zero_init: bool = True,
    ):
        super().__init__()
        time_hidden = max(1, int(round(hidden_size * ffn_hidden_mult)))
        self.fit_horizon = int(fit_horizon)
        self.action_dim = int(action_dim)
        self.rot_mix = nn.Linear(2 * hidden_size, hidden_size)
        self.action_embed = nn.Linear(action_dim, hidden_size)
        self.cat_proj = nn.Linear(4 * hidden_size, hidden_size)
        self.action_time_embed = nn.Parameter(torch.zeros(1, fit_horizon, hidden_size))
        nn.init.normal_(self.action_time_embed, mean=0.0, std=0.02)
        self.time_norm = nn.LayerNorm(hidden_size)
        self.time_ffn = nn.Sequential(
            nn.Linear(hidden_size, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, hidden_size),
        )
        self.time_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_hidden = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, action_dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _basis_lift(phi: Tensor, hidden: Tensor) -> Tensor:
        return torch.einsum("dk,bkh->bdh", phi.to(dtype=hidden.dtype, device=hidden.device), hidden)

    def _base_action_features(self, base_actions_raw: Tensor, horizon: int) -> Tensor:
        base = base_actions_raw[:, :horizon].to(dtype=torch.float32)
        if base.shape[-1] < self.action_dim:
            base = F.pad(base, (0, self.action_dim - base.shape[-1]))
        return base[..., : self.action_dim]

    def forward(
        self,
        *,
        tr_hidden: Tensor,
        rot_hidden: Tensor,
        grip_hidden: Tensor,
        base_actions_raw: Tensor,
        basis_tr: Tensor,
        basis_rot: Tensor,
        basis_grip: Tensor,
        **_: Tensor,
    ) -> Tensor:
        horizon = min(self.fit_horizon, base_actions_raw.shape[1])
        tr_hidden = tr_hidden.to(dtype=torch.float32)
        rot_hidden = rot_hidden.to(dtype=torch.float32)
        grip_hidden = grip_hidden.to(dtype=torch.float32)
        basis_tr = basis_tr[:horizon].to(device=tr_hidden.device, dtype=tr_hidden.dtype)
        basis_rot = basis_rot[:horizon].to(device=rot_hidden.device, dtype=rot_hidden.dtype)
        basis_grip = basis_grip[:horizon].to(device=grip_hidden.device, dtype=grip_hidden.dtype)
        dphi_tr = _delta_basis(basis_tr)
        dphi_rot = _delta_basis(basis_rot)

        z_tr = self._basis_lift(dphi_tr, tr_hidden)
        z_rot = self.rot_mix(
            torch.cat(
                [
                    self._basis_lift(basis_rot, rot_hidden),
                    self._basis_lift(dphi_rot, rot_hidden),
                ],
                dim=-1,
            )
        )
        z_grip = self._basis_lift(basis_grip, grip_hidden)
        z_action = self.action_embed(self._base_action_features(base_actions_raw, horizon))
        z = self.cat_proj(torch.cat([z_tr, z_rot, z_grip, z_action], dim=-1))
        z = z + self.action_time_embed[:, :horizon, :].to(device=z.device, dtype=z.dtype)
        z = z + self.time_alpha.to(dtype=z.dtype) * self.time_ffn(self.time_norm(z))
        z = self.out_norm(z)
        z = F.silu(self.out_hidden(z))
        return self.out_proj(z)


class HiVAResidualTransformerBlock(nn.Module):
    """Action-time residual block with self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_mult: float = 4.0,
        alpha_init: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_heads` ({num_heads})."
            )
        ffn_hidden = max(1, int(round(hidden_size * ffn_hidden_mult)))
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.cross_q_norm = nn.LayerNorm(hidden_size)
        self.cross_kv_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, hidden_size),
        )
        self.ffn_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, q: Tensor, memory: Tensor) -> Tensor:
        q_ln = self.self_norm(q)
        self_out, _ = self.self_attn(q_ln, q_ln, q_ln, need_weights=False)
        q = q + self.self_alpha.to(dtype=q.dtype) * self_out
        q_ln = self.cross_q_norm(q)
        memory_ln = self.cross_kv_norm(memory)
        cross_out, _ = self.cross_attn(q_ln, memory_ln, memory_ln, need_weights=False)
        q = q + self.cross_alpha.to(dtype=q.dtype) * cross_out
        q = q + self.ffn_alpha.to(dtype=q.dtype) * self.ffn(self.ffn_norm(q))
        return q


class HiVABasisXAttnTransformerResidualHead(nn.Module):
    """Basis-aware action-time queries cross-attend to coefficient hidden states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        n_ctrl: int,
        fit_horizon: int,
        action_dim: int,
        num_blocks: int = 4,
        num_heads: int = 4,
        ffn_hidden_mult: float = 4.0,
        alpha_init: float = 0.1,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.fit_horizon = int(fit_horizon)
        self.action_dim = int(action_dim)
        self.n_ctrl = int(n_ctrl)
        self.tr_basis_proj = nn.Linear(2 * n_ctrl, hidden_size)
        self.rot_basis_proj = nn.Linear(2 * n_ctrl, hidden_size)
        self.grip_basis_proj = nn.Linear(n_ctrl, hidden_size)
        self.base_action_proj = nn.Linear(action_dim, hidden_size)
        self.action_time_embed = nn.Parameter(torch.zeros(1, fit_horizon, hidden_size))
        self.modality_embed = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.coeff_index_embed = nn.Parameter(torch.zeros(1, n_ctrl, hidden_size))
        nn.init.normal_(self.action_time_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.coeff_index_embed, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList(
            [
                HiVAResidualTransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_hidden_mult=ffn_hidden_mult,
                    alpha_init=alpha_init,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, action_dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def _base_action_features(self, base_actions_raw: Tensor, horizon: int) -> Tensor:
        base = base_actions_raw[:, :horizon].to(dtype=torch.float32)
        if base.shape[-1] < self.action_dim:
            base = F.pad(base, (0, self.action_dim - base.shape[-1]))
        return base[..., : self.action_dim]

    def _memory(self, tr_hidden: Tensor, rot_hidden: Tensor, grip_hidden: Tensor) -> Tensor:
        dtype = tr_hidden.dtype
        device = tr_hidden.device
        modality = self.modality_embed.to(device=device, dtype=dtype)
        index = self.coeff_index_embed.to(device=device, dtype=dtype)
        return torch.cat(
            [
                tr_hidden + modality[:, 0:1] + index,
                rot_hidden + modality[:, 1:2] + index,
                grip_hidden + modality[:, 2:3] + index,
            ],
            dim=1,
        )

    def forward(
        self,
        *,
        tr_hidden: Tensor,
        rot_hidden: Tensor,
        grip_hidden: Tensor,
        base_actions_raw: Tensor,
        basis_tr: Tensor,
        basis_rot: Tensor,
        basis_grip: Tensor,
        **_: Tensor,
    ) -> Tensor:
        horizon = min(self.fit_horizon, base_actions_raw.shape[1])
        tr_hidden = tr_hidden.to(dtype=torch.float32)
        rot_hidden = rot_hidden.to(dtype=torch.float32)
        grip_hidden = grip_hidden.to(dtype=torch.float32)
        basis_tr = basis_tr[:horizon].to(device=tr_hidden.device, dtype=tr_hidden.dtype)
        basis_rot = basis_rot[:horizon].to(device=tr_hidden.device, dtype=tr_hidden.dtype)
        basis_grip = basis_grip[:horizon].to(device=tr_hidden.device, dtype=tr_hidden.dtype)
        dphi_tr = _delta_basis(basis_tr)
        dphi_rot = _delta_basis(basis_rot)
        q_basis = (
            self.tr_basis_proj(torch.cat([basis_tr, dphi_tr], dim=-1))
            + self.rot_basis_proj(torch.cat([basis_rot, dphi_rot], dim=-1))
            + self.grip_basis_proj(basis_grip)
        )
        q = q_basis.unsqueeze(0).expand(base_actions_raw.shape[0], -1, -1)
        q = q + self.base_action_proj(self._base_action_features(base_actions_raw, horizon))
        q = q + self.action_time_embed[:, :horizon, :].to(device=q.device, dtype=q.dtype)
        q = self.query_norm(q)
        memory = self._memory(tr_hidden, rot_hidden, grip_hidden)
        for block in self.blocks:
            q = block(q, memory)
        return self.out_proj(self.out_norm(q))


class HiVAResidualFlowBasisAdapter(nn.Module):
    """Inject B-spline basis information with a residual adapter.

    Default fusion is action_emb = fused_emb + alpha * f([fused_emb, basis_emb]).
    The final adapter layer is zero-initialized so the copied SmolVLA suffix path is
    preserved exactly at initialization while gradients still reach the adapter.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        n_ctrl: int,
        hidden_mult: float = 2.0,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        adapter_hidden = max(1, int(round(hidden_size * hidden_mult)))
        self.tr_basis_proj = nn.Linear(2 * n_ctrl, hidden_size)
        self.rot_basis_proj = nn.Linear(2 * n_ctrl, hidden_size)
        self.grip_basis_proj = nn.Linear(n_ctrl, hidden_size)
        self.fused_norm = nn.LayerNorm(hidden_size)
        self.basis_norm = nn.LayerNorm(hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(2 * hidden_size, adapter_hidden),
            nn.SiLU(),
            nn.Linear(adapter_hidden, hidden_size),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def build_basis_emb(
        self,
        *,
        basis_tr: Tensor,
        basis_rot: Tensor,
        basis_grip: Tensor,
        batch_size: int,
        horizon: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        basis_tr = basis_tr[:horizon].to(device=device, dtype=dtype)
        basis_rot = basis_rot[:horizon].to(device=device, dtype=dtype)
        basis_grip = basis_grip[:horizon].to(device=device, dtype=dtype)
        dphi_tr = _delta_basis(basis_tr)
        dphi_rot = _delta_basis(basis_rot)
        basis_emb = (
            self.tr_basis_proj(torch.cat([basis_tr, dphi_tr], dim=-1))
            + self.rot_basis_proj(torch.cat([basis_rot, dphi_rot], dim=-1))
            + self.grip_basis_proj(basis_grip)
        )
        return basis_emb.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, *, fused_emb: Tensor, basis_emb: Tensor) -> Tensor:
        adapter_in = torch.cat([self.fused_norm(fused_emb), self.basis_norm(basis_emb)], dim=-1)
        basis_delta = self.adapter(adapter_in)
        return fused_emb + self.alpha.to(dtype=fused_emb.dtype) * basis_delta


class HiVAResidualFlowCoeffContextAdapter(nn.Module):
    """Optional v2 adapter: residual suffix tokens cross-attend to HiVA coefficient hidden states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        n_ctrl: int,
        num_heads: int = 4,
        ffn_hidden_mult: float = 4.0,
        alpha_init: float = 0.1,
        dropout: float = 0.0,
        zero_init_outputs: bool = True,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"`hidden_size` ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        ffn_hidden = max(1, int(round(hidden_size * ffn_hidden_mult)))
        self.modality_embed = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.coeff_index_embed = nn.Parameter(torch.zeros(1, n_ctrl, hidden_size))
        nn.init.normal_(self.modality_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.coeff_index_embed, mean=0.0, std=0.02)
        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, hidden_size),
        )
        self.ffn_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        if zero_init_outputs:
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)

    def _memory(self, tr_hidden: Tensor, rot_hidden: Tensor, grip_hidden: Tensor) -> Tensor:
        dtype = tr_hidden.dtype
        device = tr_hidden.device
        modality = self.modality_embed.to(device=device, dtype=dtype)
        index = self.coeff_index_embed.to(device=device, dtype=dtype)
        return torch.cat(
            [
                tr_hidden + modality[:, 0:1] + index,
                rot_hidden + modality[:, 1:2] + index,
                grip_hidden + modality[:, 2:3] + index,
            ],
            dim=1,
        )

    def forward(self, suffix_out: Tensor, *, tr_hidden: Tensor, rot_hidden: Tensor, grip_hidden: Tensor) -> Tensor:
        memory = self._memory(
            tr_hidden.to(dtype=suffix_out.dtype),
            rot_hidden.to(dtype=suffix_out.dtype),
            grip_hidden.to(dtype=suffix_out.dtype),
        )
        cross_out, _ = self.cross_attn(
            self.q_norm(suffix_out),
            self.kv_norm(memory),
            self.kv_norm(memory),
            need_weights=False,
        )
        suffix_out = suffix_out + self.cross_alpha.to(dtype=suffix_out.dtype) * cross_out
        suffix_out = suffix_out + self.ffn_alpha.to(dtype=suffix_out.dtype) * self.ffn(self.ffn_norm(suffix_out))
        return suffix_out


class HiVAResidualSmolVLAFlowHead(nn.Module):
    """SmolVLA-style flow head that denoises scaled residual-action latents."""

    def __init__(
        self,
        *,
        hidden_size: int,
        max_action_dim: int,
        n_ctrl: int,
        fit_horizon: int,
        min_period: float,
        max_period: float,
        conditioning: str = "v1_minimal",
        out_head_init: str = "copy",
        out_head_small_init_std: float = 1e-3,
        basis_hidden_mult: float = 2.0,
        basis_alpha_init: float = 0.1,
        coeff_context_alpha_init: float = 0.1,
        coeff_context_ffn_hidden_mult: float = 4.0,
        coeff_context_heads: int = 4,
        coeff_context_dropout: float = 0.0,
        coeff_context_zero_init: bool = True,
    ):
        super().__init__()
        if conditioning not in ("v1_minimal", "v2_coeff_xattn"):
            raise ValueError(f"Unknown residual-flow conditioning mode: {conditioning!r}")
        if out_head_init not in ("copy", "small", "zero"):
            raise ValueError(f"Unknown residual-flow output-head init: {out_head_init!r}")
        self.hidden_size = int(hidden_size)
        self.max_action_dim = int(max_action_dim)
        self.fit_horizon = int(fit_horizon)
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.conditioning = str(conditioning)
        self.out_head_init = str(out_head_init)
        self.out_head_small_init_std = float(out_head_small_init_std)

        self.action_in_proj = nn.Linear(max_action_dim, hidden_size)
        self.base_action_proj = nn.Linear(max_action_dim, hidden_size)
        self.residual_fuse_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * hidden_size, hidden_size)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)
        self.basis_adapter = HiVAResidualFlowBasisAdapter(
            hidden_size=hidden_size,
            n_ctrl=n_ctrl,
            hidden_mult=basis_hidden_mult,
            alpha_init=basis_alpha_init,
        )
        self.coeff_context_adapter = None
        if conditioning == "v2_coeff_xattn":
            self.coeff_context_adapter = HiVAResidualFlowCoeffContextAdapter(
                hidden_size=hidden_size,
                n_ctrl=n_ctrl,
                num_heads=coeff_context_heads,
                ffn_hidden_mult=coeff_context_ffn_hidden_mult,
                alpha_init=coeff_context_alpha_init,
                dropout=coeff_context_dropout,
                zero_init_outputs=coeff_context_zero_init,
            )
        self.residual_action_out_proj = nn.Linear(hidden_size, max_action_dim)
        self._init_fuse_identity()
        if out_head_init == "small":
            nn.init.normal_(self.residual_action_out_proj.weight, mean=0.0, std=self.out_head_small_init_std)
            nn.init.zeros_(self.residual_action_out_proj.bias)
        elif out_head_init == "zero":
            nn.init.zeros_(self.residual_action_out_proj.weight)
            nn.init.zeros_(self.residual_action_out_proj.bias)

    def _init_fuse_identity(self) -> None:
        with torch.no_grad():
            self.residual_fuse_proj.weight.zero_()
            self.residual_fuse_proj.bias.zero_()
            eye = torch.eye(
                self.hidden_size,
                dtype=self.residual_fuse_proj.weight.dtype,
                device=self.residual_fuse_proj.weight.device,
            )
            self.residual_fuse_proj.weight[:, : self.hidden_size].copy_(eye)
            self.residual_fuse_proj.weight[:, self.hidden_size :].zero_()

    def initialize_from_action_modules(
        self,
        *,
        action_in_proj: nn.Linear,
        action_time_mlp_in: nn.Linear,
        action_time_mlp_out: nn.Linear,
        action_out_proj: nn.Linear,
    ) -> None:
        with torch.no_grad():
            self.action_in_proj.weight.copy_(action_in_proj.weight)
            self.action_in_proj.bias.copy_(action_in_proj.bias)
            self.base_action_proj.weight.copy_(action_in_proj.weight)
            self.base_action_proj.bias.copy_(action_in_proj.bias)
            self.action_time_mlp_in.weight.copy_(action_time_mlp_in.weight)
            self.action_time_mlp_in.bias.copy_(action_time_mlp_in.bias)
            self.action_time_mlp_out.weight.copy_(action_time_mlp_out.weight)
            self.action_time_mlp_out.bias.copy_(action_time_mlp_out.bias)
            self._init_fuse_identity()
            if self.out_head_init == "copy":
                self.residual_action_out_proj.weight.copy_(action_out_proj.weight)
                self.residual_action_out_proj.bias.copy_(action_out_proj.bias)
            elif self.out_head_init == "small":
                nn.init.normal_(self.residual_action_out_proj.weight, mean=0.0, std=self.out_head_small_init_std)
                nn.init.zeros_(self.residual_action_out_proj.bias)
            elif self.out_head_init == "zero":
                nn.init.zeros_(self.residual_action_out_proj.weight)
                nn.init.zeros_(self.residual_action_out_proj.bias)

    def initialize_from_smolvla_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, list[str]]:
        """Initialize residual-flow raw-action suffix modules from an original SmolVLA checkpoint."""
        copied: list[str] = []
        missing: list[str] = []
        skipped_shape: list[str] = []

        def copy_linear(src_name: str, dst_name: str, dst_module: nn.Linear) -> None:
            for attr in ("weight", "bias"):
                src_key = f"model.{src_name}.{attr}"
                dst_tensor = getattr(dst_module, attr)
                value = state_dict.get(src_key)
                label = f"{src_key}->{dst_name}.{attr}"
                if value is None:
                    missing.append(label)
                    continue
                if tuple(value.shape) != tuple(dst_tensor.shape):
                    skipped_shape.append(
                        f"{label}: checkpoint{tuple(value.shape)}->model{tuple(dst_tensor.shape)}"
                    )
                    continue
                dst_tensor.copy_(value.to(device=dst_tensor.device, dtype=dst_tensor.dtype))
                copied.append(label)

        with torch.no_grad():
            copy_linear("action_in_proj", "hiva_residual_flow_head.action_in_proj", self.action_in_proj)
            copy_linear("action_in_proj", "hiva_residual_flow_head.base_action_proj", self.base_action_proj)
            copy_linear("action_time_mlp_in", "hiva_residual_flow_head.action_time_mlp_in", self.action_time_mlp_in)
            copy_linear("action_time_mlp_out", "hiva_residual_flow_head.action_time_mlp_out", self.action_time_mlp_out)
            self._init_fuse_identity()
            if self.out_head_init == "copy":
                copy_linear(
                    "action_out_proj",
                    "hiva_residual_flow_head.residual_action_out_proj",
                    self.residual_action_out_proj,
                )
            elif self.out_head_init == "small":
                nn.init.normal_(self.residual_action_out_proj.weight, mean=0.0, std=self.out_head_small_init_std)
                nn.init.zeros_(self.residual_action_out_proj.bias)
            elif self.out_head_init == "zero":
                nn.init.zeros_(self.residual_action_out_proj.weight)
                nn.init.zeros_(self.residual_action_out_proj.bias)

        return {"copied": copied, "missing": missing, "skipped_shape": skipped_shape}

    def _apply_time_mlp(self, token_emb: Tensor, timestep: Tensor) -> Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            self.min_period,
            self.max_period,
            device=token_emb.device,
        ).to(dtype=token_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(token_emb)
        token_time_emb = torch.cat([token_emb, time_emb], dim=-1)
        token_time_emb = self.action_time_mlp_in(token_time_emb)
        token_time_emb = F.silu(token_time_emb)
        return self.action_time_mlp_out(token_time_emb)

    def embed_suffix(
        self,
        *,
        noisy_z: Tensor,
        base_action_norm: Tensor,
        timestep: Tensor,
        basis_tr: Tensor,
        basis_rot: Tensor,
        basis_grip: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual_emb = self.action_in_proj(noisy_z)
        base_emb = self.base_action_proj(base_action_norm)
        fused_emb = self.residual_fuse_proj(torch.cat([residual_emb, base_emb], dim=-1))
        basis_emb = self.basis_adapter.build_basis_emb(
            basis_tr=basis_tr,
            basis_rot=basis_rot,
            basis_grip=basis_grip,
            batch_size=fused_emb.shape[0],
            horizon=fused_emb.shape[1],
            dtype=fused_emb.dtype,
            device=fused_emb.device,
        )
        action_emb = self.basis_adapter(fused_emb=fused_emb, basis_emb=basis_emb)
        action_time_emb = self._apply_time_mlp(action_emb, timestep)
        bsize, horizon = action_time_emb.shape[:2]
        pad_masks = torch.ones(bsize, horizon, dtype=torch.bool, device=action_time_emb.device)
        att_masks = torch.ones(bsize, horizon, dtype=torch.bool, device=action_time_emb.device)
        return action_time_emb, pad_masks, att_masks

    def output_from_suffix(
        self,
        suffix_out: Tensor,
        *,
        tr_hidden: Tensor | None = None,
        rot_hidden: Tensor | None = None,
        grip_hidden: Tensor | None = None,
    ) -> Tensor:
        suffix_out = suffix_out.to(dtype=torch.float32)
        if self.coeff_context_adapter is not None:
            if tr_hidden is None or rot_hidden is None or grip_hidden is None:
                raise RuntimeError("Residual-flow v2 requires coefficient hidden states.")
            suffix_out = self.coeff_context_adapter(
                suffix_out,
                tr_hidden=tr_hidden,
                rot_hidden=rot_hidden,
                grip_hidden=grip_hidden,
            )
        return self.residual_action_out_proj(suffix_out)


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
    safe_theta = theta.clamp_min(1e-6)
    safe_theta2 = safe_theta * safe_theta
    # Keep both torch.where branches finite. The decoded-action loss backprops through this helper,
    # and PyTorch still evaluates the inactive branch where theta can be exactly zero.
    a = torch.where(small, 1 - theta2 / 6 + theta2 * theta2 / 120, torch.sin(theta) / safe_theta)
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - torch.cos(theta)) / safe_theta2,
    )
    k = _skew(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(*rotvec.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def _matrix_to_rotvec(matrix: Tensor) -> Tensor:
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    omega = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    cos_theta = ((trace - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    sin_theta = 0.5 * torch.linalg.norm(omega, dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    theta2 = theta * theta
    small = theta < 1e-4
    scale = torch.where(
        small,
        0.5 + theta2 / 12 + 7 * theta2 * theta2 / 720,
        theta / (2 * sin_theta.clamp_min(1e-7)),
    )
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
    """SmolVLA policy that predicts B-spline coefficient macro-actions and an execution horizon."""

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
        self._maybe_initialize_residual_flow_modules()
        self._sync_model_action_normalization()
        self._duration_execution_map = self._parse_duration_execution_map()
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
        if not self.config.init_smolvla_strict:
            loaded_state = load_safetensors_file(str(model_file), device=str(self.config.device))
            target_state = self.state_dict()
            compatible_state = {}
            skipped_shape = []
            for key, value in loaded_state.items():
                target = target_state.get(key)
                if target is not None and target.shape != value.shape:
                    skipped_shape.append((key, tuple(value.shape), tuple(target.shape)))
                    continue
                compatible_state[key] = value
            missing_keys, unexpected_keys = self.load_state_dict(compatible_state, strict=False)
            if skipped_shape:
                preview = ", ".join(
                    f"{key}: checkpoint{src_shape}->model{dst_shape}"
                    for key, src_shape, dst_shape in skipped_shape[:12]
                )
                suffix = "" if len(skipped_shape) <= 12 else f", ... (+{len(skipped_shape) - 12} more)"
                logging.warning("Skipped %d shape-mismatched init keys: %s%s", len(skipped_shape), preview, suffix)
            if missing_keys:
                logging.warning("Missing key(s) when loading model: %s", set(missing_keys))
            if unexpected_keys:
                logging.warning("Unexpected key(s) when loading model: %s", set(unexpected_keys))
            return

        type(self)._load_as_safetensor(
            self,
            str(model_file),
            self.config.device,
            strict=self.config.init_smolvla_strict,
        )

    def _maybe_initialize_residual_flow_modules(self) -> None:
        if not self.config.hiva_residual_flow_enabled:
            return
        if not hasattr(self.model, "initialize_hiva_residual_flow_from_action_modules"):
            return

        init_path = getattr(self.config, "hiva_residual_flow_init_smolvla_checkpoint_path", None)
        if init_path:
            checkpoint_dir = Path(init_path)
            model_file = checkpoint_dir / SAFETENSORS_SINGLE_FILE
            if model_file.exists() and hasattr(self.model, "initialize_hiva_residual_flow_from_smolvla_state_dict"):
                logging.info("Initializing HiVA residual-flow modules/expert from original SmolVLA checkpoint: %s", checkpoint_dir)
                loaded_state = self._load_residual_flow_init_tensors(model_file)
                report = self.model.initialize_hiva_residual_flow_from_smolvla_state_dict(loaded_state)
                copied = report.get("copied", [])
                missing = report.get("missing", [])
                skipped_shape = report.get("skipped_shape", [])
                if copied:
                    logging.info("Copied %d residual-flow init tensor(s) from %s.", len(copied), checkpoint_dir)
                    if missing:
                        logging.warning("Missing residual-flow init tensor(s): %s", set(missing))
                    if skipped_shape:
                        logging.warning("Skipped residual-flow shape-mismatched tensor(s): %s", set(skipped_shape))
                    return
                logging.warning(
                    "No residual-flow tensors were copied from %s; falling back to current action modules.",
                    checkpoint_dir,
                )
            else:
                logging.warning("Skipping residual-flow SmolVLA init: %s does not exist.", model_file)

        logging.info("Initializing HiVA residual-flow suffix modules from current action modules.")
        self.model.initialize_hiva_residual_flow_from_action_modules()

    def _load_residual_flow_init_tensors(self, model_file: Path) -> dict[str, Tensor]:
        """Load only original-SmVLA tensors needed by the residual-flow branch."""
        prefixes = (
            "model.action_in_proj.",
            "model.action_time_mlp_in.",
            "model.action_time_mlp_out.",
            "model.action_out_proj.",
            "model.vlm_with_expert.lm_expert.",
        )
        loaded_state: dict[str, Tensor] = {}
        with safe_open(str(model_file), framework="pt", device=str(self.config.device)) as handle:
            for key in handle.keys():
                if key.startswith(prefixes):
                    loaded_state[key] = handle.get_tensor(key)
        return loaded_state

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
        self._sync_model_action_normalization()

    def _sync_model_action_normalization(self) -> None:
        if hasattr(self.model, "set_action_normalization"):
            # Keep the learnable residual bound tied to ACTION std from dataset/config stats.
            self.model.set_action_normalization(self._hiva_action_mean.flatten(), self._hiva_action_std.flatten())

    def _normalize_raw_actions(self, raw_actions: Tensor) -> Tensor:
        mean = self._hiva_action_mean.to(device=raw_actions.device, dtype=raw_actions.dtype)
        std = self._hiva_action_std.to(device=raw_actions.device, dtype=raw_actions.dtype)
        return (raw_actions - mean) / (std + 1e-8)

    def _unnormalize_normalized_actions(self, actions: Tensor) -> Tensor:
        """Invert ACTION mean/std normalization for decoded-action training loss."""
        mean = self._hiva_action_mean.to(device=actions.device, dtype=actions.dtype)
        std = self._hiva_action_std.to(device=actions.device, dtype=actions.dtype)
        action_dim = mean.shape[-1]
        return actions[..., :action_dim] * (std + 1e-8) + mean

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._per_env_action_queues = None
        self._last_duration_steps = None
        self._last_execution_horizon = None
        self._last_execution_horizons = None
        self._duration_inference_count = 0
        self._duration_inference_counts_by_env = None
        self._execution_horizon_sum = 0
        self._execution_horizon_sums_by_env = None
        self._execution_horizon_history = []
        self._execution_horizon_histories = None

    def _parse_duration_execution_map(self) -> dict[int, int]:
        raw_map = self.config.hiva_duration_execution_map
        if raw_map is None:
            return {}
        raw_map = str(raw_map).strip()
        if raw_map.lower() in ("", "none", "null"):
            return {}

        mapping: dict[int, int] = {}
        for item in raw_map.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "`hiva_duration_execution_map` entries must use SRC:DST format, "
                    f"for example '15:12'. Got {item!r} in {raw_map!r}."
                )
            src_raw, dst_raw = item.split(":", 1)
            src = int(src_raw.strip())
            dst = int(dst_raw.strip())
            if src <= 0 or dst <= 0:
                raise ValueError("Duration execution map values must be positive integers.")
            mapping[src] = dst
        return mapping

    def _batch_size_from_batch(self, batch: dict[str, Tensor]) -> int:
        if OBS_STATE in batch:
            return int(batch[OBS_STATE].shape[0])
        for value in batch.values():
            if torch.is_tensor(value):
                return int(value.shape[0])
        raise ValueError("Could not infer batch size from policy batch.")

    def _ensure_per_env_action_queues(self, batch_size: int) -> None:
        if self._per_env_action_queues is not None and len(self._per_env_action_queues) == batch_size:
            return

        self._per_env_action_queues = [
            deque(maxlen=self.config.n_action_steps) for _ in range(batch_size)
        ]
        self._last_execution_horizons = [None for _ in range(batch_size)]
        self._duration_inference_counts_by_env = [0 for _ in range(batch_size)]
        self._execution_horizon_sums_by_env = [0 for _ in range(batch_size)]
        self._execution_horizon_histories = [[] for _ in range(batch_size)]

    def _duration_steps_for_execution(self, duration_steps: Tensor) -> Tensor:
        execution_steps = duration_steps.reshape(-1).to(dtype=torch.long)
        if not self._duration_execution_map:
            return execution_steps

        mapped_steps = execution_steps.clone()
        for src, dst in self._duration_execution_map.items():
            mapped_steps = torch.where(
                execution_steps == int(src),
                torch.full_like(mapped_steps, int(dst)),
                mapped_steps,
            )
        return mapped_steps

    def _execution_horizons_from_duration(self, duration_steps: Tensor, actions: Tensor) -> Tensor:
        execution_steps = self._duration_steps_for_execution(duration_steps)
        max_horizon = min(actions.shape[1], int(self.config.n_action_steps))
        return execution_steps.clamp(min=1, max=max_horizon).to(dtype=torch.long)

    def _record_execution_horizon(self, env_ix: int, horizon: int) -> None:
        self._last_execution_horizon = horizon
        self._duration_inference_count += 1
        self._execution_horizon_sum += horizon
        self._execution_horizon_history.append(horizon)

        if self._last_execution_horizons is None:
            return
        self._last_execution_horizons[env_ix] = horizon
        self._duration_inference_counts_by_env[env_ix] += 1
        self._execution_horizon_sums_by_env[env_ix] += horizon
        self._execution_horizon_histories[env_ix].append(horizon)

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
        batch_size = self._batch_size_from_batch(batch)
        self._ensure_per_env_action_queues(batch_size)

        empty_envs = [
            env_ix for env_ix, queue in enumerate(self._per_env_action_queues) if len(queue) == 0
        ]
        if empty_envs:
            actions, duration_steps = self._get_action_chunk(batch, noise)
            execution_horizons = self._execution_horizons_from_duration(duration_steps, actions)
            if execution_horizons.numel() != len(self._per_env_action_queues):
                raise ValueError(
                    "Duration predictions must match evaluation batch size. "
                    f"Got {execution_horizons.numel()} predictions for "
                    f"{len(self._per_env_action_queues)} env queues."
                )
            for env_ix in empty_envs:
                horizon = int(execution_horizons[env_ix].item())
                self._record_execution_horizon(env_ix, horizon)
                self._per_env_action_queues[env_ix].extend(actions[env_ix, :horizon])

        return torch.stack([queue.popleft() for queue in self._per_env_action_queues], dim=0)

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        if self.config.adapt_to_pi_aloha:
            raise NotImplementedError("HiVA coefficient SmolVLA is currently implemented for LIBERO actions.")

        required_keys = (
            "hiva_theta_tr_raw",
            "hiva_theta_rot_raw",
            "hiva_theta_grip_raw",
            "duration_class",
            "duration_label",
        )
        needs_future_actions = (
            self.config.hiva_decoded_action_loss_weight > 0
            or self.config.hiva_residual_flow_enabled
        )
        if needs_future_actions and ACTION not in batch:
            raise KeyError(
                f"`{ACTION}` is missing from the training batch. "
                "Decoded action loss / residual-flow training needs the future raw-action target chunk."
            )
        for key in required_keys:
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
        duration_class_target = batch["duration_class"].to(device=state.device, dtype=torch.long)
        duration_label_target = batch["duration_label"].to(device=state.device, dtype=torch.float32)
        target_actions_raw = None
        target_actions_is_pad = None
        hiva_fit_real_steps = None
        if needs_future_actions:
            target_actions_norm = self.prepare_action(batch).to(device=state.device, dtype=torch.float32)
            target_actions_raw = self._unnormalize_normalized_actions(target_actions_norm)
            if "action_is_pad" in batch and batch["action_is_pad"] is not None:
                target_actions_is_pad = batch["action_is_pad"].to(device=state.device, dtype=torch.bool)
            if "hiva_fit_real_steps" in batch:
                hiva_fit_real_steps = batch["hiva_fit_real_steps"].to(device=state.device, dtype=torch.long)
            elif "hiva_real_steps" in batch:
                hiva_fit_real_steps = batch["hiva_real_steps"].to(device=state.device, dtype=torch.long)

        per_sample_loss, loss_dict = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            targets,
            duration_class_target,
            duration_label_target,
            target_actions_raw=target_actions_raw,
            target_actions_is_pad=target_actions_is_pad,
            hiva_fit_real_steps=hiva_fit_real_steps,
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
            "hiva_duration_token|hiva_duration_head|hiva_duration_head\\..*|"
            "hiva_duration_in_proj|hiva_duration_out_proj|"
            "hiva_residual_head|hiva_residual_head\\..*|"
            "hiva_residual_flow_head|hiva_residual_flow_head\\..*|"
            "action_time_mlp_in|action_time_mlp_out"
        )
        if (
            self.config.hiva_residual_flow_enabled
            and getattr(self.config, "hiva_residual_flow_use_separate_expert", False)
        ):
            expert_target = r"model\.hiva_residual_lm_expert\..*\.(q|v)_proj"
        else:
            expert_target = r"model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj"
        target_modules = rf"({expert_target}|model\.({common}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }


class HiVACoeffVLAFlowMatching(VLAFlowMatching):
    """Flow matching over HiVA B-spline coefficient tokens and optional continuous duration scalar."""

    def __init__(self, config: HiVACoeffSmolVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config = config
        hidden = self.vlm_with_expert.expert_hidden_size
        for module in (self.action_in_proj, self.action_out_proj):
            for param in module.parameters():
                param.requires_grad = False

        self.hiva_tr_in_proj = nn.Linear(3, hidden)
        self.hiva_rot_in_proj = nn.Linear(3, hidden)
        self.hiva_grip_in_proj = nn.Linear(1, hidden)
        self.hiva_tr_out_proj = nn.Linear(hidden, 3)
        self.hiva_rot_out_proj = nn.Linear(hidden, 3)
        self.hiva_grip_out_proj = nn.Linear(hidden, 1)

        self.hiva_duration_token = None
        self.hiva_duration_head = None
        self.hiva_duration_in_proj = None
        self.hiva_duration_out_proj = None
        if config.hiva_duration_prediction_type == "categorical":
            if config.hiva_duration_readout == "token":
                self.hiva_duration_token = nn.Parameter(torch.zeros(1, 1, hidden))
                nn.init.normal_(self.hiva_duration_token, mean=0.0, std=0.02)
                if config.hiva_duration_head_type == "linear":
                    self.hiva_duration_head = nn.Linear(hidden, len(config.hiva_duration_classes))
                elif config.hiva_duration_head_type == "residual_ffn":
                    self.hiva_duration_head = HiVAResidualFFNDurationHead(
                        hidden,
                        len(config.hiva_duration_classes),
                        hidden_mult=config.hiva_duration_ffn_hidden_mult,
                        alpha_init=config.hiva_duration_ffn_alpha_init,
                    )
                else:
                    raise ValueError(f"Unknown hiva_duration_head_type={config.hiva_duration_head_type!r}.")
            elif config.hiva_duration_readout == "coeff_modality_pool":
                self.hiva_duration_head = HiVACoeffModalityPoolDurationHead(
                    hidden,
                    len(config.hiva_duration_classes),
                    hidden_mult=config.hiva_duration_ffn_hidden_mult,
                    alpha_init=config.hiva_duration_ffn_alpha_init,
                )
            else:
                raise ValueError(f"Unknown hiva_duration_readout={config.hiva_duration_readout!r}.")
        else:
            self.hiva_duration_in_proj = nn.Linear(1, hidden)
            self.hiva_duration_out_proj = nn.Linear(hidden, 1)

        action_dim = self.config.action_feature.shape[0] if self.config.action_feature is not None else 7
        self.register_buffer("hiva_action_mean_for_residual", torch.zeros(1, 1, action_dim), persistent=True)
        self.register_buffer("hiva_action_std_for_residual", torch.ones(1, 1, action_dim), persistent=True)
        self.hiva_residual_head = None
        if config.hiva_residual_enabled:
            common_residual_kwargs = {
                "hidden_size": hidden,
                "fit_horizon": config.hiva_residual_horizon,
                "action_dim": action_dim,
                "ffn_hidden_mult": config.hiva_residual_ffn_hidden_mult,
                "alpha_init": config.hiva_residual_alpha_init,
                "zero_init": config.hiva_residual_zero_init,
            }
            if config.hiva_residual_mode == "token_to_time":
                self.hiva_residual_head = HiVAFullHorizonResidualHead(
                    hidden_size=hidden,
                    num_coeff_tokens=3 * config.hiva_k,
                    fit_horizon=config.hiva_residual_horizon,
                    action_dim=action_dim,
                    ffn_hidden_mult=config.hiva_residual_ffn_hidden_mult,
                    token_time_hidden_mult=config.hiva_residual_token_time_hidden_mult,
                    alpha_init=config.hiva_residual_alpha_init,
                    zero_init=config.hiva_residual_zero_init,
                )
            elif config.hiva_residual_mode == "basis_hidden_action":
                self.hiva_residual_head = HiVABasisHiddenActionResidualHead(**common_residual_kwargs)
            elif config.hiva_residual_mode == "basis_xattn_transformer":
                self.hiva_residual_head = HiVABasisXAttnTransformerResidualHead(
                    hidden_size=hidden,
                    n_ctrl=config.hiva_k,
                    fit_horizon=config.hiva_residual_horizon,
                    action_dim=action_dim,
                    num_blocks=config.hiva_residual_num_blocks,
                    num_heads=config.hiva_residual_cross_attn_heads,
                    ffn_hidden_mult=config.hiva_residual_ffn_hidden_mult,
                    alpha_init=config.hiva_residual_alpha_init,
                    dropout=config.hiva_residual_attn_dropout,
                    zero_init=config.hiva_residual_zero_init,
                )
            else:
                raise ValueError(f"Unknown hiva_residual_mode={config.hiva_residual_mode!r}.")

        self.hiva_residual_flow_head = None
        self.hiva_residual_lm_expert = None
        if config.hiva_residual_flow_enabled:
            if getattr(config, "hiva_residual_flow_use_separate_expert", True):
                # Start from the same architecture as the current action expert. The policy-level
                # initializer will overwrite this copy with original SmolVLA expert weights when
                # hiva_residual_flow_init_smolvla_checkpoint_path is provided.
                self.hiva_residual_lm_expert = copy.deepcopy(self.vlm_with_expert.lm_expert)
            self.hiva_residual_flow_head = HiVAResidualSmolVLAFlowHead(
                hidden_size=hidden,
                max_action_dim=config.max_action_dim,
                n_ctrl=config.hiva_k,
                fit_horizon=config.hiva_residual_flow_horizon,
                min_period=config.min_period,
                max_period=config.max_period,
                conditioning=config.hiva_residual_flow_conditioning,
                out_head_init=config.hiva_residual_flow_out_head_init,
                out_head_small_init_std=config.hiva_residual_flow_small_init_std,
                basis_hidden_mult=config.hiva_residual_flow_basis_hidden_mult,
                basis_alpha_init=config.hiva_residual_flow_basis_alpha_init,
                coeff_context_alpha_init=config.hiva_residual_flow_coeff_context_alpha_init,
                coeff_context_ffn_hidden_mult=config.hiva_residual_flow_coeff_context_ffn_hidden_mult,
                coeff_context_heads=config.hiva_residual_flow_coeff_context_heads,
                coeff_context_dropout=config.hiva_residual_flow_coeff_context_dropout,
                coeff_context_zero_init=config.hiva_residual_flow_coeff_context_zero_init,
            )
            self.initialize_hiva_residual_flow_from_action_modules()

        self._coeff_summary = self._load_coeff_summary()
        self._register_coeff_stats()
        self._register_duration_stats()
        self._register_basis_buffers()

    def set_action_normalization(self, mean: Tensor, std: Tensor) -> None:
        action_dim = self.hiva_action_std_for_residual.shape[-1]
        mean_tensor = torch.as_tensor(
            mean,
            dtype=torch.float32,
            device=self.hiva_action_mean_for_residual.device,
        ).flatten()[:action_dim]
        std_tensor = torch.as_tensor(
            std,
            dtype=torch.float32,
            device=self.hiva_action_std_for_residual.device,
        ).flatten()[:action_dim]
        self.hiva_action_mean_for_residual.copy_(mean_tensor.view(1, 1, action_dim))
        self.hiva_action_std_for_residual.copy_(std_tensor.view(1, 1, action_dim).clamp_min(1e-6))

    @property
    def hiva_suffix_len(self) -> int:
        duration_len = (
            0
            if (
                self.config.hiva_duration_prediction_type == "categorical"
                and self.config.hiva_duration_readout == "coeff_modality_pool"
            )
            else 1
        )
        return duration_len + 3 * self.config.hiva_k

    def _load_coeff_summary(self) -> dict | None:
        if self.config.hiva_coeff_sidecar_summary_path:
            summary_path = Path(self.config.hiva_coeff_sidecar_summary_path)
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logging.warning("HiVA coefficient summary does not exist: %s", summary_path)
        return None

    def _register_coeff_stats(self) -> None:
        stats = None if self._coeff_summary is None else self._coeff_summary.get("coef_stats")

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

    def _infer_duration_mean_std(self) -> tuple[float, float]:
        if self.config.hiva_duration_mean is not None and self.config.hiva_duration_std is not None:
            return float(self.config.hiva_duration_mean), max(float(self.config.hiva_duration_std), 1e-6)

        hist = None if self._coeff_summary is None else self._coeff_summary.get("duration_hist")
        if hist:
            values = []
            weights = []
            for key, count in hist.items():
                values.append(float(key))
                weights.append(float(count))
            value_t = torch.tensor(values, dtype=torch.float64)
            weight_t = torch.tensor(weights, dtype=torch.float64)
            denom = weight_t.sum().clamp_min(1.0)
            mean = float((value_t * weight_t).sum() / denom)
            var = float(((value_t - mean) ** 2 * weight_t).sum() / denom)
            return mean, max(var**0.5, 1e-6)

        values = torch.tensor(self.config.hiva_duration_classes, dtype=torch.float64)
        return float(values.mean()), max(float(values.std(unbiased=False)), 1e-6)

    def _register_duration_stats(self) -> None:
        mean, std = self._infer_duration_mean_std()
        self.register_buffer("hiva_duration_mean", torch.tensor(mean, dtype=torch.float32), persistent=True)
        self.register_buffer("hiva_duration_std", torch.tensor(std, dtype=torch.float32), persistent=True)

    def _register_basis_buffers(self) -> None:
        # Keep legacy duration-specific bases so older sidecars/checkpoints continue to work.
        for duration in self.config.hiva_duration_classes:
            basis = _clamped_bspline_basis(
                int(duration),
                n_ctrl=self.config.hiva_k,
                degree=self.config.hiva_degree,
            )
            self.register_buffer(f"hiva_basis_{duration}", basis, persistent=False)
            self.register_buffer(
                f"hiva_basis_tr_{duration}",
                _clamped_bspline_basis(
                    int(duration),
                    n_ctrl=self.config.hiva_k,
                    degree=self.config.hiva_degree_tr,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"hiva_basis_rot_{duration}",
                _clamped_bspline_basis(
                    int(duration),
                    n_ctrl=self.config.hiva_k,
                    degree=self.config.hiva_degree_rot,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"hiva_basis_grip_{duration}",
                _clamped_bspline_basis(
                    int(duration),
                    n_ctrl=self.config.hiva_k,
                    degree=self.config.hiva_degree_grip,
                ),
                persistent=False,
            )

        canonical_basis = _clamped_bspline_basis(
            int(self.config.hiva_fit_horizon),
            n_ctrl=self.config.hiva_k,
            degree=self.config.hiva_degree,
        )
        self.register_buffer("hiva_basis_canonical", canonical_basis, persistent=False)
        self.register_buffer(
            "hiva_basis_canonical_tr",
            _clamped_bspline_basis(
                int(self.config.hiva_fit_horizon),
                n_ctrl=self.config.hiva_k,
                degree=self.config.hiva_degree_tr,
            ),
            persistent=False,
        )
        self.register_buffer(
            "hiva_basis_canonical_rot",
            _clamped_bspline_basis(
                int(self.config.hiva_fit_horizon),
                n_ctrl=self.config.hiva_k,
                degree=self.config.hiva_degree_rot,
            ),
            persistent=False,
        )
        self.register_buffer(
            "hiva_basis_canonical_grip",
            _clamped_bspline_basis(
                int(self.config.hiva_fit_horizon),
                n_ctrl=self.config.hiva_k,
                degree=self.config.hiva_degree_grip,
            ),
            persistent=False,
        )

    def _uses_canonical_basis(self) -> bool:
        return self.config.hiva_basis_mode in ("canonical_hp", "canonical_mt", "canonical_lp_mt")

    def _basis(self, duration: int, *, device, dtype) -> Tensor:
        return getattr(self, f"hiva_basis_{int(duration)}").to(device=device, dtype=dtype)

    def _basis_set(self, duration: int, *, device, dtype) -> tuple[Tensor, Tensor, Tensor]:
        duration = int(duration)
        return (
            getattr(self, f"hiva_basis_tr_{duration}").to(device=device, dtype=dtype),
            getattr(self, f"hiva_basis_rot_{duration}").to(device=device, dtype=dtype),
            getattr(self, f"hiva_basis_grip_{duration}").to(device=device, dtype=dtype),
        )

    def _canonical_basis(self, *, device, dtype) -> Tensor:
        return self.hiva_basis_canonical.to(device=device, dtype=dtype)

    def _canonical_basis_set(self, *, device, dtype) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.hiva_basis_canonical_tr.to(device=device, dtype=dtype),
            self.hiva_basis_canonical_rot.to(device=device, dtype=dtype),
            self.hiva_basis_canonical_grip.to(device=device, dtype=dtype),
        )

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

    def normalize_duration(self, duration_raw: Tensor) -> Tensor:
        duration_raw = duration_raw.to(dtype=torch.float32)
        if self.config.hiva_duration_cont_norm == "bounded":
            return 2.0 * (duration_raw - 1.0) / float(self.config.hiva_dmax - 1) - 1.0
        if self.config.hiva_duration_cont_norm == "mean_std":
            return (duration_raw - self.hiva_duration_mean) / self.hiva_duration_std.clamp_min(1e-6)
        raise ValueError(f"Unknown hiva_duration_cont_norm={self.config.hiva_duration_cont_norm!r}.")

    def unnormalize_duration(self, duration_norm: Tensor) -> Tensor:
        duration_norm = duration_norm.to(dtype=torch.float32)
        if self.config.hiva_duration_cont_norm == "bounded":
            return 1.0 + 0.5 * (duration_norm + 1.0) * float(self.config.hiva_dmax - 1)
        if self.config.hiva_duration_cont_norm == "mean_std":
            return duration_norm * self.hiva_duration_std + self.hiva_duration_mean
        raise ValueError(f"Unknown hiva_duration_cont_norm={self.config.hiva_duration_cont_norm!r}.")

    def _duration_steps_from_continuous(self, duration_raw: Tensor) -> Tensor:
        return torch.floor(duration_raw.reshape(-1)).long().clamp(min=1, max=int(self.config.hiva_dmax))

    def _sample_coeff_noise(self, coeffs: HiVACoeffTargets) -> HiVACoeffTargets:
        return {key: self.sample_noise(value.shape, value.device) for key, value in coeffs.items()}

    def _mix_coeffs(self, targets: HiVACoeffTargets, noise: HiVACoeffTargets, time: Tensor):
        time_expanded = time[:, None, None]
        x_t = {key: time_expanded * noise[key] + (1 - time_expanded) * targets[key] for key in targets}
        u_t = {key: noise[key] - targets[key] for key in targets}
        return x_t, u_t

    def _mix_duration(self, duration_target_norm: Tensor, duration_noise: Tensor, time: Tensor):
        time_expanded = time[:, None, None]
        duration_t = time_expanded * duration_noise + (1 - time_expanded) * duration_target_norm
        duration_u = duration_noise - duration_target_norm
        return duration_t, duration_u

    def duration_noisy_weights(self, time: Tensor) -> Tensor:
        sigma = self.config.hiva_duration_noisy_sigma
        return torch.exp(-(time.to(dtype=torch.float32) ** 2) / (sigma**2))

    def _hiva_suffix_att_list(self, suffix_len: int) -> list[int]:
        """Return the cumulative block-mask placeholder for the HiVA coefficient suffix.

        Token/continuous suffix order is [duration][translation coeffs][rotation coeffs][gripper coeffs].
        The coeff_modality_pool categorical readout removes the duration token and uses
        [translation coeffs][rotation coeffs][gripper coeffs].
        SmolVLA's make_att_2d_masks treats 1 as a new cumulative attention block and 0 as
        sharing the previous block. For asymmetric duration_reads_coeffs, this list is only a
        placeholder because the real suffix mask is built by _make_hiva_suffix_2d_masks.
        """
        if suffix_len < 1:
            raise ValueError(f"HiVA suffix must contain at least one token, got {suffix_len}.")
        if (
            self.config.hiva_duration_prediction_type == "categorical"
            and self.config.hiva_duration_readout == "coeff_modality_pool"
        ):
            return [1] + [0] * (suffix_len - 1)
        if suffix_len < 2:
            raise ValueError(f"HiVA suffix must contain duration plus coefficient tokens, got {suffix_len}.")

        mode = self.config.hiva_suffix_attention
        if mode == "duration_prefix":
            # Duration attends prefix + itself; coefficient tokens attend prefix + duration + all coeffs.
            return [1, 1] + [0] * (suffix_len - 2)
        if mode == "duration_reads_coeffs":
            # Placeholder only: custom 2D mask lets duration read coeffs while coeffs ignore duration.
            return [1] + [0] * (suffix_len - 1)
        if mode == "full":
            # One bidirectional suffix block: duration and coefficients all attend each other.
            return [1] + [0] * (suffix_len - 1)
        if mode == "causal":
            # Original cumulative/causal suffix ordering.
            return [1] * suffix_len
        raise ValueError(f"Unknown hiva_suffix_attention={mode!r}.")

    def _make_hiva_suffix_2d_masks(self, suffix_pad_masks: Tensor, suffix_att_masks: Tensor) -> Tensor:
        """Build suffix self-attention masks for HiVA coefficient tokens.

        Most modes use SmolVLA's cumulative block-mask helper. duration_reads_coeffs is asymmetric:
        duration attends itself and all coefficient keys, while coefficient queries attend the
        coefficient block but not the duration key. Prefix/context attention is added separately, so
        all suffix queries still read the multimodal prefix/state tokens.
        """
        if (
            self.config.hiva_duration_prediction_type == "categorical"
            and self.config.hiva_duration_readout == "coeff_modality_pool"
        ):
            return make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        if self.config.hiva_suffix_attention != "duration_reads_coeffs":
            return make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        if suffix_pad_masks.ndim != 2:
            raise ValueError(f"suffix_pad_masks must be rank 2, got rank {suffix_pad_masks.ndim}.")
        batch_size, suffix_len = suffix_pad_masks.shape
        if suffix_len < 2:
            raise ValueError(f"HiVA suffix must contain duration plus coefficient tokens, got {suffix_len}.")

        valid = suffix_pad_masks.bool()
        suffix_att_2d_masks = torch.zeros(
            batch_size,
            suffix_len,
            suffix_len,
            dtype=torch.bool,
            device=suffix_pad_masks.device,
        )
        suffix_att_2d_masks[:, 0, :] = valid[:, 0:1] & valid
        coeff_valid = valid[:, 1:]
        suffix_att_2d_masks[:, 1:, 1:] = coeff_valid[:, :, None] & coeff_valid[:, None, :]
        return suffix_att_2d_masks

    def _apply_time_mlp(self, token_emb: Tensor, timestep: Tensor) -> Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=token_emb.device,
        ).to(dtype=token_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(token_emb)
        token_time_emb = torch.cat([token_emb, time_emb], dim=-1)
        token_time_emb = self.action_time_mlp_in(token_time_emb)
        token_time_emb = F.silu(token_time_emb)
        return self.action_time_mlp_out(token_time_emb)

    def embed_hiva_suffix(self, coeffs: HiVACoeffTargets, timestep: Tensor, duration_t: Tensor | None = None):
        tr_emb = self.hiva_tr_in_proj(coeffs["tr"])
        rot_emb = self.hiva_rot_in_proj(coeffs["rot"])
        grip_emb = self.hiva_grip_in_proj(coeffs["grip"])
        coeff_emb = torch.cat([tr_emb, rot_emb, grip_emb], dim=1)
        coeff_time_emb = self._apply_time_mlp(coeff_emb, timestep)

        device = coeff_emb.device
        bsize = coeff_emb.shape[0]
        dtype = coeff_emb.dtype
        if self.config.hiva_duration_prediction_type == "continuous_fm":
            if duration_t is None:
                raise ValueError("continuous_fm duration requires `duration_t` in embed_hiva_suffix.")
            duration_t = duration_t.to(device=device, dtype=dtype).reshape(bsize, 1, 1)
            duration_emb = self.hiva_duration_in_proj(duration_t)
            duration_emb = self._apply_time_mlp(duration_emb, timestep)
            embs = torch.cat([duration_emb, coeff_time_emb], dim=1)
        elif self.config.hiva_duration_readout == "token":
            duration_emb = self.hiva_duration_token.to(device=device, dtype=dtype).expand(bsize, -1, -1)
            embs = torch.cat([duration_emb, coeff_time_emb], dim=1)
        else:
            embs = coeff_time_emb

        pad_masks = torch.ones(bsize, embs.shape[1], dtype=torch.bool, device=device)
        att = self._hiva_suffix_att_list(embs.shape[1])
        att_masks = torch.tensor(att, dtype=torch.bool, device=device)[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    def _split_hiva_suffix_out(self, suffix_out: Tensor) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
        suffix_out = suffix_out[:, -self.hiva_suffix_len :].to(dtype=torch.float32)
        if (
            self.config.hiva_duration_prediction_type == "categorical"
            and self.config.hiva_duration_readout == "coeff_modality_pool"
        ):
            duration_out = None
            coeff_out = suffix_out
        else:
            duration_out = suffix_out[:, 0]
            coeff_out = suffix_out[:, 1:]
        k = self.config.hiva_k
        tr_out = coeff_out[:, :k]
        rot_out = coeff_out[:, k : 2 * k]
        grip_out = coeff_out[:, 2 * k : 3 * k]
        return duration_out, tr_out, rot_out, grip_out

    def _heads_from_suffix_out(self, suffix_out: Tensor) -> tuple[HiVACoeffTargets, Tensor, Tensor]:
        duration_out, tr_out, rot_out, grip_out = self._split_hiva_suffix_out(suffix_out)
        pred = {
            "tr": self.hiva_tr_out_proj(tr_out),
            "rot": self.hiva_rot_out_proj(rot_out),
            "grip": self.hiva_grip_out_proj(grip_out),
        }
        if self.config.hiva_duration_prediction_type == "continuous_fm":
            if duration_out is None:
                raise RuntimeError("Continuous duration mode expected a duration hidden state.")
            duration_out_value = self.hiva_duration_out_proj(duration_out).reshape(-1, 1, 1)
        elif self.config.hiva_duration_readout == "coeff_modality_pool":
            duration_out_value = self.hiva_duration_head(tr_out, rot_out, grip_out)
        else:
            if duration_out is None:
                raise RuntimeError("Token duration readout expected a duration hidden state.")
            duration_out_value = self.hiva_duration_head(duration_out)
        coeff_hidden = torch.cat([tr_out, rot_out, grip_out], dim=1)
        return pred, duration_out_value, coeff_hidden

    def _forward_hiva_suffix_with_prefix_cache(
        self,
        prefix_pad_masks,
        past_key_values,
        coeffs: HiVACoeffTargets,
        timestep: Tensor,
        duration_t: Tensor | None = None,
        use_cache=None,
    ) -> tuple[HiVACoeffTargets, Tensor, Tensor]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_hiva_suffix(
            coeffs, timestep, duration_t=duration_t
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = self._make_hiva_suffix_2d_masks(suffix_pad_masks, suffix_att_masks)
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

    def _residual_scale_tensor(self, *, device, dtype, action_dim: int) -> Tensor:
        # Default to ACTION std from dataset/config stats. Explicit scale knobs override by modality.
        std = self.hiva_action_std_for_residual.to(device=device, dtype=dtype)[..., :action_dim]
        scale = std.clone()
        if self.config.hiva_residual_scale_tr is not None and action_dim > 0:
            scale[..., : min(3, action_dim)] = float(self.config.hiva_residual_scale_tr)
        if self.config.hiva_residual_scale_rot is not None and action_dim > 3:
            scale[..., 3 : min(6, action_dim)] = float(self.config.hiva_residual_scale_rot)
        if self.config.hiva_residual_scale_grip is not None and action_dim > 6:
            scale[..., 6:7] = float(self.config.hiva_residual_scale_grip)
        return (scale * float(self.config.hiva_residual_scale_mult)).clamp_min(1e-8)

    def _split_coeff_hidden(self, coeff_hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        k = self.config.hiva_k
        return coeff_hidden[:, :k], coeff_hidden[:, k : 2 * k], coeff_hidden[:, 2 * k : 3 * k]

    def initialize_hiva_residual_flow_from_action_modules(self) -> None:
        if self.hiva_residual_flow_head is None:
            return
        self.hiva_residual_flow_head.initialize_from_action_modules(
            action_in_proj=self.action_in_proj,
            action_time_mlp_in=self.action_time_mlp_in,
            action_time_mlp_out=self.action_time_mlp_out,
            action_out_proj=self.action_out_proj,
        )

    def initialize_hiva_residual_flow_from_smolvla_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, list[str]]:
        """Initialize residual-flow head and optional separate expert from original SmolVLA."""
        copied: list[str] = []
        missing: list[str] = []
        skipped_shape: list[str] = []
        if self.hiva_residual_flow_head is not None:
            head_report = self.hiva_residual_flow_head.initialize_from_smolvla_state_dict(state_dict)
            copied.extend(head_report.get("copied", []))
            missing.extend(head_report.get("missing", []))
            skipped_shape.extend(head_report.get("skipped_shape", []))

        if self.hiva_residual_lm_expert is not None:
            expert_state = self.hiva_residual_lm_expert.state_dict()
            compatible_state: dict[str, Tensor] = {}
            expert_prefix = "model.vlm_with_expert.lm_expert."
            for key, target in expert_state.items():
                src_key = expert_prefix + key
                value = state_dict.get(src_key)
                label = f"{src_key}->hiva_residual_lm_expert.{key}"
                if value is None:
                    missing.append(label)
                    continue
                if tuple(value.shape) != tuple(target.shape):
                    skipped_shape.append(
                        f"{label}: checkpoint{tuple(value.shape)}->model{tuple(target.shape)}"
                    )
                    continue
                compatible_state[key] = value.to(device=target.device, dtype=target.dtype)
                copied.append(label)
            if compatible_state:
                self.hiva_residual_lm_expert.load_state_dict(compatible_state, strict=False)

        return {"copied": copied, "missing": missing, "skipped_shape": skipped_shape}

    def _forward_with_residual_expert(
        self,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        """Run residual suffix tokens through the original-SmVLA residual expert.

        The VLM/prefix side remains shared with the Stage-0 HiVA coefficient model.
        When a separate residual expert is enabled, we build the VLM/expert layer pair
        explicitly instead of temporarily monkey-patching ``self.vlm_with_expert.lm_expert``.
        This keeps the registered Stage-0 coefficient expert untouched while the residual
        suffix uses its own original-SmVLA expert weights.
        """
        if self.hiva_residual_lm_expert is None:
            return self.vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided for residual-expert forward.")

        models = [self.vlm_with_expert.get_vlm_model().text_model, self.hiva_residual_lm_expert]
        model_layers = self.vlm_with_expert.get_model_layers(models)
        batch_size = None
        for hidden_states in inputs_embeds:
            if hidden_states is not None:
                batch_size = hidden_states.shape[0]
                break
        if batch_size is None:
            raise ValueError("At least one residual-expert input embedding must be non-None.")

        num_layers = self.vlm_with_expert.num_vlm_layers
        head_dim = self.vlm_with_expert.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.vlm_with_expert.attention_mode
                or (
                    self.vlm_with_expert.self_attn_every_n_layers > 0
                    and layer_idx % self.vlm_with_expert.self_attn_every_n_layers == 0
                )
            ):
                att_outputs, past_key_values = self.vlm_with_expert.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.vlm_with_expert.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue

                end = start + hidden_states.shape[1]
                layer_dtype = layer.self_attn.o_proj.weight.dtype
                if hidden_states.dtype != layer_dtype:
                    hidden_states = hidden_states.to(dtype=layer_dtype)
                if att_output.dtype != layer_dtype:
                    att_output = att_output.to(dtype=layer_dtype)
                att_out = att_output[:, start:end]
                out_emb = layer.self_attn.o_proj(att_out)
                out_emb = out_emb + hidden_states
                after_first_residual = out_emb.clone()
                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)
                out_emb = out_emb + after_first_residual
                outputs_embeds.append(out_emb)
                start = end if len(att_outputs) == 1 else 0
            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                outputs_embeds.append(models[i].norm(hidden_states))
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def _normalize_raw_actions_for_residual(self, raw_actions: Tensor) -> Tensor:
        mean = self.hiva_action_mean_for_residual.to(device=raw_actions.device, dtype=raw_actions.dtype)
        std = self.hiva_action_std_for_residual.to(device=raw_actions.device, dtype=raw_actions.dtype)
        return (raw_actions - mean) / std.clamp_min(1e-8)

    def _unnormalize_actions_for_residual(self, norm_actions: Tensor) -> Tensor:
        mean = self.hiva_action_mean_for_residual.to(device=norm_actions.device, dtype=norm_actions.dtype)
        std = self.hiva_action_std_for_residual.to(device=norm_actions.device, dtype=norm_actions.dtype)
        return norm_actions * std.clamp_min(1e-8) + mean

    def _residual_flow_scale_tensor(self, *, device, dtype, action_dim: int) -> Tensor:
        values = torch.ones(action_dim, dtype=dtype, device=device)
        if action_dim > 0:
            values[: min(3, action_dim)] = float(self.config.hiva_residual_flow_scale_tr)
        if action_dim > 3:
            values[3 : min(6, action_dim)] = float(self.config.hiva_residual_flow_scale_rot)
        if action_dim > 6:
            values[6:7] = float(self.config.hiva_residual_flow_scale_grip)
        return values.view(1, 1, action_dim).clamp_min(1e-8)

    def _residual_flow_zero_logs(self, *, device: torch.device | None = None) -> tuple[Tensor | None, dict]:
        zero_tensor = None if device is None else torch.zeros((), device=device)
        return zero_tensor, {
            "hiva_residual_flow_enabled": float(self.config.hiva_residual_flow_enabled),
            "hiva_residual_flow_conditioning": self.config.hiva_residual_flow_conditioning,
            "hiva_residual_flow_conditioning_index": {
                "v1_minimal": 1,
                "v2_coeff_xattn": 2,
            }.get(self.config.hiva_residual_flow_conditioning, -1),
            "hiva_residual_flow_loss": 0.0,
            "hiva_residual_flow_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_loss": 0.0,
            "hiva_residual_flow_decoded_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_tr_loss": 0.0,
            "hiva_residual_flow_decoded_tr_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_rot_loss": 0.0,
            "hiva_residual_flow_decoded_rot_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_grip_loss": 0.0,
            "hiva_residual_flow_decoded_grip_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_prefix_tr_loss": 0.0,
            "hiva_residual_flow_decoded_prefix_rot_loss": 0.0,
            "hiva_residual_flow_decoded_prefix_grip_loss": 0.0,
            "hiva_residual_flow_decoded_prefix_action_loss": 0.0,
            "hiva_residual_flow_decoded_prefix_tr_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_prefix_rot_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_prefix_grip_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_prefix_action_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_post_duration_tr_loss": 0.0,
            "hiva_residual_flow_decoded_post_duration_rot_loss": 0.0,
            "hiva_residual_flow_decoded_post_duration_grip_loss": 0.0,
            "hiva_residual_flow_decoded_post_duration_action_loss": 0.0,
            "hiva_residual_flow_decoded_post_duration_tr_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_post_duration_rot_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_post_duration_grip_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_post_duration_action_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_preview_tr_loss": 0.0,
            "hiva_residual_flow_decoded_preview_rot_loss": 0.0,
            "hiva_residual_flow_decoded_preview_grip_loss": 0.0,
            "hiva_residual_flow_decoded_preview_action_loss": 0.0,
            "hiva_residual_flow_decoded_preview_tr_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_preview_rot_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_preview_grip_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_preview_action_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_terminal_tr_loss": 0.0,
            "hiva_residual_flow_decoded_terminal_rot_loss": 0.0,
            "hiva_residual_flow_decoded_terminal_grip_loss": 0.0,
            "hiva_residual_flow_decoded_terminal_action_loss": 0.0,
            "hiva_residual_flow_decoded_terminal_tr_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_terminal_rot_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_terminal_grip_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_terminal_action_loss_scaled": 0.0,
            "hiva_residual_flow_decoded_prefix_weight_mean": 0.0,
            "hiva_residual_flow_decoded_post_duration_exec_weight_mean": 0.0,
            "hiva_residual_flow_decoded_preview_weight_mean": 0.0,
            "hiva_residual_flow_decoded_terminal_weight_mean": 0.0,
            "hiva_residual_flow_decoded_weight_mean": 0.0,
            "hiva_residual_flow_z_star_abs_mean": 0.0,
            "hiva_residual_flow_z_star_abs_max": 0.0,
            "hiva_residual_flow_z0_abs_mean": 0.0,
            "hiva_residual_flow_z0_abs_max": 0.0,
            "hiva_residual_flow_tanh_abs_mean": 0.0,
            "hiva_residual_flow_tanh_abs_max": 0.0,
            "hiva_residual_flow_delta_abs_mean": 0.0,
            "hiva_residual_flow_delta_abs_max": 0.0,
            "hiva_residual_flow_valid_frac": 0.0,
            "hiva_residual_flow_scale_tr": float(self.config.hiva_residual_flow_scale_tr),
            "hiva_residual_flow_scale_rot": float(self.config.hiva_residual_flow_scale_rot),
            "hiva_residual_flow_scale_grip": float(self.config.hiva_residual_flow_scale_grip),
            "hiva_residual_flow_inference_weight": float(self.config.hiva_residual_flow_inference_weight),
            "hiva_residual_flow_use_separate_expert": float(getattr(self.config, "hiva_residual_flow_use_separate_expert", False)),
            "hiva_residual_flow_basis_alpha": 0.0,
            "hiva_residual_flow_coeff_cross_alpha": 0.0,
            "hiva_residual_flow_coeff_ffn_alpha": 0.0,
        }

    def _valid_action_mask(
        self,
        *,
        bsize: int,
        horizon: int,
        device,
        target_actions_is_pad: Tensor | None,
        hiva_fit_real_steps: Tensor | None,
    ) -> Tensor:
        if target_actions_is_pad is not None:
            return ~target_actions_is_pad[:, :horizon].to(device=device, dtype=torch.bool)
        if hiva_fit_real_steps is not None:
            tau = torch.arange(1, horizon + 1, device=device)[None, :]
            real_steps = hiva_fit_real_steps.to(device=device, dtype=torch.long).reshape(-1, 1)
            return tau <= real_steps.clamp_min(0)
        return torch.ones(bsize, horizon, dtype=torch.bool, device=device)

    def _forward_residual_flow_suffix_with_prefix_cache(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        noisy_z: Tensor,
        base_action_norm: Tensor,
        timestep: Tensor,
        coeff_hidden: Tensor | None,
        use_cache=None,
    ) -> Tensor:
        if self.hiva_residual_flow_head is None:
            raise RuntimeError("Residual-flow branch is enabled but the residual-flow head is missing.")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        basis_tr, basis_rot, basis_grip = self._canonical_basis_set(device=noisy_z.device, dtype=noisy_z.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.hiva_residual_flow_head.embed_suffix(
            noisy_z=noisy_z,
            base_action_norm=base_action_norm,
            timestep=timestep,
            basis_tr=basis_tr,
            basis_rot=basis_rot,
            basis_grip=basis_grip,
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self._forward_with_residual_expert(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -suffix_len:].to(dtype=torch.float32)
        tr_hidden = rot_hidden = grip_hidden = None
        if coeff_hidden is not None:
            tr_hidden, rot_hidden, grip_hidden = self._split_coeff_hidden(coeff_hidden)
        return self.hiva_residual_flow_head.output_from_suffix(
            suffix_out,
            tr_hidden=tr_hidden,
            rot_hidden=rot_hidden,
            grip_hidden=grip_hidden,
        )

    def _residual_flow_training_loss(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: HiVACoeffTargets,
        pred: HiVACoeffTargets,
        coeff_hidden: Tensor | None,
        time: Tensor,
        duration_label_target: Tensor,
        target_actions_raw: Tensor | None,
        target_actions_is_pad: Tensor | None,
        hiva_fit_real_steps: Tensor | None,
    ) -> tuple[Tensor, dict]:
        bsize = time.shape[0]
        zero = torch.zeros(bsize, dtype=time.dtype, device=time.device)
        if not self.config.hiva_residual_flow_enabled:
            _unused, logs = self._residual_flow_zero_logs(device=time.device)
            return zero, logs
        if self.hiva_residual_flow_head is None:
            raise RuntimeError("Residual-flow branch is enabled but the residual-flow head is missing.")
        if target_actions_raw is None:
            raise ValueError("target_actions_raw is required when hiva_residual_flow_enabled=True.")
        if coeff_hidden is None and self.config.hiva_residual_flow_conditioning == "v2_coeff_xattn":
            raise RuntimeError("Residual-flow v2 requires coefficient hidden states.")

        time_expanded = time[:, None, None]
        theta0_norm = {key: x_t[key] - time_expanded * pred[key] for key in x_t}
        theta0_raw = self.unnormalize_coeffs(theta0_norm)
        duration_dummy = torch.full((bsize,), int(self.config.hiva_dmax), device=time.device, dtype=torch.long)
        base_actions_raw = self.decode_coefficients_to_raw_actions(theta0_raw, duration_dummy)
        if self.config.hiva_residual_flow_detach_base:
            base_actions_raw = base_actions_raw.detach()
        if self.config.hiva_residual_flow_detach_coeff_context and coeff_hidden is not None:
            coeff_hidden = coeff_hidden.detach()

        horizon = min(
            int(self.config.hiva_residual_flow_horizon),
            base_actions_raw.shape[1],
            target_actions_raw.shape[1],
            int(self.config.hiva_fit_horizon),
        )
        action_dim = min(base_actions_raw.shape[-1], target_actions_raw.shape[-1])
        base_actions_raw = base_actions_raw[:, :horizon, :action_dim]
        target_actions_raw = target_actions_raw[:, :horizon, :action_dim]
        valid_mask = self._valid_action_mask(
            bsize=bsize,
            horizon=horizon,
            device=time.device,
            target_actions_is_pad=target_actions_is_pad,
            hiva_fit_real_steps=hiva_fit_real_steps,
        )
        valid_float = valid_mask.to(dtype=base_actions_raw.dtype)

        base_action_norm = self._normalize_raw_actions_for_residual(base_actions_raw)
        target_action_norm = self._normalize_raw_actions_for_residual(target_actions_raw)
        scale = self._residual_flow_scale_tensor(device=time.device, dtype=base_action_norm.dtype, action_dim=action_dim)
        z_star = (target_action_norm - base_action_norm) / scale
        z_star_pad = _pad_last_dim(z_star, self.config.max_action_dim)
        base_action_pad = _pad_last_dim(base_action_norm, self.config.max_action_dim)
        eps = self.sample_noise(z_star_pad.shape, z_star_pad.device).to(dtype=z_star_pad.dtype)
        x_res = time_expanded.to(dtype=z_star_pad.dtype) * eps + (1.0 - time_expanded.to(dtype=z_star_pad.dtype)) * z_star_pad
        u_res = eps - z_star_pad
        v_res = self._forward_residual_flow_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            noisy_z=x_res,
            base_action_norm=base_action_pad,
            timestep=time,
            coeff_hidden=coeff_hidden,
            use_cache=True,
        )
        v_res_actual = v_res[:, :horizon, :action_dim]
        u_res_actual = u_res[:, :horizon, :action_dim]
        flow_per_dim = F.mse_loss(v_res_actual, u_res_actual, reduction="none")
        flow_per_step = flow_per_dim.mean(dim=-1)
        denom = valid_float.sum(dim=1).clamp_min(1e-6)
        flow_loss = (flow_per_step * valid_float).sum(dim=1) / denom
        z0_hat = x_res[:, :horizon, :action_dim] - time_expanded.to(dtype=x_res.dtype) * v_res_actual
        tanh_z0 = torch.tanh(z0_hat)
        delta_norm = scale * tanh_z0
        final_action_norm = base_action_norm + float(self.config.hiva_residual_flow_inference_weight) * delta_norm
        final_actions_raw = self._unnormalize_actions_for_residual(final_action_norm)
        if action_dim > 6:
            final_actions_raw = torch.cat(
                [
                    final_actions_raw[..., :6],
                    final_actions_raw[..., 6:7].clamp(-1.0, 1.0),
                    final_actions_raw[..., 7:],
                ],
                dim=-1,
            )
        # Match the inference path: decode the normalized residual correction back to raw action
        # space, clamp the gripper to the legal command range, then re-normalize before the
        # auxiliary residual-flow decoded loss. This avoids supervising values that would be
        # clipped away at rollout time.
        final_action_norm_for_loss = self._normalize_raw_actions_for_residual(final_actions_raw)
        decoded_per_dim = F.smooth_l1_loss(
            final_action_norm_for_loss,
            target_action_norm,
            reduction="none",
            beta=float(self.config.hiva_residual_flow_decoded_loss_beta),
        )
        decoded_per_step = decoded_per_dim.mean(dim=-1)
        decoded_loss = (decoded_per_step * valid_float).sum(dim=1) / denom

        # Logging-only breakdown of the auxiliary decoded loss. The residual-flow
        # objective above is unchanged; these metrics expose modality and region
        # scales with the same region definitions as the deterministic decoded loss.
        tau = torch.arange(1, horizon + 1, device=time.device)[None, :]
        d_star = duration_label_target.to(device=time.device, dtype=torch.long).reshape(-1, 1)
        prefix_mask = tau <= d_star
        post_duration_exec_mask = (tau > d_star) & (tau <= int(self.config.hiva_dmax))
        preview_mask = tau > int(self.config.hiva_dmax)

        target_action_norm_for_region = target_action_norm
        if self.config.hiva_decoded_terminal_weight > 0:
            target_hold_raw = self._build_terminal_hold_target(target_actions_raw, valid_mask)
            target_action_norm_for_region = self._normalize_raw_actions_for_residual(target_hold_raw)

        decoded_region_per_dim = F.smooth_l1_loss(
            final_action_norm_for_loss,
            target_action_norm_for_region,
            reduction="none",
            beta=float(self.config.hiva_residual_flow_decoded_loss_beta),
        )

        def modality_per_step(per_dim: Tensor, start: int, end: int) -> Tensor:
            if start >= action_dim:
                return torch.zeros(bsize, horizon, dtype=per_dim.dtype, device=per_dim.device)
            end = min(end, action_dim)
            return per_dim[..., start:end].mean(dim=-1)

        tr_per_step = modality_per_step(decoded_region_per_dim, 0, 3)
        rot_per_step = modality_per_step(decoded_region_per_dim, 3, 6)
        grip_per_step = modality_per_step(decoded_region_per_dim, 6, 7)
        tr_decoded = (tr_per_step * valid_float).sum(dim=1) / denom
        rot_decoded = (rot_per_step * valid_float).sum(dim=1) / denom
        grip_decoded = (grip_per_step * valid_float).sum(dim=1) / denom

        weights = torch.zeros(bsize, horizon, dtype=valid_float.dtype, device=time.device)
        weights = torch.where(
            prefix_mask & valid_mask,
            torch.full_like(weights, float(self.config.hiva_decoded_prefix_weight)),
            weights,
        )
        weights = torch.where(
            post_duration_exec_mask & valid_mask,
            torch.full_like(weights, float(self.config.hiva_decoded_post_duration_exec_weight)),
            weights,
        )
        weights = torch.where(
            preview_mask & valid_mask,
            torch.full_like(weights, float(self.config.hiva_decoded_preview_weight)),
            weights,
        )
        if self.config.hiva_decoded_terminal_weight > 0:
            weights = torch.where(
                ~valid_mask,
                torch.full_like(weights, float(self.config.hiva_decoded_terminal_weight)),
                weights,
            )

        prefix_region = prefix_mask.expand_as(weights) & valid_mask
        post_region = post_duration_exec_mask.expand_as(weights) & valid_mask
        preview_region = preview_mask.expand_as(weights) & valid_mask
        terminal_region = ~valid_mask

        def safe_mean(x: Tensor) -> float:
            return x.mean().item() if x.numel() else 0.0

        def region_stats(per_step: Tensor, mask: Tensor, region_weight: float) -> tuple[float, float]:
            selected = per_step[mask]
            if selected.numel() == 0:
                return 0.0, 0.0
            loss = selected.mean()
            return loss.item(), (loss * float(region_weight)).item()

        prefix_weights = weights[prefix_region]
        post_weights = weights[post_region]
        preview_weights = weights[preview_region]
        terminal_weights = weights[terminal_region]
        prefix_tr_loss, prefix_tr_loss_scaled = region_stats(
            tr_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        prefix_rot_loss, prefix_rot_loss_scaled = region_stats(
            rot_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        prefix_grip_loss, prefix_grip_loss_scaled = region_stats(
            grip_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        post_tr_loss, post_tr_loss_scaled = region_stats(
            tr_per_step, post_region, self.config.hiva_decoded_post_duration_exec_weight
        )
        post_rot_loss, post_rot_loss_scaled = region_stats(
            rot_per_step, post_region, self.config.hiva_decoded_post_duration_exec_weight
        )
        post_grip_loss, post_grip_loss_scaled = region_stats(
            grip_per_step, post_region, self.config.hiva_decoded_post_duration_exec_weight
        )
        preview_tr_loss, preview_tr_loss_scaled = region_stats(
            tr_per_step, preview_region, self.config.hiva_decoded_preview_weight
        )
        preview_rot_loss, preview_rot_loss_scaled = region_stats(
            rot_per_step, preview_region, self.config.hiva_decoded_preview_weight
        )
        preview_grip_loss, preview_grip_loss_scaled = region_stats(
            grip_per_step, preview_region, self.config.hiva_decoded_preview_weight
        )
        terminal_tr_loss, terminal_tr_loss_scaled = region_stats(
            tr_per_step, terminal_region, self.config.hiva_decoded_terminal_weight
        )
        terminal_rot_loss, terminal_rot_loss_scaled = region_stats(
            rot_per_step, terminal_region, self.config.hiva_decoded_terminal_weight
        )
        terminal_grip_loss, terminal_grip_loss_scaled = region_stats(
            grip_per_step, terminal_region, self.config.hiva_decoded_terminal_weight
        )
        prefix_action_loss = prefix_tr_loss + prefix_rot_loss + prefix_grip_loss
        prefix_action_loss_scaled = prefix_tr_loss_scaled + prefix_rot_loss_scaled + prefix_grip_loss_scaled
        post_action_loss = post_tr_loss + post_rot_loss + post_grip_loss
        post_action_loss_scaled = post_tr_loss_scaled + post_rot_loss_scaled + post_grip_loss_scaled
        preview_action_loss = preview_tr_loss + preview_rot_loss + preview_grip_loss
        preview_action_loss_scaled = preview_tr_loss_scaled + preview_rot_loss_scaled + preview_grip_loss_scaled
        terminal_action_loss = terminal_tr_loss + terminal_rot_loss + terminal_grip_loss
        terminal_action_loss_scaled = terminal_tr_loss_scaled + terminal_rot_loss_scaled + terminal_grip_loss_scaled

        total = (
            float(self.config.hiva_residual_flow_loss_weight) * flow_loss
            + float(self.config.hiva_residual_flow_decoded_loss_weight) * decoded_loss
        )

        adapter = self.hiva_residual_flow_head.basis_adapter
        coeff_adapter = self.hiva_residual_flow_head.coeff_context_adapter
        logs = {
            "hiva_residual_flow_enabled": 1.0,
            "hiva_residual_flow_conditioning": self.config.hiva_residual_flow_conditioning,
            "hiva_residual_flow_conditioning_index": {
                "v1_minimal": 1,
                "v2_coeff_xattn": 2,
            }.get(self.config.hiva_residual_flow_conditioning, -1),
            "hiva_residual_flow_loss": flow_loss.mean().item(),
            "hiva_residual_flow_loss_scaled": (float(self.config.hiva_residual_flow_loss_weight) * flow_loss.mean()).item(),
            "hiva_residual_flow_decoded_loss": decoded_loss.mean().item(),
            "hiva_residual_flow_decoded_loss_scaled": (
                float(self.config.hiva_residual_flow_decoded_loss_weight) * decoded_loss.mean()
            ).item(),
            "hiva_residual_flow_decoded_tr_loss": tr_decoded.mean().item(),
            "hiva_residual_flow_decoded_tr_loss_scaled": (
                float(self.config.hiva_residual_flow_decoded_loss_weight) * tr_decoded.mean()
            ).item(),
            "hiva_residual_flow_decoded_rot_loss": rot_decoded.mean().item(),
            "hiva_residual_flow_decoded_rot_loss_scaled": (
                float(self.config.hiva_residual_flow_decoded_loss_weight) * rot_decoded.mean()
            ).item(),
            "hiva_residual_flow_decoded_grip_loss": grip_decoded.mean().item(),
            "hiva_residual_flow_decoded_grip_loss_scaled": (
                float(self.config.hiva_residual_flow_decoded_loss_weight) * grip_decoded.mean()
            ).item(),
            "hiva_residual_flow_decoded_prefix_tr_loss": prefix_tr_loss,
            "hiva_residual_flow_decoded_prefix_rot_loss": prefix_rot_loss,
            "hiva_residual_flow_decoded_prefix_grip_loss": prefix_grip_loss,
            "hiva_residual_flow_decoded_prefix_action_loss": prefix_action_loss,
            "hiva_residual_flow_decoded_prefix_tr_loss_scaled": prefix_tr_loss_scaled,
            "hiva_residual_flow_decoded_prefix_rot_loss_scaled": prefix_rot_loss_scaled,
            "hiva_residual_flow_decoded_prefix_grip_loss_scaled": prefix_grip_loss_scaled,
            "hiva_residual_flow_decoded_prefix_action_loss_scaled": prefix_action_loss_scaled,
            "hiva_residual_flow_decoded_post_duration_tr_loss": post_tr_loss,
            "hiva_residual_flow_decoded_post_duration_rot_loss": post_rot_loss,
            "hiva_residual_flow_decoded_post_duration_grip_loss": post_grip_loss,
            "hiva_residual_flow_decoded_post_duration_action_loss": post_action_loss,
            "hiva_residual_flow_decoded_post_duration_tr_loss_scaled": post_tr_loss_scaled,
            "hiva_residual_flow_decoded_post_duration_rot_loss_scaled": post_rot_loss_scaled,
            "hiva_residual_flow_decoded_post_duration_grip_loss_scaled": post_grip_loss_scaled,
            "hiva_residual_flow_decoded_post_duration_action_loss_scaled": post_action_loss_scaled,
            "hiva_residual_flow_decoded_preview_tr_loss": preview_tr_loss,
            "hiva_residual_flow_decoded_preview_rot_loss": preview_rot_loss,
            "hiva_residual_flow_decoded_preview_grip_loss": preview_grip_loss,
            "hiva_residual_flow_decoded_preview_action_loss": preview_action_loss,
            "hiva_residual_flow_decoded_preview_tr_loss_scaled": preview_tr_loss_scaled,
            "hiva_residual_flow_decoded_preview_rot_loss_scaled": preview_rot_loss_scaled,
            "hiva_residual_flow_decoded_preview_grip_loss_scaled": preview_grip_loss_scaled,
            "hiva_residual_flow_decoded_preview_action_loss_scaled": preview_action_loss_scaled,
            "hiva_residual_flow_decoded_terminal_tr_loss": terminal_tr_loss,
            "hiva_residual_flow_decoded_terminal_rot_loss": terminal_rot_loss,
            "hiva_residual_flow_decoded_terminal_grip_loss": terminal_grip_loss,
            "hiva_residual_flow_decoded_terminal_action_loss": terminal_action_loss,
            "hiva_residual_flow_decoded_terminal_tr_loss_scaled": terminal_tr_loss_scaled,
            "hiva_residual_flow_decoded_terminal_rot_loss_scaled": terminal_rot_loss_scaled,
            "hiva_residual_flow_decoded_terminal_grip_loss_scaled": terminal_grip_loss_scaled,
            "hiva_residual_flow_decoded_terminal_action_loss_scaled": terminal_action_loss_scaled,
            "hiva_residual_flow_decoded_prefix_weight_mean": safe_mean(prefix_weights),
            "hiva_residual_flow_decoded_post_duration_exec_weight_mean": safe_mean(post_weights),
            "hiva_residual_flow_decoded_preview_weight_mean": safe_mean(preview_weights),
            "hiva_residual_flow_decoded_terminal_weight_mean": safe_mean(terminal_weights),
            "hiva_residual_flow_decoded_weight_mean": weights.mean().item(),
            "hiva_residual_flow_z_star_abs_mean": z_star.abs().mean().item(),
            "hiva_residual_flow_z_star_abs_max": z_star.abs().amax().item(),
            "hiva_residual_flow_z0_abs_mean": z0_hat.abs().mean().item(),
            "hiva_residual_flow_z0_abs_max": z0_hat.abs().amax().item(),
            "hiva_residual_flow_tanh_abs_mean": tanh_z0.abs().mean().item(),
            "hiva_residual_flow_tanh_abs_max": tanh_z0.abs().amax().item(),
            "hiva_residual_flow_delta_abs_mean": delta_norm.abs().mean().item(),
            "hiva_residual_flow_delta_abs_max": delta_norm.abs().amax().item(),
            "hiva_residual_flow_valid_frac": valid_float.mean().item(),
            "hiva_residual_flow_scale_tr": float(self.config.hiva_residual_flow_scale_tr),
            "hiva_residual_flow_scale_rot": float(self.config.hiva_residual_flow_scale_rot),
            "hiva_residual_flow_scale_grip": float(self.config.hiva_residual_flow_scale_grip),
            "hiva_residual_flow_inference_weight": float(self.config.hiva_residual_flow_inference_weight),
            "hiva_residual_flow_use_separate_expert": float(self.hiva_residual_lm_expert is not None),
            "hiva_residual_flow_basis_alpha": adapter.alpha.detach().float().item(),
            "hiva_residual_flow_coeff_cross_alpha": (
                coeff_adapter.cross_alpha.detach().float().item() if coeff_adapter is not None else 0.0
            ),
            "hiva_residual_flow_coeff_ffn_alpha": (
                coeff_adapter.ffn_alpha.detach().float().item() if coeff_adapter is not None else 0.0
            ),
        }
        return total, logs

    def _sample_residual_flow_actions(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        base_actions_raw: Tensor,
        coeff_hidden: Tensor | None,
    ) -> Tensor:
        if not self.config.hiva_residual_flow_enabled:
            return base_actions_raw
        if self.hiva_residual_flow_head is None:
            raise RuntimeError("Residual-flow branch is enabled but the residual-flow head is missing.")
        if coeff_hidden is None and self.config.hiva_residual_flow_conditioning == "v2_coeff_xattn":
            raise RuntimeError("Residual-flow v2 requires coefficient hidden states.")
        bsize, horizon_full, action_dim = base_actions_raw.shape
        horizon = min(int(self.config.hiva_residual_flow_horizon), horizon_full)
        base_prefix_raw = base_actions_raw[:, :horizon, :action_dim]
        base_norm = self._normalize_raw_actions_for_residual(base_prefix_raw)
        base_pad = _pad_last_dim(base_norm, self.config.max_action_dim)
        z_t = self.sample_noise((bsize, horizon, self.config.max_action_dim), base_actions_raw.device).to(
            dtype=base_actions_raw.dtype
        )
        steps = int(self.config.hiva_residual_flow_steps)
        dt = -1.0 / float(steps)
        for step in range(steps):
            t_value = 1.0 + step * dt
            timestep = torch.full((bsize,), t_value, dtype=torch.float32, device=base_actions_raw.device)
            v_t = self._forward_residual_flow_suffix_with_prefix_cache(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                noisy_z=z_t,
                base_action_norm=base_pad,
                timestep=timestep,
                coeff_hidden=coeff_hidden,
            )
            z_t = z_t + dt * v_t
        scale = self._residual_flow_scale_tensor(device=base_actions_raw.device, dtype=base_norm.dtype, action_dim=action_dim)
        delta_norm = scale * torch.tanh(z_t[:, :horizon, :action_dim])
        final_prefix_norm = base_norm + float(self.config.hiva_residual_flow_inference_weight) * delta_norm
        final_prefix_raw = self._unnormalize_actions_for_residual(final_prefix_norm)
        if action_dim > 6:
            final_prefix_raw = torch.cat(
                [
                    final_prefix_raw[..., :6],
                    final_prefix_raw[..., 6:7].clamp(-1.0, 1.0),
                    final_prefix_raw[..., 7:],
                ],
                dim=-1,
            )
        if horizon < horizon_full:
            return torch.cat([final_prefix_raw, base_actions_raw[:, horizon:]], dim=1)
        return final_prefix_raw

    @staticmethod
    def _scalar_param_value(module: nn.Module | None, name: str) -> float:
        param = getattr(module, name, None)
        if isinstance(param, Tensor):
            return param.detach().float().item()
        return 0.0

    def _apply_action_residual(
        self,
        base_actions_raw: Tensor,
        coeff_hidden: Tensor | None,
    ) -> tuple[Tensor, dict]:
        _bsize, horizon, action_dim = base_actions_raw.shape
        logs = {
            "hiva_residual_enabled": float(self.config.hiva_residual_enabled),
            "hiva_residual_mode": self.config.hiva_residual_mode,
            "hiva_residual_mode_index": {
                "token_to_time": 0,
                "basis_hidden_action": 1,
                "basis_xattn_transformer": 2,
            }.get(self.config.hiva_residual_mode, -1),
            "hiva_residual_tanh_abs_mean": 0.0,
            "hiva_residual_tanh_abs_max": 0.0,
            "hiva_residual_abs_mean": 0.0,
            "hiva_residual_abs_max": 0.0,
            "hiva_residual_tr_abs_mean": 0.0,
            "hiva_residual_tr_abs_max": 0.0,
            "hiva_residual_rot_abs_mean": 0.0,
            "hiva_residual_rot_abs_max": 0.0,
            "hiva_residual_grip_abs_mean": 0.0,
            "hiva_residual_grip_abs_max": 0.0,
            "hiva_residual_tr_tanh_abs_mean": 0.0,
            "hiva_residual_tr_tanh_abs_max": 0.0,
            "hiva_residual_rot_tanh_abs_mean": 0.0,
            "hiva_residual_rot_tanh_abs_max": 0.0,
            "hiva_residual_grip_tanh_abs_mean": 0.0,
            "hiva_residual_grip_tanh_abs_max": 0.0,
            "hiva_residual_scale_mean": 0.0,
            "hiva_residual_scale_mult": float(self.config.hiva_residual_scale_mult),
            "hiva_residual_inference_weight": float(self.config.hiva_residual_inference_weight),
            "hiva_residual_coeff_alpha": 0.0,
            "hiva_residual_time_alpha": 0.0,
        }
        if not self.config.hiva_residual_enabled:
            return base_actions_raw, logs
        if self.hiva_residual_head is None or coeff_hidden is None:
            raise RuntimeError("Residual branch is enabled but residual head/hidden states are unavailable.")

        if self.config.hiva_residual_mode == "token_to_time":
            residual_raw = self.hiva_residual_head(coeff_hidden)
        else:
            tr_hidden, rot_hidden, grip_hidden = self._split_coeff_hidden(coeff_hidden)
            basis_tr, basis_rot, basis_grip = self._canonical_basis_set(
                device=base_actions_raw.device,
                dtype=base_actions_raw.dtype,
            )
            residual_raw = self.hiva_residual_head(
                tr_hidden=tr_hidden,
                rot_hidden=rot_hidden,
                grip_hidden=grip_hidden,
                base_actions_raw=base_actions_raw,
                basis_tr=basis_tr,
                basis_rot=basis_rot,
                basis_grip=basis_grip,
            )
        residual_horizon = min(int(residual_raw.shape[1]), horizon)
        residual_raw = residual_raw[:, :residual_horizon, :action_dim]
        scale = self._residual_scale_tensor(
            device=base_actions_raw.device,
            dtype=base_actions_raw.dtype,
            action_dim=action_dim,
        )
        tanh_residual = torch.tanh(residual_raw.to(dtype=base_actions_raw.dtype))
        residual = scale * tanh_residual
        residual_full = F.pad(residual, (0, 0, 0, horizon - residual_horizon))
        tanh_residual_full = F.pad(tanh_residual, (0, 0, 0, horizon - residual_horizon))
        final_actions = base_actions_raw + float(self.config.hiva_residual_inference_weight) * residual_full
        if action_dim > 6:
            # Keep this functional: decoded-action loss backprops through final_actions, and
            # in-place slice updates can invalidate autograd's saved views.
            final_actions = torch.cat(
                [
                    final_actions[..., :6],
                    final_actions[..., 6:7].clamp(-1.0, 1.0),
                    final_actions[..., 7:],
                ],
                dim=-1,
            )

        def modality_abs_stats(tensor: Tensor, start: int, end: int) -> tuple[float, float]:
            if start >= action_dim:
                return 0.0, 0.0
            selected = tensor[..., start : min(end, action_dim)].abs()
            if selected.numel() == 0:
                return 0.0, 0.0
            return selected.mean().item(), selected.amax().item()

        tr_abs_mean, tr_abs_max = modality_abs_stats(residual_full, 0, 3)
        rot_abs_mean, rot_abs_max = modality_abs_stats(residual_full, 3, 6)
        grip_abs_mean, grip_abs_max = modality_abs_stats(residual_full, 6, 7)
        tr_tanh_abs_mean, tr_tanh_abs_max = modality_abs_stats(tanh_residual_full, 0, 3)
        rot_tanh_abs_mean, rot_tanh_abs_max = modality_abs_stats(tanh_residual_full, 3, 6)
        grip_tanh_abs_mean, grip_tanh_abs_max = modality_abs_stats(tanh_residual_full, 6, 7)

        logs = {
            "hiva_residual_enabled": 1.0,
            "hiva_residual_mode": self.config.hiva_residual_mode,
            "hiva_residual_mode_index": {
                "token_to_time": 0,
                "basis_hidden_action": 1,
                "basis_xattn_transformer": 2,
            }.get(self.config.hiva_residual_mode, -1),
            "hiva_residual_tanh_abs_mean": tanh_residual.abs().mean().item(),
            "hiva_residual_tanh_abs_max": tanh_residual.abs().amax().item(),
            "hiva_residual_abs_mean": residual_full.abs().mean().item(),
            "hiva_residual_abs_max": residual_full.abs().amax().item(),
            "hiva_residual_tr_abs_mean": tr_abs_mean,
            "hiva_residual_tr_abs_max": tr_abs_max,
            "hiva_residual_rot_abs_mean": rot_abs_mean,
            "hiva_residual_rot_abs_max": rot_abs_max,
            "hiva_residual_grip_abs_mean": grip_abs_mean,
            "hiva_residual_grip_abs_max": grip_abs_max,
            "hiva_residual_tr_tanh_abs_mean": tr_tanh_abs_mean,
            "hiva_residual_tr_tanh_abs_max": tr_tanh_abs_max,
            "hiva_residual_rot_tanh_abs_mean": rot_tanh_abs_mean,
            "hiva_residual_rot_tanh_abs_max": rot_tanh_abs_max,
            "hiva_residual_grip_tanh_abs_mean": grip_tanh_abs_mean,
            "hiva_residual_grip_tanh_abs_max": grip_tanh_abs_max,
            "hiva_residual_scale_mean": scale.mean().item(),
            "hiva_residual_scale_mult": float(self.config.hiva_residual_scale_mult),
            "hiva_residual_inference_weight": float(self.config.hiva_residual_inference_weight),
            "hiva_residual_coeff_alpha": self._scalar_param_value(self.hiva_residual_head, "coeff_alpha"),
            "hiva_residual_time_alpha": self._scalar_param_value(self.hiva_residual_head, "time_alpha"),
        }
        return final_actions, logs

    def _build_terminal_hold_target(self, target_actions_raw: Tensor, valid_mask: Tensor) -> Tensor:
        """Fill invalid end-of-episode target steps with a low-weight terminal-hold action."""
        target = target_actions_raw.clone()
        bsize, horizon, action_dim = target.shape
        valid_count = valid_mask.long().sum(dim=1).clamp_min(1)
        last_idx = (valid_count - 1).clamp(max=horizon - 1)
        batch_idx = torch.arange(bsize, device=target.device)
        last_action = target[batch_idx, last_idx]

        terminal = torch.zeros_like(target)
        if action_dim > 6:
            terminal[..., 6] = last_action[:, 6:7]
        return torch.where(valid_mask[:, :, None], target, terminal)

    def _decoded_action_loss(
        self,
        *,
        x_t: HiVACoeffTargets,
        pred: HiVACoeffTargets,
        coeff_hidden: Tensor | None,
        time: Tensor,
        duration_label_target: Tensor,
        target_actions_raw: Tensor | None,
        target_actions_is_pad: Tensor | None,
        hiva_fit_real_steps: Tensor | None,
    ) -> tuple[Tensor, dict]:
        bsize = time.shape[0]
        zero = torch.zeros(bsize, dtype=time.dtype, device=time.device)
        logs = {
            "hiva_decoded_action_loss": 0.0,
            "hiva_decoded_action_loss_scaled": 0.0,
            "hiva_decoded_tr_loss": 0.0,
            "hiva_decoded_tr_loss_scaled": 0.0,
            "hiva_decoded_rot_loss": 0.0,
            "hiva_decoded_rot_loss_scaled": 0.0,
            "hiva_decoded_grip_loss": 0.0,
            "hiva_decoded_grip_loss_scaled": 0.0,
            "hiva_decoded_prefix_tr_loss": 0.0,
            "hiva_decoded_prefix_rot_loss": 0.0,
            "hiva_decoded_prefix_grip_loss": 0.0,
            "hiva_base_prefix_tr_loss": 0.0,
            "hiva_final_prefix_tr_loss": 0.0,
            "hiva_delta_prefix_tr_loss": 0.0,
            "hiva_base_prefix_rot_loss": 0.0,
            "hiva_final_prefix_rot_loss": 0.0,
            "hiva_delta_prefix_rot_loss": 0.0,
            "hiva_base_prefix_grip_loss": 0.0,
            "hiva_final_prefix_grip_loss": 0.0,
            "hiva_delta_prefix_grip_loss": 0.0,
            "hiva_decoded_prefix_action_loss": 0.0,
            "hiva_decoded_prefix_tr_loss_scaled": 0.0,
            "hiva_decoded_prefix_rot_loss_scaled": 0.0,
            "hiva_decoded_prefix_grip_loss_scaled": 0.0,
            "hiva_decoded_prefix_action_loss_scaled": 0.0,
            "hiva_decoded_post_duration_tr_loss": 0.0,
            "hiva_decoded_post_duration_rot_loss": 0.0,
            "hiva_decoded_post_duration_grip_loss": 0.0,
            "hiva_decoded_post_duration_action_loss": 0.0,
            "hiva_decoded_post_duration_tr_loss_scaled": 0.0,
            "hiva_decoded_post_duration_rot_loss_scaled": 0.0,
            "hiva_decoded_post_duration_grip_loss_scaled": 0.0,
            "hiva_decoded_post_duration_action_loss_scaled": 0.0,
            "hiva_decoded_preview_tr_loss": 0.0,
            "hiva_decoded_preview_rot_loss": 0.0,
            "hiva_decoded_preview_grip_loss": 0.0,
            "hiva_decoded_preview_action_loss": 0.0,
            "hiva_decoded_preview_tr_loss_scaled": 0.0,
            "hiva_decoded_preview_rot_loss_scaled": 0.0,
            "hiva_decoded_preview_grip_loss_scaled": 0.0,
            "hiva_decoded_preview_action_loss_scaled": 0.0,
            "hiva_decoded_terminal_tr_loss": 0.0,
            "hiva_decoded_terminal_rot_loss": 0.0,
            "hiva_decoded_terminal_grip_loss": 0.0,
            "hiva_decoded_terminal_action_loss": 0.0,
            "hiva_decoded_terminal_tr_loss_scaled": 0.0,
            "hiva_decoded_terminal_rot_loss_scaled": 0.0,
            "hiva_decoded_terminal_grip_loss_scaled": 0.0,
            "hiva_decoded_terminal_action_loss_scaled": 0.0,
            "hiva_decoded_valid_frac": 0.0,
            "hiva_decoded_prefix_weight_mean": 0.0,
            "hiva_decoded_post_duration_exec_weight_mean": 0.0,
            "hiva_decoded_preview_weight_mean": 0.0,
            "hiva_decoded_terminal_weight_mean": 0.0,
            "hiva_decoded_weight_mean": 0.0,
            "hiva_residual_enabled": float(self.config.hiva_residual_enabled),
            "hiva_residual_mode": self.config.hiva_residual_mode,
            "hiva_residual_mode_index": {
                "token_to_time": 0,
                "basis_hidden_action": 1,
                "basis_xattn_transformer": 2,
            }.get(self.config.hiva_residual_mode, -1),
            "hiva_residual_tanh_abs_mean": 0.0,
            "hiva_residual_tanh_abs_max": 0.0,
            "hiva_residual_abs_mean": 0.0,
            "hiva_residual_abs_max": 0.0,
            "hiva_residual_tr_abs_mean": 0.0,
            "hiva_residual_tr_abs_max": 0.0,
            "hiva_residual_rot_abs_mean": 0.0,
            "hiva_residual_rot_abs_max": 0.0,
            "hiva_residual_grip_abs_mean": 0.0,
            "hiva_residual_grip_abs_max": 0.0,
            "hiva_residual_tr_tanh_abs_mean": 0.0,
            "hiva_residual_tr_tanh_abs_max": 0.0,
            "hiva_residual_rot_tanh_abs_mean": 0.0,
            "hiva_residual_rot_tanh_abs_max": 0.0,
            "hiva_residual_grip_tanh_abs_mean": 0.0,
            "hiva_residual_grip_tanh_abs_max": 0.0,
            "hiva_residual_scale_mean": 0.0,
            "hiva_residual_scale_mult": float(self.config.hiva_residual_scale_mult),
            "hiva_residual_coeff_alpha": 0.0,
            "hiva_residual_time_alpha": 0.0,
        }
        if self.config.hiva_decoded_action_loss_weight <= 0:
            return zero, logs
        if target_actions_raw is None:
            raise ValueError("target_actions_raw is required when hiva_decoded_action_loss_weight > 0.")

        time_expanded = time[:, None, None]
        theta0_norm = {key: x_t[key] - time_expanded * pred[key] for key in x_t}
        theta0_raw = self.unnormalize_coeffs(theta0_norm)
        duration_dummy = torch.full((bsize,), int(self.config.hiva_dmax), device=time.device, dtype=torch.long)
        base_actions_raw = self.decode_coefficients_to_raw_actions(theta0_raw, duration_dummy)
        pred_actions_raw, residual_logs = self._apply_action_residual(base_actions_raw, coeff_hidden)

        horizon = min(pred_actions_raw.shape[1], target_actions_raw.shape[1], int(self.config.hiva_fit_horizon))
        action_dim = min(pred_actions_raw.shape[-1], target_actions_raw.shape[-1])
        base_actions_raw = base_actions_raw[:, :horizon, :action_dim]
        pred_actions_raw = pred_actions_raw[:, :horizon, :action_dim]
        target_actions_raw = target_actions_raw[:, :horizon, :action_dim]

        tau = torch.arange(1, horizon + 1, device=time.device)[None, :]
        d_star = duration_label_target.to(device=time.device, dtype=torch.long).reshape(-1, 1)
        prefix_mask = tau <= d_star
        post_duration_exec_mask = (tau > d_star) & (tau <= int(self.config.hiva_dmax))
        preview_mask = tau > int(self.config.hiva_dmax)

        if target_actions_is_pad is not None:
            valid_mask = ~target_actions_is_pad[:, :horizon].to(device=time.device, dtype=torch.bool)
        elif hiva_fit_real_steps is not None:
            real_steps = hiva_fit_real_steps.to(device=time.device, dtype=torch.long).reshape(-1, 1)
            valid_mask = tau <= real_steps.clamp_min(0)
        else:
            valid_mask = torch.ones(bsize, horizon, dtype=torch.bool, device=time.device)

        target_for_loss = target_actions_raw
        if self.config.hiva_decoded_terminal_weight > 0:
            target_for_loss = self._build_terminal_hold_target(target_actions_raw, valid_mask)

        weights = torch.zeros(bsize, horizon, dtype=pred_actions_raw.dtype, device=time.device)
        weights = torch.where(
            prefix_mask & valid_mask,
            torch.full_like(weights, float(self.config.hiva_decoded_prefix_weight)),
            weights,
        )
        weights = torch.where(
            post_duration_exec_mask & valid_mask,
            torch.full_like(weights, float(self.config.hiva_decoded_post_duration_exec_weight)),
            weights,
        )
        weights = torch.where(
            preview_mask & valid_mask,
            torch.full_like(weights, float(self.config.hiva_decoded_preview_weight)),
            weights,
        )
        if self.config.hiva_decoded_terminal_weight > 0:
            weights = torch.where(
                ~valid_mask,
                torch.full_like(weights, float(self.config.hiva_decoded_terminal_weight)),
                weights,
            )

        def modality_per_step_loss(actions_raw: Tensor, start: int, end: int, beta: float) -> Tensor:
            if start >= action_dim:
                return torch.zeros(bsize, horizon, dtype=time.dtype, device=time.device)
            end = min(end, action_dim)
            per_dim = F.smooth_l1_loss(
                actions_raw[..., start:end],
                target_for_loss[..., start:end],
                reduction="none",
                beta=float(beta),
            )
            return per_dim.mean(dim=-1)

        def modality_loss(start: int, end: int) -> Tensor:
            if start == 0:
                beta = self.config.hiva_decoded_tr_loss_beta
            elif start == 3:
                beta = self.config.hiva_decoded_rot_loss_beta
            else:
                beta = self.config.hiva_decoded_grip_loss_beta
            per_step = modality_per_step_loss(pred_actions_raw, start, end, beta)
            denom = weights.sum(dim=1).clamp_min(1e-6)
            return (per_step * weights).sum(dim=1) / denom

        tr_per_step = modality_per_step_loss(
            pred_actions_raw, 0, 3, self.config.hiva_decoded_tr_loss_beta
        )
        rot_per_step = modality_per_step_loss(
            pred_actions_raw, 3, 6, self.config.hiva_decoded_rot_loss_beta
        )
        grip_per_step = modality_per_step_loss(
            pred_actions_raw, 6, 7, self.config.hiva_decoded_grip_loss_beta
        )
        base_tr_per_step = modality_per_step_loss(
            base_actions_raw, 0, 3, self.config.hiva_decoded_tr_loss_beta
        )
        base_rot_per_step = modality_per_step_loss(
            base_actions_raw, 3, 6, self.config.hiva_decoded_rot_loss_beta
        )
        base_grip_per_step = modality_per_step_loss(
            base_actions_raw, 6, 7, self.config.hiva_decoded_grip_loss_beta
        )
        tr_decoded = modality_loss(0, 3)
        rot_decoded = modality_loss(3, 6)
        grip_decoded = modality_loss(6, 7)
        decoded_total = (
            self.config.hiva_decoded_tr_loss_weight * tr_decoded
            + self.config.hiva_decoded_rot_loss_weight * rot_decoded
            + self.config.hiva_decoded_grip_loss_weight * grip_decoded
        )

        valid_float = valid_mask.to(dtype=weights.dtype)
        prefix_weights = weights[prefix_mask.expand_as(weights) & valid_mask]
        post_weights = weights[post_duration_exec_mask.expand_as(weights) & valid_mask]
        preview_weights = weights[preview_mask.expand_as(weights) & valid_mask]
        terminal_weights = weights[~valid_mask]

        def safe_mean(x: Tensor) -> float:
            return x.mean().item() if x.numel() else 0.0

        def region_stats(per_step: Tensor, mask: Tensor, region_weight: float) -> tuple[float, float]:
            selected = per_step[mask]
            if selected.numel() == 0:
                return 0.0, 0.0
            loss = selected.mean()
            return loss.item(), (loss * float(region_weight)).item()

        prefix_region = prefix_mask.expand_as(weights) & valid_mask
        post_region = post_duration_exec_mask.expand_as(weights) & valid_mask
        preview_region = preview_mask.expand_as(weights) & valid_mask
        terminal_region = ~valid_mask
        prefix_tr_loss, prefix_tr_loss_scaled = region_stats(
            tr_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        prefix_rot_loss, prefix_rot_loss_scaled = region_stats(
            rot_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        prefix_grip_loss, prefix_grip_loss_scaled = region_stats(
            grip_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        base_prefix_tr_loss, _base_prefix_tr_loss_scaled = region_stats(
            base_tr_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        base_prefix_rot_loss, _base_prefix_rot_loss_scaled = region_stats(
            base_rot_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        base_prefix_grip_loss, _base_prefix_grip_loss_scaled = region_stats(
            base_grip_per_step, prefix_region, self.config.hiva_decoded_prefix_weight
        )
        delta_prefix_tr_loss = base_prefix_tr_loss - prefix_tr_loss
        delta_prefix_rot_loss = base_prefix_rot_loss - prefix_rot_loss
        delta_prefix_grip_loss = base_prefix_grip_loss - prefix_grip_loss
        post_tr_loss, post_tr_loss_scaled = region_stats(
            tr_per_step, post_region, self.config.hiva_decoded_post_duration_exec_weight
        )
        post_rot_loss, post_rot_loss_scaled = region_stats(
            rot_per_step, post_region, self.config.hiva_decoded_post_duration_exec_weight
        )
        post_grip_loss, post_grip_loss_scaled = region_stats(
            grip_per_step, post_region, self.config.hiva_decoded_post_duration_exec_weight
        )
        preview_tr_loss, preview_tr_loss_scaled = region_stats(
            tr_per_step, preview_region, self.config.hiva_decoded_preview_weight
        )
        preview_rot_loss, preview_rot_loss_scaled = region_stats(
            rot_per_step, preview_region, self.config.hiva_decoded_preview_weight
        )
        preview_grip_loss, preview_grip_loss_scaled = region_stats(
            grip_per_step, preview_region, self.config.hiva_decoded_preview_weight
        )
        terminal_tr_loss, terminal_tr_loss_scaled = region_stats(
            tr_per_step, terminal_region, self.config.hiva_decoded_terminal_weight
        )
        terminal_rot_loss, terminal_rot_loss_scaled = region_stats(
            rot_per_step, terminal_region, self.config.hiva_decoded_terminal_weight
        )
        terminal_grip_loss, terminal_grip_loss_scaled = region_stats(
            grip_per_step, terminal_region, self.config.hiva_decoded_terminal_weight
        )
        prefix_action_loss = prefix_tr_loss + prefix_rot_loss + prefix_grip_loss
        prefix_action_loss_scaled = (
            prefix_tr_loss_scaled + prefix_rot_loss_scaled + prefix_grip_loss_scaled
        )
        post_action_loss = post_tr_loss + post_rot_loss + post_grip_loss
        post_action_loss_scaled = post_tr_loss_scaled + post_rot_loss_scaled + post_grip_loss_scaled
        preview_action_loss = preview_tr_loss + preview_rot_loss + preview_grip_loss
        preview_action_loss_scaled = (
            preview_tr_loss_scaled + preview_rot_loss_scaled + preview_grip_loss_scaled
        )
        terminal_action_loss = terminal_tr_loss + terminal_rot_loss + terminal_grip_loss
        terminal_action_loss_scaled = (
            terminal_tr_loss_scaled + terminal_rot_loss_scaled + terminal_grip_loss_scaled
        )

        logs = {
            "hiva_decoded_action_loss": decoded_total.mean().item(),
            "hiva_decoded_action_loss_scaled": (
                self.config.hiva_decoded_action_loss_weight * decoded_total.mean()
            ).item(),
            "hiva_decoded_tr_loss": tr_decoded.mean().item(),
            "hiva_decoded_tr_loss_scaled": (
                self.config.hiva_decoded_tr_loss_weight * tr_decoded.mean()
            ).item(),
            "hiva_decoded_rot_loss": rot_decoded.mean().item(),
            "hiva_decoded_rot_loss_scaled": (
                self.config.hiva_decoded_rot_loss_weight * rot_decoded.mean()
            ).item(),
            "hiva_decoded_grip_loss": grip_decoded.mean().item(),
            "hiva_decoded_grip_loss_scaled": (
                self.config.hiva_decoded_grip_loss_weight * grip_decoded.mean()
            ).item(),
            "hiva_decoded_prefix_tr_loss": prefix_tr_loss,
            "hiva_decoded_prefix_rot_loss": prefix_rot_loss,
            "hiva_decoded_prefix_grip_loss": prefix_grip_loss,
            "hiva_base_prefix_tr_loss": base_prefix_tr_loss,
            "hiva_final_prefix_tr_loss": prefix_tr_loss,
            "hiva_delta_prefix_tr_loss": delta_prefix_tr_loss,
            "hiva_base_prefix_rot_loss": base_prefix_rot_loss,
            "hiva_final_prefix_rot_loss": prefix_rot_loss,
            "hiva_delta_prefix_rot_loss": delta_prefix_rot_loss,
            "hiva_base_prefix_grip_loss": base_prefix_grip_loss,
            "hiva_final_prefix_grip_loss": prefix_grip_loss,
            "hiva_delta_prefix_grip_loss": delta_prefix_grip_loss,
            "hiva_decoded_prefix_action_loss": prefix_action_loss,
            "hiva_decoded_prefix_tr_loss_scaled": prefix_tr_loss_scaled,
            "hiva_decoded_prefix_rot_loss_scaled": prefix_rot_loss_scaled,
            "hiva_decoded_prefix_grip_loss_scaled": prefix_grip_loss_scaled,
            "hiva_decoded_prefix_action_loss_scaled": prefix_action_loss_scaled,
            "hiva_decoded_post_duration_tr_loss": post_tr_loss,
            "hiva_decoded_post_duration_rot_loss": post_rot_loss,
            "hiva_decoded_post_duration_grip_loss": post_grip_loss,
            "hiva_decoded_post_duration_action_loss": post_action_loss,
            "hiva_decoded_post_duration_tr_loss_scaled": post_tr_loss_scaled,
            "hiva_decoded_post_duration_rot_loss_scaled": post_rot_loss_scaled,
            "hiva_decoded_post_duration_grip_loss_scaled": post_grip_loss_scaled,
            "hiva_decoded_post_duration_action_loss_scaled": post_action_loss_scaled,
            "hiva_decoded_preview_tr_loss": preview_tr_loss,
            "hiva_decoded_preview_rot_loss": preview_rot_loss,
            "hiva_decoded_preview_grip_loss": preview_grip_loss,
            "hiva_decoded_preview_action_loss": preview_action_loss,
            "hiva_decoded_preview_tr_loss_scaled": preview_tr_loss_scaled,
            "hiva_decoded_preview_rot_loss_scaled": preview_rot_loss_scaled,
            "hiva_decoded_preview_grip_loss_scaled": preview_grip_loss_scaled,
            "hiva_decoded_preview_action_loss_scaled": preview_action_loss_scaled,
            "hiva_decoded_terminal_tr_loss": terminal_tr_loss,
            "hiva_decoded_terminal_rot_loss": terminal_rot_loss,
            "hiva_decoded_terminal_grip_loss": terminal_grip_loss,
            "hiva_decoded_terminal_action_loss": terminal_action_loss,
            "hiva_decoded_terminal_tr_loss_scaled": terminal_tr_loss_scaled,
            "hiva_decoded_terminal_rot_loss_scaled": terminal_rot_loss_scaled,
            "hiva_decoded_terminal_grip_loss_scaled": terminal_grip_loss_scaled,
            "hiva_decoded_terminal_action_loss_scaled": terminal_action_loss_scaled,
            "hiva_decoded_valid_frac": valid_float.mean().item(),
            "hiva_decoded_prefix_weight_mean": safe_mean(prefix_weights),
            "hiva_decoded_post_duration_exec_weight_mean": safe_mean(post_weights),
            "hiva_decoded_preview_weight_mean": safe_mean(preview_weights),
            "hiva_decoded_terminal_weight_mean": safe_mean(terminal_weights),
            "hiva_decoded_weight_mean": weights.mean().item(),
        }
        logs.update(residual_logs)
        return decoded_total, logs

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        targets: HiVACoeffTargets,
        duration_class_target: Tensor,
        duration_label_target: Tensor,
        target_actions_raw: Tensor | None = None,
        target_actions_is_pad: Tensor | None = None,
        hiva_fit_real_steps: Tensor | None = None,
        noise: HiVACoeffTargets | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        targets_norm = self.normalize_coeffs(targets)
        if noise is None:
            noise = self._sample_coeff_noise(targets_norm)
        if time is None:
            time = self.sample_time(state.shape[0], state.device)

        x_t, u_t = self._mix_coeffs(targets_norm, noise, time)
        duration_t = None
        duration_u = None
        if self.config.hiva_duration_prediction_type == "continuous_fm":
            duration_target_norm = self.normalize_duration(duration_label_target).reshape(-1, 1, 1)
            duration_noise = self.sample_noise(duration_target_norm.shape, duration_target_norm.device)
            duration_t, duration_u = self._mix_duration(duration_target_norm, duration_noise, time)

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
        pred, duration_logits, coeff_hidden = self._forward_hiva_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=x_t,
            timestep=time,
            duration_t=duration_t,
            use_cache=True,
        )
        clean_duration_logits = None
        if self.config.hiva_duration_clean_loss_weight > 0:
            clean_time = torch.zeros_like(time, dtype=torch.float32, device=time.device)
            clean_duration_logits = self.predict_duration_logits(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                coeffs=targets_norm,
                timestep=clean_time,
                use_cache=True,
            )

        tr_loss = F.mse_loss(pred["tr"], u_t["tr"], reduction="none").mean(dim=(1, 2))
        rot_loss = F.mse_loss(pred["rot"], u_t["rot"], reduction="none").mean(dim=(1, 2))
        grip_loss = F.mse_loss(pred["grip"], u_t["grip"], reduction="none").mean(dim=(1, 2))
        decoded_action_loss, decoded_logs = self._decoded_action_loss(
            x_t=x_t,
            pred=pred,
            coeff_hidden=coeff_hidden,
            time=time,
            duration_label_target=duration_label_target,
            target_actions_raw=target_actions_raw,
            target_actions_is_pad=target_actions_is_pad,
            hiva_fit_real_steps=hiva_fit_real_steps,
        )
        residual_flow_loss, residual_flow_logs = self._residual_flow_training_loss(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            pred=pred,
            coeff_hidden=coeff_hidden,
            time=time,
            duration_label_target=duration_label_target,
            target_actions_raw=target_actions_raw,
            target_actions_is_pad=target_actions_is_pad,
            hiva_fit_real_steps=hiva_fit_real_steps,
        )

        if self.config.hiva_duration_prediction_type == "continuous_fm":
            duration_fm_loss = F.mse_loss(duration_logits, duration_u, reduction="none").mean(dim=(1, 2))
            per_sample_loss = (
                self.config.hiva_tr_loss_weight * tr_loss
                + self.config.hiva_rot_loss_weight * rot_loss
                + self.config.hiva_grip_loss_weight * grip_loss
                + self.config.hiva_duration_fm_loss_weight * duration_fm_loss
                + self.config.hiva_decoded_action_loss_weight * decoded_action_loss
                + residual_flow_loss
            )
            time_expanded = time[:, None, None]
            duration0_norm_hat = duration_t - time_expanded * duration_logits
            duration0_raw_hat = self.unnormalize_duration(duration0_norm_hat).reshape(-1)
            duration_steps_pred = self._duration_steps_from_continuous(duration0_raw_hat)
            duration_label_flat = duration_label_target.reshape(-1)
            duration_abs_err = (duration0_raw_hat - duration_label_flat).abs()
            fm_total_reduced = (
                self.config.hiva_tr_loss_weight * tr_loss.mean()
                + self.config.hiva_rot_loss_weight * rot_loss.mean()
                + self.config.hiva_grip_loss_weight * grip_loss.mean()
            )
            duration_scaled_reduced = self.config.hiva_duration_fm_loss_weight * duration_fm_loss.mean()
            loss_dict = {
                "hiva_tr_fm_loss": tr_loss.mean().item(),
                "hiva_rot_fm_loss": rot_loss.mean().item(),
                "hiva_grip_fm_loss": grip_loss.mean().item(),
                "hiva_duration_prediction_type": self.config.hiva_duration_prediction_type,
                "hiva_duration_fm_loss": duration_fm_loss.mean().item(),
                "hiva_duration_fm_loss_scaled": duration_scaled_reduced.item(),
                "hiva_duration_noisy_loss": duration_fm_loss.mean().item(),
                "hiva_duration_noisy_loss_scaled": duration_scaled_reduced.item(),
                "hiva_duration_cont_norm": self.config.hiva_duration_cont_norm,
                "hiva_duration_cont_pred_mean": duration0_raw_hat.mean().item(),
                "hiva_duration_cont_pred_min": duration0_raw_hat.min().item(),
                "hiva_duration_cont_pred_max": duration0_raw_hat.max().item(),
                "hiva_duration_cont_pred_std": duration0_raw_hat.std(unbiased=False).item(),
                "hiva_duration_target_mean": duration_label_flat.float().mean().item(),
                "hiva_duration_cont_mae": duration_abs_err.mean().item(),
                "hiva_duration_cont_rmse": torch.sqrt((duration_abs_err * duration_abs_err).mean()).item(),
                "hiva_duration_floor_acc": (
                    duration_steps_pred == duration_label_flat.long()
                ).float().mean().item(),
                "hiva_duration_acc": (duration_steps_pred == duration_label_flat.long()).float().mean().item(),
                "hiva_duration_noisy_acc": (duration_steps_pred == duration_label_flat.long()).float().mean().item(),
                "hiva_duration_pred_mean": duration_steps_pred.float().mean().item(),
                "hiva_duration_to_fm_ratio": (
                    duration_scaled_reduced / fm_total_reduced.detach().clamp_min(1e-6)
                ).item(),
                "hiva_fm_total": fm_total_reduced.item(),
                "hiva_tr_loss_weight": self.config.hiva_tr_loss_weight,
                "hiva_rot_loss_weight": self.config.hiva_rot_loss_weight,
                "hiva_grip_loss_weight": self.config.hiva_grip_loss_weight,
                "hiva_duration_fm_loss_weight": self.config.hiva_duration_fm_loss_weight,
                "hiva_decoded_action_loss_weight": self.config.hiva_decoded_action_loss_weight,
                "hiva_decoded_tr_loss_weight": self.config.hiva_decoded_tr_loss_weight,
                "hiva_decoded_rot_loss_weight": self.config.hiva_decoded_rot_loss_weight,
                "hiva_decoded_grip_loss_weight": self.config.hiva_decoded_grip_loss_weight,
                "hiva_decoded_prefix_weight": self.config.hiva_decoded_prefix_weight,
                "hiva_decoded_post_duration_exec_weight": self.config.hiva_decoded_post_duration_exec_weight,
                "hiva_decoded_preview_weight": self.config.hiva_decoded_preview_weight,
                "hiva_decoded_terminal_weight": self.config.hiva_decoded_terminal_weight,
                "hiva_decoded_loss_beta": self.config.hiva_decoded_loss_beta,
                "hiva_decoded_tr_loss_beta": self.config.hiva_decoded_tr_loss_beta,
                "hiva_decoded_rot_loss_beta": self.config.hiva_decoded_rot_loss_beta,
                "hiva_decoded_grip_loss_beta": self.config.hiva_decoded_grip_loss_beta,
                "hiva_duration_mean": self.hiva_duration_mean.detach().float().item(),
                "hiva_duration_std": self.hiva_duration_std.detach().float().item(),
                "hiva_duration_loss": self.config.hiva_duration_loss,
                "hiva_duration_readout": self.config.hiva_duration_readout,
                "hiva_suffix_attention": self.config.hiva_suffix_attention,
                "hiva_duration_head_type": self.config.hiva_duration_head_type,
                "hiva_basis_mode": self.config.hiva_basis_mode,
                "hiva_dmax": self.config.hiva_dmax,
                "hiva_fit_horizon": self.config.hiva_fit_horizon,
                "hiva_k": self.config.hiva_k,
                "hiva_degree": self.config.hiva_degree,
                "hiva_degree_tr": self.config.hiva_degree_tr,
                "hiva_degree_rot": self.config.hiva_degree_rot,
                "hiva_degree_grip": self.config.hiva_degree_grip,
                "hiva_suffix_len": self.hiva_suffix_len,
            }
            loss_dict.update(decoded_logs)
            loss_dict.update(residual_flow_logs)
            return per_sample_loss, loss_dict

        duration_ce = F.cross_entropy(duration_logits.to(dtype=torch.float32), duration_class_target, reduction="none")
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

        if clean_duration_logits is not None:
            duration_clean_loss = F.cross_entropy(
            clean_duration_logits.to(dtype=torch.float32), duration_class_target, reduction="none"
            )
            duration_clean_loss_reduced = duration_clean_loss.mean()
        else:
            duration_clean_loss = torch.zeros_like(duration_ce)
            duration_clean_loss_reduced = duration_ce.new_tensor(0.0)

        duration_noisy_loss_scaled_reduced = (
            self.config.hiva_duration_noisy_loss_weight * duration_noisy_loss_reduced
        )
        duration_clean_loss_scaled_reduced = (
            self.config.hiva_duration_clean_loss_weight * duration_clean_loss_reduced
        )
        duration_total_loss_scaled_reduced = duration_noisy_loss_scaled_reduced + duration_clean_loss_scaled_reduced
        per_sample_loss = (
            self.config.hiva_tr_loss_weight * tr_loss
            + self.config.hiva_rot_loss_weight * rot_loss
            + self.config.hiva_grip_loss_weight * grip_loss
            + self.config.hiva_duration_noisy_loss_weight * duration_noisy_loss
            + self.config.hiva_duration_clean_loss_weight * duration_clean_loss
            + self.config.hiva_decoded_action_loss_weight * decoded_action_loss
            + residual_flow_loss
        )

        duration_values = torch.as_tensor(
            self.config.hiva_duration_classes,
            device=duration_logits.device,
            dtype=torch.long,
        )
        duration_pred = duration_logits.argmax(dim=-1)
        duration_correct = (duration_pred == duration_class_target).float()
        if clean_duration_logits is not None:
            clean_duration_pred = clean_duration_logits.argmax(dim=-1)
            duration_clean_correct = (clean_duration_pred == duration_class_target).float()
            duration_clean_pred_mean = duration_values[clean_duration_pred].float().mean()
        else:
            duration_clean_correct = torch.zeros_like(duration_correct)
            duration_clean_pred_mean = duration_ce.new_tensor(0.0)
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
            duration_total_loss_scaled_reduced / fm_total_reduced.detach().clamp_min(1e-6)
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
            "hiva_duration_clean_loss": duration_clean_loss_reduced.item(),
            "hiva_duration_clean_loss_scaled": duration_clean_loss_scaled_reduced.item(),
            "hiva_duration_total_loss_scaled": duration_total_loss_scaled_reduced.item(),
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
            "hiva_duration_clean_acc": duration_clean_correct.mean().item(),
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
            "hiva_duration_target_mean": duration_values[duration_class_target].float().mean().item(),
            "hiva_duration_clean_pred_mean": duration_clean_pred_mean.item(),
            "hiva_duration_noisy_weight_mean": duration_noisy_weights.mean().item(),
            "hiva_duration_noisy_weight_min": duration_noisy_weights.min().item(),
            "hiva_duration_noisy_weight_max": duration_noisy_weights.max().item(),
            "hiva_tr_loss_weight": self.config.hiva_tr_loss_weight,
            "hiva_rot_loss_weight": self.config.hiva_rot_loss_weight,
            "hiva_grip_loss_weight": self.config.hiva_grip_loss_weight,
            "hiva_duration_noisy_loss_weight": self.config.hiva_duration_noisy_loss_weight,
            "hiva_duration_clean_loss_weight": self.config.hiva_duration_clean_loss_weight,
            "hiva_duration_clean_enabled": float(self.config.hiva_duration_clean_loss_weight > 0),
            "hiva_decoded_action_loss_weight": self.config.hiva_decoded_action_loss_weight,
            "hiva_decoded_tr_loss_weight": self.config.hiva_decoded_tr_loss_weight,
            "hiva_decoded_rot_loss_weight": self.config.hiva_decoded_rot_loss_weight,
            "hiva_decoded_grip_loss_weight": self.config.hiva_decoded_grip_loss_weight,
            "hiva_decoded_prefix_weight": self.config.hiva_decoded_prefix_weight,
            "hiva_decoded_post_duration_exec_weight": self.config.hiva_decoded_post_duration_exec_weight,
            "hiva_decoded_preview_weight": self.config.hiva_decoded_preview_weight,
            "hiva_decoded_terminal_weight": self.config.hiva_decoded_terminal_weight,
            "hiva_decoded_loss_beta": self.config.hiva_decoded_loss_beta,
            "hiva_decoded_tr_loss_beta": self.config.hiva_decoded_tr_loss_beta,
            "hiva_decoded_rot_loss_beta": self.config.hiva_decoded_rot_loss_beta,
            "hiva_decoded_grip_loss_beta": self.config.hiva_decoded_grip_loss_beta,
            "hiva_duration_noisy_sigma": self.config.hiva_duration_noisy_sigma,
            "hiva_duration_loss": self.config.hiva_duration_loss,
            "hiva_duration_readout": self.config.hiva_duration_readout,
            "hiva_suffix_attention": self.config.hiva_suffix_attention,
            "hiva_duration_head_type": self.config.hiva_duration_head_type,
            "hiva_duration_ffn_hidden_mult": self.config.hiva_duration_ffn_hidden_mult,
            "hiva_duration_ffn_alpha_init": self.config.hiva_duration_ffn_alpha_init,
            "hiva_duration_ffn_alpha": (
                self.hiva_duration_head.alpha.detach().float().item()
                if hasattr(self.hiva_duration_head, "alpha")
                else 0.0
            ),
            "hiva_basis_mode": self.config.hiva_basis_mode,
            "hiva_dmax": self.config.hiva_dmax,
            "hiva_fit_horizon": self.config.hiva_fit_horizon,
            "hiva_k": self.config.hiva_k,
            "hiva_degree": self.config.hiva_degree,
            "hiva_degree_tr": self.config.hiva_degree_tr,
            "hiva_degree_rot": self.config.hiva_degree_rot,
            "hiva_degree_grip": self.config.hiva_degree_grip,
            "hiva_suffix_len": self.hiva_suffix_len,
        }
        loss_dict.update(decoded_logs)
        loss_dict.update(residual_flow_logs)
        return per_sample_loss, loss_dict

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t: HiVACoeffTargets,
        timestep: Tensor,
        duration_t: Tensor | None = None,
        use_cache=None,
    ) -> tuple[HiVACoeffTargets, Tensor]:
        pred, duration_output, _coeff_hidden = self._forward_hiva_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=x_t,
            timestep=timestep,
            duration_t=duration_t,
            use_cache=use_cache,
        )
        return pred, duration_output

    def predict_duration_logits(
        self,
        prefix_pad_masks,
        past_key_values,
        coeffs: HiVACoeffTargets,
        timestep,
        use_cache=None,
    ):
        if self.config.hiva_duration_prediction_type != "categorical":
            raise RuntimeError("predict_duration_logits is only available for categorical duration mode.")
        _pred, duration_logits, _coeff_hidden = self._forward_hiva_suffix_with_prefix_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            coeffs=coeffs,
            timestep=timestep,
            use_cache=use_cache,
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
        duration_t = None
        if self.config.hiva_duration_prediction_type == "continuous_fm":
            duration_t = self.sample_noise((bsize, 1, 1), device)

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
            v_t, duration_v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
                duration_t=duration_t,
            )
            x_t = {key: x_t[key] + dt * v_t[key] for key in x_t}
            if self.config.hiva_duration_prediction_type == "continuous_fm":
                duration_t = duration_t + dt * duration_v_t

        coeff_hidden = None
        final_timestep = torch.zeros(bsize, dtype=torch.float32, device=device)
        if self.config.hiva_duration_prediction_type == "continuous_fm":
            duration_raw = self.unnormalize_duration(duration_t).reshape(-1)
            duration_steps = self._duration_steps_from_continuous(duration_raw)
            if self.config.hiva_residual_enabled or self.config.hiva_residual_flow_enabled:
                _pred_final, _duration_output, coeff_hidden = self._forward_hiva_suffix_with_prefix_cache(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    coeffs=x_t,
                    timestep=final_timestep,
                    duration_t=duration_t,
                )
        else:
            _pred_final, duration_logits, coeff_hidden = self._forward_hiva_suffix_with_prefix_cache(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                coeffs=x_t,
                timestep=final_timestep,
            )
            duration_steps = self._duration_steps_from_logits(duration_logits)
        coeffs_raw = self.unnormalize_coeffs(x_t)
        base_actions = self.decode_coefficients_to_raw_actions(coeffs_raw, duration_steps)
        if self.config.hiva_residual_flow_enabled:
            raw_actions = self._sample_residual_flow_actions(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                base_actions_raw=base_actions,
                coeff_hidden=coeff_hidden,
            )
        else:
            raw_actions, _residual_logs = self._apply_action_residual(base_actions, coeff_hidden)
        return raw_actions, duration_steps

    def _decode_basis_to_raw_actions(
        self,
        coeffs_raw: HiVACoeffTargets,
        phi_tr: Tensor,
        phi_rot: Tensor | None = None,
        phi_grip: Tensor | None = None,
    ) -> Tensor:
        """Decode raw LIBERO actions from B-spline bases shared by a batch.

        Translation and rotation coefficients decode cumulative labels; gripper coefficients decode
        absolute continuous commands directly. Old sidecars use one basis for all modalities; mixed-degree
        sidecars can pass separate translation, rotation, and gripper bases with the same horizon.
        """
        phi_rot = phi_tr if phi_rot is None else phi_rot
        phi_grip = phi_tr if phi_grip is None else phi_grip
        bsize = coeffs_raw["tr"].shape[0]
        device = coeffs_raw["tr"].device
        dtype = coeffs_raw["tr"].dtype
        horizon = int(phi_tr.shape[0])
        if int(phi_rot.shape[0]) != horizon or int(phi_grip.shape[0]) != horizon:
            raise ValueError("Translation, rotation, and gripper bases must share the same horizon.")
        raw_actions = torch.zeros(
            bsize,
            horizon,
            self.config.action_feature.shape[0],
            dtype=dtype,
            device=device,
        )
        eye = torch.eye(3, dtype=dtype, device=device)

        p_hat = torch.einsum("dk,bkc->bdc", phi_tr, coeffs_raw["tr"])
        p0 = torch.zeros(bsize, 1, 3, dtype=dtype, device=device)
        tr_delta = torch.diff(torch.cat([p0, p_hat], dim=1), dim=1)

        rho_hat = torch.einsum("dk,bkc->bdc", phi_rot, coeffs_raw["rot"])
        rot_mats = _rotvec_to_matrix(rho_hat.reshape(-1, 3)).reshape(bsize, horizon, 3, 3)
        prev = torch.cat([eye.expand(bsize, 1, 3, 3), rot_mats[:, :-1]], dim=1)
        delta = prev.transpose(-1, -2) @ rot_mats
        raw_rot = _matrix_to_rotvec(delta.reshape(-1, 3, 3)).reshape(bsize, horizon, 3)
        raw_rot = raw_rot / float(self.config.hiva_rot_scale_eta)

        grip = torch.einsum("dk,bkc->bdc", phi_grip, coeffs_raw["grip"]).clamp(-1.0, 1.0)
        decoded = torch.cat([tr_delta, raw_rot, grip], dim=-1)
        if decoded.shape[-1] < raw_actions.shape[-1]:
            decoded = F.pad(decoded, (0, raw_actions.shape[-1] - decoded.shape[-1]))
        return decoded[..., : raw_actions.shape[-1]]

    def decode_coefficients_to_raw_actions(self, coeffs_raw: HiVACoeffTargets, duration_steps: Tensor) -> Tensor:
        """Decode coefficients to raw LIBERO action chunks.

        Legacy duration_specific mode decodes with the predicted-duration basis.
        Canonical HP/MT/LP-MT modes always decode a full hiva_fit_horizon chunk with the canonical
        basis; duration only controls how many prefix actions are queued for execution.
        """
        device = duration_steps.device
        dtype = coeffs_raw["tr"].dtype
        if self._uses_canonical_basis():
            phi_tr, phi_rot, phi_grip = self._canonical_basis_set(device=device, dtype=dtype)
            return self._decode_basis_to_raw_actions(coeffs_raw, phi_tr, phi_rot, phi_grip)

        bsize = duration_steps.shape[0]
        raw_actions = torch.zeros(
            bsize,
            self.config.hiva_fit_horizon,
            self.config.action_feature.shape[0],
            dtype=dtype,
            device=device,
        )

        for duration in self.config.hiva_duration_classes:
            mask = duration_steps == int(duration)
            if not mask.any():
                continue
            phi_tr, phi_rot, phi_grip = self._basis_set(int(duration), device=device, dtype=dtype)
            sub_coeffs = {
                "tr": coeffs_raw["tr"][mask],
                "rot": coeffs_raw["rot"][mask],
                "grip": coeffs_raw["grip"][mask],
            }
            decoded = self._decode_basis_to_raw_actions(sub_coeffs, phi_tr, phi_rot, phi_grip)
            raw_actions[mask, : int(duration), : decoded.shape[-1]] = decoded[:, : int(duration)]

        return raw_actions
