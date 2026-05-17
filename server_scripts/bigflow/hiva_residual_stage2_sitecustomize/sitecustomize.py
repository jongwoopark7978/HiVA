"""HiVA staged training freeze hooks for LeRobot training.

This module is loaded automatically when its directory is first on PYTHONPATH.
It keeps the normal `lerobot-train` entrypoint intact while controlling which
HiVA parameters are trainable for residual-adapter experiments.

Supported modes:
  HIVA_TRAIN_RESIDUAL_ONLY=true
      Stage 1. Train only model.hiva_residual_head.*

  HIVA_TRAIN_STAGE2_ACTION_HEADS=true
      Stage 2. Train residual + coefficient output heads + duration head:
        model.hiva_residual_head.*
        model.hiva_tr_out_proj.*
        model.hiva_rot_out_proj.*
        model.hiva_grip_out_proj.*
        model.hiva_duration_head.*
      Everything else is frozen, including the VLM/action expert transformer,
      coefficient input projections, and action_time_mlp_in/out.

Optional override:
  HIVA_STAGE2_TRAINABLE_SUBSTRINGS="model.foo,model.bar"
      Comma-separated substring allowlist for Stage 2 trainable parameters.
"""

from __future__ import annotations

import os
from typing import Iterable


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y", "on"}


def _parse_csv(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _install_hiva_freeze_patch(*, mode: str, trainable_substrings: Iterable[str]) -> None:
    from lerobot.policies.smolvla_hiva_coeff.modeling_smolvla_hiva_coeff import (
        HiVACoeffSmolVLAPolicy,
    )

    trainable_substrings = tuple(trainable_substrings)
    _original_init = HiVACoeffSmolVLAPolicy.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if mode in {"stage1_residual_only", "stage2_action_heads"}:
            if not getattr(self.config, "hiva_residual_enabled", False):
                raise RuntimeError(f"{mode} requires hiva_residual_enabled=true.")
            if getattr(self.model, "hiva_residual_head", None) is None:
                raise RuntimeError(f"{mode} requested but model.hiva_residual_head is None.")

        trainable_names: list[str] = []
        frozen_names: list[str] = []
        for name, param in self.named_parameters():
            should_train = any(substr in name for substr in trainable_substrings)
            param.requires_grad = should_train
            if should_train:
                trainable_names.append(name)
            else:
                frozen_names.append(name)

        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        print(
            f"[HiVA {mode} patch] trainable params: {trainable_count:,} / {total_count:,}; "
            f"trainable tensors: {len(trainable_names)}; frozen tensors: {len(frozen_names)}",
            flush=True,
        )
        print(
            f"[HiVA {mode} patch] trainable substrings: {list(trainable_substrings)}",
            flush=True,
        )
        print(
            f"[HiVA {mode} patch] first trainable tensors: {trainable_names[:40]}",
            flush=True,
        )
        if not trainable_names:
            raise RuntimeError(f"{mode} patch found zero trainable tensors. Check module names.")

    def _get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    HiVACoeffSmolVLAPolicy.__init__ = _patched_init
    HiVACoeffSmolVLAPolicy.get_optim_params = _get_optim_params


_stage1 = _truthy(os.environ.get("HIVA_TRAIN_RESIDUAL_ONLY"))
_stage2 = _truthy(os.environ.get("HIVA_TRAIN_STAGE2_ACTION_HEADS"))

if _stage1 and _stage2:
    raise RuntimeError("Set only one of HIVA_TRAIN_RESIDUAL_ONLY or HIVA_TRAIN_STAGE2_ACTION_HEADS.")

if _stage1:
    _install_hiva_freeze_patch(
        mode="stage1_residual_only",
        trainable_substrings=("model.hiva_residual_head",),
    )
elif _stage2:
    override = _parse_csv(os.environ.get("HIVA_STAGE2_TRAINABLE_SUBSTRINGS"))
    trainable = override or [
        "model.hiva_residual_head",
        "model.hiva_tr_out_proj",
        "model.hiva_rot_out_proj",
        "model.hiva_grip_out_proj",
        "model.hiva_duration_head",
    ]
    _install_hiva_freeze_patch(mode="stage2_action_heads", trainable_substrings=trainable)
