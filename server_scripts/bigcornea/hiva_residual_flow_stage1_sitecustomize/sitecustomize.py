"""Residual-flow-only stage-1 training patch for HiVA coefficient policies.

Enabled only when HIVA_TRAIN_RESIDUAL_FLOW_ONLY is truthy. The normal
lerobot-train entrypoint stays intact, but every parameter except
model.hiva_residual_flow_head.* and the optional model.hiva_residual_lm_expert.*
is frozen and the optimizer only receives those trainable tensors.
"""

from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y", "on"}


if _truthy(os.environ.get("HIVA_TRAIN_RESIDUAL_FLOW_ONLY")):
    from lerobot.policies.smolvla_hiva_coeff.modeling_smolvla_hiva_coeff import (
        HiVACoeffSmolVLAPolicy,
    )

    _original_init = HiVACoeffSmolVLAPolicy.__init__

    def _residual_flow_only_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if not getattr(self.config, "hiva_residual_flow_enabled", False):
            raise RuntimeError(
                "HIVA_TRAIN_RESIDUAL_FLOW_ONLY requires hiva_residual_flow_enabled=true."
            )
        if getattr(self.model, "hiva_residual_flow_head", None) is None:
            raise RuntimeError(
                "Residual-flow-only training requested but model.hiva_residual_flow_head is None."
            )

        trainable_names = []
        frozen_names = []
        for name, param in self.named_parameters():
            train_residual_flow = (
                "hiva_residual_flow_head" in name
                or "hiva_residual_lm_expert" in name
            )
            param.requires_grad = train_residual_flow
            if train_residual_flow:
                trainable_names.append(name)
            else:
                frozen_names.append(name)

        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        print(
            "[HIVA residual-flow-only patch] "
            f"trainable params: {trainable_count:,} / {total_count:,}; "
            f"trainable tensors: {len(trainable_names)}; frozen tensors: {len(frozen_names)}",
            flush=True,
        )
        print(
            "[HIVA residual-flow-only patch] first trainable tensors: "
            f"{trainable_names[:20]}",
            flush=True,
        )

    def _residual_flow_only_get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    HiVACoeffSmolVLAPolicy.__init__ = _residual_flow_only_init
    HiVACoeffSmolVLAPolicy.get_optim_params = _residual_flow_only_get_optim_params
