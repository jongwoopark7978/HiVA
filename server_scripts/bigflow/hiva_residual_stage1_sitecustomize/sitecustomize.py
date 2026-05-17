"""Residual-only stage-1 training patch for HiVA coefficient policies.

Enabled only when HIVA_TRAIN_RESIDUAL_ONLY is set to a truthy value. The patch
keeps the normal lerobot-train entrypoint intact while freezing every parameter
except model.hiva_residual_head and making the optimizer see only those tensors.
"""

from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y", "on"}


if _truthy(os.environ.get("HIVA_TRAIN_RESIDUAL_ONLY")):
    from lerobot.policies.smolvla_hiva_coeff.modeling_smolvla_hiva_coeff import (
        HiVACoeffSmolVLAPolicy,
    )

    _original_init = HiVACoeffSmolVLAPolicy.__init__

    def _residual_only_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if not getattr(self.config, "hiva_residual_enabled", False):
            raise RuntimeError("HIVA_TRAIN_RESIDUAL_ONLY requires hiva_residual_enabled=true.")
        if getattr(self.model, "hiva_residual_head", None) is None:
            raise RuntimeError("Residual-only training requested but model.hiva_residual_head is None.")

        trainable_names = []
        frozen_names = []
        for name, param in self.named_parameters():
            train_residual = "hiva_residual_head" in name
            param.requires_grad = train_residual
            if train_residual:
                trainable_names.append(name)
            else:
                frozen_names.append(name)

        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        print(
            "[HIVA residual-only patch] "
            f"trainable params: {trainable_count:,} / {total_count:,}; "
            f"trainable tensors: {len(trainable_names)}; frozen tensors: {len(frozen_names)}",
            flush=True,
        )
        print(
            "[HIVA residual-only patch] first trainable tensors: "
            f"{trainable_names[:20]}",
            flush=True,
        )

    def _residual_only_get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    HiVACoeffSmolVLAPolicy.__init__ = _residual_only_init
    HiVACoeffSmolVLAPolicy.get_optim_params = _residual_only_get_optim_params
