from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_smolvla_hiva_coeff import HiVACoeffSmolVLAConfig


def make_smolvla_hiva_coeff_pre_post_processors(
    config: HiVACoeffSmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Reuse SmolVLA observation/action processors for the coefficient policy.

    The policy predicts normalized raw LIBERO action chunks after decoding B-spline coefficients, so
    the standard SmolVLA postprocessor still performs the final ACTION unnormalization.
    """

    return make_smolvla_pre_post_processors(config=config, dataset_stats=dataset_stats)
