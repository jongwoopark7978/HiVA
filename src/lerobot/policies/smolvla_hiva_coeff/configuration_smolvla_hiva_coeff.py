from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


DEFAULT_HIVA_COEFF_SIDECAR = (
    "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/"
    "libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.parquet"
)
DEFAULT_HIVA_COEFF_SUMMARY = (
    "/nfs/bigcornea.cs.stonybrook.edu/add_disk2/jongwoopark/"
    "libero_hiva_coeff_sidecar_d6_10_15_k6_all_episodes.summary.json"
)
DEFAULT_INIT_SMOLVLA_CHECKPOINT = "/home/jongwoopark/lerobot/smolvla_libero"


@PreTrainedConfig.register_subclass("smolvla_hiva_coeff")
@dataclass
class HiVACoeffSmolVLAConfig(SmolVLAConfig):
    """SmolVLA backbone with HiVA duration and B-spline coefficient suffix tokens.

    This implements the B.4 loss from the HiVA pilot note: coefficient flow matching plus duration
    cross-entropy. It deliberately does not include the B.3 path or tail losses.
    """

    # The canonical LIBERO SmolVLA checkpoint uses these architecture settings. Keeping the same
    # defaults lets the VLM/expert weights load by name from `init_smolvla_checkpoint_path`.
    chunk_size: int = 15
    n_action_steps: int = 15
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"
    load_vlm_weights: bool = False
    num_vlm_layers: int = 0
    expert_width_multiplier: float = 0.5
    prefix_length: int = 0

    # HiVA coefficient representation.
    hiva_duration_classes: tuple[int, ...] = (6, 10, 15)
    hiva_dmax: int = 15
    hiva_k: int = 6
    hiva_degree: int = 3
    hiva_rot_scale_eta: float = 0.5
    hiva_coeff_sidecar_path: str | None = DEFAULT_HIVA_COEFF_SIDECAR
    hiva_coeff_sidecar_summary_path: str | None = DEFAULT_HIVA_COEFF_SUMMARY
    hiva_normalize_coefficients: bool = True
    hiva_coeff_norm_eps: float = 1e-6
    # Coefficient basis/decode mode:
    # - "duration_specific": legacy mode. Each duration d uses its own d x K basis at
    #   sidecar fitting and inference decode time.
    # - "canonical_hp": canonical Dmax hold-pad mode. Coefficients are fit against
    #   one Dmax x K basis after holding the target after GT duration d*. During inference,
    #   duration only controls how many decoded actions are queued for execution.
    # - "canonical_mt": same architecture as canonical_hp, reserved for max-target sidecars.
    hiva_basis_mode: str = "duration_specific"

    # B.4 loss weights.
    hiva_tr_loss_weight: float = 1.0
    hiva_rot_loss_weight: float = 1.0
    hiva_grip_loss_weight: float = 1.0
    hiva_duration_noisy_loss_weight: float = 1.0
    hiva_duration_noisy_sigma: float = 0.25
    # Duration CE variants:
    # - "ce_mean": simple mean CE over the batch. This is the default for the cleaner-suffix ablation,
    #   where the duration token cannot attend coefficient tokens and is trained from prefix context.
    # - "mean": preserves the original mean(CE * noisy_weight) behavior.
    # - "duration_noisy_weights": normalizes by sum(noisy_weight), i.e. sum(CE*w)/sum(w).
    hiva_duration_loss: str | None = None
    duration_loss: str | None = None
    # Coefficient suffix attention variants:
    # - "duration_prefix": [duration][coeffs] uses mask [1,1,0,...,0]. The duration token attends
    #   only to prefix/context plus itself; coefficient tokens attend prefix, duration, and all coeffs.
    # - "full": [1,0,...,0], one bidirectional suffix block where duration and coeffs attend each other.
    # - "causal": [1,1,...,1], original causal suffix ordering for debugging only.
    #
    # Keep this as None internally so older checkpoints/scripts that only specify hiva_duration_loss="mean"
    # or "duration_noisy_weights" fall back to the original full suffix attention.
    hiva_suffix_attention: str | None = None

    # Base SmolVLA initialization.
    init_smolvla_checkpoint_path: str | None = DEFAULT_INIT_SMOLVLA_CHECKPOINT
    init_smolvla_strict: bool = False

    def __post_init__(self):
        super().__post_init__()

        self.hiva_duration_classes = tuple(int(d) for d in self.hiva_duration_classes)
        self.duration_classes = self.hiva_duration_classes
        if self.duration_loss is not None:
            self.hiva_duration_loss = self.duration_loss
        if self.hiva_duration_loss is None:
            # Legacy coefficient HiVA checkpoints/scripts predate this explicit mode and used
            # mean(CE*w(t)) with lambda=0.1. New cleaner-suffix defaults use CE mean with lambda=1.0.
            self.hiva_duration_loss = (
                "mean" if self.hiva_duration_noisy_loss_weight < 1.0 else "ce_mean"
            )
        if self.hiva_suffix_attention is None:
            self.hiva_suffix_attention = "duration_prefix" if self.hiva_duration_loss == "ce_mean" else "full"
        self.hiva_basis_mode = str(self.hiva_basis_mode)

        if len(self.hiva_duration_classes) < 2:
            raise ValueError("`hiva_duration_classes` must contain at least two durations.")
        if tuple(sorted(set(self.hiva_duration_classes))) != self.hiva_duration_classes:
            raise ValueError("`hiva_duration_classes` must be strictly increasing.")
        if any(d <= 0 for d in self.hiva_duration_classes):
            raise ValueError("All HiVA durations must be positive.")
        if self.hiva_dmax != max(self.hiva_duration_classes):
            raise ValueError(
                "`hiva_dmax` must equal the largest duration class. "
                f"Got hiva_dmax={self.hiva_dmax}, classes={self.hiva_duration_classes}."
            )
        if self.chunk_size < self.hiva_dmax:
            raise ValueError(
                "`chunk_size` must be at least `hiva_dmax` so decoded action chunks can be returned."
            )
        if self.n_action_steps < self.hiva_dmax:
            raise ValueError(
                "`n_action_steps` must be at least `hiva_dmax` so inference can execute the longest "
                "duration class."
            )
        if self.hiva_k < self.hiva_degree + 1:
            raise ValueError("`hiva_k` must be at least `hiva_degree + 1`.")
        if self.hiva_rot_scale_eta <= 0:
            raise ValueError("`hiva_rot_scale_eta` must be positive.")
        if self.hiva_coeff_norm_eps <= 0:
            raise ValueError("`hiva_coeff_norm_eps` must be positive.")
        for name in (
            "hiva_tr_loss_weight",
            "hiva_rot_loss_weight",
            "hiva_grip_loss_weight",
            "hiva_duration_noisy_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be non-negative.")
        if self.hiva_duration_noisy_sigma <= 0:
            raise ValueError("`hiva_duration_noisy_sigma` must be positive.")
        if self.hiva_duration_loss not in ("ce_mean", "mean", "duration_noisy_weights"):
            raise ValueError(
                "`hiva_duration_loss` must be one of: 'ce_mean', 'mean', 'duration_noisy_weights'. "
                f"Got {self.hiva_duration_loss!r}."
            )
        if self.hiva_suffix_attention not in ("duration_prefix", "full", "causal"):
            raise ValueError(
                "`hiva_suffix_attention` must be one of: 'duration_prefix', 'full', 'causal'. "
                f"Got {self.hiva_suffix_attention!r}."
            )
        if self.hiva_basis_mode not in ("duration_specific", "canonical_hp", "canonical_mt"):
            raise ValueError(
                "`hiva_basis_mode` must be one of: 'duration_specific', 'canonical_hp', 'canonical_mt'. "
                f"Got {self.hiva_basis_mode!r}."
            )

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.hiva_dmax))
