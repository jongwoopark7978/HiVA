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
    """SmolVLA backbone with HiVA B-spline coefficient suffix tokens.

    This policy supports the legacy categorical duration head and a continuous duration
    flow-matching mode. Canonical basis modes decode one fixed B-spline horizon; duration controls
    how many decoded actions are queued for execution.
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
    # Optional inference-only remap from predicted duration class to executed action horizon.
    # Example: "15:12" keeps the model's 15-step prediction but queues only 12 actions.
    hiva_duration_execution_map: str | None = None
    # Maximum executable duration. Categorical mode predicts one class from hiva_duration_classes;
    # continuous mode denoises a scalar and clamps it to [1, hiva_dmax].
    hiva_dmax: int = 15
    # B-spline fitting/decoding horizon. Defaults to hiva_dmax. LP-MT sets this larger.
    hiva_fit_horizon: int | None = None
    hiva_k: int = 6
    hiva_degree: int = 3
    # Optional per-modality B-spline degrees. When omitted, each modality uses hiva_degree.
    hiva_degree_tr: int | None = None
    hiva_degree_rot: int | None = None
    hiva_degree_grip: int | None = None
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
    # - "canonical_mt": canonical max-target mode with hiva_fit_horizon == hiva_dmax.
    # - "canonical_lp_mt": long-preview max-target mode with hiva_fit_horizon > hiva_dmax.
    hiva_basis_mode: str = "duration_specific"

    # Coefficient flow-matching loss weights.
    hiva_tr_loss_weight: float = 1.0
    hiva_rot_loss_weight: float = 1.0
    hiva_grip_loss_weight: float = 1.0

    # Duration prediction mode:
    # - "categorical": legacy learned duration query + classification head over hiva_duration_classes.
    # - "continuous_fm": duration is a noisy scalar suffix token trained with flow matching.
    hiva_duration_prediction_type: str = "categorical"

    # Legacy categorical duration settings.
    hiva_duration_noisy_loss_weight: float = 1.0
    # Optional teacher-forced duration loss from clean coefficients. When positive, the model runs
    # one extra suffix pass with normalized clean coeffs at t=0 and reuses the prefix KV cache.
    hiva_duration_clean_loss_weight: float = 0.0
    hiva_duration_noisy_sigma: float = 0.25
    # Duration CE variants:
    # - "ce_mean": simple mean CE over the batch. This is the default for the cleaner-suffix ablation,
    #   where the duration token cannot attend coefficient tokens and is trained from prefix context.
    # - "mean": preserves the original mean(CE * noisy_weight) behavior.
    # - "duration_noisy_weights": normalizes by sum(noisy_weight), i.e. sum(CE*w)/sum(w).
    hiva_duration_loss: str | None = None
    duration_loss: str | None = None

    # Categorical duration readout variants:
    # - "token": legacy path. Append one learned duration suffix token and classify its hidden state.
    # - "coeff_modality_pool": no duration suffix token. Run only coefficient suffix tokens through the
    #   action expert, mean-pool final translation/rotation/gripper hidden states separately, concatenate
    #   them, and classify duration from a residual FFN + MLP readout.
    # Continuous-FM duration ignores this field because it uses a noisy scalar duration suffix token.
    hiva_duration_readout: str = "token"

    # Continuous duration flow-matching settings.
    hiva_duration_fm_loss_weight: float = 1.0
    # Option A/default: "bounded" maps d to [-1, 1] using hiva_dmax:
    #   d_norm = 2 * (d - 1) / (hiva_dmax - 1) - 1.
    # Option B: "mean_std" maps d using duration histogram stats from the coefficient sidecar
    # summary, or explicit hiva_duration_mean/std if provided. It does not require a new sidecar
    # unless the desired target is different from the existing duration_label column.
    hiva_duration_cont_norm: str = "bounded"
    hiva_duration_mean: float | None = None
    hiva_duration_std: float | None = None
    # Coefficient suffix attention variants:
    # - "duration_prefix": [duration][coeffs] uses mask [1,1,0,...,0]. The duration token attends
    #   only to prefix/context plus itself; coefficient tokens attend prefix, duration, and all coeffs.
    # - "duration_reads_coeffs": custom asymmetric 2D mask. The duration token can read all clean/noisy
    #   coefficient tokens, while coefficient tokens cannot read the duration token.
    # - "full": [1,0,...,0], one bidirectional suffix block where duration and coeffs attend each other.
    # - "causal": [1,1,...,1], original causal suffix ordering for debugging only.
    #
    # Keep this as None internally so older checkpoints/scripts that only specify hiva_duration_loss="mean"
    # or "duration_noisy_weights" fall back to the original full suffix attention.
    hiva_suffix_attention: str | None = None

    # Duration-head architecture variants:
    # - "linear": legacy single Linear(H, num_duration_classes) head. This remains the default so
    #   existing coefficient HiVA checkpoints and finetuning scripts keep the same architecture.
    # - "residual_ffn": stronger duration head, h_tilde = h + alpha * FFN(LN(h)), then classify.
    # - "none": set automatically for continuous_fm because no categorical duration head is used.
    hiva_duration_head_type: str = "linear"
    hiva_duration_ffn_hidden_mult: float = 4.0
    hiva_duration_ffn_alpha_init: float = 0.1

    # Optional decoded raw-action auxiliary loss. Disabled by default so existing MT and LP-MT
    # scripts/checkpoints keep their coefficient-FM-only behavior unless explicitly enabled.
    hiva_decoded_action_loss_weight: float = 0.0
    hiva_decoded_tr_loss_weight: float = 1.0
    hiva_decoded_rot_loss_weight: float = 1.0
    hiva_decoded_grip_loss_weight: float = 1.0
    hiva_decoded_prefix_weight: float = 1.0
    hiva_decoded_post_duration_exec_weight: float = 0.5
    hiva_decoded_preview_weight: float = 0.1
    hiva_decoded_terminal_weight: float = 0.0
    hiva_decoded_loss_beta: float = 0.1
    hiva_decoded_tr_loss_beta: float = 0.1
    hiva_decoded_rot_loss_beta: float = 0.1
    hiva_decoded_grip_loss_beta: float = 0.1

    # Optional full-horizon action residual branch. The residual is predicted from final coefficient
    # token hidden states, added in raw ACTION space, and bounded by ACTION std from dataset stats
    # unless a modality-specific scale override is provided.
    hiva_residual_enabled: bool = False
    # Residual architecture:
    # - "token_to_time": existing learned coefficient-token -> action-time mixer.
    # - "basis_hidden_action": basis-aware action-time FFN using coeff hidden states and base actions.
    # - "basis_xattn_transformer": basis-aware action-time queries cross-attend to coeff hidden states.
    hiva_residual_mode: str = "token_to_time"
    hiva_residual_horizon: int | None = None
    hiva_residual_scale_tr: float | None = None
    hiva_residual_scale_rot: float | None = None
    hiva_residual_scale_grip: float | None = None
    hiva_residual_scale_mult: float = 1.0
    # Inference/eval-time multiplier for the residual before adding it to the decoded base action.
    # This keeps the trained residual scale intact while allowing sweeps of base + lambda * residual.
    hiva_residual_inference_weight: float = 1.0
    hiva_residual_ffn_hidden_mult: float = 4.0
    hiva_residual_token_time_hidden_mult: float = 2.0
    hiva_residual_alpha_init: float = 0.1
    hiva_residual_zero_init: bool = True
    hiva_residual_num_blocks: int = 4
    hiva_residual_cross_attn_heads: int = 4
    hiva_residual_attn_dropout: float = 0.0

    # Optional SmolVLA-style residual flow branch. This denoises a scaled residual latent
    # z* = (ACTION*_norm - ACTION_base_norm) / s, then decodes the correction as
    # ACTION_final_norm = ACTION_base_norm + lambda * s * tanh(z0_hat).
    # Conditioning variants:
    # - "v1_minimal": original SmolVLA multimodal prefix + HiVA base action + B-spline basis.
    # - "v2_coeff_xattn": v1 plus a coefficient-hidden cross-attention adapter.
    hiva_residual_flow_enabled: bool = False
    hiva_residual_flow_conditioning: str = "v1_minimal"
    hiva_residual_flow_horizon: int | None = None
    hiva_residual_flow_steps: int | None = None
    hiva_residual_flow_loss_weight: float = 1.0
    hiva_residual_flow_decoded_loss_weight: float = 1.0
    hiva_residual_flow_scale_tr: float = 3.0
    hiva_residual_flow_scale_rot: float = 3.0
    hiva_residual_flow_scale_grip: float = 0.5
    hiva_residual_flow_inference_weight: float = 1.0
    hiva_residual_flow_out_head_init: str = "copy"
    hiva_residual_flow_small_init_std: float = 1e-3
    hiva_residual_flow_basis_hidden_mult: float = 2.0
    hiva_residual_flow_basis_alpha_init: float = 0.1
    hiva_residual_flow_coeff_context_alpha_init: float = 0.1
    hiva_residual_flow_coeff_context_ffn_hidden_mult: float = 4.0
    hiva_residual_flow_coeff_context_heads: int = 4
    hiva_residual_flow_coeff_context_dropout: float = 0.0
    hiva_residual_flow_coeff_context_zero_init: bool = True
    hiva_residual_flow_decoded_loss_beta: float = 0.1
    # Detach base/coeff context during residual-flow training so Stage 1 can update only the residual denoiser.
    hiva_residual_flow_detach_base: bool = True
    hiva_residual_flow_detach_coeff_context: bool = True
    # Dual-expert residual flow. When enabled, the coefficient suffix keeps the Stage-0 HiVA
    # action expert while the residual-flow suffix uses a separate raw-action SmolVLA expert.
    # This avoids forcing one adapted expert to serve both B-spline coefficient denoising and
    # residual raw-action denoising.
    hiva_residual_flow_use_separate_expert: bool = True
    # Optional source checkpoint for residual-flow raw-action suffix modules and the separate
    # residual expert. Keep this separate from init_smolvla_checkpoint_path so Stage-0 HiVA
    # initializes the coefficient path while original SmolVLA initializes the residual path.
    hiva_residual_flow_init_smolvla_checkpoint_path: str | None = DEFAULT_INIT_SMOLVLA_CHECKPOINT
    # Base SmolVLA / Stage-0 HiVA initialization.
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
            self.hiva_suffix_attention = (
                "full"
                if (
                    self.hiva_duration_prediction_type == "continuous_fm"
                    or self.hiva_duration_readout == "coeff_modality_pool"
                )
                else "duration_prefix"
                if self.hiva_duration_loss == "ce_mean"
                else "full"
            )
        if self.hiva_residual_flow_init_smolvla_checkpoint_path == "":
            self.hiva_residual_flow_init_smolvla_checkpoint_path = None
        if self.hiva_duration_prediction_type == "continuous_fm":
            self.hiva_duration_head_type = "none"
        elif self.hiva_duration_readout == "coeff_modality_pool":
            # The pooled 3H readout always uses its own residual FFN + MLP classifier.
            self.hiva_duration_head_type = "residual_ffn"
        self.hiva_basis_mode = str(self.hiva_basis_mode)
        if self.hiva_fit_horizon is None:
            self.hiva_fit_horizon = self.hiva_dmax
        self.hiva_fit_horizon = int(self.hiva_fit_horizon)
        if self.hiva_degree_tr is None:
            self.hiva_degree_tr = self.hiva_degree
        if self.hiva_degree_rot is None:
            self.hiva_degree_rot = self.hiva_degree
        if self.hiva_degree_grip is None:
            self.hiva_degree_grip = self.hiva_degree
        self.hiva_degree_tr = int(self.hiva_degree_tr)
        self.hiva_degree_rot = int(self.hiva_degree_rot)
        self.hiva_degree_grip = int(self.hiva_degree_grip)
        if self.hiva_residual_horizon is None:
            self.hiva_residual_horizon = self.hiva_fit_horizon
        self.hiva_residual_horizon = int(self.hiva_residual_horizon)
        if self.hiva_residual_flow_horizon is None:
            self.hiva_residual_flow_horizon = self.hiva_fit_horizon
        self.hiva_residual_flow_horizon = int(self.hiva_residual_flow_horizon)
        if self.hiva_residual_flow_steps is None:
            self.hiva_residual_flow_steps = self.num_steps
        self.hiva_residual_flow_steps = int(self.hiva_residual_flow_steps)

        if len(self.hiva_duration_classes) < 2:
            raise ValueError("`hiva_duration_classes` must contain at least two durations.")
        if tuple(sorted(set(self.hiva_duration_classes))) != self.hiva_duration_classes:
            raise ValueError("`hiva_duration_classes` must be strictly increasing.")
        if any(d <= 0 for d in self.hiva_duration_classes):
            raise ValueError("All HiVA durations must be positive.")
        if self.hiva_dmax != max(self.hiva_duration_classes):
            raise ValueError(
                "`hiva_dmax` must equal the largest executable duration class. "
                f"Got hiva_dmax={self.hiva_dmax}, classes={self.hiva_duration_classes}."
            )
        if self.hiva_fit_horizon < self.hiva_dmax:
            raise ValueError("`hiva_fit_horizon` must be >= `hiva_dmax`.")
        if self.chunk_size < self.hiva_fit_horizon:
            raise ValueError(
                "`chunk_size` must be at least `hiva_fit_horizon` so decoded preview chunks can be returned."
            )
        if self.n_action_steps < self.hiva_dmax:
            raise ValueError(
                "`n_action_steps` must be at least `hiva_dmax` so inference can execute the longest "
                "duration class."
            )
        if self.hiva_k < self.hiva_degree + 1:
            raise ValueError("`hiva_k` must be at least `hiva_degree + 1`.")
        for name in ("hiva_degree_tr", "hiva_degree_rot", "hiva_degree_grip"):
            degree = getattr(self, name)
            if degree < 0:
                raise ValueError(f"`{name}` must be non-negative.")
            if self.hiva_k < degree + 1:
                raise ValueError(f"`hiva_k` must be at least `{name} + 1`.")
        if self.hiva_rot_scale_eta <= 0:
            raise ValueError("`hiva_rot_scale_eta` must be positive.")
        if self.hiva_coeff_norm_eps <= 0:
            raise ValueError("`hiva_coeff_norm_eps` must be positive.")
        for name in (
            "hiva_tr_loss_weight",
            "hiva_rot_loss_weight",
            "hiva_grip_loss_weight",
            "hiva_duration_noisy_loss_weight",
            "hiva_duration_clean_loss_weight",
            "hiva_duration_fm_loss_weight",
            "hiva_decoded_action_loss_weight",
            "hiva_decoded_tr_loss_weight",
            "hiva_decoded_rot_loss_weight",
            "hiva_decoded_grip_loss_weight",
            "hiva_decoded_prefix_weight",
            "hiva_decoded_post_duration_exec_weight",
            "hiva_decoded_preview_weight",
            "hiva_decoded_terminal_weight",
            "hiva_residual_flow_loss_weight",
            "hiva_residual_flow_decoded_loss_weight",
            "hiva_residual_flow_scale_tr",
            "hiva_residual_flow_scale_rot",
            "hiva_residual_flow_scale_grip",
            "hiva_residual_flow_inference_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be non-negative.")
        for name in (
            "hiva_residual_scale_tr",
            "hiva_residual_scale_rot",
            "hiva_residual_scale_grip",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"`{name}` must be non-negative when provided.")
        if self.hiva_duration_noisy_sigma <= 0:
            raise ValueError("`hiva_duration_noisy_sigma` must be positive.")
        if self.hiva_duration_loss not in ("ce_mean", "mean", "duration_noisy_weights"):
            raise ValueError(
                "`hiva_duration_loss` must be one of: 'ce_mean', 'mean', 'duration_noisy_weights'. "
                f"Got {self.hiva_duration_loss!r}."
            )
        if self.hiva_duration_prediction_type not in ("categorical", "continuous_fm"):
            raise ValueError(
                "`hiva_duration_prediction_type` must be one of: 'categorical', 'continuous_fm'. "
                f"Got {self.hiva_duration_prediction_type!r}."
            )
        if self.hiva_duration_readout not in ("token", "coeff_modality_pool"):
            raise ValueError(
                "`hiva_duration_readout` must be one of: 'token', 'coeff_modality_pool'. "
                f"Got {self.hiva_duration_readout!r}."
            )
        if self.hiva_duration_cont_norm not in ("bounded", "mean_std"):
            raise ValueError(
                "`hiva_duration_cont_norm` must be one of: 'bounded', 'mean_std'. "
                f"Got {self.hiva_duration_cont_norm!r}."
            )
        if self.hiva_suffix_attention not in ("duration_prefix", "duration_reads_coeffs", "full", "causal"):
            raise ValueError(
                "`hiva_suffix_attention` must be one of: 'duration_prefix', 'duration_reads_coeffs', "
                f"'full', 'causal'. Got {self.hiva_suffix_attention!r}."
            )
        if self.hiva_duration_head_type not in ("linear", "residual_ffn", "none"):
            raise ValueError(
                "`hiva_duration_head_type` must be one of: 'linear', 'residual_ffn', 'none'. "
                f"Got {self.hiva_duration_head_type!r}."
            )
        if self.hiva_duration_ffn_hidden_mult <= 0:
            raise ValueError("`hiva_duration_ffn_hidden_mult` must be positive.")
        if self.hiva_duration_ffn_alpha_init < 0:
            raise ValueError("`hiva_duration_ffn_alpha_init` must be non-negative.")
        if self.hiva_residual_horizon <= 0 or self.hiva_residual_horizon > self.hiva_fit_horizon:
            raise ValueError("`hiva_residual_horizon` must be in [1, hiva_fit_horizon].")
        if self.hiva_residual_flow_horizon <= 0 or self.hiva_residual_flow_horizon > self.hiva_fit_horizon:
            raise ValueError("`hiva_residual_flow_horizon` must be in [1, hiva_fit_horizon].")
        if self.hiva_residual_flow_steps <= 0:
            raise ValueError("`hiva_residual_flow_steps` must be positive.")
        if self.hiva_residual_flow_conditioning not in ("v1_minimal", "v2_coeff_xattn"):
            raise ValueError(
                "`hiva_residual_flow_conditioning` must be one of: 'v1_minimal', 'v2_coeff_xattn'. "
                f"Got {self.hiva_residual_flow_conditioning!r}."
            )
        if self.hiva_residual_flow_out_head_init not in ("copy", "small", "zero"):
            raise ValueError(
                "`hiva_residual_flow_out_head_init` must be one of: 'copy', 'small', 'zero'. "
                f"Got {self.hiva_residual_flow_out_head_init!r}."
            )
        if self.hiva_residual_flow_small_init_std <= 0:
            raise ValueError("`hiva_residual_flow_small_init_std` must be positive.")
        if self.hiva_residual_flow_basis_hidden_mult <= 0:
            raise ValueError("`hiva_residual_flow_basis_hidden_mult` must be positive.")
        if self.hiva_residual_flow_basis_alpha_init < 0:
            raise ValueError("`hiva_residual_flow_basis_alpha_init` must be non-negative.")
        if self.hiva_residual_flow_coeff_context_alpha_init < 0:
            raise ValueError("`hiva_residual_flow_coeff_context_alpha_init` must be non-negative.")
        if self.hiva_residual_flow_coeff_context_ffn_hidden_mult <= 0:
            raise ValueError("`hiva_residual_flow_coeff_context_ffn_hidden_mult` must be positive.")
        if self.hiva_residual_flow_coeff_context_heads <= 0:
            raise ValueError("`hiva_residual_flow_coeff_context_heads` must be positive.")
        if not 0 <= self.hiva_residual_flow_coeff_context_dropout <= 1:
            raise ValueError("`hiva_residual_flow_coeff_context_dropout` must be in [0, 1].")
        if self.hiva_residual_flow_decoded_loss_beta <= 0:
            raise ValueError("`hiva_residual_flow_decoded_loss_beta` must be positive.")
        if self.hiva_residual_enabled and self.hiva_residual_flow_enabled:
            raise ValueError(
                "Enable either the deterministic HiVA residual branch or the residual-flow branch, not both."
            )
        if self.hiva_residual_mode not in (
            "token_to_time",
            "basis_hidden_action",
            "basis_xattn_transformer",
        ):
            raise ValueError(
                "`hiva_residual_mode` must be one of: 'token_to_time', 'basis_hidden_action', "
                f"'basis_xattn_transformer'. Got {self.hiva_residual_mode!r}."
            )
        if self.hiva_residual_scale_mult <= 0:
            raise ValueError("`hiva_residual_scale_mult` must be positive.")
        if self.hiva_residual_inference_weight < 0:
            raise ValueError("`hiva_residual_inference_weight` must be non-negative.")
        if self.hiva_residual_ffn_hidden_mult <= 0:
            raise ValueError("`hiva_residual_ffn_hidden_mult` must be positive.")
        if self.hiva_residual_token_time_hidden_mult <= 0:
            raise ValueError("`hiva_residual_token_time_hidden_mult` must be positive.")
        if self.hiva_residual_alpha_init < 0:
            raise ValueError("`hiva_residual_alpha_init` must be non-negative.")
        if self.hiva_residual_num_blocks <= 0:
            raise ValueError("`hiva_residual_num_blocks` must be positive.")
        if self.hiva_residual_cross_attn_heads <= 0:
            raise ValueError("`hiva_residual_cross_attn_heads` must be positive.")
        if not 0 <= self.hiva_residual_attn_dropout <= 1:
            raise ValueError("`hiva_residual_attn_dropout` must be in [0, 1].")
        if self.hiva_basis_mode not in ("duration_specific", "canonical_hp", "canonical_mt", "canonical_lp_mt"):
            raise ValueError(
                "`hiva_basis_mode` must be one of: 'duration_specific', 'canonical_hp', "
                f"'canonical_mt', 'canonical_lp_mt'. Got {self.hiva_basis_mode!r}."
            )
        if self.hiva_basis_mode == "canonical_lp_mt" and self.hiva_fit_horizon <= self.hiva_dmax:
            raise ValueError("`canonical_lp_mt` requires hiva_fit_horizon > hiva_dmax.")
        if self.hiva_residual_flow_enabled and self.hiva_basis_mode == "duration_specific":
            raise ValueError("Residual-flow Stage 1 requires a canonical HiVA basis mode.")
        if self.hiva_decoded_loss_beta <= 0:
            raise ValueError("`hiva_decoded_loss_beta` must be positive.")
        if self.hiva_decoded_tr_loss_beta <= 0:
            raise ValueError("`hiva_decoded_tr_loss_beta` must be positive.")
        if self.hiva_decoded_rot_loss_beta <= 0:
            raise ValueError("`hiva_decoded_rot_loss_beta` must be positive.")
        if self.hiva_decoded_grip_loss_beta <= 0:
            raise ValueError("`hiva_decoded_grip_loss_beta` must be positive.")
        if self.hiva_decoded_action_loss_weight > 0 and self.hiva_basis_mode == "duration_specific":
            raise ValueError("Decoded action loss is intended for canonical HiVA basis modes.")
        if self.hiva_duration_prediction_type == "continuous_fm":
            if self.hiva_basis_mode == "duration_specific":
                raise ValueError("continuous_fm duration requires a canonical hiva_basis_mode.")
            if self.hiva_duration_readout != "token":
                raise ValueError("`hiva_duration_readout` is only configurable for categorical duration.")
            if self.hiva_duration_clean_loss_weight != 0:
                raise ValueError("`hiva_duration_clean_loss_weight` must be 0 for continuous_fm duration.")
            if self.hiva_suffix_attention not in ("full", "duration_reads_coeffs"):
                raise ValueError(
                    "continuous_fm duration currently supports hiva_suffix_attention='full' or "
                    f"'duration_reads_coeffs'. Got {self.hiva_suffix_attention!r}."
                )
        elif self.hiva_duration_head_type == "none":
            raise ValueError("`hiva_duration_head_type='none'` is only valid for continuous_fm duration.")

    @property
    def action_delta_indices(self) -> list:
        # The dataset must provide enough future actions for the coefficient fitting/preview horizon.
        return list(range(self.hiva_fit_horizon))
