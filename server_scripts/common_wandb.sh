#!/usr/bin/env bash

# Shared optional W&B configuration for finetuning scripts.
#
# Usage:
#   source "${REPO_ROOT}/server_scripts/common_wandb.sh"
#   build_wandb_args
#   print_wandb_config
#   lerobot-train ... "${WANDB_ARGS[@]}"
#
# Keep WANDB_ENABLE=false by default so older launch commands behave the same.
# Do not put API keys in scripts; use an existing `wandb login`/~/.netrc or set
# WANDB_API_KEY in the shell environment before launching.

build_wandb_args() {
  WANDB_ENABLE="${WANDB_ENABLE:-false}"
  WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
  WANDB_ENTITY="${WANDB_ENTITY:-}"
  WANDB_MODE="${WANDB_MODE:-online}"
  WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
  WANDB_NOTES="${WANDB_NOTES:-}"

  WANDB_ARGS=(
    --wandb.enable="${WANDB_ENABLE}"
    --wandb.project="${WANDB_PROJECT}"
    --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
  )

  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}")
  fi

  if [[ -n "${WANDB_MODE}" ]]; then
    WANDB_ARGS+=(--wandb.mode="${WANDB_MODE}")
  fi

  if [[ -n "${WANDB_NOTES}" ]]; then
    WANDB_ARGS+=(--wandb.notes="${WANDB_NOTES}")
  fi
}

print_wandb_config() {
  echo "WANDB_ENABLE=${WANDB_ENABLE}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "WANDB_ENTITY=${WANDB_ENTITY:-<unset>}"
  echo "WANDB_MODE=${WANDB_MODE}"
  echo "WANDB_DISABLE_ARTIFACT=${WANDB_DISABLE_ARTIFACT}"
}

build_run_id() {
  # Use nanoseconds plus the shell pid so tmux/background sweeps do not collide
  # when multiple stages are launched close together.
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S_%N)_pid$$}"
}

guard_train_output_dir() {
  local output_dir="$1"
  local resume="${2:-false}"

  if [[ "${resume}" == "true" ]]; then
    if [[ ! -e "${output_dir}/checkpoints/last" ]]; then
      echo "ERROR: RESUME=true but no last checkpoint exists under: ${output_dir}" >&2
      echo "Launch with a fresh RUN_NAME/RUN_ID, or point OUTPUT_DIR to a checkpointed run." >&2
      exit 2
    fi
    return
  fi

  if [[ -e "${output_dir}" ]]; then
    echo "ERROR: OUTPUT_DIR already exists and RESUME is not true: ${output_dir}" >&2
    echo "Use a fresh RUN_NAME/RUN_ID, or set RESUME=true only for a checkpointed run." >&2
    exit 2
  fi
}
