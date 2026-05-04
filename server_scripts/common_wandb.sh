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
