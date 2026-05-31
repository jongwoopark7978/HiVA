#!/usr/bin/env bash

setup_hf_datasets_cache() {
  local min_free_gb="${HF_DATASETS_CACHE_MIN_FREE_GB:-100}"
  local fallback_root="${HF_CACHE_ROOT:-/nfs/bigcornea.cs.stonybrook.edu/add_disk3/jongwoopark/cache}"
  local tmp_default="/tmp/jongwoo_hf_datasets_cache"
  local chosen="${HF_DATASETS_CACHE:-${tmp_default}}"

  mkdir -p "${chosen}" 2>/dev/null || chosen="${fallback_root}/huggingface/datasets"

  local available_kb
  available_kb="$(df -Pk "${chosen}" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -z "${available_kb}" || "${available_kb}" -lt $((min_free_gb * 1024 * 1024)) ]]; then
    if [[ "${chosen}" == "${tmp_default}" || -z "${HF_DATASETS_CACHE:-}" ]]; then
      chosen="${fallback_root}/huggingface/datasets"
      mkdir -p "${chosen}"
      available_kb="$(df -Pk "${chosen}" 2>/dev/null | awk 'NR==2 {print $4}')"
      if [[ -z "${available_kb}" || "${available_kb}" -lt $((min_free_gb * 1024 * 1024)) ]]; then
        echo "ERROR: fallback HF_DATASETS_CACHE=${chosen} has less than ${min_free_gb}GB free." >&2
        echo "Set HF_CACHE_ROOT or HF_DATASETS_CACHE to a larger filesystem." >&2
        exit 2
      fi
    else
      echo "ERROR: HF_DATASETS_CACHE=${chosen} has less than ${min_free_gb}GB free." >&2
      echo "Set HF_DATASETS_CACHE to a larger filesystem or lower HF_DATASETS_CACHE_MIN_FREE_GB." >&2
      exit 2
    fi
  fi

  export HF_DATASETS_CACHE="${chosen}"
  export TMPDIR="${TMPDIR:-${fallback_root}/tmp}"
  mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
}
