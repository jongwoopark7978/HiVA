#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-$HOME/lerobot}"
DST="${DST:-$HOME/HiVA}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-}"
LEROBOT_COMMIT_MESSAGE="${LEROBOT_COMMIT_MESSAGE:-Update lerobot sources for HiVA sync}"
HIVA_COMMIT_MESSAGE="${HIVA_COMMIT_MESSAGE:-Sync from lerobot}"

require_git_repo() {
  local repo="$1"
  if ! git -C "$repo" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "Not a git repository: $repo" >&2
    exit 1
  fi
}

copy_or_delete_path() {
  local status="$1"
  local path="$2"
  local new_path="${3:-}"

  case "$status" in
    D)
      rm -f "$DST/$path"
      printf '%s\n' "$path" >>"$STAGE_PATHS"
      ;;
    R*)
      rm -f "$DST/$path"
      mkdir -p "$DST/$(dirname "$new_path")"
      cp -a "$SRC/$new_path" "$DST/$new_path"
      printf '%s\n%s\n' "$path" "$new_path" >>"$STAGE_PATHS"
      ;;
    C*)
      mkdir -p "$DST/$(dirname "$new_path")"
      cp -a "$SRC/$new_path" "$DST/$new_path"
      printf '%s\n' "$new_path" >>"$STAGE_PATHS"
      ;;
    *)
      mkdir -p "$DST/$(dirname "$path")"
      cp -a "$SRC/$path" "$DST/$path"
      printf '%s\n' "$path" >>"$STAGE_PATHS"
      ;;
  esac
}

require_git_repo "$SRC"
require_git_repo "$DST"

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git -C "$DST" branch --show-current)"
fi
if [[ -z "$BRANCH" ]]; then
  echo "Could not determine HiVA branch. Set BRANCH=<branch> and rerun." >&2
  exit 1
fi

if [[ -n "$(git -C "$SRC" status --porcelain)" ]]; then
  git -C "$SRC" add -A
  git -C "$SRC" commit -m "$LEROBOT_COMMIT_MESSAGE"
else
  echo "No changed or untracked files to commit in $SRC"
fi

SOURCE_COMMIT="$(git -C "$SRC" rev-parse HEAD)"
SOURCE_COMMIT_SHORT="$(git -C "$SRC" rev-parse --short HEAD)"
CHANGE_LIST="$(mktemp)"
STAGE_PATHS="$(mktemp)"
trap 'rm -f "$CHANGE_LIST" "$STAGE_PATHS"' EXIT

git -C "$SRC" diff-tree -r --name-status --no-commit-id "$SOURCE_COMMIT" >"$CHANGE_LIST"

if [[ ! -s "$CHANGE_LIST" ]]; then
  echo "Source commit $SOURCE_COMMIT_SHORT has no file changes to sync."
else
  while IFS=$'\t' read -r status path new_path; do
    copy_or_delete_path "$status" "$path" "${new_path:-}"
  done <"$CHANGE_LIST"
fi

printf '%s\n' "$SOURCE_COMMIT" >"$DST/LEROBOT_SOURCE_COMMIT.txt"
printf '%s\n' "LEROBOT_SOURCE_COMMIT.txt" >>"$STAGE_PATHS"

if [[ -s "$STAGE_PATHS" ]]; then
  sort -u "$STAGE_PATHS" | git -C "$DST" add -A --pathspec-from-file=-
fi

if ! git -C "$DST" diff --cached --quiet; then
  git -C "$DST" commit \
    -m "$HIVA_COMMIT_MESSAGE $SOURCE_COMMIT_SHORT" \
    -m "Source lerobot commit: $SOURCE_COMMIT"
else
  echo "No HiVA file changes to commit after syncing $SOURCE_COMMIT_SHORT"
fi

git -C "$DST" push "$REMOTE" "$BRANCH"
