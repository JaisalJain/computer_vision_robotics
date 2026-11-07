#!/usr/bin/env bash
# Clone research code that is not cleanly pip-installable into third_party/ at pinned commits.
# These repos declare numpy<2 (or similar) in their packaging metadata although the code we use
# runs on numpy 2.x, so they are imported from source via cvlab.io.add_third_party() instead of
# being pip-installed (keeps `pip check` clean). Idempotent; third_party/ is gitignored.
#
# Usage: bash scripts/setup_third_party.sh [depth-anything-3|vggt|co-tracker ...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

declare -A URL=(
  [depth-anything-3]=https://github.com/ByteDance-Seed/Depth-Anything-3
  [vggt]=https://github.com/facebookresearch/vggt
  [co-tracker]=https://github.com/facebookresearch/co-tracker
)
declare -A COMMIT=(
  [depth-anything-3]=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
  [vggt]=a288dd0f14786c93483e45524328726ab7b1b4ce
  [co-tracker]=82e02e8029753ad4ef13cf06be7f4fc5facdda4d
)

clone() {
  local name=$1 dir="third_party/$1"
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet --filter=blob:none "${URL[$name]}" "$dir"
  fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "${COMMIT[$name]}" ]; then
    git -C "$dir" fetch --quiet origin "${COMMIT[$name]}" 2>/dev/null || git -C "$dir" fetch --quiet origin
    git -C "$dir" -c advice.detachedHead=false checkout --quiet "${COMMIT[$name]}"
  fi
  echo "$name @ $(git -C "$dir" rev-parse --short HEAD)"
}

if [ "$#" -eq 0 ]; then set -- depth-anything-3 vggt co-tracker; fi
for name in "$@"; do
  [ -n "${URL[$name]:-}" ] || { echo "unknown third-party repo: $name" >&2; exit 1; }
  clone "$name"
done
