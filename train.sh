#!/usr/bin/env bash
# Run once per node; the launcher creates that node's training ranks.
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -r "$BUNDLE/config/cluster.env" ]]; then source "$BUNDLE/config/cluster.env"; fi
TRAIN_ENV="${TRAIN_ENV:-$BUNDLE/config/train.env}"
if [[ -f "$TRAIN_ENV" ]]; then
    source "$TRAIN_ENV"
elif [[ "$TRAIN_ENV" != "$BUNDLE/config/train.env" ]]; then
    printf 'Training configuration does not exist: %s\n' "$TRAIN_ENV" >&2
    exit 1
fi
source "$BUNDLE/scripts/activate.sh"
exec bash "$BUNDLE/cosmos-training/mrlv/launch_wam_litegen.sh" "$@"
