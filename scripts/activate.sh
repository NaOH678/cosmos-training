#!/usr/bin/env bash
# Source after bootstrap.sh, inside the destination CANN task.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Use: source %s\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

_cosmos_migration_activate() {
    local bundle prefix cann_env candidate result=0 restore_nounset=0
    bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    prefix="${COSMOS_NPU_ENV:-$bundle/.venv-npu}"
    if [[ "$(uname -m)" != aarch64 ]]; then
        printf 'Requires Linux aarch64; current architecture: %s\n' "$(uname -m)" >&2
        return 1
    fi
    if [[ ! -x "$prefix/bin/python" ]]; then
        printf 'Python environment is missing: %s; run bash run.sh --bootstrap-only on this node first.\n' "$prefix" >&2
        return 1
    fi
    export LITEGEN_ROOT="${LITEGEN_ROOT:-$bundle/../LiteGen.dev}"
    "$prefix/bin/python" "$bundle/scripts/check_litegen.py" || return $?
    cann_env="${CANN_ENV:-}"
    if [[ -z "$cann_env" ]]; then
        for candidate in /usr/local/Ascend/cann/set_env.sh \
            /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
            if [[ -r "$candidate" ]]; then cann_env="$candidate"; break; fi
        done
    fi
    if [[ -z "$cann_env" || ! -r "$cann_env" ]]; then
        printf 'Cannot find CANN set_env.sh; set CANN_ENV in config/cluster.env.\n' >&2
        return 1
    fi
    if [[ $- == *u* ]]; then restore_nounset=1; set +u; fi
    source "$cann_env" || result=$?
    if (( restore_nounset )); then set -u; fi
    if (( result != 0 )); then return "$result"; fi
    export COSMOS_NPU_ENV="$prefix" TRAIN_PYTHON="$prefix/bin/python"
    export PATH="$prefix/bin:$PATH"
    export PYTHONPATH="$bundle/cosmos-training:$LITEGEN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export LITEGEN_DEVICE=npu COSMOS_DEVICE=npu COMPILE=false
    export PYTHONNOUSERSITE=1
    if [[ -x "$prefix/bin/ffmpeg" ]]; then export IMAGEIO_FFMPEG_EXE="$prefix/bin/ffmpeg"; fi
    printf 'Portable cosmos-npu: %s\nCANN: %s\n' "$TRAIN_PYTHON" "$cann_env"
}
_cosmos_migration_activate
