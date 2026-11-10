#!/usr/bin/env bash
# Recreate the environment in the destination; no relocated Conda binaries.
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
    printf 'Usage: bash scripts/bootstrap.sh [--plan]\nPython: native ARM Python 3.12; destination: COSMOS_NPU_ENV or <checkout>/.venv-npu\n'
    exit 0
fi
PYTHON="${NPU_BOOTSTRAP_PYTHON:-python3.12}"
export LITEGEN_ROOT="${LITEGEN_ROOT:-$BUNDLE/../LiteGen.dev}"
PREFIX="${COSMOS_NPU_ENV:-$BUNDLE/.venv-npu}"
[[ "$PREFIX" = /* ]] || PREFIX="$BUNDLE/$PREFIX"
PREFIX="$(realpath -m "$PREFIX")"
REPORT_ROOT="${NPU_BOOTSTRAP_REPORT_ROOT:-$BUNDLE/outputs/environment}"
if [[ "${1:-}" == --plan ]]; then
    printf 'Bootstrap Python: %s\nEnvironment: %s\nLock: %s\n' "$PYTHON" "$PREFIX" "$BUNDLE/environment/requirements.lock"
    exit 0
fi
[[ $# == 0 ]] || { printf 'Unknown bootstrap arguments.\n' >&2; exit 2; }
if [[ -f "$PREFIX/.cosmos-shared" ]]; then
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    unset PYTHONPATH
    "$PREFIX/bin/python" "$BUNDLE/scripts/check_environment.py" --require-ready \
        --lock "$BUNDLE/environment/requirements.lock" --report "$REPORT_ROOT/installed.json"
    "$PREFIX/bin/python" -m pip check
    printf 'Prepared shared environment verified without installation: %s\n' "$PREFIX"
    exit 0
fi
"$PYTHON" "$BUNDLE/scripts/check_litegen.py"
"$PYTHON" - <<'PY'
import platform, sys
if platform.machine() != 'aarch64' or sys.version_info[:2] != (3, 12):
    raise SystemExit('Environment lock targets Linux aarch64 / Python 3.12; use native image Python 3.12.')
PY
mkdir -p "$(dirname "$PREFIX")" "$REPORT_ROOT"
exec 9>"$PREFIX.bootstrap.lock"
flock 9
if [[ ! -e "$PREFIX" ]]; then
    "$PYTHON" -m venv "$PREFIX"
fi
[[ -x "$PREFIX/bin/python" ]] || { printf 'Environment exists without a usable Python: %s\n' "$PREFIX" >&2; exit 1; }
"$PREFIX/bin/python" - "$PREFIX" <<'PY'
from pathlib import Path
import platform, sys
if platform.machine() != 'aarch64' or sys.version_info[:2] != (3, 12):
    raise SystemExit('Existing COSMOS_NPU_ENV must contain ARM Python 3.12; choose a new environment prefix.')
if Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit('Environment Python resolves to a different prefix; choose a new environment prefix.')
PY
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1
# Inspect/install only this venv. CANN adds unrelated profiler distributions
# through PYTHONPATH; activation restores CANN's Python paths for training.
# Keep LD_LIBRARY_PATH intact for the native interpreter and accelerator stack.
unset PYTHONPATH
READY="$PREFIX/.cosmos-npu-lock"
LOCK_HASH="$(sha256sum "$BUNDLE/environment/requirements.lock")"
EXPECTED="${LOCK_HASH%% *}:$PREFIX"
if [[ ! -r "$READY" || "$(cat "$READY")" != "$EXPECTED" ]]; then
    # antlr4 4.9.3 and iopath 0.1.10 are source-only. Build its hash-checked sdist with the
    # locked setuptools, without fetching unpinned isolated build dependencies.
    BUILD_LOCK="$PREFIX/.build-requirements.lock"
    "$PREFIX/bin/python" - "$BUNDLE/environment/requirements.lock" "$BUILD_LOCK" <<'PYBUILD'
import re, sys
from pathlib import Path
blocks = re.split(r"(?=^[A-Za-z0-9_.-]+==)", Path(sys.argv[1]).read_text(), flags=re.MULTILINE)
selected = [block for block in blocks if block.startswith(("setuptools==", "packaging=="))]
if len(selected) != 2:
    raise SystemExit("Expected setuptools and packaging in the delivered lock")
Path(sys.argv[2]).write_text("".join(selected))
PYBUILD
    "$PREFIX/bin/python" -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
        --no-deps -r "$BUILD_LOCK"
    "$PREFIX/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
        --no-binary=antlr4-python3-runtime,iopath --no-build-isolation --require-hashes \
        -r "$BUNDLE/environment/requirements.lock"
fi
# Install the external platform core without its optional CUDA/development groups.
"$PREFIX/bin/python" -m pip install --no-build-isolation --no-deps \
    -e "$LITEGEN_ROOT" -e "$BUNDLE/cosmos-training"
"$PREFIX/bin/python" -m pip check
"$PREFIX/bin/python" "$BUNDLE/scripts/check_environment.py" \
    --lock "$BUNDLE/environment/requirements.lock" \
    --report "$REPORT_ROOT/installed.json"
FFMPEG="$("$PREFIX/bin/python" - <<'PY'
from pathlib import Path
import importlib.util
root=Path(importlib.util.find_spec('imageio_ffmpeg').origin).parent/'binaries'
files=list(root.glob('ffmpeg-linux-aarch64-*'))
if len(files)!=1:
    raise SystemExit('Expected exactly one bundled ARM ffmpeg binary')
print(files[0])
PY
)"
ln -sfn "$FFMPEG" "$PREFIX/bin/ffmpeg"
printf '%s\n' "$EXPECTED" > "$READY"
printf 'Environment ready: %s\n' "$PREFIX"
