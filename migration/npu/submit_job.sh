#!/usr/bin/env bash
# Run once as the platform task command. The Python driver owns all workers.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${NPU_BOOTSTRAP_PYTHON:-}" ]]; then
    exec "$NPU_BOOTSTRAP_PYTHON" -u "$SCRIPT_DIR/job_runner.py" "$@"
fi
for candidate in python3.12 /usr/bin/python3.12 python3 /usr/bin/python3; do
    interpreter="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -n "$interpreter" ]] && "$interpreter" -c \
        'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        exec "$interpreter" -u "$SCRIPT_DIR/job_runner.py" "$@"
    fi
done
printf 'No native Python >=3.10 found in the task image; set NPU_BOOTSTRAP_PYTHON.\n' >&2
exit 2
