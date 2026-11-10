"""Validate all pinned distributions without importing Torch or initializing NPU."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

from check_litegen import check_source


def locked_versions(path):
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", path.read_text(), re.MULTILINE))


def check_ready(prefix, lock):
    marker = prefix / ".cosmos-npu-lock"
    expected = f"{hashlib.sha256(lock.read_bytes()).hexdigest()}:{prefix.resolve()}"
    if not marker.is_file() or marker.read_text().strip() != expected:
        raise ValueError("Environment is incomplete, moved, or built from a different dependency lock; prepare a new environment before submitting the job.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    expected = locked_versions(args.lock)
    if not expected:
        raise ValueError("Empty environment lock")
    actual, errors = {}, {}
    for name, version in expected.items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
        if actual[name] != version:
            errors[name] = {"expected": version, "actual": actual[name]}
    environment_errors = []
    if args.require_ready:
        try:
            check_ready(Path(sys.prefix), args.lock)
        except ValueError as error:
            environment_errors.append(str(error))
        if platform.system() != "Linux" or platform.machine() != "aarch64" or sys.version_info[:2] != (3, 12):
            environment_errors.append("Requires Linux aarch64 and Python 3.12.")
    report = {
        "status": "failed" if errors or environment_errors else "passed",
        "python": sys.version,
        "prefix": sys.prefix,
        "architecture": platform.machine(),
        "packages": actual,
        "mismatches": errors,
        "environment_errors": environment_errors,
        "litegen": check_source(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Locked distributions: {len(expected)}; mismatches: {len(errors)}")
    for error in environment_errors:
        print(error, file=sys.stderr)
    return bool(errors or environment_errors)


if __name__ == "__main__":
    raise SystemExit(main())
