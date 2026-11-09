"""Validate all pinned distributions without importing Torch or initializing NPU."""

import argparse
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

from check_litegen import check_source


def locked_versions(path):
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", path.read_text(), re.MULTILINE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
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
    report = {
        "status": "failed" if errors else "passed",
        "python": sys.version,
        "prefix": sys.prefix,
        "architecture": platform.machine(),
        "packages": actual,
        "mismatches": errors,
        "litegen": check_source(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Locked distributions: {len(expected)}; mismatches: {len(errors)}")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
