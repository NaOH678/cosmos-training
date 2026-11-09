"""Verify the separately cloned LiteGen revision before installation or training."""

import json
import os
import subprocess
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def check_source():
    lock = json.loads((BUNDLE / "config/litegen.lock.json").read_text())
    root = Path(os.environ.get("LITEGEN_ROOT", BUNDLE.parent / "LiteGen.dev")).resolve()
    if root == BUNDLE or BUNDLE in root.parents:
        raise ValueError("Clone LiteGen outside this project and set LITEGEN_ROOT to its checkout.")
    if not (root / "litegen/__init__.py").is_file():
        raise ValueError(f"Missing LiteGen checkout: {root}; see docs/litegen_integration.md")
    # Clean only Git's subprocess environment; CANN libraries remain available to training.
    env = {key: value for key, value in os.environ.items() if key not in {"LD_LIBRARY_PATH", "LD_PRELOAD"}}

    def git(*args):
        return subprocess.check_output(["/usr/bin/git", "-C", str(root), *args], env=env, text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError(f"LITEGEN_ROOT must be the Git checkout root: {root}")
    commit = git("rev-parse", "HEAD")
    if commit != lock["commit"]:
        raise ValueError(f"LiteGen revision mismatch: expected {lock['commit']}, got {commit}; see docs/litegen_integration.md")
    if git("status", "--porcelain", "--untracked-files=all", "--", "litegen", "pyproject.toml"):
        raise ValueError("LiteGen runtime/packaging has uncommitted changes; validate and pin a committed revision.")
    return {**lock, "root": str(root), "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(check_source(), ensure_ascii=False, indent=2))
