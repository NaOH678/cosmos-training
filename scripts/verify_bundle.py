"""Check exported file hashes without importing either training framework."""

import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    errors = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(entry["path"])
    return {"status": "failed" if errors else "passed", "files": len(manifest["files"]), "errors": errors}


if __name__ == "__main__":
    result = verify(Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
