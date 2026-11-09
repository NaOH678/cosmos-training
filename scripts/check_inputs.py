"""Verify external WAM inputs and their source manifest fingerprints; stdlib only."""

import argparse
import hashlib
import json
import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(inventory, paths, *, full=False):
    checked, errors = [], []
    for entry in inventory["files"]:
        path = paths[entry["root"]] / entry["relative_path"]
        if not path.is_file():
            errors.append(f"Missing file: {path}")
        elif path.stat().st_size != entry["bytes"]:
            errors.append(f"Different size: {path}")
        elif entry.get("sha256") and fingerprint(path) != entry["sha256"]:
            errors.append(f"SHA256 mismatch: {path}")
        else:
            checked.append(str(path))
    for entry in inventory["directories"]:
        path = paths[entry["root"]] / entry["relative_path"]
        if not path.is_dir():
            errors.append(f"Missing directory: {path}")
    # Check all latent/frame files and a deterministic action-record sample.
    # --all-payloads checks every action file too (many remote metadata requests).
    checked_payloads = 0
    if not errors:
        data = paths["SIM_POINTFK_BUNDLE"] / "datasets/cosmos-cache"
        selection = data / "pointflow_fk_handguided1024/manifest.json"
        latent_manifest = data / "vae_window_latents/window_manifest.json"
        rows = json.loads(selection.read_text())["windows"]
        latent_rows = json.loads(latent_manifest.read_text())["episodes"]
        chosen = rows if full else rows[:: max(1, len(rows) // 32)] + rows[-1:]
        payloads = {data / "pointflow_fk_handguided1024" / row["path"] for row in chosen}
        payloads.update(data / "video_frames" / f"{row['episode']}.npy" for row in rows)
        for row in latent_rows:
            payloads.add(data / "vae_window_latents" / row["path"])
        for path in sorted(payloads):
            if not path.is_file() or path.stat().st_size == 0:
                if len(errors) < 25:
                    errors.append(f"Missing/empty payload: {path}")
            else:
                checked_payloads += 1
    return {
        "status": "failed" if errors else "passed",
        "checked_files": checked,
        "checked_payload_files": checked_payloads,
        "errors": errors,
        "scope": "manifest/config SHA256; weights, all frames/latents and selected action payload existence; no full payload hashes",
        "all_action_payloads_checked": full,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-payloads", action="store_true")
    args = parser.parse_args()
    oss = Path(os.environ.get("OSS_ROOT", "/oss")) / "shichaojian"
    model = Path(os.environ.get("EDGE_DROID_MODEL_PATH", oss / "models/cosmos3-edge-droid"))
    # A separate root for VAE allows arbitrary model/tokenizer mount locations.
    vae = Path(os.environ.get("WAN_VAE_PATH", model / "vae/Wan2.2_VAE.pth"))
    paths = {
        "EDGE_DROID_MODEL_PATH": model,
        "BASE_CHECKPOINT_PATH": Path(os.environ.get("BASE_CHECKPOINT_PATH", oss / "models/cosmos3-edge-droid-dcp")),
        "SIM_POINTFK_BUNDLE": Path(
            os.environ.get("SIM_POINTFK_BUNDLE", oss / "wam_datasets_20261029/sim_task27_ball_wam")
        ),
        "WAN_VAE_PATH": vae.parent,
    }
    inventory = json.loads((BUNDLE / "external_inputs.json").read_text())
    # Inventory stores a relative VAE filename but the override may rename it.
    for entry in inventory["files"]:
        if entry["root"] == "WAN_VAE_PATH":
            entry["relative_path"] = vae.name
    result = check(inventory, paths, full=args.all_payloads)
    report = BUNDLE / "outputs/input_check.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Input report: {report}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
