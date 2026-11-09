"""Pin real/simulation action statistics in a run and validate the resume contract."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from cosmos_framework.data.generator.action.real_normalization import RealActionNormalizer


def validate_real(config, digest, *, fk_camera_profile, variant):
    job_name = OmegaConf.select(config, "job.name", default="")
    if not job_name.endswith(f"_{variant}"):
        raise ValueError("Existing run uses a different model variant; use a new OUTPUT_ROOT")
    if variant != "wam":
        profile = OmegaConf.select(
            config,
            "dataloader_train.dataloader.datasets.singlerighthand.dataset.fk_camera_profile",
            default="legacy",
        )
        if profile != fk_camera_profile:
            raise ValueError("Existing run uses different FK camera extrinsics; use a new OUTPUT_ROOT")
    saved = OmegaConf.select(config, "dataloader_train.dataloader.datasets.singlerighthand.dataset.action_stats_sha256")
    if saved != digest:
        raise ValueError("Existing run has a different/raw action space; use a new OUTPUT_ROOT")


def validate_sim(config, stats, digest):
    for split in ("train", "val"):
        key = f"dataloader_{split}.dataloader.datasets.sim_pointfk.dataset"
        dataset = OmegaConf.select(config, key)
        if dataset is None:
            raise ValueError("Existing run is not a simulation PointFK/WAM run")
        saved = dataset.get("action_stats_sha256")
        if saved:
            if saved != digest:
                raise ValueError("Existing run has different action statistics; use a new OUTPUT_ROOT")
        else:
            # Older runs read the bundle directly. Require its recorded selection hash
            # before backfilling a missing snapshot; never infer statistics from a name.
            source = Path(dataset["bundle"]) / "normalization_train_only.json"
            selection = json.loads((Path(dataset["selection_root"]) / "manifest.json").read_text())
            if source.resolve() != Path(stats).resolve() or selection.get("action_stats_sha256") != digest:
                raise ValueError("Cannot bind legacy run to the supplied action statistics")


def prepare(stats, root, *, kind="real", fk_camera_profile="legacy", variant="wam"):
    payload = Path(stats).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if kind == "real":
        RealActionNormalizer(stats, expected_sha256=digest)
    else:
        norm = json.loads(payload)
        if not norm["source"].startswith("training episodes only") or norm.get("forward_clamp") is not None:
            raise ValueError("Simulation statistics must use training-only actions without clipping")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".action_stats.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = root / "action_stats.json"
        configs = list(root.glob("cosmos3_action/action_sft/*/config.yaml"))
        for config in configs:
            cfg = OmegaConf.load(config)
            if kind == "real":
                validate_real(cfg, digest, fk_camera_profile=fk_camera_profile, variant=variant)
            else:
                validate_sim(cfg, stats, digest)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing normalization snapshot differs; use a new OUTPUT_ROOT")
        else:
            if any(root.glob("**/checkpoints/**/.metadata")) and (kind == "real" or not configs):
                raise ValueError("Existing checkpoint without a verifiable normalization snapshot")
            temp = root / ".action_stats.json.tmp"
            temp.write_bytes(payload)
            temp.replace(target)
    print(target.resolve())
    print(digest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--kind", choices=("real", "sim"), default="real")
    p.add_argument("--fk-camera-profile", choices=("legacy", "dagger"), default="legacy")
    p.add_argument("--variant", choices=("wam", "nonlocal", "local"), default="wam")
    a = p.parse_args()
    prepare(a.stats, a.output_root, kind=a.kind, fk_camera_profile=a.fk_camera_profile, variant=a.variant)
