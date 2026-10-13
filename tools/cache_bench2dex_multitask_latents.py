"""Robot-independent Bench2Dex RGB/window audit and task21-style accelerated VAE cache."""

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tools.bench2dex_rgb_contract import CAMERAS, compose_rgb

REPO = Path(__file__).resolve().parents[1]
TASKS = {
    "80": ("task80_desk", "80_gaming_desk_setup"),
    "07": ("task07_citrus", "07_citrus_plate_loading"),
    "06": ("task06_fruit", "06_fruit_bowl_loading"),
    "27": ("task27_ball", "27_ball_box_loading"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def semantic_contract(contract, *, encoding=False):
    """Code hashes describe provenance; input hashes bind each episode's data."""
    ignored = {"code_sha256"}
    if encoding:
        # The aggregate changes when preparation code is reorganized. The
        # per-episode alignment/RGB/state hashes are checked separately below.
        ignored.add("alignment_manifest_sha256")
    result = {k: v for k, v in contract.items() if k not in ignored}
    if encoding:
        result.setdefault("encoding_revision", 1)
    return result


def plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    return value


def decode_rgb(dataset, index):
    import cv2

    raw = np.asarray(dataset[index])
    if raw.ndim == 1:
        bgr = cv2.imdecode(raw.astype(np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Invalid JPEG: {dataset.name}/{index}")
        raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f"Invalid RGB shape: {dataset.name}/{index}")
    return raw


def prepare_episode(job):
    import cv2
    import h5py

    source, expected, output, contract = job
    source, output = Path(source), Path(output)
    name = source.stem
    cv2.setNumThreads(1)
    meta_path = output / "alignment" / f"{name}.json"
    if meta_path.exists():
        record = json.loads(meta_path.read_text())
        if (
            semantic_contract(record["preparation_contract"]) != semantic_contract(contract)
            or record["source_sha256"] != expected
        ):
            raise ValueError(f"Incompatible prepared episode: {name}; choose a fresh output")
        st = source.stat()
        if record["source_stat"] != [st.st_size, st.st_mtime_ns]:
            if digest(source) != expected:
                raise ValueError(f"Source changed: {source}")
        for key in ("rgb", "state", "calibration"):
            path = output / record[key]["path"]
            if not path.exists() or path.stat().st_size != record[key]["bytes"]:
                raise ValueError(f"Missing/truncated prepared {key}: {name}")
        return record
    source_hash = digest(source)
    if source_hash != expected:
        raise ValueError(f"HDF5 SHA256 mismatch: {source}")
    with h5py.File(source) as f:
        meta = {k: plain(v[()]) for k, v in f["meta"].items() if isinstance(v, h5py.Dataset)}
        if meta["scene_name"] != contract["task"] or float(meta["effective_fps"]) != 20:
            raise ValueError(f"Wrong task or effective FPS: {source}")
        if plain(f["action/control_mode"][()]) != "joint_position" or plain(f["action/action_type"][()]) != "absolute":
            raise ValueError(f"Unexpected action semantics: {source}")
        steps = np.asarray(f["time/sim_step"][:])
        if not np.all(np.diff(steps) > 0):
            raise ValueError(f"Nonmonotonic frames: {source}")
        marker = int(meta["homing_start_sim_step"])
        if marker < 0:
            raise ValueError(f"Missing homing marker: {source}")
        n = int(np.searchsorted(steps, marker))
        if n < 33:
            raise ValueError(f"Too short before homing: {source}")
        names = plain(f["robot/joint_names"][:])
        action_names = plain(f["action/action_names"][:])
        if (
            len(names) != len(set(names))
            or len(action_names) != len(set(action_names))
            or set(names) != set(action_names)
        ):
            raise ValueError(f"Ambiguous state/action names: {source}")
        state = np.asarray(f["robot/qpos"][:n], np.float32)
        action = np.asarray(f["action/commanded"][:n], np.float32)[:, [action_names.index(k) for k in names]]
        if state.shape != (n, len(names)) or action.shape != state.shape or not np.isfinite(state).all():
            raise ValueError(f"Malformed state/action: {source}")
        valid = np.asarray(f["action/action_valid"][:n], bool)
        if valid.shape != (n,):
            raise ValueError(f"Unexpected action validity shape: {source}")
        valid &= np.isfinite(action).all(1)
        valid[:-1] &= np.diff(steps[:n]) == int(meta["step_stride"])
        # Same rule as task21: 33 observations and 32 valid outgoing action transitions.
        starts = np.arange(n - 32, dtype=np.int64)
        bad = np.r_[0, np.cumsum(~valid)]
        starts = starts[bad[starts + 32] == bad[starts]]
        if not len(starts):
            raise ValueError(f"No valid windows: {source}")
        action[~valid] = 0
        state_path = output / "episodes" / f"{name}.npz"
        with state_path.with_suffix(".tmp").open("wb") as stream:
            np.savez(stream, state=state, action=action, action_valid=valid, sim_step=steps[:n], joint_names=names)
        os.replace(state_path.with_suffix(".tmp"), state_path)
        # Preserve full per-frame camera calibration, including moving wrist cameras and fisheye fields.
        calibration = {"source_sim_step": steps, "retained_frame_ids": np.arange(n), "window_start_frames": starts}
        camera_info = {}
        for camera in CAMERAS:
            group = f[f"cameras/{camera}"]
            if len(group["rgb"]) != len(steps):
                raise ValueError(f"Camera/frame count mismatch: {source}/{camera}")
            fields = {}
            for key, value in group.items():
                if key == "rgb" or not isinstance(value, h5py.Dataset):
                    continue
                if value.dtype.kind in "biufc":
                    calibration[f"{camera}__{key}"] = value[()]
                    fields[key] = dict(npz_key=f"{camera}__{key}", shape=list(value.shape))
                else:
                    fields[key] = plain(value[()])
            camera_info[camera] = dict(fields=fields, rgb_shape_hwc=list(decode_rgb(group["rgb"], 0).shape))
        calibration_path = output / "alignment" / f"{name}.calibration.npz"
        with calibration_path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, **calibration)
        os.replace(calibration_path.with_suffix(".tmp"), calibration_path)
        rgb_path = output / "video_frames" / f"{name}.npy"
        shape = (n, 3, 720, 640)
        pixel_hash = hashlib.sha256()
        selected_hashes = {}
        probe_ids = {0, n // 2, n - 1}
        with rgb_path.with_suffix(".tmp").open("wb") as stream:
            np.lib.format.write_array_header_1_0(stream, dict(descr="|u1", fortran_order=False, shape=shape))
            for t in range(n):
                rgb = compose_rgb({c: decode_rgb(f[f"cameras/{c}/rgb"], t) for c in CAMERAS})
                payload = rgb.tobytes()
                pixel_hash.update(payload)
                stream.write(payload)
                if t in probe_ids:
                    selected_hashes[str(t)] = hashlib.sha256(payload).hexdigest()
        os.replace(rgb_path.with_suffix(".tmp"), rgb_path)

    def file_record(path):
        return dict(path=str(path.relative_to(output)), bytes=path.stat().st_size, sha256=digest(path))

    st = source.stat()
    record = dict(
        name=name,
        preparation_contract=contract,
        source=str(source),
        source_sha256=source_hash,
        source_stat=[st.st_size, st.st_mtime_ns],
        metadata=meta,
        cameras=camera_info,
        joint_names=names,
        source_frames=len(steps),
        num_frames=n,
        start_frames=starts.tolist(),
        frame_mapping="source HDF5 row indices; each start uses start..start+32 inclusive",
        rgb={
            **file_record(rgb_path),
            "shape": list(shape),
            "pixel_sha256_TCHW": pixel_hash.hexdigest(),
            "probe_frame_sha256_CHW": selected_hashes,
        },
        state=file_record(state_path),
        calibration=file_record(calibration_path),
    )
    atomic_json(meta_path, record)
    print(f"PREPARED {contract['task']} {name}: {n} frames, {len(starts)} windows, {len(names)} joints", flush=True)
    return record


class PreparedFrames:
    """Task21 acceleration: mmap RGB once; reuse 128 spatially transformed frames across stride1 windows."""

    def __init__(self, path):
        self.rgb = np.load(path, mmap_mode="r")
        self.frames = OrderedDict()

    def prepare(self, start):
        import torch

        from cosmos_framework.data.generator.action.transforms import find_closest_target_size, reflection_pad_to_target

        for index in range(start, start + 33):
            if index not in self.frames:
                frame = torch.from_numpy(np.array(self.rgb[index], copy=True))
                tw, th = find_closest_target_size(*frame.shape[-2:], "480")
                p = reflection_pad_to_target({"video": frame}, ["video"], True, tw, th)
                if p["image_size"].tolist() != [640, 640, 640, 569]:
                    raise ValueError("Unexpected video spatial contract")
                self.frames[index] = p["video"]
        video = torch.stack([self.frames[i] for i in range(start, start + 33)], dim=1)
        while len(self.frames) > 128:
            self.frames.popitem(last=False)
        return video


def encode_worker(device, records, output, vae_path, contract):
    import torch

    from tools.cache_window_vae_latents import encode_window_latents, load_window_vae

    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.manual_seed(contract["seed"])
    torch.cuda.manual_seed_all(contract["seed"])
    output = Path(output)
    out = output / "vae_window_latents"
    tokenizer = None
    results = []
    for ep in records:
        name = ep["name"]
        starts = ep["start_frames"]
        if contract["max_windows"]:
            starts = starts[: contract["max_windows"]]
        binding = dict(
            contract=contract,
            name=name,
            start_frames=starts,
            source_sha256=ep["source_sha256"],
            rgb_sha256=ep["rgb"]["sha256"],
            state_sha256=ep["state"]["sha256"],
            alignment_sha256=digest(output / "alignment" / f"{name}.json"),
        )
        meta, target = out / f"{name}.json", out / f"{name}.npy"
        if meta.exists():
            prior = json.loads(meta.read_text())
            compatible = semantic_contract(prior["contract"], encoding=True) == semantic_contract(
                contract, encoding=True
            )
            if (
                not compatible
                or any(prior.get(k) != v for k, v in binding.items() if k != "contract")
                or not target.exists()
                or target.stat().st_size != prior["file_bytes"]
            ):
                raise ValueError(f"Incompatible or truncated latent: {target}")
            results.append(prior)
            print(f"GPU{device} SKIP {name}", flush=True)
            continue
        if tokenizer is None:
            tokenizer = load_window_vae(vae_path)
        frames = PreparedFrames(output / ep["rgb"]["path"])
        shape = (len(starts), 48, 9, 40, 40)
        begin = time.monotonic()

        def prepare_batch(row):
            return frames.prepare(starts[row]).unsqueeze(0)

        def progress(done):
            if done % 100 == 0:
                speed = done / (time.monotonic() - begin)
                print(f"GPU{device} {name} {done}/{len(starts)} {speed:.2f} windows/s", flush=True)

        encode_window_latents(
            tokenizer,
            prepare_batch,
            len(starts),
            target,
            f"cuda:{device}",
            expected_shape=shape[1:],
            progress=progress,
        )
        binding.update(
            path=target.name,
            shape=list(shape),
            image_size=[640, 640, 640, 569],
            windows=len(starts),
            file_bytes=target.stat().st_size,
            latent_sha256=digest(target),
        )
        atomic_json(meta, binding)
        results.append(binding)
        print(f"GPU{device} DONE {name}: {len(starts)} windows", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("/data/shichaojian/raw_data/bench2dex"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae-path", default="/data/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth")
    parser.add_argument("--wait-downloads", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0, help="Smoke only; use a separate --output")
    parser.add_argument("--max-windows", type=int, default=0, help="Smoke only; use a separate --output")
    args = parser.parse_args()
    if (args.limit_episodes or args.max_windows) and args.output is None:
        parser.error("Smoke limits require a separate --output")
    short, task = TASKS[args.task]
    root = args.raw_root / short
    output = args.output or root / "cosmos-cache"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = json.loads((REPO / "pointflow_outputs/bench2dex_task_lengths" / f"{task}.files.json").read_text())
        records = sorted(records, key=lambda r: r["path"])
        if args.limit_episodes:
            records = records[: args.limit_episodes]
        while True:
            pending = [
                r for r in records if not (root / r["path"]).exists() or (root / r["path"]).stat().st_size != r["size"]
            ]
            if not pending:
                break
            if not args.wait_downloads:
                raise RuntimeError(f"{len(pending)} source files missing/incomplete; finish download first")
            print(f"WAIT {task}: {len(records) - len(pending)}/{len(records)} complete source files", flush=True)
            time.sleep(30)
        for sub in ("episodes", "video_frames", "alignment", "vae_window_latents", "provenance"):
            (output / sub).mkdir(exist_ok=True)
        source_files = [
            Path(__file__),
            REPO / "tools/cache_window_vae_latents.py",
            REPO / "tools/bench2dex_rgb_contract.py",
            REPO / "cosmos_framework/data/generator/action/transforms.py",
            REPO / "cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py",
        ]
        code_hashes = {str(p.relative_to(REPO)): digest(p) for p in source_files}
        prep_contract = dict(
            schema="bench2dex_multirobot_rgb_v1",
            task=task,
            cameras=list(CAMERAS),
            fps=20,
            view_width=640,
            rgb_layout="TCHW",
            preserve_raw_wrist_fisheye=True,
            action_order="recorded robot/joint_names; action reordered by name",
            code_sha256=code_hashes,
        )
        for p in source_files:
            dst = output / "provenance/source" / code_hashes[str(p.relative_to(REPO))] / p.relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(p, dst)
        context = multiprocessing.get_context("spawn")
        jobs = [(str(root / r["path"]), r["lfs"]["oid"], str(output), prep_contract) for r in records]
        episodes = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for future in as_completed([pool.submit(prepare_episode, j) for j in jobs]):
                episodes.append(future.result())
                atomic_json(
                    output / "progress.json", dict(phase="prepare", completed=len(episodes), total=len(records))
                )
        episodes.sort(key=lambda r: r["name"])
        manifest = dict(contract=prep_contract, episodes=episodes)
        atomic_json(output / "alignment_manifest.json", manifest)
        atomic_json(
            output / "window_plan.json",
            dict(
                fps=20,
                frames_per_window=33,
                sample_stride=1,
                episodes=[
                    dict(
                        name=e["name"],
                        source_sha256=e["source_sha256"],
                        num_frames=e["num_frames"],
                        start_frames=e["start_frames"],
                        frame_ids_rule="range(start,start+33)",
                    )
                    for e in episodes
                ],
            ),
        )
        print(f"PREPARATION COMPLETE {task}: {sum(len(e['start_frames']) for e in episodes)} windows", flush=True)
        if args.prepare_only:
            return
        devices = [int(x) for x in args.devices.split(",")]
        if len(devices) != len(set(devices)) or not devices:
            raise ValueError("Devices must be distinct and nonempty")
        import torch

        if not torch.cuda.is_available() or max(devices) >= torch.cuda.device_count():
            raise RuntimeError("Requested GPUs unavailable; CPU preparation retained for resume")
        from tools.cache_window_vae_latents import ENCODING_REVISION

        contract = dict(
            schema="bench2dex_valid_window_latents_v1",
            fps=20,
            chunk_length=32,
            sample_stride=1,
            resolution="480",
            storage="bfloat16_bits_in_uint16",
            vae_batch_size=1,
            seed=args.seed,
            encoding_revision=ENCODING_REVISION,
            max_windows=args.max_windows,
            alignment_manifest_sha256=digest(output / "alignment_manifest.json"),
            window_plan_sha256=digest(output / "window_plan.json"),
            vae_sha256=digest(args.vae_path),
            code_sha256=code_hashes,
        )
        import importlib.metadata
        import platform

        atomic_json(
            output / "provenance/environment.json",
            dict(
                python=platform.python_version(),
                packages={
                    p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "opencv-python", "h5py")
                },
                cuda=torch.version.cuda,
                devices=[torch.cuda.get_device_name(i) for i in devices],
                seed=args.seed,
            ),
        )
        latent_records = []
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as pool:
            futures = [
                pool.submit(encode_worker, gpu, episodes[i :: len(devices)], str(output), args.vae_path, contract)
                for i, gpu in enumerate(devices)
            ]
            for future in as_completed(futures):
                latent_records.extend(future.result())
        atomic_json(
            output / "vae_window_latents/window_manifest.json",
            dict(contract=contract, episodes=sorted(latent_records, key=lambda r: r["name"])),
        )
        atomic_json(
            output / "progress.json",
            dict(phase="complete", episodes=len(episodes), windows=sum(r["windows"] for r in latent_records)),
        )
        print(f"ALL LATENTS COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
