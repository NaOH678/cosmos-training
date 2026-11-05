"""Eval-only entry: patches the unified joint eval with a lightweight many-window sampler, then runs the normal trainer.
Env: DIAG_OUT (dir for npz), DIAG_SHARD, DIAG_NSHARD, DIAG_STRIDE (window stride), DIAG_ORIG (1: also run callback's original fixed val cases),
DIAG_LIMIT (max extra windows, smoke),
DIAG_PROBE=1 (+DIAG_DECODE_LIST, DIAG_VIDEO_DIR): GT-injection probe, see probe.py.
DIAG_NSAMPLES (default 1; N>1: N joint samples per window with seeds seed+k*1000003, k=0 = original seed; npz holds stacked [N,...] arrays; use a separate DIAG_OUT)."""
import copy, hashlib, json, os, runpy, sys, traceback
from pathlib import Path
import numpy as np
import torch


def diag_run_joint_eval(callback, model, iteration):
    from omegaconf import OmegaConf
    from cosmos_framework.callbacks.joint_pointflow_fk_eval import sample_joint_case, numpy
    from cosmos_framework.callbacks.pointflow_eval_cases import evaluation_rng
    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.lazy_config import instantiate
    import copy as _copy

    out = Path(os.environ["DIAG_OUT"]); out.mkdir(parents=True, exist_ok=True)
    shard, nshard = int(os.environ.get("DIAG_SHARD", 0)), int(os.environ.get("DIAG_NSHARD", 1))
    stride, limit = int(os.environ.get("DIAG_STRIDE", 24)), int(os.environ.get("DIAG_LIMIT", 0))
    nsamp = int(os.environ.get("DIAG_NSAMPLES", 1))
    do_orig = os.environ.get("DIAG_ORIG", "0") == "1"
    loader = callback.config.dataloader_val
    cfg = OmegaConf.to_container(loader, resolve=True) if OmegaConf.is_config(loader) else _copy.deepcopy(loader)
    dcfg = _copy.deepcopy(next(iter(cfg["dataloader"]["datasets"].values()))["dataset"])
    dcfg.update(iterable_shuffle=False, cfg_dropout_rate=0.0, use_image_augmentation=False)
    dataset = instantiate(dcfg)
    pack_cfg = {k: v for k, v in cfg.items() if k not in ("_target_", "dataloader")}
    pack_cfg.update(max_samples_per_batch=1, max_sequence_length=None)

    def make_batch(sample):
        inner = torch.utils.data.DataLoader([sample], batch_size=1, num_workers=0, collate_fn=custom_collate_fn)
        return next(iter(PackingDataLoader(dataloader=inner, **pack_cfg)))

    pctx = dict(dataset=dataset, make_batch=make_batch, sf=None)

    def run(tag, idx, cpu_batch, seed, orig):
        row = dataset.rows[idx]
        name = f"{row['episode']}_{row['start_frame']:05d}"
        if (out / f"{name}.npz").exists():
            return
        if os.environ.get("DIAG_PROBE") == "1":   # FDM/IDM GT-injection probe (probe.py); same windows/seeds
            import probe
            meta = dict(episode=row["episode"], start_frame=int(row["start_frame"]), index=int(idx), seed=int(seed), orig=bool(orig), steps=callback.sampling_steps)
            probe.probe_window(model, callback, pctx, idx, cpu_batch, seed, meta, out, name)
            return
        seeds = [seed + k * 1000003 for k in range(nsamp)]
        outs = []
        for sd in seeds:
            with evaluation_rng(sd):
                batch = misc.to(_copy.deepcopy(cpu_batch), device="cuda")
                s = sample_joint_case(model, batch, seed=sd, steps=callback.sampling_steps)
            outs.append({k: numpy(s[k][0]) for k in ("action", "fk", "pointflow")})
            del s, batch
        meta = json.dumps(dict(episode=row["episode"], start_frame=int(row["start_frame"]), path=row["path"], index=int(idx),
                               seed=int(seed), orig=bool(orig), case_id=tag, steps=callback.sampling_steps,
                               selected_root=str(dataset.selected), nsamples=nsamp, seeds=[int(x) for x in seeds]))
        stack = (lambda k: np.stack([o[k] for o in outs])) if nsamp > 1 else (lambda k: outs[0][k])
        np.savez_compressed(out / f"{name}.npz", action=stack("action"), fk=stack("fk"), pointflow=stack("pointflow"), meta=meta)
        print(f"[diag] saved {name} orig={orig}", flush=True)

    if do_orig:
        osh = os.environ.get("DIAG_ORIG_SHARDED", "0") == "1"   # 1: spread the original fixed cases over shards too
        for j, (identity, cpu_batch) in enumerate(zip(*callback._fixed, strict=True)):
            if identity["split"] != "val" or (osh and j % nshard != shard):
                continue
            try:
                run(identity["case_id"], identity["index"], cpu_batch, identity["seed"], True)
            except Exception:
                traceback.print_exc()
    cand = [i for i, r in enumerate(dataset.rows) if (r["start_frame"] - 1) % stride == 0]
    cand = [i for k, i in enumerate(cand) if k % nshard == shard]
    if limit:
        cand = cand[:limit]
    print(f"[diag] shard {shard}/{nshard}: {len(cand)} extra windows", flush=True)
    for idx in cand:
        try:
            sample = dataset[idx]
            p = sample.get("pointflow")
            if p is None or not len(p["inputs"]["point_ids"]) or not np.asarray(p["targets"]["valid"]).any() or sample.get("fk") is None:
                print(f"[diag] skip {idx} (no labels)", flush=True); continue
            meta = p["metadata"]
            seed = int.from_bytes(hashlib.sha256(f"{callback.seed}:val:{meta['episode']}:{int(meta['start_frame'])}".encode()).digest()[:4], "little")
            with evaluation_rng(seed):
                cpu_batch = make_batch(sample)
            run(f"extra_{idx}", idx, cpu_batch, seed, False)
        except Exception:
            traceback.print_exc()
    print("[diag] DONE", flush=True)


import cosmos_framework.callbacks.joint_pointflow_fk_eval as jm
jm.run_joint_eval = diag_run_joint_eval
runpy.run_module("cosmos_framework.scripts.train", run_name="__main__", alter_sys=True)
