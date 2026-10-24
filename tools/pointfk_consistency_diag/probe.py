"""FDM/IDM ground-truth injection probe (offline, diag tool only; no model/training code is touched).

At EVERY sampling step the chosen modality's noisy tokens are REPLACED by the ground truth noised to the current sigma
    x_t = m*ref + (1-m)*((1-sigma)*x_gt + sigma*eps)     (rectified flow: sigma = timestep/1000, v = eps - x_gt)
eps = the window's fixed initial noise; m = the model's own condition mask (conditioning frame / state row stay as they are).
The model's velocity for that modality is discarded and replaced by the exact path velocity (eps - x_gt) so the sampler state
follows the same path. All other modalities denoise normally. Implemented by wrapping `model.sampler` and `model._prepare_inference_data`
(instance-level monkeypatch, restored after each call).
Variants: a baseline | a2 baseline other seed | b GT video | c GT action | c2 action held at row 0 | c3 action of another window (relative motion)
| d future video of another episode (its GT latents, noised along its own path) | e future video held at pure noise every step.
d / e are the video-perturbation robustness test (D0, 10-24): how far the action moves when the video plan is wrong / absent.
"""
import copy, json, os
from pathlib import Path
import numpy as np
import torch

VARIANTS = ("a", "a2", "b", "c", "c2", "c3", "d", "e")
SEED_STEP = 1000003


def install(model):
    if getattr(model, "_probe_installed", False):
        return
    orig = model._prepare_inference_data
    cap = {}

    def prep(*a, **k):
        r = orig(*a, **k)
        cap["r"] = r
        return r

    model.__dict__["_probe_orig_prepare"] = orig
    model.__dict__["_probe_cap"] = cap
    model.__dict__["_prepare_inference_data"] = prep
    model.__dict__["_probe_installed"] = True


def layout(r):
    plans, g = r[0], r[1]
    ref, hna = r[5][0], r[7]
    assert len(r[4]) == 1, "probe supports one sample"
    nv = g.num_vision_items_per_sample[0] if g.num_vision_items_per_sample is not None else 1
    dims = [int(x.numel()) for x in g.x0_tokens_vision[:nv]]
    vis = (sum(dims[:-1]), sum(dims))
    act, ashape = None, None
    if hna and plans[0].has_action:
        ashape = tuple(g.x0_tokens_action[0].shape)
        act = (vis[1], vis[1] + int(g.x0_tokens_action[0].numel()))
    return vis, act, tuple(g.x0_tokens_vision[nv - 1].shape), ashape


class InjectSampler:
    """Drop-in for model.sampler: wraps velocity_fn so the injected modalities follow the GT path."""

    def __init__(self, orig, cap, inject, stats):
        self.orig, self.cap, self.inject, self.stats = orig, cap, inject, stats

    def __call__(self, velocity_fn, initial_noise, **kw):
        r = self.cap["r"]
        vis, act, _, ashape = layout(r)
        ref, msk, eps = r[5][0], r[6][0], initial_noise[0]
        segs = {}
        if "video" in self.inject:
            xv = self.inject["video"]   # full GT clip latents (the inference ref only encodes the conditioning frame; future = 0)
            segs["video"] = (slice(*vis), (ref[slice(*vis)].clone() if xv is None else xv.reshape(-1).to(ref)))
        if "action" in self.inject:
            x0 = self.inject["action"]
            segs["action"] = (slice(*act), (ref[slice(*act)].clone() if x0 is None else x0.reshape(-1).to(ref)))
        noise_sl = slice(*vis) if self.inject.get("video_noise") else None
        sig_log = self.stats.setdefault("sigmas", [])

        def vfn(noise_x, timestep, *a, **k):
            sig = float(timestep.reshape(-1)[0]) / 1000.0
            sig_log.append(sig)
            x = noise_x[0].clone()
            for sl, x0 in segs.values():
                m = msk[sl].to(x)
                x[sl] = (m * ref[sl].to(x) + (1 - m) * ((1 - sig) * x0.to(x) + sig * eps[sl].to(x)))
            if noise_sl is not None:
                m = msk[noise_sl].to(x)
                x[noise_sl] = m * ref[noise_sl].to(x) + (1 - m) * eps[noise_sl].to(x)
            v = velocity_fn([x] + list(noise_x[1:]), timestep, *a, **k)
            v0 = v[0].clone()
            for sl, x0 in segs.values():
                m = msk[sl].to(v0)
                v0[sl] = (1 - m) * (eps[sl].to(v0) - x0.to(v0)) + m * v0[sl]
            return [v0] + list(v[1:])

        out = self.orig(vfn, initial_noise, **kw)
        if noise_sl is not None:
            m = msk[noise_sl].to(out[0])
            out[0][noise_sl] = m * ref[noise_sl].to(out[0]) + (1 - m) * eps[noise_sl].to(out[0])
        for name, (sl, x0) in segs.items():
            m = msk[sl].to(out[0])
            fin = m * ref[sl].to(out[0]) + (1 - m) * x0.to(out[0])
            raw = out[0][sl].float()
            self.stats[f"{name}_raw_rel_dev"] = float((raw - fin.float()).norm() / (fin.float().norm() + 1e-8))
            out[0][sl] = fin          # sigma->0 end state of the injected path (removes the tiny t0<1000 offset)
        return out


def run_variant(model, cb, cpu_batch, seed, inject=None):
    from cosmos_framework.callbacks.joint_pointflow_fk_eval import sample_joint_case, numpy
    from cosmos_framework.callbacks.pointflow_eval_cases import evaluation_rng
    from cosmos_framework.utils import misc
    stats = {}
    had = "sampler" in model.__dict__
    orig = model.sampler
    if inject:
        object.__setattr__(model, "sampler", InjectSampler(orig, model._probe_cap, inject, stats))
    try:
        with evaluation_rng(seed):
            batch = misc.to(copy.deepcopy(cpu_batch), device="cuda")
            s = sample_joint_case(model, batch, seed=seed, steps=cb.sampling_steps)
    finally:
        if had:
            model.__dict__["sampler"] = orig
        else:
            model.__dict__.pop("sampler", None)
    res = {k: numpy(s[k][0]) for k in ("action", "fk", "pointflow")}
    res["vision"] = s["vision"][0].detach().float().cpu().numpy()
    return res, s["vision"][0], stats


def partner_index(dataset, idx, step=50):
    n = len(dataset.rows)
    ep = dataset.rows[idx]["episode"]
    for k in range(step, n):
        j = (idx + k) % n
        if dataset.rows[j]["episode"] != ep:
            yield j


def ctx_decode_sf(model, lat):
    from cosmos_framework.callbacks.pointflow_eval import _decode_dream_frames
    with torch.no_grad():
        f = _decode_dream_frames(model, {"vision": [lat]})
    return f, f[0].shape[0] // lat.shape[-2]


def write_video(cols, labels, path_mp4, path_jpg, fps=15):
    import cv2
    n = min(len(c) for c in cols)
    h, w = cols[0][0].shape[:2]

    def lab(fr, t):
        fr = fr.copy()
        cv2.putText(fr, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(fr, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return fr
    rows = [np.concatenate([lab(c[i], t) for c, t in zip(cols, labels)], axis=1) for i in range(n)]
    wr = cv2.VideoWriter(str(path_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (rows[0].shape[1], rows[0].shape[0]))
    for r in rows:
        wr.write(r)
    wr.release()
    cv2.imwrite(str(path_jpg), np.concatenate([rows[0], rows[-1]], axis=0))


def probe_window(model, cb, ctx, idx, cpu_batch, seed, meta, out_dir, name):
    """ctx: dict(dataset, make_batch, sf=None). Saves <name>.npz with all variants."""
    from cosmos_framework.utils import misc
    from cosmos_framework.callbacks.joint_pointflow_fk_eval import attach_fk_overlay, numpy
    from cosmos_framework.callbacks.pointflow_eval_cases import evaluation_rng
    install(model)
    cap = model._probe_cap
    out = {}
    stats = {}
    with torch.no_grad():
        res_a, lat_a, _ = run_variant(model, cb, cpu_batch, seed)
        r = cap["r"]
        vis, act, vshape, ashape = layout(r)
        ref = r[5][0]
        with evaluation_rng(seed):
            gb = misc.to(copy.deepcopy(cpu_batch), device="cuda")
            gfull = model.get_data_and_condition(gb)      # default (None) encodes every pixel frame
        xv_full = gfull.x0_tokens_vision[-1].reshape(vshape).float()
        r0 = ref[slice(*vis)].reshape(vshape)[:, 0].float()
        stats["gt_f0_rel_diff"] = float((xv_full[:, 0] - r0).norm() / (r0.norm() + 1e-8))   # full-clip vs first-frame-only encode
        xv_full[:, 0] = r0
        gt_vis = xv_full.cpu().numpy()
        del gb, gfull
        gt_act = ref[slice(*act)].reshape(ashape)
        msk_act = r[6][0][slice(*act)].reshape(ashape)
        cond_rows = msk_act[:, 0].cpu().numpy()
        out["a"] = res_a
        out["a2"], _, _ = run_variant(model, cb, cpu_batch, seed + SEED_STEP)
        out["b"], _, stats["b"] = run_variant(model, cb, cpu_batch, seed, {"video": xv_full})
        out["c"], _, stats["c"] = run_variant(model, cb, cpu_batch, seed, {"action": None})
        x_static = gt_act.clone()
        x_static[1:] = gt_act[0:1]
        out["c2"], _, stats["c2"] = run_variant(model, cb, cpu_batch, seed, {"action": x_static})
        # swap: another window >=50 rows away in a different episode; row 0 (state) stays ours, its relative motion is added
        pj = None
        for j in partner_index(ctx["dataset"], idx):
            s = ctx["dataset"][j]
            if s.get("pointflow") is not None and len(s["pointflow"]["inputs"]["point_ids"]) and s.get("fk") is not None:
                pj = j
                break
        prow = ctx["dataset"].rows[pj]
        with evaluation_rng(seed):
            pb = ctx["make_batch"](ctx["dataset"][pj])
            pb = misc.to(copy.deepcopy(pb), device="cuda")
        # direct (uncaptured) call: only to read the partner's normalised action
        pr = model._probe_orig_prepare(pb, [seed], False)
        pact = pr[5][0][slice(*layout(pr)[1])].reshape(ashape)
        x_swap = gt_act[0:1] + (pact - pact[0:1])
        x_swap[cond_rows > 0.5] = gt_act[cond_rows > 0.5]
        out["c3"], _, stats["c3"] = run_variant(model, cb, cpu_batch, seed, {"action": x_swap})
        # D0 video perturbation: the partner's full GT clip as the future video (our conditioning frame stays)
        with evaluation_rng(seed):
            gp = model.get_data_and_condition(pb)
        xv_partner = gp.x0_tokens_vision[-1].reshape(vshape).float()
        xv_partner[:, 0] = r0
        del gp
        out["d"], _, stats["d"] = run_variant(model, cb, cpu_batch, seed, {"video": xv_partner})
        out["e"], _, stats["e"] = run_variant(model, cb, cpu_batch, seed, {"video_noise": True})
        # hand keypoints in the dream canvas (reuses the eval's FK projector)
        if ctx.get("sf") is None:
            _, ctx["sf"] = ctx_decode_sf(model, lat_a)
        sf = ctx["sf"]
        H, W = vshape[-2] * sf, vshape[-1] * sf
        fk = cpu_batch["fk"][0]
        rec = {"frames_bgr": [np.zeros((H, W, 3), np.uint8)] * (numpy(fk["targets"]["displacement"]).shape[0] + 1)}
        attach_fk_overlay(rec, fk, numpy(fk["targets"]["displacement"]), cpu_batch["pointflow"][0], dream_hw=(H, W))
    arrs = {}
    for v in VARIANTS:
        for k, a in out[v].items():
            arrs[f"{v}_{k}"] = a.astype(np.float16) if k == "vision" else a
    arrs["gt_vision"] = gt_vis.astype(np.float16)
    arrs["fk_gt_uv"], arrs["fk_gt_valid"] = rec["fk_gt_uv"], rec["fk_gt_valid"]
    stats["b"]["gt_f0_rel_diff"] = stats["gt_f0_rel_diff"]
    meta = dict(meta, sf=int(sf), partner=dict(episode=prow["episode"], start_frame=int(prow["start_frame"])), stats=stats,
                cond_rows=cond_rows.tolist(), sigmas=stats["b"].get("sigmas"))
    np.savez_compressed(Path(out_dir) / f"{name}.npz", meta=json.dumps(meta), **arrs)
    # optional video decode for the selected subset
    dl = os.environ.get("DIAG_DECODE_LIST")
    if dl and (lambda nm: name in nm or "*" in nm)(set(Path(dl).read_text().split())):
        vd = Path(os.environ["DIAG_VIDEO_DIR"]); vd.mkdir(parents=True, exist_ok=True)
        from cosmos_framework.callbacks.pointflow_eval import _decode_dream_frames
        lats = {"GT": torch.from_numpy(gt_vis)}
        for v, lab in (("a", "a base"), ("c", "c GT act"), ("c2", "c2 static"), ("c3", "c3 swap")):
            lats[lab] = torch.from_numpy(out[v]["vision"])
        cols, labels = [], []
        with torch.no_grad():
            for lab, lat in lats.items():
                cols.append(_decode_dream_frames(model, {"vision": [lat.to(lat_a.device, lat_a.dtype)]}))
                labels.append(lab)
        write_video(cols, labels, vd / f"{name}.mp4", vd / f"{name}.jpg")
    print(f"[probe] saved {name} b_dev={stats['b'].get('video_raw_rel_dev'):.2e} c_dev={stats['c'].get('action_raw_rel_dev'):.2e}", flush=True)
