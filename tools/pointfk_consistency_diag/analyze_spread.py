"""Across-sample spread of the PREDICTED fingertip landing position (CPU only, read-only).
usage: analyze_spread.py PREDS_DIR OUT_DIR [--samples-csv CSV]
PREDS_DIR/*.npz: action[N,33,52], fk[N,32,42,3], pointflow[N,32,1024,3] (analyze_multisample inputs).

Frame/definitions (everything in the sim WORLD frame, z up, so "horizontal" = world xy; cm):
 * active hand = hand with larger GT finger closure (samples.csv closure_left/right).  Fingertip centroid = mean of the 5 *_tip keypoints.
 * GT trajectory = realised FK GT (anchor + fk.targets.displacement) rows 1..32; also FK(GT action) = commanded.
 * target object = the ball (soccer/tennis/golf/table-tennis; box3d center_world in rgbd.hdf5) with the smallest GT fingertip-centroid
   distance over the window; t* = frame (1..32) of that closest approach (3D distance).  Fallback reported separately: last frame (32).
 * predicted traj: (a) FK head: GT-anchor + R^T d_head ; (b) FK(predicted action) in world.  cam<-world (R,t) from Kabsch of FK(GT act)[0] vs anchor.
 * spread at t*: sd_xy = sqrt(var_x+var_y) over the N samples (RMS radial deviation from the sample mean), max pairwise xy distance.
 * error of a sample = xy distance to GT fingertip centroid at t*.  best/median/avg(=random pick)/ensemble-mean-position.
 * offset-to-object of a sample = xy distance predicted fingertip centroid(t*) -> GT object centre(t*)  (the closed-loop "predicted offset to ball" analogue).
 * GT-free proxies: (P1) xy distance predicted centroid -> PointFlow-predicted object centroid at t* (PF points whose anchor lies within r_obj+1.5cm of
   the object centre at frame 0; object membership uses GT only to define the points -- needs a segmenter at test time); (P2) 3D distance to the same;
   (P3) medoid of the sample positions at t*.
"""
import argparse, csv, glob, json, os, sys
import numpy as np, torch, h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from cosmos_framework.model.generator import pointfk_consistency as pc
from cosmos_framework.model.generator.robot_fk_torch import TorchRobotFK, estimate_rigid_transform

B = "/data/shichaojian/sim_data/bench2dex_task27_ball_training"
SEL = B + "/datasets/cosmos-cache/pointflow_fk_handguided1024"
STATS = B + "/normalization_train_only.json"
SRC = "/data/shichaojian/sim_data/bench2dex_task27_ball"
TIPS = {"left": [4, 8, 12, 16, 20], "right": [25, 29, 33, 37, 41]}
BALLS = ["obj_053_mini_soccer_ball_2", "obj_056_tennis_ball_3", "obj_058_golf_ball_4", "obj_144_table_tennis_5"]
names, offset, scale = pc.load_action_stats(STATS)
J = len(names)
FK = TorchRobotFK.from_json(pc.DEFAULT_URDF, pc.DEFAULT_FK_JSON, names)
rows = {(w["episode"], w["start_frame"]): w["path"] for w in json.load(open(SEL + "/manifest.json"))["windows"] if w["split"] == "validation"}
_obj = {}
def objects(ep):
    if ep not in _obj:
        with h5py.File(f"{SRC}/{ep}/rgbd.hdf5", "r") as f:
            _obj[ep] = {o: (f[f"labels/box3d/{o}/center_world"][:], float(f[f"labels/box3d/{o}/size_lwh"][0].max()) / 2) for o in BALLS if f"labels/box3d/{o}" in f}
    return _obj[ep]

def pairmax(x):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1); return float(d.max())
def spread(x): return float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preds"); ap.add_argument("out")
    ap.add_argument("--samples-csv", default="/data/shichaojian/runs/consistency_multisample_20261014/samples.csv"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for r in csv.DictReader(open(a.samples_csv)):
        if r["k"] == "0": meta[(r["episode"], int(r["start_frame"]))] = r
    out = []
    for f in sorted(glob.glob(f"{a.preds}/*.npz")):
        P = np.load(f); m = json.loads(str(P["meta"])); ep, s = m["episode"], m["start_frame"]
        mr = meta[(ep, s)]; hand = "left" if float(mr["closure_left"]) >= float(mr["closure_right"]) else "right"
        # camera<-world and camera-frame object anchor for PF membership
        pt = torch.load(f"{SEL}/{rows[(ep, s)]}", weights_only=False, map_location="cpu")
        T = lambda x: torch.as_tensor(np.asarray(x)).float()
        anchor = T(pt["fk"]["inputs"]["anchor_xyz"]); a_gt = T(pt["action"])[:, :J]
        with torch.no_grad():
            w_gt = FK(a_gt[None])[0]; w_pr = FK(torch.from_numpy(P["action"]).float()[:, :, :J])
            R, tv = estimate_rigid_transform(w_gt[0][None], anchor[None]); R, tv = R[0], tv[0]
        resid = float((w_gt[0] @ R.T + tv - anchor).norm(dim=-1).max() * 100)
        Rt = R.T; tips = TIPS[hand]; w0 = w_gt[0]
        gt_real = (w0[None] + T(pt["fk"]["targets"]["displacement"]) @ Rt.T)[:, tips].mean(1).numpy()
        gt_cmd = w_gt[1:, tips].mean(1).numpy()
        head = (w0[None, None] + torch.from_numpy(P["fk"]).float() @ Rt.T)[:, :, tips].mean(2).numpy()
        act = w_pr[:, 1:, tips].mean(2).numpy()
        raw = np.asarray(pt["raw_frame_ids"]).astype(int); ob = objects(ep)
        best = None
        for o, (c, r) in ob.items():
            d = np.linalg.norm(gt_real - c[raw[1:]], axis=1); k = int(d.argmin())
            if best is None or d[k] < best[0]: best = (d[k], o, k, r)
        dmin, oname, tstar, orad = best
        oc = ob[oname][0][raw]                                   # [33,3]
        # PF points on the object
        apf = T(pt["pointflow"]["inputs"]["anchor_xyz"]).numpy()
        o0_cam = (R @ torch.from_numpy(oc[0]).float() + tv).numpy()
        sel = np.linalg.norm(apf - o0_cam, axis=1) <= orad + 0.015
        pf = P["pointflow"]                                       # [N,32,1024,3] camera-frame displacement
        N = len(pf)
        rec = dict(episode=ep, start_frame=s, phase=mr["phase"], hand=hand, obj=oname, tstar=tstar + 1, d_gt_obj_cm=dmin * 100,
                   n_obj_pf=int(sel.sum()), fit_resid_cm=resid)
        for tag, t in (("cl", tstar), ("last", 31)):
            g = gt_real[t, :2]; gc = gt_cmd[t, :2]; ocw = oc[t + 1, :2]
            for pname, X in (("head", head), ("act", act)):
                x = X[:, t, :2]
                e_real = np.linalg.norm(x - g, axis=1) * 100; e_cmd = np.linalg.norm(x - gc, axis=1) * 100
                off = np.linalg.norm(x - ocw, axis=1) * 100
                pre = f"{pname}_{tag}_"
                rec[pre + "sd_xy_cm"] = spread(x) * 100; rec[pre + "maxpair_xy_cm"] = pairmax(x) * 100
                rec[pre + "sd_z_cm"] = float(X[:, t, 2].std() * 100)
                rec[pre + "err_best"] = e_real.min(); rec[pre + "err_med"] = np.median(e_real); rec[pre + "err_avg"] = e_real.mean()
                rec[pre + "err_ensmean"] = float(np.linalg.norm(x.mean(0) - g) * 100); rec[pre + "err_worst"] = e_real.max()
                rec[pre + "errcmd_best"] = e_cmd.min(); rec[pre + "errcmd_med"] = np.median(e_cmd); rec[pre + "errcmd_avg"] = e_cmd.mean()
                rec[pre + "off_best"] = off.min(); rec[pre + "off_med"] = np.median(off); rec[pre + "off_max"] = off.max(); rec[pre + "off_mean"] = off.mean()
                # selectors (value = err vs GT realised of the selected sample)
                d_med = np.linalg.norm(x[:, None] - x[None], axis=-1).sum(1); rec[pre + "err_medoid"] = e_real[int(d_med.argmin())]
                rec[pre + "err_sel_gtobj"] = e_real[int(off.argmin())]          # picks smallest offset to GT object
                rec[pre + "off_sel_gtobj"] = off.min()
                rec[pre + "rank_corr_off_err"] = _spear(off, e_real)
                if sel.sum() >= 3:
                    # PF-predicted object centroid per sample (world xy via camera->world); fingertip pred in camera frame via R,tv
                    oc_pred_cam = (torch.from_numpy(apf[sel])[None] + torch.from_numpy(pf[:, t, sel])).mean(1)     # [N,3] cam
                    oc_pred_w = ((oc_pred_cam - tv) @ R).numpy()                                                  # R^T(c - t)
                    p1 = np.linalg.norm(X[:, t, :2] - oc_pred_w[:, :2], axis=1); p2 = np.linalg.norm(X[:, t] - oc_pred_w, axis=1)
                    rec[pre + "err_sel_p1"] = e_real[int(p1.argmin())]; rec[pre + "err_sel_p2"] = e_real[int(p2.argmin())]
                    rec[pre + "off_sel_p1"] = off[int(p1.argmin())]; rec[pre + "off_sel_p2"] = off[int(p2.argmin())]
                    rec[pre + "rank_corr_p1_err"] = _spear(p1, e_real); rec[pre + "rank_corr_p1_off"] = _spear(p1, off)
                    rec[pre + "pfobj_sd_xy_cm"] = spread(oc_pred_w[:, :2]) * 100
                else:
                    for k in ("err_sel_p1", "err_sel_p2", "off_sel_p1", "off_sel_p2", "rank_corr_p1_err", "rank_corr_p1_off", "pfobj_sd_xy_cm"): rec[pre + k] = np.nan
                rec[pre + "off_medoid"] = off[int(d_med.argmin())]
        out.append(rec)
        print(ep, s, rec["phase"], f"t*={rec['tstar']} d={rec['d_gt_obj_cm']:.1f}cm head sd={rec['head_cl_sd_xy_cm']:.2f} act sd={rec['act_cl_sd_xy_cm']:.2f} resid={resid:.3f}", flush=True)
    cols = list(out[0]);
    for r in out:
        for c in r:
            if c not in cols: cols.append(c)
    with open(f"{a.out}/windows_spread.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out)
    summarize(out, a.out)

def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0: return np.nan
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])

def summarize(out, odir):
    def sub(f): return [r for r in out if f(r)]
    groups = [("grasp (all)", sub(lambda r: r["phase"] == "grasp")), ("grasp, contact (GT closest approach < 6 cm)", sub(lambda r: r["phase"] == "grasp" and r["d_gt_obj_cm"] < 6)),
              ("other", sub(lambda r: r["phase"] == "other")), ("other, contact (<6 cm)", sub(lambda r: r["phase"] == "other" and r["d_gt_obj_cm"] < 6))]
    md = ["# Multi-sample spread of predicted fingertip landing position (nolocal 40k, N=8)\n",
          "World frame, cm, horizontal = world xy.  Definitions: see the script docstring (analyze_spread.py).  t* = frame of GT fingertip-centroid closest approach to the nearest ball (box3d), 'last' = frame 32.",
          "Spread = RMS radial deviation of the 8 predicted positions from their mean (sd_xy) and the max pairwise distance.  Errors are xy distance to the GT (realised) fingertip centroid.\n",
          f"windows: {len(out)} (grasp {sum(r['phase']=='grasp' for r in out)}, other {sum(r['phase']=='other' for r in out)}); Kabsch cam<-world fit max residual {max(r['fit_resid_cm'] for r in out):.3f} cm.\n"]
    nz = lambda rs, k: np.array([r[k] for r in rs], float)
    mm = lambda v: (f"{np.nanmean(v):.2f} / {np.nanmedian(v):.2f} / {np.nanpercentile(v, 90):.2f}" if np.isfinite(v).any() else "n/a")
    for tag, tl in (("cl", "closest-approach frame t*"), ("last", "last frame (32)")):
        md.append(f"## Spread and error at {tl}\n")
        for gname, rs in groups:
            if not rs: continue
            md += [f"### {gname}, n={len(rs)}  (cells: mean / median / p90 over windows, cm)\n", "| quantity | FK head | FK(pred action) |", "|---|---|---|"]
            for k, lab in (("sd_xy_cm", "across-sample sd_xy"), ("maxpair_xy_cm", "max pairwise xy"), ("sd_z_cm", "across-sample sd_z (height)"), ("err_best", "xy error: best-of-8"), ("err_med", "xy error: median-of-8"),
                           ("err_avg", "xy error: avg over samples (random pick)"), ("err_ensmean", "xy error of ensemble-mean position"), ("err_worst", "xy error: worst-of-8"),
                           ("errcmd_avg", "xy error vs commanded GT: avg"), ("off_med", "xy offset to GT ball: median-of-8"), ("off_best", "xy offset to GT ball: best-of-8"),
                           ("off_mean", "xy offset to GT ball: mean-of-8"), ("off_max", "xy offset to GT ball: worst-of-8")):
                md.append(f"| {lab} | " + " | ".join(mm(nz(rs, f"{p}_{tag}_{k}")) for p in ("head", "act")) + " |")
            fr = lambda p, th: np.mean(nz(rs, f"{p}_{tag}_sd_xy_cm") > th)
            md.append("| fraction of windows with sd_xy > 1 cm | " + " | ".join(f"{fr(p, 1):.2f}" for p in ("head", "act")) + " |")
            md.append("| fraction of windows with max pairwise > 2 cm | " + " | ".join(f"{np.mean(nz(rs, f'{p}_{tag}_maxpair_xy_cm') > 2):.2f}" for p in ("head", "act")) + " |")
            md.append("")
            md += ["Selectors (mean over windows of the selected sample's xy error vs GT / its offset to GT ball, cm); GT-free ones need no GT at test time except where marked:\n",
                   "| selector | FK head err | FK head offset | FK(act) err | FK(act) offset |", "|---|---|---|---|---|"]
            sels = [("oracle best-of-8 (GT error)", "err_best", None), ("random / avg", "err_avg", "off_mean"), ("median-of-8 (value)", "err_med", "off_med"), ("medoid (GT-free)", "err_medoid", "off_medoid"),
                    ("P1: min xy dist to PF-predicted object (GT-free given segmentation)", "err_sel_p1", "off_sel_p1"), ("P2: min 3D dist to PF-predicted object", "err_sel_p2", "off_sel_p2"),
                    ("min offset to GT ball (closed-loop 'best aligned'; needs GT)", "err_sel_gtobj", "off_sel_gtobj")]
            for lab, ke, ko in sels:
                cells = []
                for p in ("head", "act"):
                    cells += [f"{np.nanmean(nz(rs, f'{p}_{tag}_{ke}')):.2f}", f"{np.nanmean(nz(rs, f'{p}_{tag}_{ko}')):.2f}" if ko else "-"]
                md.append(f"| {lab} | " + " | ".join(cells) + " |")
            md.append(f"\nPF object points: median {int(np.median(nz(rs, 'n_obj_pf')))} per window, windows with >=3: {int((nz(rs, 'n_obj_pf') >= 3).sum())}/{len(rs)}; "
                      f"mean within-window Spearman(P1 distance, GT error): head {np.nanmean(nz(rs, f'head_{tag}_rank_corr_p1_err')):.2f}, act {np.nanmean(nz(rs, f'act_{tag}_rank_corr_p1_err')):.2f}; "
                      f"Spearman(offset-to-GT-ball, GT error): head {np.nanmean(nz(rs, f'head_{tag}_rank_corr_off_err')):.2f}, act {np.nanmean(nz(rs, f'act_{tag}_rank_corr_off_err')):.2f}\n")
    open(f"{odir}/summary.md", "w").write("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
