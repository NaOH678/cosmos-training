"""Does the PLAN (offline, nolocal 40k PF/FK) follow the table height?  CPU only, read-only.
usage: CUDA_VISIBLE_DEVICES= analyze_table_follow.py OUT_DIR [--probe-preds DIR]
Conventions are those of analyze_spread.py / analyze_hand_obj.py (imported): world frame via Kabsch(FK(GT action)[0] -> anchors), box3d ball centres,
active hand = larger GT closure, fingertip centroid = mean of 5 *_tip keypoints, t* = frame (1..32) of closest GT fingertip-centroid approach to a ball.
Sample 'a' of fdm_idm_probe_20261014 (216 windows).  Table offset = meta/scene_generalization_sample/table_height/height_offset_m of the source episode.
Table-top height (world z) of a window = median over balls of (centre_z - r) at the window's first frame (same definition as analyze_hand_obj.table_top, ball-based).
Heights are above that table top.  Series per window (active hand):
  gt   : realised GT (anchor + GT displacement)       head : FK head (GT anchor + predicted displacement)     act : FK(predicted action)
Quantities: palm (wrist keypoint) height at t*, fingertip-centroid height at t*, lowest fingertip over frames 1..32, lowest of the 21 hand keypoints over 1..32.
Slope = OLS of height vs table offset (m/m); follow = 1 + slope (1 = hand moves with the table, 0 = hand stays at fixed world height).
Only a few distinct table heights exist among the validation episodes: slopes are reported at window level and from per-table means; no CI claimed beyond leave-one-table-out range.
"""
import argparse, csv, json, os, sys, collections
import numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_spread as S
import analyze_hand_obj as H
J = S.J; T = lambda x: torch.as_tensor(np.asarray(x)).float()
PALM = {"left": 0, "right": 21}

def table_offset(ep):
    with h5py.File(f"{S.SRC}/{ep}/rgbd.hdf5", "r") as f:
        return float(json.loads(f["meta/scene_generalization_sample"][()].decode())["spatial"]["table_height"]["height_offset_m"])

def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0: return np.nan
    xm = x - x.mean(); return float((xm * (y - y.mean())).sum() / (xm * xm).sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--probe-preds", default="/data/shichaojian/runs/fdm_idm_probe_20261014/preds/A")
    ap.add_argument("--samples-csv", default="/data/shichaojian/runs/consistency_multisample_20261014/samples.csv"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {(r["episode"], int(r["start_frame"])): r for r in csv.DictReader(open(a.samples_csv)) if r["k"] == "0"}
    P = H.load_set(a.probe_preds, "probe"); recs = []; toff = {}
    for (ep, s) in sorted(P):
        mr = meta[(ep, s)]; hand = "left" if float(mr["closure_left"]) >= float(mr["closure_right"]) else "right"; tips = S.TIPS[hand]; pm = PALM[hand]
        if ep not in toff: toff[ep] = table_offset(ep)
        pt = torch.load(f"{S.SEL}/{S.rows[(ep, s)]}", weights_only=False, map_location="cpu")
        anchor = T(pt["fk"]["inputs"]["anchor_xyz"]); a_gt = T(pt["action"])[:, :J]
        with torch.no_grad():
            w_gt = S.FK(a_gt[None])[0]; R, tv = S.estimate_rigid_transform(w_gt[0][None], anchor[None]); R, tv = R[0], tv[0]
            act_, fk_, _ = P[(ep, s)]; w_pr = S.FK(torch.from_numpy(act_).float()[:, :, :J])
        Rt = R.T; w0 = w_gt[0]
        ser = {"gt": (w0[None] + T(pt["fk"]["targets"]["displacement"]) @ Rt.T).numpy(),            # [32,42,3] frames 1..32
               "head": (w0[None] + torch.from_numpy(fk_[0]).float() @ Rt.T).numpy(),
               "act": w_pr[0, 1:].numpy()}
        raw = np.asarray(pt["raw_frame_ids"]).astype(int); ob = S.objects(ep)
        top = float(np.median([c[raw[0], 2] - r for c, r in ob.values()]))
        gt_tip = ser["gt"][:, tips].mean(1); best = None
        for o, (c, r) in ob.items():
            d = np.linalg.norm(gt_tip - c[raw[1:]], axis=1); k = int(d.argmin())
            if best is None or d[k] < best[0]: best = (d[k], o, k, r)
        dmin, oname, ts, orad = best
        rec = dict(episode=ep, start_frame=s, phase=mr["phase"], hand=hand, obj=oname, tstar=ts + 1, d_gt_obj_cm=dmin * 100, table_offset_cm=toff[ep] * 100, table_top_z=top,
                   gt_tip_minus_ballc_cm=(gt_tip[ts, 2] - ob[oname][0][raw[ts + 1], 2]) * 100)
        for nm, X in ser.items():
            hnd = X[:, 21 * (hand == "right"):21 * (hand == "right") + 21]
            rec[f"{nm}_palm_t*"] = (X[ts, pm, 2] - top) * 100
            rec[f"{nm}_tipc_t*"] = (X[ts, tips].mean(0)[2] - top) * 100
            rec[f"{nm}_tipmin_t*"] = (X[ts, tips, 2].min() - top) * 100
            rec[f"{nm}_tipmin_traj"] = (X[:, tips, 2].min() - top) * 100
            rec[f"{nm}_ptmin_traj"] = (hnd[:, :, 2].min() - top) * 100
            rec[f"{nm}_palm_min_traj"] = (X[:, pm, 2].min() - top) * 100
        recs.append(rec); print(ep, s, rec["phase"], hand, oname[:14], f"d={dmin*100:.1f} off={toff[ep]*100:.1f}cm top={top:.3f} palm gt/head/act={rec['gt_palm_t*']:.1f}/{rec['head_palm_t*']:.1f}/{rec['act_palm_t*']:.1f}", flush=True)
    cols = list(recs[0]);
    with open(f"{a.out}/windows_table_follow.csv", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(recs)
    summarize(recs, toff, a.out)

def summarize(recs, toff, odir):
    nz = lambda rs, k: np.array([r[k] for r in rs], float)
    md = ["# Does the plan follow the table height? (nolocal 40k, offline validation windows, sample a, CPU)\n", "Definitions: see the docstring of analyze_table_follow.py.  Heights in cm above the table top (ball-based), table offset in cm; slope = d height / d table offset (dimensionless, m/m), follow = 1 + slope.\n"]
    md.append("Table offsets of the validation episodes (cm): " + ", ".join(f"{e}: {v*100:+.2f}" for e, v in sorted(toff.items())) + f"  -> {len({round(v,6) for v in toff.values()})} distinct table heights.\n")
    groups = [("grasp, contact (GT closest approach <6 cm)", lambda r: r["phase"] == "grasp" and r["d_gt_obj_cm"] < 6), ("grasp, all", lambda r: r["phase"] == "grasp"),
              ("other, contact (<6 cm)", lambda r: r["phase"] == "other" and r["d_gt_obj_cm"] < 6), ("grasp, contact, golf ball only", lambda r: r["phase"] == "grasp" and r["d_gt_obj_cm"] < 6 and "golf" in r["obj"]), ("all windows", lambda r: True)]
    qs = [("palm_t*", "palm height at t*"), ("tipc_t*", "fingertip-centroid height at t*"), ("tipmin_t*", "lowest fingertip at t*"), ("tipmin_traj", "lowest fingertip over frames 1-32"),
          ("ptmin_traj", "lowest of 21 hand keypoints over 1-32"), ("palm_min_traj", "lowest palm over 1-32")]
    for gname, f in groups:
        rs = [r for r in recs if f(r)]
        if len(rs) < 3: continue
        x = nz(rs, "table_offset_cm") / 100
        md += [f"## {gname}  (n={len(rs)} windows, {len({r['episode'] for r in rs})} episodes, {len(set(np.round(x,6)))} distinct table heights)\n",
               "| quantity | series | mean height (cm) | window-level slope | follow=1+slope | per-table-mean slope | leave-one-table-out slope range |", "|---|---|---|---|---|---|---|"]
        eps = sorted({r["episode"] for r in rs})
        for q, ql in qs:
            for nm in ("gt", "head", "act"):
                y = nz(rs, f"{nm}_{q}"); sl = ols(x, y / 100)
                # per-table means (windows grouped by distinct offset)
                ux = sorted(set(np.round(x, 6))); mx = [u for u in ux]; my = [y[np.round(x, 6) == u].mean() / 100 for u in ux]
                ps = ols(mx, my)
                loo = []
                for u in ux:
                    m = np.round(x, 6) != u
                    if len(set(np.round(x[m], 6))) >= 2: loo.append(ols(x[m], y[m] / 100))
                md.append(f"| {ql} | {nm} | {y.mean():.2f} | {sl:+.2f} | {1+sl:.2f} | {ps:+.2f} | " + (f"[{min(loo):+.2f}, {max(loo):+.2f}]" if loo else "n/a") + " |")
        md.append("")
        md += ["Plan error (pred - GT) slope vs table offset (dimensionless; plan follows LESS than GT iff slope(pred-GT) < 0; the closed-loop rollout deficit vs demos is about -0.33 = follow 0.57 - 0.90):\n", "| quantity | series | slope(pred - GT) | window-level mean pred - GT (cm) |", "|---|---|---|---|"]
        for q, ql in qs:
            for nm in ("head", "act"):
                d = nz(rs, f"{nm}_{q}") - nz(rs, f"gt_{q}"); md.append(f"| {ql} | {nm} | {ols(x, d / 100):+.2f} | {d.mean():+.2f} |")
        md.append("")
    md += ["## Per-episode means (grasp, contact): palm height at t* (cm)\n", "| episode | table offset | n | gt | head | act | head-gt | act-gt |", "|---|---|---|---|---|---|---|---|"]
    for e in sorted({r["episode"] for r in recs}):
        rs = [r for r in recs if r["episode"] == e and r["phase"] == "grasp" and r["d_gt_obj_cm"] < 6]
        if rs: md.append(f"| {e} | {toff[e]*100:+.2f} | {len(rs)} | {nz(rs,'gt_palm_t*').mean():.2f} | {nz(rs,'head_palm_t*').mean():.2f} | {nz(rs,'act_palm_t*').mean():.2f} | {(nz(rs,'head_palm_t*')-nz(rs,'gt_palm_t*')).mean():+.2f} | {(nz(rs,'act_palm_t*')-nz(rs,'gt_palm_t*')).mean():+.2f} |")
    open(f"{odir}/summary.md", "w").write("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
