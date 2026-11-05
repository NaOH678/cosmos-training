"""Hand-vs-ball error decomposition on the offline validation windows (CPU only, read-only; nolocal 40k PF/FK model).
usage: analyze_hand_obj.py OUT_DIR [--ms-preds DIR] [--probe-preds DIR] [--samples-csv CSV]
Reuses the frame/geometry conventions of analyze_spread.py (imported): world frame via Kabsch fit of FK(GT action) to the FK anchors,
ball poses = box3d center_world (soccer/tennis/golf/table-tennis), active hand = larger GT closure, fingertip centroid = mean of the 5 *_tip keypoints.

(1) ON-BALL ANCHORS (frame 0).  A PointFlow anchor (world xyz) lies on ball B when  | |p - c_B| - r_B | <= 0.7 cm  (r_B = max(size_lwh)/2, c_B = box3d centre at
    window frame 0)  AND  z >= table_top + 0.5 cm  (drops table points under/next to the ball).  table_top = max( median over the scene's balls of (centre_z - r) at frame 0 , mode of the anchor-z histogram (2 mm bins) within +-5 cm of it ) when they
    differ by <3 cm (else the ball-based one); both logged (csv table_top = used value, table_top_anchor_mode).
    Caveat: hand surface points inside the shell cannot be excluded (hand is usually >5 cm away at frame 0 except in <5 cm bin).
    Alternative criterion (csv col n_on_alt_B): |p-c| <= r + 0.5 cm and z >= top + 0.5 cm.
    Hand-to-ball gap at frame 0 = |GT fingertip centroid(active hand) - c_B| - r_B  (cm), bins >10 / 5-10 / <5.
(2) ERRORS (target ball = ball with the closest GT fingertip-centroid approach over frames 1..32; t* = that frame; 'last' = frame 32), world frame, cm.
    hand_pred = FK-head fingertip centroid (or FK(predicted action) centroid, 'act');  hand_gt = realised GT fingertip centroid.
    ball_pred = mean over on-ball anchors valid at that frame of (anchor + PF predicted displacement);  ball_gt = mean of (anchor + GT displacement) of the same anchors.
    E_hand = |hand_pred - hand_gt|, E_ball = |ball_pred - ball_gt|, E_rel = |(hand_pred - ball_pred) - (hand_gt - ball_gt)|  (3D and xy).
    E_ball/E_rel use windows with >=3 on-ball valid anchors (the n<3 are excluded; 1-2 are shown separately in the count-bin table).
    Sets: 'probe_a' = fdm_idm_probe sample a (single baseline sample); 'ms8' = consistency_multisample 8 samples (per-sample errors averaged over the 8).
"""
import argparse, csv, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_spread as S
J = S.J; T = lambda x: torch.as_tensor(np.asarray(x)).float()
SHELL, ZMARGIN = 0.007, 0.005

def load_set(path, kind):
    """-> dict (ep,s) -> (action[N,33,52], fk[N,32,42,3], pf[N,32,1024,3])"""
    out = {}
    for f in sorted(glob.glob(path + "/*.npz")):
        P = np.load(f, allow_pickle=True); m = json.loads(str(P["meta"]))
        if kind == "ms": out[(m["episode"], m["start_frame"])] = (P["action"], P["fk"], P["pointflow"])
        else: out[(m["episode"], m["start_frame"])] = (P["a_action"][None], P["a_fk"][None], P["a_pointflow"][None])
    return out

def table_top(oc0, apf_w):
    z = apf_w[:, 2]; tops = [c[2] - r for c, r in oc0.values()]; tb = float(np.median(tops))
    sel = z[(z > tb - 0.05) & (z < tb + 0.05)]
    if len(sel) < 20: return tb, np.nan
    h, e = np.histogram(sel, bins=np.arange(tb - 0.05, tb + 0.05, 0.002)); k = int(h.argmax()); return tb, float(e[k] + 0.001)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--ms-preds", default="/data/shichaojian/runs/consistency_multisample_20261014/preds/A")
    ap.add_argument("--probe-preds", default="/data/shichaojian/runs/fdm_idm_probe_20261014/preds/A")
    ap.add_argument("--samples-csv", default="/data/shichaojian/runs/consistency_multisample_20261014/samples.csv"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {(r["episode"], int(r["start_frame"])): r for r in csv.DictReader(open(a.samples_csv)) if r["k"] == "0"}
    sets = {"probe_a": load_set(a.probe_preds, "probe"), "ms8": load_set(a.ms_preds, "ms")}
    ball_rows, win_rows = [], []
    for (ep, s) in sorted(set(sets["probe_a"]) & set(sets["ms8"])):
        mr = meta[(ep, s)]; hand = "left" if float(mr["closure_left"]) >= float(mr["closure_right"]) else "right"; tips = S.TIPS[hand]
        pt = torch.load(f"{S.SEL}/{S.rows[(ep, s)]}", weights_only=False, map_location="cpu")
        anchor = T(pt["fk"]["inputs"]["anchor_xyz"]); a_gt = T(pt["action"])[:, :J]
        with torch.no_grad():
            w_gt = S.FK(a_gt[None])[0]; R, tv = S.estimate_rigid_transform(w_gt[0][None], anchor[None]); R, tv = R[0], tv[0]
        Rt = R.T; w0 = w_gt[0]
        gt_real = (w0[None] + T(pt["fk"]["targets"]["displacement"]) @ Rt.T)[:, tips].mean(1).numpy()     # [32,3] frames 1..32
        hand0 = w0[tips].mean(0).numpy()
        raw = np.asarray(pt["raw_frame_ids"]).astype(int); ob = S.objects(ep)
        apf = T(pt["pointflow"]["inputs"]["anchor_xyz"]); apf_w = ((apf - tv) @ R).numpy()                   # R^T(c-t)
        gtd = (T(pt["pointflow"]["targets"]["displacement"]) @ Rt.T).numpy(); gtv = np.asarray(pt["pointflow"]["targets"]["valid"]).astype(bool)   # [32,1024,3] world, [32,1024]
        oc0 = {o: (c[raw[0]], r) for o, (c, r) in ob.items()}
        ttop_b, ttop_a = table_top(oc0, apf_w)[0], table_top(oc0, apf_w)[1]
        ttop = max(ttop_b, ttop_a) if np.isfinite(ttop_a) and abs(ttop_a - ttop_b) < 0.03 else ttop_b   # conservative: higher of the two estimates
        on = {}
        for o, (c, r) in oc0.items():
            d = np.linalg.norm(apf_w - c, axis=1); ok = apf_w[:, 2] >= ttop + ZMARGIN
            on[o] = (np.abs(d - r) <= SHELL) & ok
            ball_rows.append(dict(episode=ep, start_frame=s, phase=mr["phase"], ball=o, radius_cm=r * 100, gap_cm=(np.linalg.norm(hand0 - c) - r) * 100,
                                  n_on=int(on[o].sum()), n_on_alt=int(((d <= r + 0.005) & ok).sum()), table_top=ttop, table_top_anchor_mode=ttop_a))
        # target ball: closest GT approach
        best = None
        for o, (c, r) in ob.items():
            dd = np.linalg.norm(gt_real - c[raw[1:]], axis=1); k = int(dd.argmin())
            if best is None or dd[k] < best[0]: best = (dd[k], o, k, r)
        dmin, oname, tstar, orad = best; sel = on[oname]
        rec = dict(episode=ep, start_frame=s, phase=mr["phase"], hand=hand, obj=oname, tstar=tstar + 1, d_gt_obj_cm=dmin * 100, gap0_cm=(np.linalg.norm(hand0 - oc0[oname][0]) - orad) * 100, n_on=int(sel.sum()))
        for sname, D in sets.items():
            act_, fk_, pf_ = D[(ep, s)]; N = len(act_)
            with torch.no_grad(): w_pr = S.FK(torch.from_numpy(act_).float()[:, :, :J])
            head = (w0[None, None] + torch.from_numpy(fk_).float() @ Rt.T)[:, :, tips].mean(2).numpy()
            actp = w_pr[:, 1:, tips].mean(2).numpy()
            pf_w = np.einsum("ntpc,cd->ntpd", pf_, R.numpy())                     # R^T d  (row form d @ R)
            for tag, t in (("cl", tstar), ("last", 31)):
                v = sel & gtv[t]; nv = int(v.sum()); rec[f"{sname}_{tag}_nvalid"] = nv
                g_ball = (apf_w[v] + gtd[t, v]).mean(0) if nv else None
                for pname, X in (("head", head), ("act", actp)):
                    eh = X[:, t] - gt_real[t][None]                              # [N,3]
                    pre = f"{sname}_{tag}_{pname}_"
                    rec[pre + "Eh3"] = np.linalg.norm(eh, axis=1).mean() * 100; rec[pre + "Ehxy"] = np.linalg.norm(eh[:, :2], axis=1).mean() * 100
                    if nv >= 1:
                        p_ball = (apf_w[v][None] + pf_w[:, t, v]).mean(1)       # [N,3]
                        eb = p_ball - g_ball[None]; er = eh - eb                  # (hp-bp)-(hg-bg)
                        rec[pre + "Eb3"] = np.linalg.norm(eb, axis=1).mean() * 100; rec[pre + "Ebxy"] = np.linalg.norm(eb[:, :2], axis=1).mean() * 100
                        rec[pre + "Er3"] = np.linalg.norm(er, axis=1).mean() * 100; rec[pre + "Erxy"] = np.linalg.norm(er[:, :2], axis=1).mean() * 100
                        cs = (eh * eb).sum(1) / (np.linalg.norm(eh, axis=1) * np.linalg.norm(eb, axis=1) + 1e-9); rec[pre + "cos_hb"] = cs.mean()
                        for i, ax in enumerate("xyz"): rec[pre + "ehv" + ax] = eh[:, i].mean() * 100; rec[pre + "ebv" + ax] = eb[:, i].mean() * 100
                    else:
                        for k in ("Eb3", "Ebxy", "Er3", "Erxy", "cos_hb", "ehvx", "ehvy", "ehvz", "ebvx", "ebvy", "ebvz"): rec[pre + k] = np.nan
        win_rows.append(rec)
        print(ep, s, mr["phase"], oname[:12], f"gap0={rec['gap0_cm']:.1f} n_on={rec['n_on']} top={ttop:.3f}/{ttop_a:.3f}", flush=True)
    for fn, rows in (("windows_hand_obj.csv", win_rows), ("balls_onsurface.csv", ball_rows)):
        cols = []
        for r in rows:
            for c in r:
                if c not in cols: cols.append(c)
        with open(f"{a.out}/{fn}", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    summarize(win_rows, ball_rows, a.out)

def pct(v, q): return float(np.percentile(v, q)) if len(v) else float("nan")
def summarize(W, Bl, odir):
    nz = lambda rs, k: np.array([r.get(k, np.nan) for r in rs], float)
    md = ["# Hand vs ball error decomposition (nolocal 40k PF/FK, offline validation, CPU)\n", "Definitions: see the docstring of analyze_hand_obj.py.  cm, world frame.\n"]
    tt = np.array([r["table_top"] - r["table_top_anchor_mode"] for r in Bl if np.isfinite(r["table_top_anchor_mode"])]) * 100
    tb_ = np.array([r["table_top"] for r in Bl]); md.append(f"table_top used: median {np.median(tb_):.4f} m.")
    md.append(f"windows {len(W)} (grasp {sum(r['phase']=='grasp' for r in W)}, other {sum(r['phase']=='other' for r in W)}).  table_top(ball-based) - anchor-z-histogram mode: median {np.median(tt):.2f} cm, p10/p90 {pct(tt,10):.2f}/{pct(tt,90):.2f} cm.\n")
    md += ["## (1) Anchors on each ball surface at frame 0 (|dist-r|<=0.7 cm, z>=table+0.5 cm)\n", "Each cell: median / p10 / p90 of n_on over (window,ball) pairs; [fraction with 0 points]; (n pairs).  Bins = gap of the active fingertip centroid to the ball surface at frame 0.\n",
           "| ball | all | gap >10 cm | gap 5-10 cm | gap <5 cm |", "|---|---|---|---|---|"]
    def cell(rs):
        if not rs: return "-"
        n = np.array([r["n_on"] for r in rs], float); return f"{np.median(n):.0f} / {pct(n,10):.0f} / {pct(n,90):.0f} [{np.mean(n==0):.2f}] ({len(n)})"
    for b in S.BALLS + ["ALL"]:
        rs = [r for r in Bl if b == "ALL" or r["ball"] == b]
        md.append(f"| {b} (r={np.median([r['radius_cm'] for r in rs]):.1f} cm) | " + " | ".join(cell([r for r in rs if f(r["gap_cm"])]) for f in (lambda g: True, lambda g: g > 10, lambda g: 5 <= g <= 10, lambda g: g < 5)) + " |")
    for b in S.BALLS:   # alt criterion
        rs = [r for r in Bl if r["ball"] == b]; n = np.array([r["n_on_alt"] for r in rs], float)
        md.append(f"\nalt criterion (dist<=r+0.5cm, z>=top+0.5cm) {b}: median {np.median(n):.0f} p10 {pct(n,10):.0f} p90 {pct(n,90):.0f}, zero-frac {np.mean(n==0):.2f}")
    md.append("")
    groups = [("grasp, all", lambda r: r["phase"] == "grasp"), ("grasp, contact (GT closest approach <6 cm)", lambda r: r["phase"] == "grasp" and r["d_gt_obj_cm"] < 6),
              ("other, all", lambda r: r["phase"] == "other"), ("other, contact (<6 cm)", lambda r: r["phase"] == "other" and r["d_gt_obj_cm"] < 6)]
    md.append("## (2) Hand / ball / relative errors (target ball, windows with >=3 valid on-ball anchors; mean / median per window, cm)\n")
    for sname in ("probe_a", "ms8"):
        for tag, tl in (("cl", "t* (GT closest approach)"), ("last", "last frame (32)")):
            for pname, pl in (("head", "FK head"), ("act", "FK(predicted action)")):
                md.append(f"### {sname} ({'single baseline sample' if sname=='probe_a' else 'mean over 8 samples'}), {tl}, hand = {pl}\n")
                md += ["| group | n | E_hand 3D | E_hand xy | E_ball 3D | E_ball xy | E_rel 3D | E_rel xy | sqrt(Eh²+Eb²) 3D | frac E_rel<E_hand | mean cos(eh,eb) | pooled corr(eh,eb) | E_hand 3D (all wins) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for gname, f in groups:
                    rs = [r for r in W if f(r)]; pre = f"{sname}_{tag}_{pname}_"
                    ok = [r for r in rs if r[f"{sname}_{tag}_nvalid"] >= 3]
                    if not ok: continue
                    g = lambda k: nz(ok, pre + k); mm = lambda k: f"{np.mean(g(k)):.2f} / {np.median(g(k)):.2f}"
                    ind = np.sqrt(g("Eh3") ** 2 + g("Eb3") ** 2)
                    eh = np.concatenate([[r[pre + "ehv" + c] for r in ok] for c in "xyz"]); eb = np.concatenate([[r[pre + "ebv" + c] for r in ok] for c in "xyz"])
                    md.append(f"| {gname} | {len(ok)}/{len(rs)} | {mm('Eh3')} | {mm('Ehxy')} | {mm('Eb3')} | {mm('Ebxy')} | {mm('Er3')} | {mm('Erxy')} | {np.mean(ind):.2f} | {np.mean(g('Er3') < g('Eh3')):.2f} | {np.nanmean(g('cos_hb')):.2f} | {np.corrcoef(eh, eb)[0,1]:.2f} | {np.nanmean(nz(rs, pre + 'Eh3')):.2f} |")
                md.append("")
    md += ["## (3) Error vs number of on-ball anchors (grasp + other, target ball, at t*; FK head; mean 3D cm per window; n)\n", "| set | n_on bin | n | E_hand | E_ball | E_rel |", "|---|---|---|---|---|---|"]
    for sname in ("probe_a", "ms8"):
        for lab, lo, hi in (("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), (">10", 11, 10 ** 9)):
            rs = [r for r in W if lo <= r[f"{sname}_cl_nvalid"] <= hi]
            if not rs: continue
            pre = f"{sname}_cl_head_"; f = lambda k: f"{np.nanmean(nz(rs, pre + k)):.2f}" if np.isfinite(nz(rs, pre + k)).any() else "n/a"
            md.append(f"| {sname} | {lab} | {len(rs)} | {f('Eh3')} | {f('Eb3')} | {f('Er3')} |")
    md += ["", "### per target ball (probe_a, t*, FK head, windows with >=3 anchors; mean 3D cm)\n", "| ball | n | median n_on | E_hand | E_ball | E_rel | Spearman(n_on,E_rel) within ball |", "|---|---|---|---|---|---|---|"]
    for b in S.BALLS:
        rs = [r for r in W if r["obj"] == b and r["probe_a_cl_nvalid"] >= 3]
        if len(rs) < 3: continue
        pre = "probe_a_cl_head_"; md.append(f"| {b} | {len(rs)} | {np.median(nz(rs,'n_on')):.0f} | {np.mean(nz(rs,pre+'Eh3')):.2f} | {np.mean(nz(rs,pre+'Eb3')):.2f} | {np.mean(nz(rs,pre+'Er3')):.2f} | {_sp(nz(rs,'probe_a_cl_nvalid'), nz(rs,pre+'Er3')):.2f} |")
    for sname in ("probe_a", "ms8"):
        rs = [r for r in W if r[f"{sname}_cl_nvalid"] >= 1]; n = nz(rs, f"{sname}_cl_nvalid")
        for k in ("Eb3", "Er3"):
            md.append(f"\n{sname}: Spearman(n_on valid at t*, {k}) over {len(rs)} windows with >=1 anchor = {_sp(n, nz(rs, f'{sname}_cl_head_{k}')):.2f}")
    open(f"{odir}/summary.md", "w").write("\n".join(md)); print("\n".join(md))

def _sp(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])

if __name__ == "__main__":
    main()
