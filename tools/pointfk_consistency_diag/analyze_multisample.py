"""Multi-sample selection test.  usage: analyze_multisample.py PREDS_DIR OUT_DIR [--single DIR_OF_SINGLE_SAMPLE_NPZ]   (CPU only)
PREDS_DIR/*.npz hold stacked samples action[N,33,52], fk[N,32,42,3], pointflow[N,32,1024,3] (diag_entry.py with DIAG_NSAMPLES=N).
Per window: metrics per sample (analyze.window_metrics), selectors (argmin c_rel / PF-FK raw / sum), baselines (random, k0, medoid, mean action, oracles)."""
import argparse, csv, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an

MET = [("fkact_vs_fkact_gt", "FK(pred act) vs FK(GT act), commanded [mm]"), ("fkact_vs_fk_gt", "FK(pred act) vs FK GT, realised [mm]"), ("act_mae_rad", "action MAE [rad]")]
SELS = ["sel_crel", "sel_pffk", "sel_sum"]
METHODS = ["random(mean of N)", "k0(default)", "sel_crel", "sel_pffk", "sel_crel+pffk", "medoid", "mean_action", "oracle_best", "oracle_worst"]

def amin(v, fallback=0):
    v = np.asarray(v, float)
    return int(np.nanargmin(v)) if np.isfinite(v).any() else fallback

def ranks(x): return np.argsort(np.argsort(x)).astype(float)
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0: return np.nan
    return float(np.corrcoef(ranks(a[m]), ranks(b[m]))[0, 1])

def sign_p(w, l):  # two-sided exact binomial
    n = w + l
    if n == 0: return float("nan")
    k = min(w, l); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preds"); ap.add_argument("out"); ap.add_argument("--single", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    per_sample, W = [], []   # per-sample csv rows; per-window dict(method -> metrics)
    san = []; diffk = []
    for f in sorted(glob.glob(f"{a.preds}/*.npz")):
        P = np.load(f); m = json.loads(str(P["meta"])); ep, s = m["episode"], m["start_frame"]
        act, fkp, pf = P["action"], P["fk"], P["pointflow"]; N = len(act)
        recs = []
        for k in range(N):
            r = an.window_metrics(dict(action=act[k], fk=fkp[k], pointflow=pf[k]), ep, s)
            r["k"] = k; r["seed"] = m["seeds"][k]; r["orig24"] = int(m["orig"]); recs.append(r)
        per_sample += recs
        J = an.J; A = act[:, :, :J]
        d = np.sqrt(((A[:, None] - A[None]) ** 2).mean((2, 3)))      # N x N RMS action distance
        med = int(np.argmin(d.sum(1)))
        rm = an.window_metrics(dict(action=act.mean(0), fk=fkp.mean(0), pointflow=pf.mean(0)), ep, s)
        col = lambda key: np.array([r[key] for r in recs], float)
        crel, pffk = col("fk_head_vs_fkact_rel"), col("pffk_raw")
        ssum = crel + np.where(np.isfinite(pffk), pffk, np.nan)
        orc = col("fkact_vs_fkact_gt")
        pick = {"k0(default)": 0, "sel_crel": amin(crel), "sel_pffk": amin(pffk), "sel_crel+pffk": amin(ssum), "medoid": med,
                "oracle_best": amin(orc), "oracle_worst": int(np.nanargmax(orc))}
        w = dict(episode=ep, start_frame=s, phase=recs[0]["phase"], pick=pick, spear_crel=spearman(crel, orc), spear_pffk=spearman(pffk, orc))
        w["M"] = {nm: recs[i] for nm, i in pick.items()}
        w["M"]["mean_action"] = rm
        w["M"]["random(mean of N)"] = {k: float(np.nanmean(col(k))) for k, _ in MET}
        W.append(w)
        if a.single and os.path.exists(f"{a.single}/{os.path.basename(f)}"):
            S = np.load(f"{a.single}/{os.path.basename(f)}")["action"][:, :J]
            san.append(float(np.sqrt(((S - A[0]) ** 2).mean())))
        iu = np.triu_indices(N, 1); diffk.append(float(d[iu].mean()))
    cols = list(per_sample[0])
    with open(f"{a.out}/samples.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(per_sample)
    md = [f"# Multi-sample selection (model A nonlocal 40k, N={len(act)} samples/window)\n", f"windows: {len(W)} (grasp {sum(w['phase']=='grasp' for w in W)}, other {sum(w['phase']=='other' for w in W)}); mean/median over windows.  Lower is better for all metrics.\n"]
    md.append("## Sanity\n")
    if san: md.append(f"- k=0 vs earlier single-sample action RMS (rad): mean {np.mean(san):.5f}, max {np.max(san):.5f}, n={len(san)}/{len(W)} (threshold 0.002)")
    md.append(f"- mean pairwise action RMS between the N samples of a window (rad): mean {np.mean(diffk):.5f}, min {np.min(diffk):.5f} (>0 means samples differ)\n")
    phs = {"all": W, "grasp": [w for w in W if w["phase"] == "grasp"], "other": [w for w in W if w["phase"] == "other"]}
    for key, lab in MET:
        d_ = 4 if key == "act_mae_rad" else 2
        md += [f"## {lab}\n", "| method | all mean/med | grasp mean/med | other mean/med |", "|---|---|---|---|"]
        for me in METHODS:
            cells = []
            for p, ws in phs.items():
                v = np.array([w["M"][me][key] for w in ws], float); v = v[np.isfinite(v)]
                cells.append(f"{v.mean():.{d_}f} / {np.median(v):.{d_}f}" if len(v) else "n/a")
            md.append(f"| {me} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Win rate vs k0 (selected sample strictly better than k0; ties = same sample or equal; two-sided exact sign test)\n",
           "| selector | metric | scope | win/loss/tie | win rate (w/(w+l)) | p |", "|---|---|---|---|---|---|"]
    for me in ["sel_crel", "sel_pffk", "sel_crel+pffk", "medoid", "mean_action", "oracle_best"]:
        for key, _ in MET:
            for p in ("all", "grasp"):
                dd = np.array([w["M"][me][key] - w["M"]["k0(default)"][key] for w in phs[p]], float); dd = dd[np.isfinite(dd)]
                wn, ls = int((dd < -1e-12).sum()), int((dd > 1e-12).sum())
                md.append(f"| {me} | {key} | {p} | {wn}/{ls}/{len(dd)-wn-ls} | {wn/max(wn+ls,1):.3f} | {sign_p(wn, ls):.3g} |")
    md += ["", "## Win rate vs random (mean of N) -- selected better than the window's N-sample mean\n", "| selector | metric | scope | win/loss | p |", "|---|---|---|---|---|"]
    for me in ["sel_crel", "sel_pffk", "sel_crel+pffk", "medoid"]:
        for key, _ in MET[:1] + MET[2:]:
            for p in ("all", "grasp"):
                dd = np.array([w["M"][me][key] - w["M"]["random(mean of N)"][key] for w in phs[p]], float); dd = dd[np.isfinite(dd)]
                wn, ls = int((dd < 0).sum()), int((dd > 0).sum())
                md.append(f"| {me} | {key} | {p} | {wn}/{ls} | {sign_p(wn, ls):.3g} |")
    md += ["", "## Within-window Spearman across the N samples (vs fkact_vs_fkact_gt)\n", "| signal | scope | mean | median | n windows |", "|---|---|---|---|---|"]
    for nm, k in (("c_rel", "spear_crel"), ("PF-FK raw", "spear_pffk")):
        for p, ws in phs.items():
            v = np.array([w[k] for w in ws], float); v = v[np.isfinite(v)]
            md.append(f"| {nm} | {p} | {v.mean():.3f} | {np.median(v):.3f} | {len(v)} |")
    open(f"{a.out}/summary.md", "w").write("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
