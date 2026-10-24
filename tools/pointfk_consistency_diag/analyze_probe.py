"""CPU analysis of the FDM/IDM GT-injection probe.  usage: analyze_probe.py PROBE_PREDS_DIR OUT_DIR [PRIOR_PREDS_A_DIR]
Reuses analyze.window_metrics for every variant (a, a2, b, c, c2, c3)."""
import sys, os, json, glob, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an

PRED, OUT = sys.argv[1], sys.argv[2]
PRIOR = sys.argv[3] if len(sys.argv) > 3 else "/data/shichaojian/runs/consistency_diag_20261014/preds/A"
V = ("a", "a2", "b", "c", "c2", "c3", "d", "e")
R_CELLS = 2.0   # hand region = latent cells within 2 cells of any projected GT FK keypoint
os.makedirs(OUT, exist_ok=True)


def hand_mask(P, T, Hl, Wl, sf):
    uv, ok = P["fk_gt_uv"], P["fk_gt_valid"]
    nf = uv.shape[0] - 1
    k = nf // (T - 1)
    ys = (np.arange(Hl) + 0.5) * sf - 0.5
    xs = (np.arange(Wl) + 0.5) * sf - 0.5
    gx, gy = np.meshgrid(xs, ys)
    M = np.zeros((T, Hl, Wl), bool)
    for t in range(1, T):
        m = ok[k * t]
        if m.any():
            p = uv[k * t][m]
            d = np.sqrt((gx[None] - p[:, 0, None, None]) ** 2 + (gy[None] - p[:, 1, None, None]) ** 2).min(0)
            M[t] = d <= R_CELLS * sf
    return M


def frame_err(x, y, M):
    """per latent frame mean sq diff: whole frame, hand cells, non-hand cells."""
    d = ((x.astype(np.float32) - y.astype(np.float32)) ** 2).mean(0)    # [T,H,W]
    whole = d.reshape(d.shape[0], -1).mean(1)
    hand = np.array([d[t][M[t]].mean() if M[t].any() else np.nan for t in range(len(d))])
    oth = np.array([d[t][~M[t]].mean() for t in range(len(d))])
    return whole, hand, oth


recs, extra = [], []
for f in sorted(glob.glob(f"{PRED}/*.npz")):
    P = dict(np.load(f)); P = {k: (v[0] if k.endswith("vision") and v.ndim == 5 else v) for k, v in P.items()}
    m = json.loads(str(P["meta"]))
    ep, s = m["episode"], m["start_frame"]
    r = dict(episode=ep, start_frame=s, orig24=int(m["orig"]), file=os.path.basename(f))
    W = {v: an.window_metrics({k: P[f"{v}_{k}"] for k in ("action", "fk", "pointflow")}, ep, s) for v in V}
    r["phase"] = W["a"]["phase"]
    for v in V:
        for k, x in W[v].items():
            if k not in ("episode", "start_frame", "phase"):
                r[f"{v}__{k}"] = x
    gt = P["gt_vision"]
    T, Hl, Wl = gt.shape[1:]
    M = hand_mask(P, T, Hl, Wl, int(m["sf"]))
    r["hand_cells_mean"] = float(M[1:].sum((1, 2)).mean())
    r["hand_cells_frac"] = float(M[1:].mean())
    for v in V:
        w, h, o = frame_err(P[f"{v}_vision"], gt, M)
        r[f"{v}__vid_all"], r[f"{v}__vid_hand"], r[f"{v}__vid_other"] = float(w[1:].mean()), float(np.nanmean(h[1:])), float(o[1:].mean())
        for t in range(1, T):
            r[f"{v}__vid_t{t}"] = float(w[t]); r[f"{v}__hand_t{t}"] = float(h[t])
    for x, y in (("c", "c2"), ("c", "c3"), ("a", "a2"), ("a", "c"), ("a", "c2"), ("a", "c3"), ("c2", "c3"), ("a", "d"), ("a", "e"), ("a", "b")):
        w, h, _ = frame_err(P[f"{x}_vision"], P[f"{y}_vision"], M)
        r[f"dv_{x}_{y}_all"], r[f"dv_{x}_{y}_hand"] = float(w[1:].mean()), float(np.nanmean(h[1:]))
        r[f"dact_{x}_{y}"] = float(np.abs(P[f"{x}_action"] - P[f"{y}_action"]).mean())
        for k in ("fk", "pointflow"):
            r[f"d{k}_{x}_{y}"] = float(np.linalg.norm(P[f"{x}_{k}"] - P[f"{y}_{k}"], axis=-1).mean() * 1000)
    # sanity
    r["san_b_video_raw_dev"] = m["stats"]["b"]["video_raw_rel_dev"]
    r["san_c_action_raw_dev"] = m["stats"]["c"]["action_raw_rel_dev"]
    r["san_b_final_maxabs"] = float(np.abs(P["b_vision"].astype(np.float32) - gt.astype(np.float32)).max())
    r["san_gt_vid_scale"] = float(np.abs(gt.astype(np.float32)).std())
    r["san_b_v_t0"] = float(((P["b_vision"].astype(np.float32) - gt.astype(np.float32))[:, 0] ** 2).mean())
    pf = f"{PRIOR}/{os.path.basename(f)}"
    if os.path.exists(pf):
        r["san_a_vs_prior_rms"] = float(np.sqrt(((np.load(pf)["action"] - P["a_action"]) ** 2).mean()))
    r["partner"] = f'{m["partner"]["episode"]}_{m["partner"]["start_frame"]}'
    recs.append(r)

keys = list(recs[0])
with open(f"{OUT}/windows.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(recs)

ph = {"all": recs, "grasp": [r for r in recs if r["phase"] == "grasp"], "other": [r for r in recs if r["phase"] == "other"]}
col = lambda rs, k: np.array([r[k] for r in rs], float)
mean = lambda rs, k: float(np.nanmean(col(rs, k)))
med = lambda rs, k: float(np.nanmedian(col(rs, k)))
L = [f"# FDM/IDM ground-truth injection probe + video perturbation (offline, model {os.environ.get('DIAG_KEY', 'A')}, task27)\n",
     f"windows: all {len(recs)} (orig24 {sum(r['orig24'] for r in recs)}), grasp {len(ph['grasp'])}, other {len(ph['other'])}. Hand region = latent cells within {R_CELLS:g} cells of a projected GT FK keypoint "
     f"(mean {np.mean([r['hand_cells_frac'] for r in recs]) * 100:.1f}% of cells). Video error = latent MSE vs GT future latents (frames 1..T-1). Errors mm unless noted.\n"]

# ---- sanity
L.append("## Sanity\n")
pr = [r["san_a_vs_prior_rms"] for r in recs if "san_a_vs_prior_rms" in r]
L.append(f"1. baseline a vs earlier single-sample preds: {len(pr)} windows matched, action RMS mean {np.mean(pr):.5f} max {np.max(pr):.5f} rad (limit 0.002)" if pr else "1. NO matched prior windows")
L.append(f"2. inject GT action (c): action MAE vs GT mean {mean(recs, 'c__act_mae_rad'):.2e} rad (a: {mean(recs, 'a__act_mae_rad'):.4f}); FK(act) vs FK(GT act) {mean(recs, 'c__fkact_vs_fkact_gt'):.3e} mm (a: {mean(recs, 'a__fkact_vs_fkact_gt'):.2f}); raw rel dev before final override mean {np.mean([r['san_c_action_raw_dev'] for r in recs]):.2e}")
L.append(f"3. inject GT video (b): sampler-end rel dev vs GT latents mean {np.mean([r['san_b_video_raw_dev'] for r in recs]):.2e} max {np.max([r['san_b_video_raw_dev'] for r in recs]):.2e} (before the sigma->0 override); "
         f"b video latent MSE vs GT (frames 1..) mean {mean(recs, 'b__vid_all'):.2e}; conditioning frame 0 MSE {np.mean([r['san_b_v_t0'] for r in recs]):.2e}; GT latent std {np.mean([r['san_gt_vid_scale'] for r in recs]):.3f}")

# ---- IDM
L.append("\n## IDM: a (own video) vs b (GT video injected)\n")
L.append("| metric | phase | a mean | b mean | removed (mean) | a median | b median | removed (median) |\n|---|---|---|---|---|---|---|---|")
for k, lab in (("fkact_vs_fkact_gt", "FK(act) vs FK(GT act) mm"), ("fkact_vs_fk_gt", "FK(act) vs FK GT mm"), ("act_mae_rad", "action MAE rad"), ("fk_head_err", "FK head ADE mm"), ("pf_ade", "PF ADE mm")):
    for p, rs in ph.items():
        a, b, am, bm = mean(rs, f"a__{k}"), mean(rs, f"b__{k}"), med(rs, f"a__{k}"), med(rs, f"b__{k}")
        L.append(f"| {lab} | {p} ({len(rs)}) | {a:.4g} | {b:.4g} | {(a - b) / a * 100:.1f}% | {am:.4g} | {bm:.4g} | {(am - bm) / am * 100:.1f}% |")
L.append("\nHorizon profile (mean over windows, frame k of 32), FK(act) vs FK(GT act) mm; removed = (a-b)/a\n")
L.append("| phase | " + " | ".join(f"h{h} a / b / removed" for h in an.HORIZ) + " |\n|---|" + "---|" * len(an.HORIZ))
for p, rs in ph.items():
    cells = []
    for h in an.HORIZ:
        a, b = mean(rs, f"a__h{h}_fkact_vs_fkact_gt"), mean(rs, f"b__h{h}_fkact_vs_fkact_gt")
        cells.append(f"{a:.2f} / {b:.2f} / {(a - b) / a * 100:.0f}%")
    L.append(f"| {p} | " + " | ".join(cells) + " |")
L.append("\nSame for FK(act) vs FK GT (realised), mm\n")
L.append("| phase | " + " | ".join(f"h{h} a / b / removed" for h in an.HORIZ) + " |\n|---|" + "---|" * len(an.HORIZ))
for p, rs in ph.items():
    cells = []
    for h in an.HORIZ:
        a, b = mean(rs, f"a__h{h}_fkact_vs_fk_gt"), mean(rs, f"b__h{h}_fkact_vs_fk_gt")
        cells.append(f"{a:.2f} / {b:.2f} / {(a - b) / a * 100:.0f}%")
    L.append(f"| {p} | " + " | ".join(cells) + " |")

# ---- FDM
L.append("\n## FDM: does the generated video follow the action?\n")
L.append("Latent MSE vs GT future latents (mean over frames 1..T-1; mean over windows). a2 = baseline with another seed.\n")
L.append("| variant | phase | whole frame | hand region | non-hand | hand/whole |\n|---|---|---|---|---|---|")
for v, lab in (("a", "a baseline"), ("a2", "a2 baseline seed2"), ("c", "c GT action"), ("c2", "c2 static action"), ("c3", "c3 swapped action"), ("b", "b GT video (ref ~0)")):
    for p, rs in ph.items():
        w, h, o = mean(rs, f"{v}__vid_all"), mean(rs, f"{v}__vid_hand"), mean(rs, f"{v}__vid_other")
        L.append(f"| {lab} | {p} | {w:.4f} | {h:.4f} | {o:.4f} | {h / max(w, 1e-12):.2f} |")
nT = max(int(k.split("_t")[1]) for k in keys if k.startswith("a__vid_t"))
L.append("\nPer latent frame, latent MSE vs GT (all windows): whole / hand\n")
L.append("| variant | " + " | ".join(f"t{t}" for t in range(1, nT + 1)) + " |\n|---|" + "---|" * nT)
for v in ("a", "a2", "c", "c2", "c3"):
    L.append(f"| {v} | " + " | ".join(f"{mean(recs, f'{v}__vid_t{t}'):.3f} / {mean(recs, f'{v}__hand_t{t}'):.3f}" for t in range(1, nT + 1)) + " |")
L.append("\nAction sensitivity of the VIDEO: latent MSE between two samples (mean over frames 1..), whole / hand. Seed noise = a vs a2.\n")
L.append("| pair | phase | whole | hand | whole / seed-noise | hand / seed-noise |\n|---|---|---|---|---|---|")
for x, y, lab in (("a", "a2", "seed noise a vs a2"), ("c", "c2", "c vs c2 (GT vs static)"), ("c", "c3", "c vs c3 (GT vs swap)"), ("c2", "c3", "c2 vs c3"), ("a", "c", "a vs c")):
    for p, rs in ph.items():
        w, h = mean(rs, f"dv_{x}_{y}_all"), mean(rs, f"dv_{x}_{y}_hand")
        nw, nh = mean(rs, "dv_a_a2_all"), mean(rs, "dv_a_a2_hand")
        L.append(f"| {lab} | {p} | {w:.4f} | {h:.4f} | {w / nw:.2f} | {h / nh:.2f} |")
L.append("\nPF / FK following the injected action (mm; FK/PF ADE vs GT; sample-to-sample mean point distance; FK head vs FK(output action)).\n")
L.append("| quantity | phase | a | a2 | c | c2 | c3 |\n|---|---|---|---|---|---|---|")
for k, lab in (("fk_head_err", "FK ADE vs GT"), ("pf_ade", "PF ADE vs GT"), ("fk_head_vs_fkact_abs", "FK head vs FK(output action), abs")):
    for p, rs in ph.items():
        L.append(f"| {lab} | {p} | " + " | ".join(f"{mean(rs, f'{v}__{k}'):.2f}" for v in ("a", "a2", "c", "c2", "c3")) + " |")
L.append("\n| sample-to-sample distance mm | phase | a vs a2 (noise) | c vs c2 | c vs c3 | a vs c |\n|---|---|---|---|---|---|")
for k, lab in (("fk", "FK"), ("pointflow", "PF")):
    for p, rs in ph.items():
        L.append(f"| {lab} | {p} | " + " | ".join(f"{mean(rs, f'd{k}_{x}_{y}'):.2f}" for x, y in (("a", "a2"), ("c", "c2"), ("c", "c3"), ("a", "c"))) + " |")
# ---- D0: video perturbation robustness (PDF section 8)
L.append("\n## Video perturbation robustness: d = another episode's video, e = video held at pure noise\n")
L.append("Action shift vs the baseline a (normalised-action MAE; FK / PF mean point distance mm). Seed noise = a vs a2.\n")
L.append("| pair | phase | action shift | x seed noise | FK shift mm | PF shift mm |\n|---|---|---|---|---|---|")
for x, y, lab in (("a", "a2", "seed noise a vs a2"), ("a", "d", "a vs d (other video)"), ("a", "e", "a vs e (noise video)"), ("a", "b", "a vs b (GT video)")):
    for p, rs in ph.items():
        if f"dact_{x}_{y}" not in rs[0]:
            continue
        s, n = mean(rs, f"dact_{x}_{y}"), mean(rs, "dact_a_a2")
        L.append(f"| {lab} | {p} | {s:.4f} | {s / n:.2f} | {mean(rs, f'dfk_{x}_{y}'):.2f} | {mean(rs, f'dpointflow_{x}_{y}'):.2f} |")
L.append("\nAction / FK(act) error vs GT under the perturbation (mean)\n")
L.append("| metric | phase | a | d | e | b (GT video) |\n|---|---|---|---|---|---|")
for k, lab in (("act_mae_rad", "action MAE rad"), ("fkact_vs_fkact_gt", "FK(act) vs FK(GT act) mm")):
    for p, rs in ph.items():
        L.append(f"| {lab} | {p} | " + " | ".join(f"{mean(rs, f'{v}__{k}'):.4g}" for v in ("a", "d", "e", "b")) + " |")
open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
