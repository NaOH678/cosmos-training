"""Cross-modal consistency in REAL closed-loop rollouts (same metric definitions as analyze.py).  CPU only.
usage: closed_loop.py run  OUT_DIR [--models ...] [--runs ...] [--workers N]   -> OUT_DIR/queries.csv
       closed_loop.py report OUT_DIR                                            -> OUT_DIR/summary.md

Per query (one replan, 32 steps) of a saved rollout, predictions/query_*.npz holds the model's RAW actions, FK head and PointFlow.
  a_0 = robot/qpos[row] (the conditioning state; the live fk anchor is exactly FK(a_0)), a_1..a_32 = executed_actions (raw rad).
  GT FK / PointFlow = recorded simulator body poses (metrics/*ground_truth.npz) via the Bench2Dex query_data (same as pointfk_truth).
  "GT action" of the relative form = the commanded action actually executed, action/commanded[row : row+32] (smoothed raw).
  Displacement form everywhere (K_k - K_0), camera frame, mm; offline analyze.py definitions otherwise."""
import sys, os, json, glob, csv, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CL_CODE_ROOT", os.path.abspath(os.path.join(HERE, "..", ".."))))   # consistency module + URDF FK
BENCH = os.environ.get("BENCH2DEX_ROOT", "/cpfs/user/shichaojian/Bench2Dex-Pointflow-fk")               # query_data / final_attempt
sys.path.insert(0, BENCH)

INFER = "/oss-shichaojian/outputs/infer/task27"
MODELS = [f"task27_cosmos_{m}_s4_{s}" for m in ("nolocal", "local") for s in (30000, 35000, 40000)]
RUNS = ["eval_200_8gpu", "eval_200_8gpu_b5_g1"]; GUID = {"eval_200_8gpu": 3, "eval_200_8gpu_b5_g1": 1}
CHANNELS = ["none", "cov_only", "inv_only", "inv_cov"]
GRASP_THR = float(os.environ.get("GRASP_THR", "0.25"))
GOLF = "obj_058"; GOLF_OFF = [-0.0065395, -0.0304375, 0.0209435]; GOLF_RAD = 0.0213   # from gap_20261013/funnel3.py
LAST_N = 5
G = {}   # per-process cache


def init(stats, urdf, fkjson):
    import torch
    torch.set_num_threads(1)
    from cosmos_framework.model.generator import pointfk_consistency as pc
    from cosmos_framework.model.generator.robot_fk_torch import TorchRobotFK
    names, offset, scale = pc.load_action_stats(stats)
    norm = json.load(open(stats))["action"]; rng = np.maximum(np.array(norm["q99"]) - np.array(norm["q01"]), 1e-3)
    G.update(pc=pc, names=names, J=len(names), rng=rng, radius=pc.pf_fk_radius(),
             fk=TorchRobotFK.from_json(urdf, fkjson, names),
             fingers={h: [i for i, n in enumerate(names) if f"{h}_finger" in n] for h in ("left", "right")})


def mm(v, m):
    return float(np.linalg.norm(v, axis=-1)[m].mean() * 1000) if m.any() else np.nan


def golf_state(f, truth, row_count):
    """per hdf5 row: hand keypoints near the golf ball / ball lifted > 3 cm above its start."""
    from scipy.spatial.transform import Rotation as Rot
    keys = [k for k in f["objects"] if k.startswith(GOLF)]
    if not keys: return None, None
    p = f["objects/" + keys[0] + "/pose_world"][:].astype(float)
    c = p[:, :3] + Rot.from_quat(p[:, 3:7]).apply(GOLF_OFF)
    lifted = (c[:, 2] - np.median(c[:5, 2])) > 0.03
    n = min(len(c), len(truth["fk_world"]))
    d = np.linalg.norm(truth["fk_world"][:n] - c[:n, None], axis=-1).min(1) - GOLF_RAD
    return d <= 0.03, lifted


def attempt_rows(job):
    import torch, h5py
    from tools.visualize_cosmos_joint_predictions import load_npz, query_data
    from cosmos_framework.model.generator.robot_fk_torch import estimate_rigid_transform
    pc, fk, J = G["pc"], G["fk"], G["J"]
    meta, attempt = job
    truth = load_npz(next((attempt / "metrics").glob("episode_*_ground_truth.npz")))
    f = h5py.File(glob.glob(str(attempt / "episodes/*/*.hdf5"))[0], "r")
    qpos, cmd = f["robot/qpos"][:].astype(np.float32), f["action/commanded"][:].astype(np.float32)
    assert [x.decode() if isinstance(x, bytes) else x for x in f["robot/joint_names"][()]] == G["names"]
    near_g, lift_g = golf_state(f, truth, len(qpos))
    lifted_run = bool(lift_g is not None and np.convolve(lift_g.astype(int), np.ones(5, int), "valid").max() >= 5) if lift_g is not None else False
    files = sorted(glob.glob(str(attempt / "predictions/*/query_*.npz")))
    out = []
    for qi, qf in enumerate(files):
        P = load_npz(qf); data = query_data(P, truth); valid = data["truth_valid"][1:]
        row = int(data["truth_rows"][0])
        if not data["truth_valid"][0] or row >= len(qpos): continue
        ex = P["executed_actions"][:, :J].astype(np.float32)
        step_ok = valid & (row + np.arange(32) < len(cmd))                      # commanded exists for step k-1
        cm = np.zeros_like(ex); n = max(0, min(32, len(cmd) - row)); cm[:n] = cmd[row:row + n]
        a_pr = torch.from_numpy(np.vstack([qpos[row][None, :J], ex]))[None]
        a_ex = torch.from_numpy(np.vstack([qpos[row][None, :J], cm]))[None]
        anchor = torch.from_numpy(P["fk_anchor_xyz"]).float()[None]
        with torch.no_grad():
            d_pr = pc.keypoints_displacement(a_pr, anchor, fk, 1.0)[0].numpy()
            d_ex = pc.keypoints_displacement(a_ex, anchor, fk, 1.0)[0].numpy()
            w0 = fk(a_pr[:, :1])[:, 0]; R, t = estimate_rigid_transform(w0, anchor)
            kab = float((torch.einsum("bij,bkj->bki", R, w0) + t - anchor).norm(dim=-1).mean() * 1000)
        fk_pr = P["fk_displacement"]; fk_gt = data["fk_truth_xyz"][1:] - data["fk_truth_xyz"][:1]
        fv = (valid[:, None] & np.isfinite(fk_gt).all(-1)); fv_ex = fv & step_ok[:, None]
        fk_gt = np.nan_to_num(fk_gt)
        r = dict(meta); r.update(query=qi, row=row, n_steps=int(valid.sum()), kabsch_mm=kab)
        r["c_abs"] = mm(fk_pr - d_pr, fv)
        r["c_rel"] = mm((d_pr - fk_pr) - (d_ex - fk_gt), fv_ex)
        r["fk_head_err"] = mm(fk_pr - fk_gt, fv)
        r["fkact_err"] = mm(d_pr - fk_gt, fv)                                   # FK(raw action) vs realised FK
        r["fkact_vs_exec"] = mm(d_pr - d_ex, fv_ex)                             # raw vs executed (smoothed) action, hand space
        r["cmd_vs_real"] = mm(d_ex - fk_gt, fv_ex)                              # sanity (b) offset reference
        ae = data["fk_error_m"][1:]; ok = valid[:, None] & np.isfinite(ae)       # pointfk_truth-compatible absolute error
        r["fkabs_sum"], r["fkabs_n"] = float(ae[ok].sum() * 1000), int(ok.sum())
        pf_pr = P["pointflow_displacement"]; pf_gt = data["pointflow_truth_xyz"][1:] - data["pointflow_truth_xyz"][:1]
        pv = valid[:, None] & np.isfinite(pf_gt).all(-1); pf_gt = np.nan_to_num(pf_gt)
        r["pf_ade"] = mm(pf_pr - pf_gt, pv)
        dist = np.linalg.norm(P["pointflow_anchor_xyz"][:, None] - P["fk_anchor_xyz"][None], axis=-1)
        near = np.flatnonzero(dist.min(1) <= G["radius"]); key = dist.argmin(1)[near]; r["n_near"] = len(near)
        if len(near):
            v = pv[:, near] & fv[:, key]
            rh = pf_pr[:, near] - fk_pr[:, key]; rg = pf_gt[:, near] - fk_gt[:, key]
            r["pffk_raw"], r["pffk_rel"], r["pffk_gt_resid"] = mm(rh, v), mm(rh - rg, v), mm(rg, v)
        else:
            r["pffk_raw"] = r["pffk_rel"] = r["pffk_gt_resid"] = np.nan
        cl = {h: float((np.abs(ex[:, idx] - ex[:1, idx]).max(0) / G["rng"][idx]).mean()) for h, idx in G["fingers"].items()}
        r["closure"] = max(cl.values()); r["phase"] = "grasp" if r["closure"] >= GRASP_THR else "other"
        r["smooth_gap_rad"] = float(np.abs(ex[step_ok] - cm[step_ok]).mean()) if step_ok.any() else np.nan
        if near_g is not None:
            w = slice(row, row + 33); nw = near_g[w].any(); pre = not lift_g[row]
            r["golf_near"], r["golf_prelift"] = int(nw), int(pre)
            r["golf_attempt"] = int(nw and pre); r["golf_lift_after"] = int(lift_g[row:row + 65].any())
        else:
            r["golf_near"] = r["golf_prelift"] = r["golf_attempt"] = r["golf_lift_after"] = -1
        r["golf_ep_lifted"] = int(lifted_run)
        out.append(r)
    for r in out: r["n_queries"] = len(out); r["from_end"] = len(out) - 1 - r["query"]
    return out


def jobs(models, runs):
    from tools.analyze_cosmos_pointfk_truth import final_attempt
    for m in models:
        for run in runs:
            base = f"{INFER}/{m}/runs/{run}"
            if not os.path.isdir(base): continue
            for ch in CHANNELS:
                for rj in sorted(glob.glob(f"{base}/{ch}/episode_*/result.json")):
                    from pathlib import Path
                    res = json.load(open(rj)); att = final_attempt(Path(rj).parent, res)
                    if att is None: continue
                    mod = m.split("_")[2]
                    yield dict(model=mod, ckpt=int(m.split("_")[-1]), guidance=GUID[run], channel=ch,
                               episode=int(Path(rj).parent.name.split("_")[1]), success=int(bool(res.get("success")))), att


def run(a):
    import multiprocessing as mp
    js = list(jobs(a.models, a.runs))[:a.max_attempts or None]; print(f"{len(js)} attempts", flush=True)
    os.makedirs(a.out, exist_ok=True); rows = []
    with mp.Pool(a.workers, initializer=init, initargs=(a.stats, a.urdf, a.fk_json)) as pool:
        for i, rs in enumerate(pool.imap_unordered(attempt_rows, js, chunksize=1)):
            rows += rs
            if i % 100 == 0: print(f"[{i}/{len(js)}] {len(rows)} queries", flush=True)
    rows.sort(key=lambda r: (r["model"], r["ckpt"], r["guidance"], r["channel"], r["episode"], r["query"]))
    with open(f"{a.out}/queries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows), "queries")


# ---------------------------------------------------------------- report
def load(path):
    rows = list(csv.DictReader(open(path)))
    num = lambda s: float(s) if s not in ("", "nan") else np.nan
    return [{k: (v if k in ("model", "channel", "phase") else num(v)) for k, v in r.items()} for r in rows]


def q(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.mean(x), np.median(x), np.percentile(x, 90), len(x)) if len(x) else (np.nan,) * 3 + (0,)


def cell(rows, k, d=1):
    a, b, c, n = q([r[k] for r in rows])
    return f"{a:.{d}f} / {b:.{d}f} / {c:.{d}f}" if n else "n/a"


def auc(score, label):  # P(score_pos > score_neg), ties 0.5
    s, y = np.asarray(score, float), np.asarray(label, bool); ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.all() or (~y).all(): return np.nan
    rk = np.empty(len(s)); o = np.argsort(s, kind="mergesort"); rk[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1: rk[m] = rk[m].mean()
    return (rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="mergesort")] = np.arange(len(x)); return r


def corr_block(rows, a, b):
    x, y = np.array([r[a] for r in rows]), np.array([r[b] for r in rows]); ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]; eps = np.array([(r["model"], r["ckpt"], r["guidance"], r["channel"], r["episode"]) for r, o in zip(rows, ok) if o], dtype=object)
    ids = {}; g = np.array([ids.setdefault(tuple(e), len(ids)) for e in eps])
    xd, yd = x.copy(), y.copy(); per = []
    for i in range(len(ids)):
        m = g == i; xd[m] -= x[m].mean(); yd[m] -= y[m].mean()
        if m.sum() >= 8: per.append(pearson(x[m], y[m]))
    per = np.array(per)
    return (f"| {a} vs {b} | {len(x)} | {pearson(x, y):.2f} | {pearson(rank(x), rank(y)):.2f} | {pearson(xd, yd):.2f} | "
            f"{np.nanmean(per):.2f} (n_ep={np.isfinite(per).sum()}) |")


METRICS = [("c_rel", "FK head vs FK(raw act), RELATIVE"), ("c_abs", "FK head vs FK(raw act), absolute"), ("fk_head_err", "FK head vs FK GT"),
           ("fkact_err", "FK(raw act) vs FK GT (realised)"), ("pffk_rel", "PF-FK near hand, rel: ‖r_hat - r_gt‖"), ("pffk_raw", "PF-FK near hand, raw: ‖r_hat‖"),
           ("pffk_gt_resid", "  (ref) ‖r_gt‖"), ("pf_ade", "PF ADE"), ("cmd_vs_real", "  (ref) FK(executed cmd) vs FK GT"),
           ("fkact_vs_exec", "  (ref) FK(raw act) vs FK(executed smoothed cmd)")]
OFFLINE = {"c_rel": 14.2, "c_abs": np.nan, "fk_head_err": 26.2, "fkact_err": np.nan, "pffk_rel": 8.2, "pffk_raw": np.nan, "pffk_gt_resid": np.nan, "pf_ade": 12.3, "cmd_vs_real": 17.5}
OFFLINE_P = {"c_rel": (15.3, 13.1), "fk_head_err": (29.1, 23.5), "pffk_rel": (8.8, 7.6), "pf_ade": (13.2, 11.4)}


def cmp_table(groups, keys, title, d=1):
    out = [f"**{title}**  (mean / median / p90 mm)\n", "| metric | " + " | ".join(f"{n} (n={len(r)})" for n, r in groups) + " |", "|---|" + "---|" * len(groups)]
    for k in keys:
        out.append(f"| {dict(METRICS)[k]} | " + " | ".join(cell(r, k, d) for _, r in groups) + " |")
    return "\n".join(out)


def report(a):
    rows = load(f"{a.out}/queries.csv"); keys = [k for k, _ in METRICS]
    md = ["# Closed-loop cross-modal consistency (2026-10-14)\n",
          f"Queries: {len(rows)} from {len({(r['model'], r['ckpt'], r['guidance'], r['channel'], r['episode']) for r in rows})} rollouts. Grasp phase = raw-action finger closure >= {GRASP_THR:g} (32 raw rows, a_t - a_1). "
          "All errors are camera-frame displacement (K_k - K_0), mm, mean over valid steps/points of each query, then mean / median / p90 over queries. "
          "a_0 = robot/qpos at the query row; GT = recorded simulator body poses; relative form uses the executed commanded action as the 'GT action'.\n"]
    # sanity
    kab = np.array([r["kabsch_mm"] for r in rows]); cvr = [r["cmd_vs_real"] for r in rows]
    md.append("## Sanity\n")
    md.append(f"- (a) Kabsch residual of FK(a_0=qpos[row]) vs fk_anchor_xyz: mean {np.nanmean(kab):.4f} mm, max {np.nanmax(kab):.4f} mm over all queries.")
    md.append(f"- (b) FK(executed commanded) vs FK(realised truth) displacement: mean / median / p90 = {cell(rows, 'cmd_vs_real')} (offline reference 17.5 mm).")
    sc = []
    for (m, c, g), rs in sorted(groupby(rows, ("model", "ckpt", "guidance")).items()):
        s, n = sum(r["fkabs_sum"] for r in rs), sum(r["fkabs_n"] for r in rs)
        j = f"{a.truth_json}/task27_cosmos_{m}_s4_{int(c)}.json"
        if g == 3 and os.path.exists(j):
            ref = json.load(open(j))["groups"]["all"]["fk"]["ade_mm"]
            sc.append(f"{m} {int(c)}: mine {s / n:.3f} vs pointfk_truth {ref:.3f}")
    md.append("- (c) FK head absolute ADE vs analysis/pointfk_truth (guidance 3, pooled over rollouts): " + "; ".join(sc) + ".\n")
    P = lambda rs, p: [r for r in rs if r["phase"] == p]
    A = [r for r in rows if r["model"] == "nolocal" and r["ckpt"] == 40000 and r["guidance"] == 3]
    md.append("## 1. Closed loop vs offline\n")
    md.append(cmp_table([("all", A), ("grasp", P(A, "grasp")), ("other", P(A, "other"))], keys, "nolocal 40000, guidance 3 (= offline model A config)"))
    md.append("\nOffline (A, 153 windows, mean): c_rel 14.2 (grasp 15.3 / other 13.1), FK head err 26.2 (29.1 / 23.5), PF-FK 8.2 (8.8 / 7.6), PF ADE 12.3 (13.2 / 11.4), cmd-vs-real 17.5.\n")
    md.append(cmp_table([("all", rows), ("grasp", P(rows, "grasp")), ("other", P(rows, "other"))], keys, "all models/guidance pooled"))
    ok = lambda rs, v: [r for r in rs if r["success"] == v]
    md.append("\n## 2. Success vs failure episodes (all pooled)\n")
    md.append(cmp_table([("success ep", ok(rows, 1)), ("failure ep", ok(rows, 0)), ("fail, last 5 q", [r for r in ok(rows, 0) if r["from_end"] < LAST_N]),
                         ("fail, earlier q", [r for r in ok(rows, 0) if r["from_end"] >= LAST_N])], keys[:6], "by episode outcome"))
    for p in ("grasp", "other"):
        md.append("\n" + cmp_table([("success", ok(P(rows, p), 1)), ("failure", ok(P(rows, p), 0)), ("fail last5", [r for r in ok(P(rows, p), 0) if r["from_end"] < LAST_N])], keys[:6], f"phase = {p}"))
    md.append("\n## 3. Golf grasp-attempt queries (hand within 3 cm of the golf ball in the 33-row window, ball not yet lifted)\n")
    gq = [r for r in rows if r["golf_attempt"] == 1]
    md.append(cmp_table([("all queries", rows), ("golf attempt", gq), ("  lift within 64 rows", [r for r in gq if r["golf_lift_after"] == 1]),
                         ("  no lift", [r for r in gq if r["golf_lift_after"] == 0]), ("attempt & grasp-phase", P(gq, "grasp")),
                         ("  grasp, lift", [r for r in P(gq, "grasp") if r["golf_lift_after"] == 1]), ("  grasp, no lift", [r for r in P(gq, "grasp") if r["golf_lift_after"] == 0])],
                        keys[:6], "golf attempt"))
    md.append("\nepisode-level, golf attempt queries: " + cmp_table([("ep lifted golf", [r for r in gq if r["golf_ep_lifted"] == 1]), ("ep never lifted", [r for r in gq if r["golf_ep_lifted"] == 0])], keys[:6], "by whether the golf ball was ever lifted"))
    md.append("\n## 4. Guidance 3 vs guidance 1 (same model and checkpoint)\n")
    md.append("| model | ckpt | n_q g3 / g1 | c_rel g3 | c_rel g1 | c_abs g3 | c_abs g1 | pffk_raw g3 | pffk_raw g1 | fk_head_err g3 | g1 | fkact_err g3 | g1 | succ-ep g3 | g1 |\n|" + "---|" * 15)
    for m in ("nolocal", "local"):
        for c in (30000, 35000, 40000):
            r3 = [r for r in rows if r["model"] == m and r["ckpt"] == c and r["guidance"] == 3]; r1 = [r for r in rows if r["model"] == m and r["ckpt"] == c and r["guidance"] == 1]
            f = lambda rs, k: f"{q([r[k] for r in rs])[0]:.1f}" if rs else "-"
            sr = lambda rs: (f"{np.mean([r['success'] for r in rs if r['query'] == 0]):.2f}" if rs else "-")
            md.append(f"| {m} | {c} | {len(r3)} / {len(r1)} | " + " | ".join(f"{f(r3, k)} | {f(r1, k)}" for k in ("c_rel", "c_abs", "pffk_raw", "fk_head_err", "fkact_err")) + f" | {sr(r3)} | {sr(r1)} |")
    md.append("\n" + cmp_table([("g3", [r for r in rows if r["guidance"] == 3]), ("g1", [r for r in rows if r["guidance"] == 1])], keys, "pooled"))
    md.append("\n## 5. nolocal vs local (pooled over ckpts, both guidance)\n")
    md.append(cmp_table([("nolocal", [r for r in rows if r["model"] == "nolocal"]), ("local", [r for r in rows if r["model"] == "local"])], keys, "by model"))
    md.append("\n## 6. Does inconsistency predict failure?\n")
    ep = {}
    for r in rows: ep.setdefault((r["model"], r["ckpt"], r["guidance"], r["channel"], r["episode"]), []).append(r)
    E = [dict(key=k, fail=1 - rs[0]["success"], **{f"{m}_{s}": f(np.array([r[m] for r in rs], float)) for m in ("c_rel", "c_abs", "pffk_raw", "fk_head_err", "fkact_err", "pf_ade", "closure")
          for s, f in (("mean", np.nanmean), ("max", np.nanmax))}) for k, rs in ep.items()]
    md.append(f"Episode-level AUC for FAILURE ({int(sum(e['fail'] for e in E))} failures / {len(E)} episodes). Pooled = all rollouts; per-run = mean of the AUC inside each (model, ckpt, guidance) group (removes model/guidance confounding).\n")
    md.append("| episode score | pooled AUC | per-run mean AUC (min..max) |\n|---|---|---|")
    for m in ("c_rel", "c_abs", "pffk_raw", "fk_head_err", "fkact_err", "pf_ade", "closure"):
        for s in ("mean", "max"):
            k = f"{m}_{s}"; pooled = auc([e[k] for e in E], [e["fail"] for e in E]); per = []
            for gk in {e["key"][:3] for e in E}:
                sub = [e for e in E if e["key"][:3] == gk]; v = auc([e[k] for e in sub], [e["fail"] for e in sub])
                if np.isfinite(v): per.append(v)
            md.append(f"| {k} | {pooled:.3f} | {np.mean(per):.3f} ({np.min(per):.2f}..{np.max(per):.2f}) |")
    md.append("\nQuery-level AUC of c_rel (and c_abs) for 'query is in the last 5 of a failed episode':")
    lastf = [int(r["success"] == 0 and r["from_end"] < LAST_N) for r in rows]
    md.append(f"- c_rel {auc([r['c_rel'] for r in rows], lastf):.3f}, c_abs {auc([r['c_abs'] for r in rows], lastf):.3f}, fk_head_err {auc([r['fk_head_err'] for r in rows], lastf):.3f}, closure {auc([r['closure'] for r in rows], lastf):.3f}.")
    md.append("\nQuery-level correlation (within the closed loop). 'within-ep' = pooled after removing each rollout's mean; 'mean per-ep r' = average per-rollout Pearson (>= 8 queries).\n")
    md.append("| pair | n | Pearson | Spearman | within-ep (demeaned) | mean per-ep r |\n|---|---|---|---|---|---|")
    for x, y in (("c_rel", "fkact_err"), ("c_abs", "fkact_err"), ("c_rel", "fk_head_err"), ("c_abs", "fk_head_err"), ("pffk_rel", "pf_ade"), ("pffk_raw", "pf_ade"), ("c_rel", "pffk_raw")):
        md.append(corr_block(rows, x, y))
    md.append("\nSame, grasp-phase queries only:\n\n| pair | n | Pearson | Spearman | within-ep | mean per-ep r |\n|---|---|---|---|---|---|")
    for x, y in (("c_rel", "fkact_err"), ("c_abs", "fkact_err")): md.append(corr_block(P(rows, "grasp"), x, y))
    open(f"{a.out}/summary.md", "w").write("\n".join(md)); print("\n".join(md))


def groupby(rows, ks):
    d = {}
    for r in rows: d.setdefault(tuple(r[k] for k in ks), []).append(r)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"]); ap.add_argument("out")
    ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--max-attempts", type=int, default=0, help="debug: only the first N attempts")
    ap.add_argument("--assets", default=None, help="dir with normalization_train_only.json, the URDF and fk21.json (default: A800 paths)")
    ap.add_argument("--truth-json", default="/oss-shichaojian/outputs/infer/analysis/pointfk_truth")
    a = ap.parse_args()
    if a.cmd == "run":
        from cosmos_framework.model.generator import pointfk_consistency as pc
        B = "/data/shichaojian/sim_data/bench2dex_task27_ball_training"
        a.stats = f"{a.assets}/normalization_train_only.json" if a.assets else B + "/normalization_train_only.json"
        a.urdf = f"{a.assets}/Multi_UR5_wuji_with_flange.urdf" if a.assets else pc.DEFAULT_URDF
        a.fk_json = f"{a.assets}/fk21.json" if a.assets else pc.DEFAULT_FK_JSON
        run(a)
    else:
        report(a)
