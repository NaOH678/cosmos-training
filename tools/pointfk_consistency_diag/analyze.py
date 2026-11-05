"""Cross-modal consistency metrics from saved joint samples.  usage: analyze.py PREDS_ROOT OUT_DIR [--train-dir-A DIR step]  (CPU only)
PREDS_ROOT/<A|B|C>/*.npz (action[33,52] physical rad, fk[32,42,3] m, pointflow[32,1024,3] m).  GT comes from the same .pt window files the dataset reads."""
import sys, json, glob, os, csv
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repo root: consistency module + URDF FK
from cosmos_framework.model.generator import pointfk_consistency as pc
from cosmos_framework.model.generator.robot_fk_torch import TorchRobotFK

B = "/data/shichaojian/sim_data/bench2dex_task27_ball_training"
SEL = B + "/datasets/cosmos-cache/pointflow_fk_handguided1024"
STATS = B + "/normalization_train_only.json"
RUNS = "/data/shichaojian/runs"
TRAIN = {  # training-time eval dirs for the sanity check
    "A": (RUNS + "/sim_task27_ptfk_nonlocal_8gpu_40k_20261010/cosmos3_action/action_sft/sim_task27_ptfk_nonlocal", "step_0040000"),
    "B": (RUNS + "/sim_task27_ptfk_local_8gpu_40k_20261010/cosmos3_action/action_sft/sim_task27_ptfk_local", "step_0040000"),
    "C": (RUNS + "/sim_task27_pointfk_nonlocal_20k/cosmos3_action/action_sft/action_policy_sim_27_ball_nonlocal", "step_0020000"),
}
NAMES = {"A": "A task27 nonlocal 40k", "B": "B task27 local 40k", "C": "C task27 nonlocal 20k"}
GRASP_THR = float(os.environ.get("GRASP_THR", "0.25"))
names, offset, scale = pc.load_action_stats(STATS)
J = len(names)
fk = TorchRobotFK.from_json(pc.DEFAULT_URDF, pc.DEFAULT_FK_JSON, names)
qlo = np.array(json.load(open(STATS))["action"]["q01"]); qhi = np.array(json.load(open(STATS))["action"]["q99"]); rng = np.maximum(qhi - qlo, 1e-3)
fingers = {h: [i for i, n in enumerate(names) if f"{h}_finger" in n] for h in ("left", "right")}
rows = {(w["episode"], w["start_frame"]): w["path"] for w in json.load(open(SEL + "/manifest.json"))["windows"] if w["split"] == "validation"}
_ep = {}
def xm_gt(ep, s):  # same GT the earlier xmodal analysis used for action MAE
    if ep not in _ep:
        _ep[ep] = np.load(f"/data/shichaojian/sim_data/bench2dex_task27_ball/{ep}/state_action.npz")["action"]
    return _ep[ep][s:s + 32]

T = lambda x: torch.as_tensor(np.asarray(x)).float()

HORIZ = (1, 8, 16, 32)
HNAMES = (("fk_head_err", "FK head vs FK GT (realised)"), ("fkact_vs_fk_gt", "FK(pred act) vs FK GT (realised)"), ("fkact_vs_fkact_gt", "FK(pred act) vs FK(GT act) (commanded)"), ("fk_head_vs_fkact_gt", "FK head vs FK(GT act) (commanded)"))

def q(x):  # mean / median / p90
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.mean(x), np.median(x), np.percentile(x, 90)) if len(x) else (np.nan,) * 3

def window_metrics(P, ep, s):
    pt = torch.load(f"{SEL}/{rows[(ep, s)]}", weights_only=False, map_location="cpu")
    a_gt = T(pt["action"])[:, :J]; a_pr = torch.from_numpy(P["action"]).float()[:, :J]
    anchor = T(pt["fk"]["inputs"]["anchor_xyz"])[None]
    fk_gt = T(pt["fk"]["targets"]["displacement"]).numpy(); fk_pr = P["fk"].astype(np.float32)
    fvalid = np.asarray(pt["fk"]["targets"]["valid"]).astype(bool)
    with torch.no_grad():
        d_pr = pc.keypoints_displacement(a_pr[None], anchor, fk, 1.0)[0].numpy()
        d_gt = pc.keypoints_displacement(a_gt[None], anchor, fk, 1.0)[0].numpy()
    mm = lambda v, m: float(np.linalg.norm(v, axis=-1)[m].mean() * 1000) if m.any() else np.nan
    r = dict(episode=ep, start_frame=s)
    r["fk_head_vs_fkact_abs"] = mm(fk_pr - d_pr, fvalid)
    r["fk_head_vs_fkact_rel"] = mm((d_pr - fk_pr) - (d_gt - fk_gt), fvalid)
    r["fkact_vs_fkact_gt"] = mm(d_pr - d_gt, fvalid)
    r["fk_head_err"] = mm(fk_pr - fk_gt, fvalid)
    r["fkact_gt_vs_fk_gt"] = mm(d_gt - fk_gt, fvalid)          # commanded-vs-realised offset reference
    # same-reference pairs (same alignment as above: displacement row k of FK(action) vs FK frame k, no extra shift)
    r["fkact_vs_fk_gt"] = mm(d_pr - fk_gt, fvalid)             # ref = expert realised (same as fk_head_err)
    r["fk_head_vs_fkact_gt"] = mm(fk_pr - d_gt, fvalid)        # ref = expert commanded (same as fkact_vs_fkact_gt)
    for nm, v in (("fk_head_err", fk_pr - fk_gt), ("fkact_vs_fk_gt", d_pr - fk_gt), ("fkact_vs_fkact_gt", d_pr - d_gt), ("fk_head_vs_fkact_gt", fk_pr - d_gt)):
        e = np.linalg.norm(v, axis=-1) * 1000
        for h in HORIZ:
            m = fvalid[h - 1]; r[f"h{h}_{nm}"] = float(e[h - 1][m].mean()) if m.any() else np.nan
    pvalid = np.asarray(pt["pointflow"]["targets"]["valid"]).astype(bool)
    pf_gt = T(pt["pointflow"]["targets"]["displacement"]).numpy(); pf_pr = P["pointflow"]
    r["pf_ade"] = mm(pf_pr - pf_gt, pvalid)
    # PF vs FK near the hand, as compute_pf_fk_loss
    apf = T(pt["pointflow"]["inputs"]["anchor_xyz"]); afk = anchor[0]
    dist = torch.cdist(apf, afk); near_d, partner = dist.min(-1)
    near = (near_d <= float(os.environ.get("POINTFK_PF_FK_RADIUS", "0.05"))).nonzero(as_tuple=True)[0].numpy(); key = partner[near].numpy()
    r["n_near"] = len(near)
    if len(near):
        v = pvalid[:, near] & fvalid[:, key]
        rh = pf_pr[:, near] - fk_pr[:, key]; rg = pf_gt[:, near] - fk_gt[:, key]
        r["pffk_rel"] = mm(rh - rg, v); r["pffk_raw"] = mm(rh, v); r["pffk_gt_resid"] = mm(rg, v)
    else:
        r["pffk_rel"] = r["pffk_raw"] = r["pffk_gt_resid"] = np.nan
    ag, ap = a_gt.numpy(), a_pr.numpy()
    r["act_mae_rad"] = float(np.abs(ap[1:] - ag[1:]).mean())
    try: r["act_mae_xm"] = float(np.abs(ap[1:] - xm_gt(ep, s)).mean())
    except Exception: r["act_mae_xm"] = np.nan
    cl = {h: float((np.abs(ag[:, idx] - ag[:1, idx]).max(0) / rng[idx]).mean()) for h, idx in fingers.items()}
    r["closure_left"], r["closure_right"] = cl["left"], cl["right"]
    r["closure"] = max(cl.values())
    r["phase"] = "grasp" if r["closure"] >= GRASP_THR else "other"
    return r

def run_model(key, preds_root, out):
    recs = []
    for f in sorted(glob.glob(f"{preds_root}/{key}/*.npz")):
        P = np.load(f); m = json.loads(str(P["meta"]))
        r = window_metrics(P, m["episode"], m["start_frame"]); r["orig24"] = int(m["orig"]); r["file"] = os.path.basename(f); recs.append(r)
    os.makedirs(out, exist_ok=True)
    if recs:
        with open(f"{out}/windows_{key}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    return recs

METRICS = [("fk_head_vs_fkact_rel", "FK head vs FK(pred act), RELATIVE (offset-cancelled)"), ("fk_head_vs_fkact_abs", "FK head vs FK(pred act), absolute"),
           ("fkact_vs_fkact_gt", "FK(pred act) vs FK(GT act)"), ("fk_head_err", "FK head vs FK GT (FK ADE)"), ("pf_ade", "PF ADE (all valid points)"),
           ("pffk_rel", "PF-FK near hand |r_hat - r_gt|"), ("pffk_raw", "PF-FK near hand raw |r_hat| = |pf_pred - fk_pred|"), ("pffk_gt_resid", "  (ref) |r_gt| GT PF-FK offset near hand"),
           ("fkact_gt_vs_fk_gt", "  (ref) FK(GT act) vs FK GT (cmd-vs-realised)"), ("act_mae_rad", "action MAE rad (vs window GT rows 1:)"),
           ("fkact_vs_fk_gt", "[same-ref, realised] FK(pred act) vs FK GT"), ("fk_head_vs_fkact_gt", "[same-ref, commanded] FK head vs FK(GT act)")]

def table(recs):
    out = ["| metric (mm unless noted) | all mean/med/p90 | grasp mean/med/p90 | other mean/med/p90 |", "|---|---|---|---|"]
    ph = {"all": recs, "grasp": [r for r in recs if r["phase"] == "grasp"], "other": [r for r in recs if r["phase"] == "other"]}
    out.insert(0, f"n windows: all {len(ph['all'])}, grasp {len(ph['grasp'])}, other {len(ph['other'])}\n")
    for k, lab in METRICS:
        cells = []
        for p in ("all", "grasp", "other"):
            a, b, c = q([r[k] for r in ph[p]]); d = 4 if k == "act_mae_rad" else 2; cells.append(f"{a:.{d}f} / {b:.{d}f} / {c:.{d}f}" if np.isfinite(a) else "n/a")
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    out += ["", "Horizon profile: median over windows (all phases) of per-frame error, mm", "", "| error | " + " | ".join(f"frame {h}" for h in HORIZ) + " |", "|---|" + "---|" * len(HORIZ)]
    for k, lab in HNAMES:
        out.append(f"| {lab} | " + " | ".join(f"{np.nanmedian([r[f'h{h}_{k}'] for r in recs]):.2f}" for h in HORIZ) + " |")
    return "\n".join(out)

def sanity(key, recs):
    d, step = TRAIN[key]; lines = []
    mine = {(r["episode"], r["start_frame"]): r for r in recs if r["orig24"]}
    pa, fa, pm, fm, rms = [], [], [], [], []
    for jd in sorted(glob.glob(f"{d}/pointflow_eval/{step}/val_*_joint")):
        js = json.load(open(jd + "/joint_sample.json")); k = (js["episode"], js["start_frame"])
        if k not in mine: continue
        pa.append(json.load(open(jd + "/metrics.json"))["all_ade_mm"]); pm.append(mine[k]["pf_ade"])
        case = js["case_id"].replace("_joint", "")
        fa.append(json.load(open(f"{d}/fk_eval/{step}/{case}/metrics.json"))["all_ade_mm"]); fm.append(mine[k]["fk_head_err"])
        tr = np.load(jd + "/joint_prediction.npz")["action"]
        mp = np.load(glob.glob(f"{PREDS}/{key}/{js['episode']}_{js['start_frame']:05d}.npz")[0])["action"]
        rms.append(float(np.sqrt(((tr - mp) ** 2).mean())))
    lines.append(f"{NAMES[key]}: matched {len(pa)}/24 orig windows | PF ADE offline {np.mean(pm):.2f} vs training-time {np.mean(pa):.2f} | FK ADE offline {np.mean(fm):.2f} vs training-time {np.mean(fa):.2f} | "
                 f"action RMS offline-vs-training-time {np.mean(rms):.4f} (max {np.max(rms):.4f}) | max per-window |dPF| {np.max(np.abs(np.array(pm)-np.array(pa))):.3f} |dFK| {np.max(np.abs(np.array(fm)-np.array(fa))):.3f} mm")
    return lines[0]

if __name__ == "__main__":
    PREDS, OUT = sys.argv[1], sys.argv[2]
    md = [f"# Cross-modal consistency diagnostic\n\nGrasp/contact phase = GT hand-finger closure >= {GRASP_THR}: max over hands of mean over that hand's finger joints of (max_t |a_t - a_0| / (q99 - q01)) over the 33-row window (train-only stats).  Values mm in camera frame, mean / median / p90 across windows.\n"]
    san = []
    for key in "ABC":
        recs = run_model(key, PREDS, OUT)
        if not recs: continue
        try: san.append(sanity(key, recs))
        except Exception as e: san.append(f"{key}: sanity failed to run: {e!r}")
        md.append(f"\n## {NAMES[key]}\n\n" + table(recs))
    md.insert(1, "\n## Sanity (offline original-24 vs training-time eval)\n\n" + "\n".join("- " + s for s in san) + "\n")
    open(f"{OUT}/summary.md", "w").write("\n".join(md)); print("\n".join(md))
