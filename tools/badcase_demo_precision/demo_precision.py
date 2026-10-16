#!/usr/bin/env python
"""Expert-demo precision metrics for task27 (ball_box_loading) and task21 (condiment_box_loading).

CPU only, read-only on the demo data.  Metric definitions are the SAME as the rollout-side scripts
(4090 node, /oss-shichaojian/outputs/infer/analysis/):
  golf_env/grasp_geom.py        first-grasp fingertip/ball geometry   -> grasp()
  badcase_30k40k/obj_extract.py stage condition, hand choice, xy_off -> cond / grasp(auto hand)
  badcase_30k40k/release.py     release height / tilt / velocity      -> release block
  badcase_30k40k/topple.py      tilt / base height at first placement -> place block
  badcase_30k40k/crowd.py       nearest-object xy distance            -> near_* fields
Only paths / IO differ (demo hdf5 instead of rollout attempt dirs; FK from fk21.npz instead of
fk_world in the ground-truth npz).

Usage:
  demo_precision.py extract --task 27|21 --out DIR      # writes DIR/task{T}_episodes.jsonl
  demo_precision.py summarize --out DIR                 # writes DIR/summary.md
"""
import argparse, glob, json, os, sys
import numpy as np, h5py

# --- constants copied from the rollout scripts (golf_extract2.py / obj_extract.py) -------------
BALLS = {'golf': ('obj_058_golf_ball_4', [-0.0065395, -0.0304375, 0.0209435]),
         'tennis': ('obj_056_tennis_ball_3', [0.0082115, -0.044278, 0.0331315]),
         'soccer': ('obj_053_mini_soccer_ball_2', [-0.0074058, 0.0132354, 0.0747246]),
         'pingpong': ('obj_144_table_tennis_5', [0.0, 0.0466254, 0.0])}
BOX27 = 'obj_212_basket_1'
T21 = dict(box='obj_154_wooden_box_1', tol=0.18, objs={
    'soy': ('obj_172_soy_sauce_2', [0.0, 0.075, 0.0], True),
    'vinegar': ('obj_173_vinegar_3', [-8.6e-06, 0.1125, -2.6e-06], True),
    'msg': ('obj_171_msg_4', [-2.0e-06, 0.0475284, 2.2e-06], False)})
# contact threshold (keypoint-to-centre) per ball.  golf/tennis: 4 cm as in the rollout scripts;
# soccer/pingpong are not analysed by the rollout scripts, radius-adjusted ad hoc thresholds (extras only).
NEAR = {'golf': 0.04, 'tennis': 0.04, 'soccer': 0.10, 'pingpong': 0.06}
COS30 = np.cos(np.radians(30))
FPS = 20
TIPS = np.array([4, 8, 12, 16, 20])          # *_tip_link within a hand's 21 keypoints
# fk21.npz is stored in the (fixed) cam_overhead optical frame (OpenCV x right, y down, z forward);
# cam_overhead/extrinsic_world_from_cam is world<-cam in the ROS convention (x fwd, y left, z up).
M_CV2ROS = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]])
T27_ROOT = '/data/shichaojian/sim_data/bench2dex_task27_ball'
T21_RAW = '/data/shichaojian/raw_data/bench2dex/task21/replay-generalization'
T21_FK = '/data/shichaojian/sim_data/bench2dex_task21_v3'


def rot(q, v):  # q [...,4] xyzw
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    v = np.asarray(v, float)
    t = 2 * np.stack([y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]], -1)
    return v + w[..., None] * t + np.cross(np.stack([x, y, z], -1), t)


def centre(pose, off):
    return pose[:, :3] + rot(pose[:, 3:7], off)


def segments(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if not m and start is not None: out.append((start, i - 1)); start = None
    if start is not None: out.append((start, len(mask) - 1))
    return out


def world_fk(h5, fk_npz):
    """[T,2,21,3] camera-frame -> world [T,42,3] (left 0..20, right 21..41)."""
    P = np.load(fk_npz)['positions_camera'].astype(float).reshape(-1, 42, 3)
    E = h5['cameras/cam_overhead/extrinsic_world_from_cam'][:].astype(float)
    assert np.abs(E - E[0]).max() < 1e-6, 'overhead camera not static'
    return P @ (E[0, :3, :3] @ M_CV2ROS).T + E[0, :3, 3], float(E[0, 2, 3])


def meta_dz(h5):
    s = json.loads(h5['meta/scene_generalization_sample'][()])
    return s['spatial']['table_height']['height_offset_m']


# ------------------------------------------------------------------------------------------------
def grasp(c, W, side, table, near_thr=0.04):
    """grasp_geom.py / obj_extract.py definition. side 0=left 1=right. c: ball centre [T,3]."""
    sl = slice(21 * side, 21 * side + 21)
    H = W[:, sl]; n = min(len(H), len(c))
    d = np.linalg.norm(H[:n] - c[:n, None], axis=2).min(1)
    near = np.flatnonzero(d < near_thr)
    if not len(near): return None
    t0 = int(near[0]); w = np.arange(t0, min(t0 + 30, n))     # first 1.5 s of contact
    tc = H[w][:, TIPS].mean(1)                                   # fingertip centroid
    k = int(w[np.argmin(np.linalg.norm(tc[:, :2] - c[w, :2], axis=1))])
    tcen = H[k, TIPS].mean(0)
    return dict(t_first_contact=t0 / FPS, k=k / FPS, xy_off=float(np.linalg.norm(tcen[:2] - c[k, :2])),
                tip_min_h=float(H[w][:, TIPS, 2].min() - table), palm_h=float(H[k, 0, 2] - table),
                tips_below_centre=int((H[k, TIPS, 2] < c[k, 2]).sum()),
                tip_centroid_h=float(tcen[2] - table), min_dist=float(d.min()))


def contact_events(c, W, h, near_thr=0.04, merge_gap=FPS, min_len=10, lift_thr=0.03):
    """Regrasp bookkeeping.  Contact = any keypoint of either hand within near_thr of ball centre.
    Segments separated by < merge_gap frames are merged.  A 'failed attempt' is a contact segment that
    starts and ends before the ball was ever lifted (>3 cm); 'regrasp' = a later contact segment starts
    while the ball has still never been lifted."""
    n = min(len(W), len(c))
    d = np.linalg.norm(W[:n] - c[:n, None], axis=2).min(1)
    segs = [s for s in segments(d < near_thr) if s[1] - s[0] + 1 >= min_len]
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < merge_gap: merged[-1] = (merged[-1][0], s[1])
        else: merged.append(s)
    lifted = np.flatnonzero(h[:n] > lift_thr)
    first_lift = int(lifted[0]) if len(lifted) else n
    pre = [s for s in merged if s[0] < first_lift]
    # sensitivity: any contact (>=3 frames) before first lift, i.e. also brushes shorter than 0.5 s
    mb = []
    for s_ in [s_ for s_ in segments(d < near_thr) if s_[1] - s_[0] + 1 >= 3]:
        if mb and s_[0] - mb[-1][1] < merge_gap: mb[-1] = (mb[-1][0], s_[1])
        else: mb.append(s_)
    brief = [s_ for s_ in mb if s_[0] < first_lift]
    return dict(n_contact_segments=len(merged), n_contact_before_lift=len(pre), regrasp=len(pre) >= 2,
                n_brief_contacts_before_lift=len(brief),
                contact_segments_s=[(a / FPS, b / FPS) for a, b in merged[:8]], first_lift_s=None if first_lift == n else first_lift / FPS)


def extract27(ep):
    h5 = h5py.File(f'{T27_ROOT}/{ep}/rgbd.hdf5', 'r')
    W, cam_z = world_fk(h5, f'{T27_ROOT}/{ep}/fk21.npz')
    P = {k: h5[f'objects/{o}/pose_world'][:].astype(float) for k, (o, _) in BALLS.items()}
    C = {k: centre(P[k], off) for k, (_, off) in BALLS.items()}
    table = C['golf'][0, 2] - 0.0214          # same table proxy as grasp_geom.py (golf radius 21.4 mm)
    out = dict(task=27, ep=ep, T=len(W), dz=meta_dz(h5), table_z=float(table), cam_z=cam_z,
               success=bool(h5['meta/success'][()]), tip_min_z_minus_table=float(W[:, [4, 8, 12, 16, 20, 25, 29, 33, 37, 41], 2].min() - table))
    for k in BALLS:
        c = C[k]; h = c[:, 2] - c[0, 2]
        n = min(len(W), len(c))
        dR = np.linalg.norm(W[:n, 21:] - c[:n, None], axis=2).min(1); dL = np.linalg.norm(W[:n, :21] - c[:n, None], axis=2).min(1)
        near = np.flatnonzero(np.minimum(dR, dL) < NEAR[k])
        r = dict(max_h=float(h.max()), final_h=float(h[-1]), touched=bool(len(near)))
        if len(near):
            t0 = near[0]; hand = 1 if dR[t0] <= dL[t0] else 0                      # obj_extract.py auto hand
            r['hand'] = 'R' if hand else 'L'
            r['auto'] = grasp(c, W, hand, table, NEAR[k])
            if k == 'golf': r['right'] = grasp(c, W, 1, table)                     # grasp_geom.py: right hand only
        r.update(contact_events(c, W, h, near_thr=NEAR[k]))
        out[k] = r
    return out


def extract21(ep):
    h5 = h5py.File(f'{T21_RAW}/{ep}.hdf5', 'r')
    W, cam_z = world_fk(h5, f'{T21_FK}/{ep}/fk21.npz')
    cfg = T21
    P = {k: h5[f'objects/{o}/pose_world'][:].astype(float) for k, (o, _, _) in cfg['objs'].items()}
    bp = h5[f"objects/{cfg['box']}/pose_world"][:].astype(float)
    C = {k: centre(P[k], off) for k, (_, off, _) in cfg['objs'].items()}
    T = len(bp); bc = centre(bp, [0, 0.075, 0])
    box_ok = rot(bp[:, 3:7], [0, 1, 0])[:, 2] > COS30
    qinv = np.c_[-bp[:, 3:6], bp[:, 6:7]]
    out = dict(task=21, ep=ep, T=T, dz=meta_dz(h5), cam_z=cam_z, success=bool(h5['meta/success'][()]),
               box_tilt_max=float(np.degrees(np.arccos(np.clip(rot(bp[:, 3:7], [0, 1, 0])[:, 2].min(), -1, 1)))))
    for k, (_, off, up_req) in cfg['objs'].items():
        c = C[k]; dxy = np.linalg.norm(c[:, :2] - bc[:, :2], axis=1)
        inside = (dxy < cfg['tol']) & (c[:, 2] > bc[:, 2] - 0.05)
        upv = rot(P[k][:, 3:7], [0, 1, 0])[:, 2]
        up = upv > COS30 if up_req else np.ones(T, bool)
        cond = inside & up & box_ok
        tilt = np.degrees(np.arccos(np.clip(upv, -1, 1)))
        d = dict(ever_cond=bool(cond.any()), final_cond=bool(cond[-1]), final_tilt=float(tilt[-1]), max_tilt_after_cond=None)
        if cond.any():
            t0 = int(np.argmax(cond)); d['first_cond_s'] = t0 / FPS
            d['max_tilt_after_cond'] = float(tilt[t0:].max())
            long = [(s, e) for s, e in segments(cond) if e - s >= 9]                   # obj_extract.py loss rule
            lost = False
            if long and long[0][1] < T - 1: lost = not bool(cond[-1])
            d['lost'] = lost
            d['topple_any'] = bool(up_req and (upv[t0:] <= COS30).any())
            others = [k2 for k2 in C if k2 != k]
            nd = lambda t: min(float(np.linalg.norm(P[k][t, :2] - C[k2][t, :2])) for k2 in others)   # crowd.py
            d['place'] = dict(tilt=float(tilt[t0]), tilt_plus1s=float(tilt[min(t0 + 20, T - 1)]),   # topple.py
                              base_above_box=float(P[k][t0, 2] - bp[t0, 2]), dxy_place=float(dxy[t0]), near=nd(t0))
            d['near_end'] = nd(T - 1)
            if k in ('soy', 'vinegar'):
                L = W[:, :21]; n = min(len(L), len(c))
                dl = np.linalg.norm(L[:n] - c[:n, None], axis=2).min(1)                # release.py: left hand >5 cm
                after = np.arange(t0, n); rel = after[np.flatnonzero(dl[after] > 0.05)]
                d['hand_d_at_place'] = float(dl[t0])
                # axis-based release (not in the rollout scripts): hand keypoints vs the bottle axis segment (base -> base+2*off_y);
                # needed because vinegar is gripped 5-6 cm from its centre, so the centre-distance rule fires at placement time.
                Lb = 2 * off[1]; base = P[k][:n, :3]; top = base + rot(P[k][:n, 3:7], [0, Lb, 0]); ab = top - base
                v = L[:n] - base[:, None]; tt = np.clip((v * ab[:, None]).sum(2) / (ab * ab).sum(1)[:, None], 0, 1)
                da = np.linalg.norm(v - tt[..., None] * ab[:, None], axis=2).min(1)
                far = da > 0.05; ra = [t for t in range(t0, n - 9) if far[t:t + 10].all()]
                d['axis_d_at_place'] = float(da[t0])
                if ra:
                    ta = ra[0]
                    loc_a = rot(qinv[ta:ta + 1], P[k][ta, :3] - bp[ta, :3])[0]
                    d['release_axis'] = dict(release_s=ta / FPS, rel_up=float(loc_a[1]), rel_tilt=float(tilt[ta]),
                                             rel_vz=float((P[k][ta, 2] - P[k][max(ta - 4, 0), 2]) * 5), near=nd(ta))
                if len(rel):
                    tr = int(rel[0])
                    loc = lambda t: rot(qinv[t:t + 1], P[k][t, :3] - bp[t, :3])[0]        # base in box frame (x, up, z)
                    L0, Le = loc(tr), loc(T - 1)
                    d['release'] = dict(release_s=tr / FPS, rel_up=float(L0[1]), rel_x=float(L0[0]), rel_z=float(L0[2]),
                                        rel_tilt=float(tilt[tr]), rel_vz=float((P[k][tr, 2] - P[k][max(tr - 4, 0), 2]) * 5),
                                        end_up=float(Le[1]), end_tilt=float(tilt[-1]), near=nd(tr),
                                        min_up_before_release=float(min(loc(t)[1] for t in range(max(t0, tr - 40), tr + 1))))
        out[k] = d
    return out


def episodes(task):
    if task == 27:
        pat = f'{T27_ROOT}/episode_??????'
    else:
        pat = f'{T21_RAW}/episode_??????.hdf5'
    return sorted(os.path.basename(p).replace('.hdf5', '') for p in glob.glob(pat))   # base episodes; "_1" are re-renders


def replay_identical(task, ep):
    """'_1' episodes are background/texture re-renders: check their object trajectories equal the base."""
    f = (lambda e: h5py.File(f'{T27_ROOT}/{e}/rgbd.hdf5', 'r')) if task == 27 else (lambda e: h5py.File(f'{T21_RAW}/{e}.hdf5', 'r'))
    a, b = f(ep), f(ep + '_1')
    key = 'objects/obj_058_golf_ball_4/pose_world' if task == 27 else 'objects/obj_172_soy_sauce_2/pose_world'
    return a[key].shape == b[key].shape and float(np.abs(a[key][:] - b[key][:]).max()) < 1e-6


# ------------------------------------------------------------------------------------------------
def pct(x, q=(10, 50, 90)):
    x = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    return None if not len(x) else [float(v) for v in np.percentile(x, q)]


def fmt(x, scale=100, nd=1):
    p = pct(x)
    return 'n/a' if p is None else f'{p[1]*scale:.{nd}f} ({p[0]*scale:.{nd}f}/{p[2]*scale:.{nd}f}) n={len([v for v in x if v is not None])}'


def slope_ci(x, y, B=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x)); bs.append(np.polyfit(x[i], y[i], 1)[0])
    return float(s), [float(v) for v in np.percentile(bs, [2.5, 97.5])], float(np.corrcoef(x, y)[0, 1])


def summarize(out):
    L = []
    R27 = [json.loads(l) for l in open(f'{out}/task27_episodes.jsonl')] if os.path.exists(f'{out}/task27_episodes.jsonl') else []
    R21 = [json.loads(l) for l in open(f'{out}/task21_episodes.jsonl')] if os.path.exists(f'{out}/task21_episodes.jsonl') else []
    if R27:
        L += [f'## task27 demos (n={len(R27)} unique demos; median (p10/p90), lengths in cm)', '']
        L += [f'- demo success flag all True: {all(r["success"] for r in R27)}; table dz range {min(r["dz"] for r in R27)*100:.1f}..{max(r["dz"] for r in R27)*100:.1f} cm, std {np.std([r["dz"] for r in R27])*100:.1f} cm',
              f'- FK-conversion sanity: lowest fingertip z minus table over all demos, p0/p50/p100 = ' + ' / '.join(f'{v*100:.1f}' for v in np.percentile([r['tip_min_z_minus_table'] for r in R27], [0, 50, 100])) + ' cm (should be about 0)',
              f'- table_z (golf z0 - 2.14 cm) vs meta dz: corr = {np.corrcoef([r["table_z"] for r in R27], [r["dz"] for r in R27])[0,1]:.4f}',
              f'- overhead camera z - 1.8 vs table dz: corr = {np.corrcoef([r["cam_z"] for r in R27], [r["dz"] for r in R27])[0,1]:.3f}', '']
        L += ['### Metric 1: first grasp geometry (fingertip centroid vs ball centre; min xy offset within 1.5 s after first contact)', '',
              '| ball | hand (R/L) | xy offset | tips below centre (mean, median) | palm height above table | tip-min height above table | first contact t (s) |', '|---|---|---|---|---|---|---|']
        for k in BALLS:
            key = 'right' if k == 'golf' else 'auto'
            G = [r[k][key] for r in R27 if r[k].get(key)]
            hands = [r[k].get('hand') for r in R27 if r[k].get('hand')]
            L.append(f'| {k} ({key}-hand def) | R{hands.count("R")}/L{hands.count("L")} | {fmt([g["xy_off"] for g in G])} | {np.mean([g["tips_below_centre"] for g in G]):.1f}, {np.median([g["tips_below_centre"] for g in G]):.0f} | '
                     f'{fmt([g["palm_h"] for g in G])} | {fmt([g["tip_min_h"] for g in G])} | {fmt([g["t_first_contact"] for g in G], 1)} |')
        L += ['', 'Rollout reference: golf xy offset placed 1.8 / fail 3.3 cm; tips below centre 5 vs 1; palm height 14.6 vs 16.4 cm; tennis lifted 2.5 / not lifted 4.7 cm.', '']
        g = [r['golf']['right'] for r in R27 if r['golf'].get('right')]
        xo = np.array([x['xy_off'] for x in g])
        L += [f'- golf xy offset distribution: fraction >3.3 cm = {np.mean(xo>0.033):.2f}, >2.5 cm = {np.mean(xo>0.025):.2f}, <=1.8 cm = {np.mean(xo<=0.018):.2f}', '']
        L += ['### Metric 2: palm height above table vs table offset at golf grasp', '']
        for k, key in (('golf', 'right'), ('tennis', 'auto')):
            rr = [r for r in R27 if r[k].get(key)]
            dz = np.array([r['dz'] for r in rr]); ph = np.array([r[k][key]['palm_h'] for r in rr])
            s, ci, cc = slope_ci(dz, ph)
            sa, cia, _ = slope_ci(dz, ph + dz)            # absolute palm z minus constant == palm_h + dz
            e = np.quantile(dz, [0, 1/3, 2/3, 1])
            tert = [np.median(ph[(dz >= e[i]) & (dz <= e[i+1])]) * 100 for i in range(3)]
            L += [f'- {k}: n={len(rr)} slope d(palm_h)/d(dz) = {s:.2f} (95% bootstrap CI {ci[0]:.2f}..{ci[1]:.2f}), r={cc:.2f}; '
                  f'=> compensation fraction = d(absolute palm z)/d(dz) = 1 + slope = {1+s:.2f} (CI {1+ci[0]:.2f}..{1+ci[1]:.2f}) (1.0 = palm follows table fully, 0 = ignores it). '
                  f'palm_h median by table tertile (low/mid/high, dz {e[0]*100:.1f}/{e[1]*100:.1f}/{e[2]*100:.1f}/{e[3]*100:.1f} cm): {tert[0]:.1f} / {tert[1]:.1f} / {tert[2]:.1f} cm; '
                  f'tertile-diff: high-low palm_h {tert[2]-tert[0]:+.1f} cm for table diff {(np.median(dz[dz>=e[2]])-np.median(dz[dz<=e[1]]))*100:.1f} cm']
        rj = os.path.join(out, 'rollout_grasp_geom.json')
        if os.path.exists(rj):
            RR = [r for r in json.load(open(rj)) if 'error' not in r and r['kind'] != 'raw' and r['ck'] in (30000, 35000, 40000) and (r['golf'] or r['golf_only'])]
            dz = np.array([r['dz_table'] for r in RR]); ph = np.array([r['palm_h'] for r in RR]); xo = np.array([r['xy_off'] for r in RR])
            s_, ci_, cc_ = slope_ci(dz, ph, B=500)
            er = np.quantile(dz, [0, 1/3, 2/3, 1]); tr = [np.median(ph[(dz >= er[i]) & (dz <= er[i+1])]) * 100 for i in range(3)]
            pp_ = np.percentile([r['xy_off'] for r in RR if r['golf']], [10, 50, 90]); pf_ = np.percentile([r['xy_off'] for r in RR if not r['golf']], [10, 50, 90])
            L += [f'- ROLLOUT (same estimator; analysis/golf_env/grasp_geom.json, local+nolocal 30k/35k/40k, golf placed + golf-only fail, n={len(RR)}): slope d(palm_h)/d(dz) = {s_:.2f} '
                  f'(CI {ci_[0]:.2f}..{ci_[1]:.2f}), r={cc_:.2f}; compensation (1 + slope) = {1+s_:.2f}; palm_h tertile medians low/mid/high {tr[0]:.1f}/{tr[1]:.1f}/{tr[2]:.1f} cm (table diff high-low {(np.median(dz[dz>=er[2]])-np.median(dz[dz<=er[1]]))*100:.1f} cm); placed: xy_off p10/50/90 {pp_[0]*100:.1f}/{pp_[1]*100:.1f}/{pp_[2]*100:.1f}, palm_h {np.median([r["palm_h"] for r in RR if r["golf"]])*100:.1f}, tips_below {np.median([r["tips_below_centre"] for r in RR if r["golf"]]):.0f}; golf-only fail: xy_off {pf_[0]*100:.1f}/{pf_[1]*100:.1f}/{pf_[2]*100:.1f}, palm_h {np.median([r["palm_h"] for r in RR if not r["golf"]])*100:.1f}, tips_below {np.median([r["tips_below_centre"] for r in RR if not r["golf"]]):.0f}; xy_off median {np.median(xo)*100:.1f} cm (p10/p90 {np.percentile(xo,10)*100:.1f}/{np.percentile(xo,90)*100:.1f}), '
                  f'palm_h median {np.median(ph)*100:.1f} cm, tips_below_centre median {np.median([r["tips_below_centre"] for r in RR]):.0f}']
        L += ['', 'Rollout reference quoted in the task brief: compensation ~40% (palm height above table changes by ~60% of the offset; 6 cm table diff -> 2.5 cm palm diff). NOTE the ROLLOUT line above recomputes it with the same estimator as the demos.', '']
        L += ['### Metric 3: regrasp (ball touched, not lifted >3 cm, then touched again)', '',
              'Contact = any keypoint of either hand within 4 cm of ball centre (soccer 10 cm, pingpong 6 cm, ad hoc); contact segments < 0.5 s ignored, gaps < 1 s merged. '
              'Regrasp = >=2 such segments start before the ball is first lifted. Sensitivity column uses segments >= 3 frames (0.15 s).', '',
              '| ball | episodes with regrasp | with any extra brief contact (>=3 frames) before first lift | never touched | never lifted |', '|---|---|---|---|---|']
        for k in BALLS:
            L.append(f'| {k} | {sum(r[k]["regrasp"] for r in R27)} | {sum(r[k]["n_brief_contacts_before_lift"]>=2 for r in R27)} | {sum(not r[k]["touched"] for r in R27)} | {sum(r[k]["max_h"]<0.03 for r in R27)} |')
        L += ['', f'Episodes where at least one ball is regrasped: {sum(any(r[k]["regrasp"] for k in BALLS) for r in R27)} / {len(R27)}; ' + ', '.join(f'{r["ep"]}:{k}' for r in R27 for k in BALLS if r[k]["regrasp"]), '']
    if R21:
        L += [f'## task21 demos (n={len(R21)} unique demos)', '',
              f'- demo success flag all True: {all(r["success"] for r in R21)}; table dz range {min(r["dz"] for r in R21)*100:.1f}..{max(r["dz"] for r in R21)*100:.1f} cm', '',
              '### Metric 4: bottle release / placement (box frame; height = bottle base above box origin, release.py definition)', '',
              'Release rule A = rollout rule (left hand >5 cm from bottle centre after first-cond; unreliable for vinegar, gripped 5-6 cm from its centre, so it fires at placement time). '
              'Rule B (new, demos only) = left hand >5 cm from the bottle AXIS segment for 10 consecutive frames.', '',
              '| bottle | rule | n | release base height above box (cm) | release tilt (deg) | release vz (m/s) | nearest-obj xy dist at release (cm) |', '|---|---|---|---|---|---|---|']
        for k in ('soy', 'vinegar'):
            X = [r[k] for r in R21 if r[k].get('ever_cond')]
            for nm, key in (('A centre', 'release'), ('B axis', 'release_axis')):
                Rl = [x[key] for x in X if x.get(key)]
                L.append(f'| {k} | {nm} | {len(Rl)} | {fmt([x["rel_up"] for x in Rl])} | {fmt([x["rel_tilt"] for x in Rl], 1)} | {fmt([x["rel_vz"] for x in Rl], 1, 2)} | {fmt([x["near"] for x in Rl])} |')
        L += ['', '| bottle | n placed | base height above box at first-cond (cm, topple.py) | tilt at first-cond (deg) | nearest-obj xy at first-cond (cm) | nearest-obj xy at end (cm) |', '|---|---|---|---|---|---|']
        for k in ('soy', 'vinegar'):
            X = [r[k] for r in R21 if r[k].get('ever_cond')]
            L.append(f'| {k} | {len(X)} | {fmt([x["place"]["base_above_box"] for x in X])} | {fmt([x["place"]["tilt"] for x in X], 1)} | {fmt([x["place"]["near"] for x in X])} | {fmt([x["near_end"] for x in X])} |')
        L += ['', 'Rollout reference (release.py/crowd.py, soy is the reliable one): no-topple release height 2.6 cm, tilt 8.6 deg, vz -0.19 m/s; toppled 6.0 cm, 16.5 deg, vz ~0. '
              'Topple rate 64% (soy) / 89% (vinegar) when nearest object < 6 cm, 3-14% at 8-14 cm.', '']
        L += ['Topple / crowding counts in demos:', '']
        for k in ('soy', 'vinegar'):
            X = [r[k] for r in R21 if r[k].get('ever_cond')]
            Rl = [x['release'] if k == 'soy' else x['release_axis'] for x in X if x.get('release' if k == 'soy' else 'release_axis')]
            L.append(f'- {k} (release rule {"A" if k == "soy" else "B"}): placed {len(X)}/{len(R21)}; lost(obj_extract rule) {sum(x["lost"] for x in X)}; toppled (tilt>30 after first-cond) {sum(x["topple_any"] for x in X)}; '
                     f'final_cond false at end {sum(not x["final_cond"] for x in X)}; release height >6 cm: {sum(x["rel_up"]>0.06 for x in Rl)}/{len(Rl)}; tilt>16.5 deg at release: {sum(x["rel_tilt"]>16.5 for x in Rl)}/{len(Rl)}; '
                     f'nearest object at end <6 cm: {sum(x["near_end"]<0.06 for x in X)}/{len(X)}, 8-14 cm: {sum(0.08<=x["near_end"]<0.14 for x in X)}/{len(X)}; max tilt after first-cond p50/p90/max: '
                     + '/'.join(f'{v:.1f}' for v in np.percentile([x['max_tilt_after_cond'] for x in X], [50, 90, 100])) + ' deg')
        L += ['Settled (end-of-episode) bottle tilt, deg, p10/50/90: ' + '; '.join(f'{k} ' + '/'.join(f'{v:.1f}' for v in np.percentile([r[k]["final_tilt"] for r in R21], [10, 50, 90])) for k in ('soy', 'vinegar', 'msg')), '']
    open(f'{out}/summary_auto.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['extract', 'summarize']); ap.add_argument('--task', type=int, choices=[21, 27])
    ap.add_argument('--out', required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == 'extract':
        fn = extract27 if a.task == 27 else extract21
        eps = episodes(a.task)
        path = f'{a.out}/task{a.task}_episodes.jsonl'
        assert not os.path.exists(path), f'{path} exists; refusing to overwrite'
        n_same = 0
        with open(path, 'w') as fh:
            for ep in eps:
                r = fn(ep)
                r['replay_1_identical'] = replay_identical(a.task, ep) if os.path.exists(
                    (f'{T27_ROOT}/{ep}_1/rgbd.hdf5') if a.task == 27 else f'{T21_RAW}/{ep}_1.hdf5') else None
                fh.write(json.dumps(r) + '\n'); fh.flush()
        print('wrote', path, len(eps))
    else:
        summarize(a.out)


if __name__ == '__main__':
    main()
