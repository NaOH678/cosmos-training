#!/usr/bin/env python3
"""Offline Viser HTML: Track4World (DA3) hand + object points, URDF hand surface, D435 hand points.

One 3D scene per head-depth frame (6 fps, real-time playback), five layers in fixed colours:

  DA3 hand / objects      ``efep_seg_v2`` labels 2 / 3 / 4, ``obs_valid & obs_unique`` -- the training
                          points as stored, in DA3's own camera (estimated intrinsics kept)
  URDF hand surface       camera-visible part of the right Wuji hand mesh, posed from qpos
  D435 hand               head depth through the same per-frame gates as the DA3 points

URDF and D435 come from ``render_urdf_surface_da3_d435.py --dump`` (``clouds.npz``; that step needs
MuJoCo, this one needs viser, which live in different venvs).  Packaged with the same offline
client as ``new_pointflow/build_dense3d_flow.py`` (native-track4w-dense worktree).

    $T4W_PY tools/build_viser_urdf_da3_d435.py --clouds <dir>/clouds.npz --output <new>.html
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NEW_POINTFLOW = Path('/mnt/afs/WorldAct-native-track4w-dense/new_pointflow')
sys.path.insert(0, str(NEW_POINTFLOW))
from export_efep_official_html import package  # noqa: E402

PF_ROOT = Path('/data/shichaojian/pf_out/9.24/sandwich/efep_seg_v2')
LAYERS = {  # node: (legend, rgb)
    'da3_hand': ('Track4World DA3 手部点（训练数据）', (255, 127, 14)),
    'da3_object': ('Track4World DA3 被操作物体点（标签 3，训练数据）', (148, 82, 204)),
    'da3_surface': ('Track4World DA3 台面/静止物体点（标签 4，训练数据）', (235, 200, 0)),
    'urdf_surface': ('URDF 手表面（相机可见部分，真值参照）', (31, 119, 180)),
    'd435_hand': ('D435 手部点（与 DA3 相同过滤）', (44, 160, 44)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clouds', type=Path, required=True)
    ap.add_argument('--episode', default='episode_0013_20260731_133649')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--point-size', type=float, default=0.003, help='metres')
    ap.add_argument('--port', type=int, default=8993)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; write a new file instead of overwriting')

    d = PF_ROOT / args.episode
    off = np.load(d / 'frame_offsets.npy')
    pos = np.load(d / 'obs_pos.npy')
    lab = np.load(d / 'obs_label.npy')
    ok = np.load(d / 'obs_valid.npy') & np.load(d / 'obs_unique.npy')
    times = np.load(d / 'timestamps_sec.npy')
    clouds = np.load(args.clouds)
    frames = [int(f) for f in clouds['frames']]

    import viser
    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    empty = np.zeros((1, 3), np.float32)
    nodes = {k: server.scene.add_point_cloud(f'/{k}', points=empty, colors=np.array([rgb], np.uint8),
                                             point_size=args.point_size, point_shape='rounded')
             for k, (_, rgb) in LAYERS.items()}
    server._run_garbage_collector(force=True)
    serializer = server.get_scene_serializer()

    counts = {k: [] for k in LAYERS}
    first_hand = None
    for i, f in enumerate(frames):
        s, e = off[f], off[f + 1]
        keep = ok[s:e]
        clouds_now = {
            'da3_hand': pos[s:e][keep & (lab[s:e] == 2)],
            'da3_object': pos[s:e][keep & (lab[s:e] == 3)],
            'da3_surface': pos[s:e][keep & (lab[s:e] == 4)],
            'urdf_surface': clouds[f'surface_{f}'],
            'd435_hand': clouds[f'd435_{f}'],
        }
        for k, p in clouds_now.items():
            p = np.asarray(p, np.float32) if len(p) else empty
            nodes[k].points = p
            nodes[k].colors = np.broadcast_to(np.array(LAYERS[k][1], np.uint8), p.shape).copy()
            counts[k].append(len(clouds_now[k]))
        if first_hand is None and len(clouds_now['urdf_surface']):
            first_hand = np.median(clouds_now['urdf_surface'], axis=0)
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        serializer.insert_sleep(float(times[nxt] - times[f]) if nxt is not None else 1 / 6)
        if i % 40 == 0:
            print(f'frame {f} ({i + 1}/{len(frames)}): ' + ', '.join(f'{k} {v[-1]}' for k, v in counts.items()), flush=True)
    recording = serializer.serialize()
    server.stop()

    # Open from the side (camera +x), where depth errors show as front/back offsets; rotate freely.
    look = first_hand if first_hand is not None else np.array([0.2, 0.1, 0.8])
    camera = dict(position=(float(look[0] + 0.55), float(look[1] - 0.15), float(look[2])),
                  look_at=look.tolist(), up=(0.0, -1.0, 0.0))
    rows = ''.join(f'<span style="color:rgb{rgb}">■</span> {name}：中位 {np.median(counts[k]):.0f} 点/帧<br>'
                   for k, (name, rgb) in LAYERS.items())
    legend = (
        '<aside style="position:fixed;left:12px;top:12px;z-index:9999;max-width:620px;'
        'padding:12px 16px;background:#ffffffed;color:#1b2838;font:13px/1.65 system-ui;'
        'border:1px solid #dce1e8;border-radius:9px">'
        f'<b>sandwich {args.episode}：DA3 vs D435 vs URDF</b><br>'
        f'{len(frames)} 个有 head 深度的帧（每 5 帧 1 个，按真实时间播放），相机坐标系，米<br>{rows}'
        'DA3 点保持 DA3 自己估计的相机参数（未换成真实内参）；URDF 由 qpos 正向运动学得到，'
        '用已验证的外参放进 D435 彩色相机坐标；D435 过滤：SAM2 v2 手 mask、腐蚀 1px、去深度边缘、0.1–6 m、drop_detached'
        '（无置信度门、无补缺口/去尖峰）<br>'
        '打开为侧视（深度方向左右展开）；拖动旋转 / 滚轮缩放 / 空格暂停</aside>')
    meta = dict(episode=args.episode, frames=len(frames), source=str(d), clouds=str(args.clouds),
                layers={k: v[0] for k, v in LAYERS.items()})
    html = package(recording, meta, camera, f'{args.episode} DA3 vs D435 vs URDF', legend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.html')
    partial.write_text(html)
    partial.replace(args.output)
    print(f'HTML {args.output} {args.output.stat().st_size}')


if __name__ == '__main__':
    main()
