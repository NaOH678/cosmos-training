#!/usr/bin/env python3
"""Offline Viser HTML of the D435 hand and object points, with the URDF hand surface as reference.

One 3D scene per head-depth frame (6 fps, real-time playback), single deck, four layers:

  D435 hand / object classes   SAM2 v2 classes 2 / 3 / 4, each through the exporter's per-frame gates
                               (mask erosion, depth edge, 0.1-6 m, drop_detached), coloured by their
                               own RGB tinted with the class colour
  URDF hand surface            camera-visible part of the right Wuji hand mesh posed from qpos (solid blue)

Made to judge the D435 object depth and where the objects sit relative to the hand.  Points come from
``render_urdf_surface_da3_d435.py --dump --dump-classes 2,3,4`` (``clouds.npz``).

    $T4W_PY tools/build_viser_d435_objects.py --clouds <dir>/clouds.npz --output <new>.html
"""

import argparse
import sys
from pathlib import Path

import numpy as np

NEW_POINTFLOW = Path('/mnt/afs/WorldAct-native-track4w-dense/new_pointflow')
sys.path.insert(0, str(NEW_POINTFLOW))
from export_efep_official_html import package  # noqa: E402

PF_ROOT = Path('/data/shichaojian/pf_out/9.24/sandwich/efep_seg_v2')
CLASSES = {  # SAM2 label: (node, legend, tint)
    2: ('d435_hand', 'D435 手部点', (44, 160, 44)),
    3: ('d435_lettuce', 'D435 生菜点（标签 3，被操作物体）', (148, 82, 204)),
    4: ('d435_bread_cheese', 'D435 面包/奶酪点（标签 4，静止物体）', (235, 200, 0)),
}
SURFACE = ('urdf_surface', 'URDF 手表面（相机可见部分，手的真值参照）', (31, 119, 180))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clouds', type=Path, required=True)
    ap.add_argument('--episode', default='episode_0013_20260731_133649')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mix', type=float, default=0.5, help='class tint weight over the point RGB')
    ap.add_argument('--point-size', type=float, default=0.003, help='metres')
    ap.add_argument('--port', type=int, default=8994)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; write a new file instead of overwriting')

    clouds = np.load(args.clouds)
    frames = [int(f) for f in clouds['frames']]
    missing = [c for c in CLASSES if f'd435c{c}_{frames[0]}' not in clouds.files]
    if missing:
        raise SystemExit(f'{args.clouds} has no D435 points for classes {missing}; rerun the dump with --dump-classes 2,3,4')
    times = np.load(PF_ROOT / args.episode / 'timestamps_sec.npy')

    import viser
    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    empty = np.zeros((1, 3), np.float32)
    nodes = {name: server.scene.add_point_cloud(f'/{name}', points=empty, colors=np.array([tint], np.uint8),
                                                point_size=args.point_size, point_shape='rounded')
             for name, _, tint in [*CLASSES.values(), SURFACE]}
    server._run_garbage_collector(force=True)
    serializer = server.get_scene_serializer()

    counts = {name: [] for name, _, _ in [*CLASSES.values(), SURFACE]}
    look = None
    for i, f in enumerate(frames):
        for c, (name, _, tint) in CLASSES.items():
            p = clouds[f'd435c{c}_{f}']
            rgb = clouds[f'd435c{c}_rgb_{f}'].astype(np.float32)
            col = (args.mix * np.array(tint, np.float32) + (1 - args.mix) * rgb).astype(np.uint8)
            nodes[name].points = p if len(p) else empty
            nodes[name].colors = col if len(p) else np.array([tint], np.uint8)
            counts[name].append(len(p))
        s = clouds[f'surface_{f}']
        nodes[SURFACE[0]].points = s if len(s) else empty
        nodes[SURFACE[0]].colors = np.broadcast_to(np.array(SURFACE[2], np.uint8), (max(len(s), 1), 3)).copy()
        counts[SURFACE[0]].append(len(s))
        if look is None and len(s):
            look = np.median(s, axis=0)
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        serializer.insert_sleep(float(times[nxt] - times[f]) if nxt is not None else 1 / 6)
        if i % 40 == 0:
            print(f'frame {f} ({i + 1}/{len(frames)}): ' + ', '.join(f'{k} {v[-1]}' for k, v in counts.items()), flush=True)
    recording = serializer.serialize()
    server.stop()

    look = look if look is not None else np.array([0.2, 0.1, 0.8])
    camera = dict(position=(float(look[0] + 0.55), float(look[1] - 0.15), float(look[2])),
                  look_at=look.tolist(), up=(0.0, -1.0, 0.0))
    rows = ''.join(f'<span style="color:rgb{tint}">■</span> {legend}：中位 {np.median(counts[name]):.0f} 点/帧<br>'
                   for name, legend, tint in [*CLASSES.values(), SURFACE])
    legend = (
        '<aside style="position:fixed;left:12px;top:12px;z-index:9999;max-width:620px;'
        'padding:12px 16px;background:#ffffffed;color:#1b2838;font:13px/1.65 system-ui;'
        'border:1px solid #dce1e8;border-radius:9px">'
        f'<b>sandwich {args.episode}：D435 深度的手与物体</b><br>'
        f'{len(frames)} 个有 head 深度的帧（每 5 帧 1 个，按真实时间播放），D435 彩色相机坐标系，米<br>{rows}'
        f'D435 点按自身 RGB 着色、叠 {int(args.mix * 100)}% 类别色；每类分别过 SAM2 v2 mask、腐蚀 1px、去深度边缘、'
        '0.1–6 m、drop_detached（与 DA3 训练点相同的逐帧门）<br>'
        'URDF 手表面由 qpos 正向运动学 + 已验证外参得到，只有手有真值；物体无真值，看形状、桌面贴合与相对手的位置<br>'
        '打开为侧视；拖动旋转 / 滚轮缩放 / 空格暂停</aside>')
    meta = dict(episode=args.episode, frames=len(frames), clouds=str(args.clouds),
                layers={name: legend for name, legend, _ in [*CLASSES.values(), SURFACE]})
    html = package(recording, meta, camera, f'{args.episode} D435 hand + objects', legend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.html')
    partial.write_text(html)
    partial.replace(args.output)
    print(f'HTML {args.output} {args.output.stat().st_size}')


if __name__ == '__main__':
    main()
