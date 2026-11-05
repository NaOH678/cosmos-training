"""Use the ORIGINAL D435 V3 pipeline; replace only native depth with FoundationStereo."""

import json
from pathlib import Path

import numpy as np
import run_dropper_d435_v2 as pipeline

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "pointflow_outputs/scissors_foundationstereo_20261105"
OUT = ROOT / "pointflow_outputs/scissors_foundationstereo_v3_parity_20261105"
Fusion = pipeline.load_module(
    "parity_original_surface", ROOT / "cosmos_framework/inference/robot_policy/causal_surface_fusion.py"
).CausalSurfaceFusion


def run():
    source = json.loads((FS / "source.json").read_text())
    streams = json.loads((FS / "calibration.json").read_text())["streams"]
    assert streams["depth"]["intrinsics"] == streams["infrared_left"]["intrinsics"]
    assert streams["depth"]["extrinsics_to_color"] == streams["infrared_left"]["extrinsics_to_color"]
    pairs = json.loads((FS / "ir_pair_index.json").read_text())
    by_frame = {x["available_rgb_step"]: x for x in pairs}
    scale = streams["depth"]["depth_scale_m"]

    def provider(frame):
        pair = by_frame.get(frame)
        if pair is None:
            return None
        return np.load(FS / "native_depth" / f"{pair['sequence']:06d}.npy") / scale

    pipeline.TASK = "scissors"
    pipeline.VERSION = "foundationstereo_v3_parity"
    pipeline.V0 = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
    pipeline.OUT = OUT
    pipeline.cv2.setNumThreads(2)
    pipeline.build(enhancement=Fusion(), native_provider=provider)
    pipeline.html(
        comparison=ROOT / "pointflow_outputs/cotracker_real_v3_d435_20261105/scissors",
        show_urdf=True,
        surface_stride=4,
        depth_label="FoundationStereo",
        display_hold_steps=0,
        comparison_label="05_D435_V3_original",
        legend_text='<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:350px;font:13px/1.5 system-ui"><b>Scissors · FoundationStereo / D435 V3同代码对照</b><br>01：原D435 V3代码，只将输入深度换成FoundationStereo；02：FoundationStereo原始点；03：原V3历史补点；04：真实URDF表面；05：原D435 V3点云。<br>保留原版mask内部筛选、相同取深度半径、相同融合与过滤。无额外PointWorld体素选点或额外EMA。原始观测到达时间相差至多一个RGB步的个别帧仍按因果时间戳处理。<br>蓝色URDF，橙色手点，物体RGB。<div id="status"></div><img id="rgb" style="width:100%"></aside>',
    )
    pipeline.save(
        OUT / "source_swap.json",
        dict(
            source=source,
            geometry_equal=True,
            units_conversion="FoundationStereo meters / recorded depth_scale_m",
            same_processing_entry="tools/run_dropper_d435_v2.py:build",
            no_extra_pointworld_sampling=True,
            no_extra_ema=True,
            uses_future_depth=False,
            upstream_cotracker_masks_reused=True,
        ),
    )


if __name__ == "__main__":
    run()
