"""FoundationStereo: causal static-motion hysteresis, per-ID fusion, continuous sampling."""

import argparse
import json
from pathlib import Path

import numpy as np
import run_dropper_d435_v2 as pipeline
from run_fs_d435_v3_parity import Fusion

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pointflow_outputs/scissors_foundationstereo_20261105"
PREVIOUS = ROOT / "pointflow_outputs/scissors_foundationstereo_v3_parity_20261105"
OUT = ROOT / "pointflow_outputs/scissors_foundationstereo_static_v4_20261105"
core = pipeline.load_module(
    "fs_causal_object_surface", ROOT / "cosmos_framework/inference/robot_policy/causal_object_surface.py"
)


class StaticFusion(Fusion):
    def __init__(self):
        super().__init__()
        self.surface = core.CausalObjectSurfaceState()

    def update(self, ids, uv, instances, labels, raw_z, z, provenance, timestamp, sensor_update, blocked):
        out, p, stats = self.surface.update(
            ids,
            uv,
            instances,
            labels,
            raw_z,
            z,
            provenance,
            timestamp,
            sensor_update,
            blocked,
            motion_estimator=self.image_displacement,
            gray=self.gray,
        )
        stats.update(
            projection_occlusion_removed=self.projection_removed, boundary_pixels_removed=self.boundary_removed
        )
        return out, p, stats


class NoObjectExtrapolation(pipeline.Stabilizer):
    def __init__(self, guard_factory):
        super().__init__(guard_factory)
        self.reacquisition = {}

    def update(self, ids, instances, z, valid, timestamp, sensor_update, cores=None):
        restarted = []
        for instance in np.unique(instances[instances > 2]):
            instance = int(instance)
            guard = self.guards.setdefault(instance, self.guard_factory())
            # Stop extrapolation in the scalar validity guard too, not just in the output points.
            guard.max_velocity = 0.0
            if not sensor_update:
                continue
            if not guard.history or timestamp - guard.history[-1][0] <= 0.3:
                self.reacquisition.pop(instance, None)
                continue
            good = (instances == instance) & valid & np.isfinite(z)
            observed = float(np.median(z[good])) if good.sum() >= 30 else None
            prior = self.reacquisition.get(instance)
            if observed is None:
                self.reacquisition.pop(instance, None)
                continue
            count = prior[2] + 1 if prior and timestamp - prior[0] <= 0.15 and abs(observed - prior[1]) <= 0.015 else 1
            self.reacquisition[instance] = (float(timestamp), observed, count)
            if count >= 2:
                guard = self.guards[instance] = self.guard_factory()
                guard.max_velocity = 0.0
                self.reacquisition.pop(instance, None)
                restarted.append(instance)
        output, provenance, stats = super().update(ids, instances, z, valid, timestamp, sensor_update, cores)
        stats.update(object_guard_velocity_disabled=True, object_guards_reinitialized=restarted)
        for i in np.flatnonzero((instances > 2) & ((provenance == 2) | (provenance == 3))):
            history = self.samples.get(int(ids[i]))
            if history and timestamp - history[-1][0] <= 0.1:
                output[i] = history[-1][1]
            else:
                output[i] = np.nan
                provenance[i] = 0
        return output, provenance, stats


def run(stage="all", limit=None):
    streams = json.loads((SRC / "calibration.json").read_text())["streams"]
    assert streams["depth"]["intrinsics"] == streams["infrared_left"]["intrinsics"]
    assert streams["depth"]["extrinsics_to_color"] == streams["infrared_left"]["extrinsics_to_color"]
    by_frame = {x["available_rgb_step"]: x for x in json.loads((SRC / "ir_pair_index.json").read_text())}

    def provider(frame):
        pair = by_frame.get(frame)
        return (
            np.load(SRC / "native_depth" / f"{pair['sequence']:06d}.npy") / streams["depth"]["depth_scale_m"]
            if pair
            else None
        )

    pipeline.TASK = "scissors"
    pipeline.VERSION = "foundationstereo_static_v4"
    pipeline.V0 = ROOT / "pointflow_outputs/cotracker_real_20261104/scissors"
    pipeline.OUT = OUT
    pipeline.cv2.setNumThreads(2)
    if stage in ["filter", "all"]:
        pipeline.build(
            enhancement=StaticFusion(),
            limit=limit,
            native_provider=provider,
            sample_depth=core.sample_continuous,
            stabilizer_factory=NoObjectExtrapolation,
        )
    if stage in ["html", "all"] and limit is None:
        legend = '<aside style="position:fixed;left:12px;top:12px;z-index:9999;background:#15202bf0;color:#eee;padding:12px;width:350px;font:13px/1.5 system-ui"><b>Scissors · FoundationStereo 静止稳定 V4</b><br>01新版：更宽容的静止判定＋运动退出迟滞；同轨迹ID表面融合；相同物体/深度层连续取样；物体中间帧只短时沿用可信Z，不按噪声速度外推。<br>02原始FoundationStereo；03历史补点；04真实URDF手表面；05上版FoundationStereo同V3处理。<br>所有新过滤仅用当前/历史帧。深度与SAM2/CoTracker复用旧产物。无额外显示旧XYZ暂留。蓝色URDF、橙色手点。<div id="status"></div><img id="rgb" style="width:100%"></aside>'
        pipeline.html(
            comparison=PREVIOUS,
            show_urdf=True,
            surface_stride=4,
            depth_label="FoundationStereo",
            display_hold_steps=0,
            comparison_label="05_previous_FoundationStereo",
            legend_text=legend,
        )
    pipeline.save(
        OUT / "method.json",
        dict(
            uses_future_frames=False,
            source_depth_model_rerun=False,
            depth_source="cached FoundationStereo",
            upstream_offline_masks_tracks_reused=True,
            sampler="4-neighbor same-instance same-depth-layer inverse-distance",
            objects_no_velocity_extrapolation=True,
            maximum_object_hold_seconds=0.1,
            temporal_cache="track ID, not floor(UV/2)",
            static_enter_observations=3,
            static_exit_motion_observations=2,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["filter", "html", "all"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.stage, args.limit)
