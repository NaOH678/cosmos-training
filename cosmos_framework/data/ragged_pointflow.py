"""Adapt episode-global sparse observations to the existing PointFlow window API."""

import json
from pathlib import Path

import numpy as np

from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window, window_rows


class RaggedPointFlow:
    def __init__(self, episode):
        self.episode = Path(episode)
        self.arrays = {
            name: np.load(self.episode / f"{name}.npy", mmap_mode="r")
            for name in (
                "frame_offsets",
                "obs_track",
                "obs_pos",
                "obs_uv",
                "obs_valid",
                "track_label",
                "frame_indices",
                "timestamps_sec",
            )
        }

    def window(self, rows):
        a = self.arrays
        first = slice(*map(int, a["frame_offsets"][rows[0] : rows[0] + 2]))
        ids = np.sort(a["obs_track"][first][np.asarray(a["obs_valid"][first], dtype=bool)])
        position = np.zeros((len(rows), len(ids), 3), np.float32)
        uv = np.zeros((len(rows), len(ids), 2), np.float32)
        valid = np.zeros((len(rows), len(ids)), bool)
        for i, row in enumerate(rows):
            source = slice(*map(int, a["frame_offsets"][row : row + 2]))
            _, ii, jj = np.intersect1d(ids, a["obs_track"][source], assume_unique=True, return_indices=True)
            if i:
                position[i], uv[i] = position[i - 1], uv[i - 1]
            position[i, ii] = a["obs_pos"][source][jj]
            uv[i, ii] = a["obs_uv"][source][jj]
            valid[i, ii] = a["obs_valid"][source][jj]
        return dict(point_ids=ids, position=position, uv_px=uv, valid=valid)

    def prepare_window(self, start_frame, *, timing=None, **kwargs):
        timing = timing or PointFlowTiming()
        a = self.arrays
        rows = window_rows(a["frame_indices"], a["timestamps_sec"], start_frame, timing.fps, timing.steps)
        window = self.window(rows)
        lookup = {int(row): i for i, row in enumerate(rows)}
        result = prepare_window(
            self.episode,
            start_frame,
            timing=timing,
            frame_provider=lambda name, row: window[name][lookup[row]],
            **kwargs,
        )
        local = result["point_ids"].copy()
        result["point_ids"] = window["point_ids"][local]
        result["region_labels"] = a["track_label"][result["point_ids"]].copy()
        return result

    def prepare_sample(self, start_frame, *, timing=None, seed=0, **kwargs):
        """Build the existing model sample, including observed-UV to RGB affine."""
        from cosmos_framework.data.pointflow_dataset import pointflow_sample

        timing = timing or PointFlowTiming()
        window = self.prepare_window(start_frame, timing=timing, seed=seed, **kwargs)
        sample = pointflow_sample(window, self.episode.name, start_frame, seed, timing)
        report = json.loads((self.episode / "report.json").read_text())
        source_wh = np.asarray(report["source_image_size_wh"], dtype=np.int64)
        scale = source_wh / window["image_size_wh"]
        sample["metadata"].update(
            geometry_source="causal_cotracker_da3",
            source_path=str(self.episode),
            uv_to_video=np.array(
                [[scale[0], 0, (scale[0] - 1) / 2], [0, scale[1], (scale[1] - 1) / 2]], dtype=np.float32
            ),
            video_size_wh=source_wh,
        )
        return sample
