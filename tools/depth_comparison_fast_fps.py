"""C++ equivalent of the existing NumPy FPS, used only by the depth experiment."""

import ctypes
from pathlib import Path

import numpy as np

_library = ctypes.CDLL(
    str(Path(__file__).resolve().parents[1] / "pointflow_outputs/depth_source_comparison_20261103/depth_fps.so")
)
_function = _library.depth_fps
_function.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]


def anchor_fps(xyz, count, initial=()):
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    initial = np.ascontiguousarray(initial, dtype=np.int64)
    count = min(count, len(xyz) - len(initial))
    result = np.empty(count, np.int64)
    if count:
        _function(xyz.ctypes.data, len(xyz), count, initial.ctypes.data, len(initial), result.ctypes.data)
    return result
