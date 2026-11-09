"""Import the WAM model/recipe when optional CUDA extensions are absent."""

import os
import subprocess
import sys


def test_wam_import_without_cuda_extensions():
    script = """
import importlib.abc
import importlib.util
import sys
import torch
blocked={'pynvml','apex','natten','flash_attn','flash_attn_interface','transformer_engine','triton'}
original=importlib.util.find_spec
def find_spec(name,*args,**kwargs):
    return None if name.split('.')[0] in blocked else original(name,*args,**kwargs)
importlib.util.find_spec=find_spec
class MissingCuda(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('CUDA extension absent: '+fullname, name=fullname.split('.')[0])
sys.meta_path.insert(0,MissingCuda())
import cosmos_framework.model.generator.omni_mot_model
import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_sim_pointfk_edge
print('WAM model/config import passed')
"""
    env = dict(os.environ, LITEGEN_DEVICE="cpu", COSMOS_DEVICE="cpu")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dataset_exports_are_lazy_and_cached(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pytest
    from cosmos_framework.data.generator.action import datasets

    expected = [
        "ActionBaseDataset", "AgiBotWorldBetaLeRobotDataset", "BridgeOrigLeRobotDataset",
        "DROIDLeRobotDataset", "DROIDMergedLeRobotDataset", "FractalLeRobotDataset",
        "HumanHandPoseLeRobotDataset", "LIBEROLeRobotDataset", "RoboMINDFrankaDataset",
        "RoboMINDURDataset", "UMILeRobotDataset",
    ]
    assert datasets.__all__ == expected
    target = object()
    loader = Mock(return_value=SimpleNamespace(DROIDLeRobotDataset=target))
    monkeypatch.setattr(datasets, "import_module", loader)
    monkeypatch.setitem(datasets.__dict__, "DROIDLeRobotDataset", None)
    del datasets.__dict__["DROIDLeRobotDataset"]
    assert datasets.DROIDLeRobotDataset is target
    assert datasets.DROIDLeRobotDataset is target
    loader.assert_called_once_with(".droid_lerobot_dataset", datasets.__name__)
    with pytest.raises(AttributeError, match="has no attribute"):
        datasets.missing_dataset
