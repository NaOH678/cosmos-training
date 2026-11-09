"""CPU checks for topology, native-policy preservation and alignment reporting."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from cosmos_framework.integrations import litegen as bridge
from mrlv.compare_wam_litegen import compare, load_recorded_metrics, load_run, numerical_config

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "reference/wam-task27-align5k-01"
CONFIG = next(BASELINE.glob("cosmos3_action/action_sft/*/config.yaml"))


def test_runtime_config_preserves_policies(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LITEGEN_DEVICE", "cuda")
    config = OmegaConf.load(CONFIG)
    config.job.path_local = str(CONFIG.parent)
    result = bridge.make_config(config)
    assert result.global_batch_size == 128
    assert result.global_seed == 42
    assert result.max_steps == 5000
    assert result.optimization.parallel.strategy == "fsdp2"
    assert result.optimization.precision.cast_model is None
    assert result.optimization.precision.fsdp2_mixed.param_dtype is None
    assert not result.optimization.memory.activation_checkpoint.enabled
    assert not result.optimization.compile.enabled
    assert not result.ema.enable


@pytest.mark.parametrize(
    "key,value",
    [
        ("context_parallel_shard_degree", 2),
        ("cfg_parallel_shard_degree", 2),
        ("data_parallel_replicate_degree", 2),
        ("data_parallel_shard_degree", 4),
    ],
)
def test_rejects_topology_changes(monkeypatch, key, value):
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = OmegaConf.load(CONFIG)
    config.model.config.parallelism[key] = value
    with pytest.raises(ValueError, match="pure FSDP2"):
        bridge.make_config(config)


def test_npu_runtime_config(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LITEGEN_DEVICE", "npu")
    config = OmegaConf.load(CONFIG)
    config.job.path_local = str(CONFIG.parent)
    config.model.config.compile.enabled = False
    result = bridge.make_config(config)
    assert result.runtime.device == "npu"
    assert result.optimization.precision.cast_model is None
    assert not result.optimization.precision.allow_tf32


def test_npu_compile_requires_separate_validation(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LITEGEN_DEVICE", "npu")
    with pytest.raises(ValueError, match="NPU compilation is not validated"):
        bridge.make_config(OmegaConf.load(CONFIG))


def test_prepare_uses_exact_native_units_and_custom_methods(monkeypatch, tmp_path):
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
    fsdp = SimpleNamespace(wrap_paths=())
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            optimization=SimpleNamespace(parallel=SimpleNamespace(fsdp2=fsdp)), results_dir=str(tmp_path)
        ),
        prepare=Mock(side_effect=lambda model, **kwargs: model),
        register_fsdp_forward_methods=Mock(),
    )
    monkeypatch.setattr(bridge, "_runtime", runtime)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    custom = Mock()
    monkeypatch.setattr("torch.distributed.fsdp.register_fsdp_forward_method", custom)
    assert bridge.prepare_network(model) is model
    assert fsdp.wrap_paths == tuple(f"language_model.model.layers.{i}" for i in range(12))
    runtime.prepare.assert_called_once_with(model, materialize_meta=False)
    runtime.register_fsdp_forward_methods.assert_called_once_with(model, ["generate_reasoner_text"])
    assert custom.call_count == 12
    assert json.loads((tmp_path / "litegen_sharding.json").read_text())["units"][-1] == ""


def test_recorded_baseline_metrics_can_be_self_compared():
    run = load_recorded_metrics(BASELINE, 5000)
    assert compare(run, run)["status"] == "passed"
    assert numerical_config(run["config"])["optimizer"]["optimizer_type"] == "FusedAdam"


def test_comparison_detects_loss_regression():
    run = load_recorded_metrics(BASELINE, 5000)
    candidate = copy.deepcopy(run)
    for row in candidate["rows"]:
        row["train/action_loss"] = float(row["train/action_loss"]) * 1.3
    result = compare(run, candidate)
    assert result["status"] == "failed"
    assert not next(c for c in result["checks"] if c["name"] == "train/action_loss/mean")["passed"]


def test_incomplete_run_cannot_pass(tmp_path):
    (tmp_path / "performance.json").write_text((BASELINE / "performance.json").read_text())
    (tmp_path / "metrics.csv").write_text("iteration,train/loss,train/video_loss,train/action_loss\n10,1,1,1\n")
    with pytest.raises(ValueError, match="complete ordered metrics"):
        load_run(tmp_path, 5000)


def test_comparison_detects_gradient_clipping_change():
    run = load_recorded_metrics(BASELINE, 5000)
    candidate = copy.deepcopy(run)
    candidate["config"]["trainer"]["callbacks"]["grad_clip"]["clip_norm"] = 999.0
    result = compare(run, candidate)
    assert not next(c for c in result["checks"] if c["name"] == "numerical_config")["passed"]


def test_shutdown_clears_runtime_on_error(monkeypatch):
    runtime = Mock()
    runtime.shutdown.side_effect = RuntimeError("teardown")
    monkeypatch.setattr(bridge, "_runtime", runtime)
    with pytest.raises(RuntimeError, match="teardown"):
        bridge.shutdown()
    assert bridge.active_runtime() is None


def test_removed_baseline_checkpoint_is_not_reported_complete():
    if (CONFIG.parent / "checkpoints/latest_checkpoint.txt").exists():
        assert load_run(BASELINE, 5000)
    else:
        with pytest.raises(OSError):
            load_run(BASELINE, 5000)


@pytest.mark.parametrize("use_litegen", [False, True])
@pytest.mark.parametrize("compile_all", [False, True])
def test_parallelize_preserves_native_transform_order(monkeypatch, use_litegen, compile_all):
    from cosmos_framework.model.generator.mot import parallelize_vfm_network as module

    model = SimpleNamespace(language_model=object())
    parallel = SimpleNamespace(cp_enabled=False, dp_enabled=True, dp_mesh=object())
    compile_config = SimpleNamespace(enabled=compile_all, compiled_region="all")
    ac_config = object()
    events = []
    language = model.language_model
    transform = Mock(side_effect=lambda value, **kwargs: events.append("language") or value)
    monkeypatch.setattr(module, "parallelize_unified_mot", transform)
    monkeypatch.setattr(module, "apply_compile", lambda value, config: events.append("compile") or value)
    monkeypatch.setattr(bridge, "active_runtime", lambda: object() if use_litegen else None)
    monkeypatch.setattr(bridge, "prepare_network", lambda value: events.append("litegen_shard") or value)
    shard = Mock(side_effect=lambda **kwargs: events.append("native_shard") or kwargs["module"])
    monkeypatch.setattr(module, "fully_shard", shard)
    monkeypatch.setattr(module, "register_fsdp_forward_method", lambda *args: events.append("register"))
    assert module.parallelize_vfm_network(model, parallel, compile_config, ac_config) is model
    transform.assert_called_once_with(
        language, parallel_dims=None if use_litegen else parallel,
        compile_config=compile_config, ac_config=ac_config, attention_io_layout="sequence_sharded",
    )
    assert events == ["language"] + (["compile"] if compile_all else []) + (
        ["litegen_shard"] if use_litegen else ["native_shard", "register"]
    )
    if use_litegen:
        shard.assert_not_called()
    else:
        shard.assert_called_once_with(module=model, mesh=parallel.dp_mesh, ignored_params=set())
