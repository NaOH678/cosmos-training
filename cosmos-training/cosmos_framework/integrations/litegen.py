"""LiteGen lifecycle bridge for Cosmos WAM's shard-first FSDP2 training.

Cosmos owns sampling, SAC/compile, RNG streams, optimizer and DCP state.
LiteGen owns the process group, accelerator selection and FSDP2 wrapping.
"""

import json
import os
from pathlib import Path

import torch
from litegen import device as accelerator

_runtime = None


def selected():
    backend = os.environ.get("COSMOS_TRAINING_BACKEND", "native")
    if backend not in {"native", "litegen"}:
        raise ValueError(f"Unknown COSMOS_TRAINING_BACKEND={backend!r}")
    return backend == "litegen"


def active_runtime():
    return _runtime


def make_config(config, *, world_size=None):
    from litegen.utils.config import MainConfig

    parallel = config.model.config.parallelism
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (
        parallel.context_parallel_shard_degree != 1
        or parallel.cfg_parallel_shard_degree != 1
        or parallel.data_parallel_replicate_degree != 1
        or parallel.data_parallel_shard_degree != world_size
        or world_size < 2
    ):
        raise ValueError("LiteGen WAM currently requires pure FSDP2: shard=WORLD_SIZE>=2, replicate=CP=CFGP=1")
    if config.trainer.distributed_parallelism != "fsdp":
        raise ValueError("LiteGen WAM requires trainer.distributed_parallelism=fsdp")
    device = accelerator.device_type()
    if device not in {"cuda", "npu"}:
        raise ValueError(f"Training requires an accelerator backend, received {device!r}")
    if device == "npu" and config.model.config.get("natten_parameter_list") is not None:
        raise ValueError("NPU neighborhood/sparse attention is not implemented; use the WAM two_way recipe")
    if device == "npu" and config.model.config.compile.enabled:
        raise ValueError("NPU compilation is not validated; set model.config.compile.enabled=false")
    batch = config.dataloader_train.dataloader.batch_size
    return MainConfig.from_dict(
        {
            "exp_name": config.job.name,
            "results_dir": config.job.path_local,
            "global_seed": config.trainer.seed,
            "global_batch_size": world_size * batch * config.trainer.grad_accum_iter,
            "max_steps": config.trainer.max_iter,
            "runtime": {"device": device},
            "optimization": {
                "parallel": {"strategy": "fsdp2", "fsdp2": {"wrap_strategy": "manual"}},
                # Cosmos casts after sharding, using its original precision setting.
                # Its optimizer, including FP32 master weights, stays application-owned.
                "precision": {
                    "cast_model": None,
                    "allow_tf32": accelerator.get_precision_state().get("matmul", False) if device == "cuda" else False,
                },
                "compile": {"enabled": False},
                "memory": {"activation_checkpoint": {"enabled": False}},
            },
            "ema": {"enable": False},
        }
    )


def initialize(config):
    global _runtime
    if _runtime is not None:
        raise RuntimeError("LiteGen runtime already initialized")
    import litegen

    from cosmos_framework.utils.distributed import set_cpu_affinity

    runtime_config = make_config(config)
    set_cpu_affinity()
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    # LiteGen has one TF32 toggle, while Cosmos can use different settings for
    # cuBLAS and cuDNN. Preserve both baseline values, including on failure.
    saved_precision = accelerator.get_precision_state()
    try:
        _runtime = litegen.initialize(runtime_config)
    finally:
        accelerator.set_precision(**saved_precision)
    if torch.distributed.get_rank() == 0:
        target = Path(config.job.path_local) / "litegen_runtime.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(runtime_config.model_dump(mode="json"), indent=2) + "\n")
    return _runtime


def prepare_network(model):
    """Wrap exactly the native decoder blocks and VFM root, before to_empty()."""
    runtime = active_runtime()
    if runtime is None:
        raise RuntimeError("LiteGen must be initialized before constructing the network")
    if getattr(model, "pointflow_branch", None) is not None or getattr(model, "fk_branch", None) is not None:
        raise ValueError("LiteGen adapter currently targets WAM without PointFlow/FK branches")
    paths = tuple(
        f"language_model.model.layers.{name}" for name, _ in model.language_model.model.layers.named_children()
    )
    if not paths:
        raise ValueError("WAM decoder has no layers to shard")
    runtime.config.optimization.parallel.fsdp2.wrap_paths = paths
    # Same helper path as Latte's gen.prepare(), with Cosmos owning materialization.
    model = runtime.prepare(model, materialize_meta=False)
    runtime.register_fsdp_forward_methods(model, ["generate_reasoner_text"])
    from torch.distributed.fsdp import register_fsdp_forward_method

    for block in model.language_model.model.layers.children():
        register_fsdp_forward_method(block, "reasoner_forward")
    if torch.distributed.get_rank() == 0:
        path = Path(runtime.config.results_dir) / "litegen_sharding.json"
        path.write_text(json.dumps({"units": [*paths, ""], "materialize_meta": False}, indent=2) + "\n")
    return model


def register_training_state(optimizer, scheduler):
    if _runtime is not None:
        # Preserve Cosmos FusedAdam, including its FP32 master weights/moments.
        # Cosmos DCP remains the persistence owner.
        _runtime.register(optimizer, scheduler)


def shutdown():
    global _runtime
    if _runtime is not None:
        try:
            _runtime.shutdown()
        finally:
            _runtime = None
