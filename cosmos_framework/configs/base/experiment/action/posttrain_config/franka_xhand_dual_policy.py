# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``franka_xhand_dual_policy`` — action-policy SFT on the dual-arm Franka + XHand
``capture-tube`` LeRobot dataset, starting from a Cosmos3-Nano base checkpoint.

Mirrors ``vision_sft_nano`` (same Hydra ``defaults`` / base config tree, EMA,
DCP checkpoints) but swaps the internal PackingDataLoader stack for the OSS
``DataPackerDataLoader`` wired to :class:`FrankaXHandDualLeRobotDataset` +
:class:`ActionDataPacker`.

Differences vs ``vision_sft_nano``:
  * ``dataloader_train`` = ``DataPackerDataLoader`` (map-style dataset, action
    bin-packing: high ``max_tokens`` + bounded ``max_batch_size``).
  * ``DataLoaderStateCallback(distributor_type="data_packer")`` added for
    stateful checkpoint/resume of the map-style dataset.
  * Optimizer also trains the action adapters (``action2llm`` / ``llm2action`` /
    ``action_modality_embed`` / ``action_pos_embed``) on top of the generator
    expert + video adapters.
  * ``strict_resume=False`` and the action adapters added to
    ``keys_to_skip_loading`` so a base checkpoint without this embodiment's
    action weights still loads (those tensors are freshly initialized).

Usage::

    DATASET_PATH=/mnt/petrelfs/shichaojian/lingbot-va/wan_va/dataset/dex-dual-capture-tube-combined-lerobot \\
    IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. \\
    torchrun --nproc_per_node=8 --master_port=12341 \\
        -m cosmos_framework.scripts.train \\
        --config=cosmos_framework/configs/base/config.py -- \\
        experiment=franka_xhand_dual_policy \\
        checkpoint.load_path=examples/checkpoints/Cosmos3-Nano
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.dataloader_state import DataLoaderStateCallback
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.action.datasets.franka_xhand_dual_lerobot_dataset import (
    FrankaXHandDualLeRobotDataset,
)
from cosmos_framework.data.vfm.action.franka_xhand_dual_packer import ActionDataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

_DATASET_ROOT = "${oc.env:DATASET_PATH}"
_RESOLUTION = "480"
_CHUNK_LENGTH = 16
_CONDITIONING_FPS = 15.0


franka_xhand_dual_policy = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action",
            name="franka_xhand_dual_policy",
            wandb_mode="offline",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, max_action_dim=64
        ),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1.0e-06,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                # action adapters (per-embodiment input/output projections + modality/pos embeds)
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            lr=5.0e-04,
            lr_multipliers={},
            optimizer_type="AdamW",
            weight_decay=0,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine",
            cycle_lengths=[12500],  # cosine spans the full run; must match trainer.max_iter (32-GPU)
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[250],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=2,
            logging_iter=1,
            max_iter=12500,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                # Stateful checkpoint/resume for the map-style DataPacker dataset.
                dataloader_state=L(DataLoaderStateCallback)(distributor_type="data_packer"),
                compile_tokenizer=dict(
                    compile_after_iterations=3,
                    enabled=False,
                    warmup_resolutions=None,
                ),
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
                expert_heatmap=dict(every_n=1000),
                grad_clip=dict(clip_norm=0.1, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                norm_monitor=dict(
                    every_n=100,
                    layer_norm_only=False,
                    log_stat_wandb=True,
                    model_key=None,
                    save_s3=False,
                    step_size=1,
                    track_activations=True,
                ),
                param_count=dict(save_s3=False),
                sequence_packing_padding=dict(every_n=50),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                wandb_2x=dict(
                    logging_iter_multipler=2,
                    save_logging_iter_multipler=1,
                    save_s3=False,
                ),
                wandb_val=dict(save_s3=False),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip EMA + freshly-initialized action adapters when loading the base
            # Cosmos3-Nano checkpoint (which may not contain this embodiment's
            # action weights).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # supply via CLI: checkpoint.load_path=examples/checkpoints/Cosmos3-Nano
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=1250,  # 12500/1250 = 10 checkpoints over the run
            # Non-strict: base checkpoint lacks the freshly-initialized action adapters.
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(DataPackerDataLoader)(
            data_source=L(FrankaXHandDualLeRobotDataset)(
                root=_DATASET_ROOT,
                fps=_CONDITIONING_FPS,
                chunk_length=_CHUNK_LENGTH,
                mode="policy",
            ),
            data_packer=L(ActionDataPacker)(
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                max_action_dim=64,
                resolution=_RESOLUTION,
                cfg_dropout_rate=0.1,
            ),
            # Action-policy convention: token budget effectively disabled, batch
            # bounded by max_batch_size (fixed number of demonstrations per step).
            max_tokens=999999,
            max_batch_size=4,
            pool_size=16,
            shuffle=True,
            seed=42,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            name="action",
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [franka_xhand_dual_policy]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
