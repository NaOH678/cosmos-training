# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge-DROID policy SFT for the single-right-hand sandwich data."""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.gpu_video_augmentation import GPUVideoAugmentation
from cosmos_framework.callbacks.pointflow_eval import PointFlowEvalCallback
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_singlerighthand_raw_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import limit_dataloader_worker_threads
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_singlerighthand_edge = copy.deepcopy(action_policy_droid_nano)
action_policy_singlerighthand_edge["job"].update(
    project="cosmos3_action",
    group="action_sft",
    name="action_policy_singlerighthand_edge",
    wandb_mode="${oc.env:WANDB_MODE,disabled}",
)

model_config = copy.deepcopy(EDGE_MODEL_CONFIG)
# The TOML schema for [model.compile] only carries ``enabled`` and
# ``compile_dynamic``; compiled_region / max_autotune_pointwise are rejected as
# extra inputs.  EDGE_MODEL_CONFIG ships compiled_region="language" and
# max_autotune_pointwise=False, so tune them here rather than editing the shared
# edge model config that vision_sft_edge also deep-copies.
#
# compiled_region stays "language" (the EDGE_MODEL_CONFIG default).  Setting it
# to "all" -- which would additionally compile the five VFM head methods wrapped
# by parallelize_vfm_network.apply_compile -- does not finish on this recipe:
# inductor's tiling selection (simd.py get_first_compatible_tiling ->
# tiling_is_compatible -> _split_iteration_ranges) reduces to a sympy integer
# polynomial gcd over the head methods' index expressions, and that gcd recurses
# ~65 frames deep in dmp_zz_heu_gcd.  Measured: still inside that single call
# after 20 minutes, never reaching iteration 1 (baseline warmup: 52s).  The only
# knob that touches the path, config.triton.max_tiles=1, is global and would
# disable tiling for the 28 blocks that are 94% of the step and currently work.
# Don't retry without addressing the index expressions themselves.
#
# max_autotune_pointwise is also not worth setting here: in torch 2.10 it is a
# no-op for pointwise kernels.  config.triton.autotune_pointwise already defaults
# to True, so triton_heuristics takes its multi-config branch unconditionally;
# the flag only still changes the reduction heuristics, which this recipe already
# steers via TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0 and
# TORCHINDUCTOR_MIX_ORDER_REDUCTION=0.  Measured: 6.51 s/iter vs a 6.45 s
# baseline over the same iterations, warmup 70s vs 62s.
model_config["tokenizer"]["encode_exact_durations"] = [33]
model_config["max_num_tokens_after_packing"] = -1
model_config["rectified_flow_training_config"]["loss_scale"] = 10.0

# PointFlow's clean sample is the point displacement in METRES, and the branch
# adds unit-variance noise to it (``pointflow_add_noise``: ``target =
# displacement / scale``, ``epsilon = randn(...)``).  At the inherited scale of
# 1.0 the measured displacement std is 0.0839 m against a noise std of 1.0 -- a
# 1:12 signal-to-noise ratio.  The branch can then drive ~99% of the
# flow-matching loss to zero by re-reading the noise off its own input, and the
# sampled trajectory degenerates into a scaled copy of the initial per-point
# noise, which renders as a fan of straight lines radiating from the point
# cluster.  Loss and ADE move in opposite directions as a result: train
# pointflow loss fell 0.237 -> 0.012 while the eval ADE ratio to the
# "predict no motion" baseline rose 1.64x -> 3.75x.
#
# Every other modality in this pipeline is already unit-scale -- the video VAE
# latent measures std 0.872, and the action is normalised by ``quantile_rot`` --
# so this restores the convention rather than introducing a special case.
#
# The scale must follow the DATA and the SELECTION, because faster-moving points
# displace more.  Pooled per-element std of the valid displacement over the
# training set (tools/scan_pointflow_selection.py with the same PointFlowSource
# seed derivation training uses):
#   legacy 10-episode dense export, select_top_n=300, min_voxel_members=3:
#     std 0.0432 m, n=3,764,532
#     (2026-09-20, pointflow_outputs/selection_scan_top300_20260920.json)
#   labeled 29-episode delivery (hand/object only, no background), same selection:
#     std 0.0751 m, n=5,641,299  (select_min_valid_steps=0)
#     std 0.0740 m, n=6,056,661  (select_min_valid_steps=16)
#     (2026-09-21, pointflow_outputs/selection_scan_labeled29_top300_mvs{0,16}_20260921.json)
#   The labeled std is ~1.7x the legacy one: every candidate is a hand/object
#   point, so the top-300 motion ranking no longer dilutes itself with a static
#   background.  --top-n 300 --min-voxel-members 3 [--min-valid-steps 16]
# A std rather than a q01/q99 span, because the noise is unit-variance Gaussian
# and the distribution measures std/robust <= 1, i.e. not tail-inflated, so
# extreme tracking outliers do not inflate it.
#
# Checkpoints trained with a different scale cannot be resumed: both the target
# and the scale the branch learned changed.  After a scale change the pointflow
# loss READS differently and its "predict zero" baseline moves with it -- judge
# the branch by ADE against the zero-motion baseline, never by the loss value.
# 0.0740 pairs with manifest_sandwich_labeled_20260921.json + top300 + mvs16;
# the legacy manifest needs 0.0432.  Override per run with
# POINTFLOW_DISPLACEMENT_SCALE (the scale follows the selection, so dense
# experiments must re-scan tools/scan_pointflow_selection.py first).
_scale_env = os.environ.get("POINTFLOW_DISPLACEMENT_SCALE")
model_config["rectified_flow_training_config"]["pointflow_displacement_scale"] = (
    float(_scale_env) if _scale_env else 0.0740
)

# PointFlow must see "clean video + noisy points" in training: the conditional-fit
# evaluation (and any deployment observing the current video) presents exactly that
# combination, but the legacy shared schedule always pairs noisy points with equally
# noisy video.  Measured on the scale-0.0432 per-point run (2026-09-20): eval UniPC-4
# ADE/zero rose monotonically (1.0-2.6 @step100 -> 1.5-6.8 @step2500) while the
# training one-step ratio fell to ~0.3 -- a train/eval condition mismatch, not a
# fitting failure.  The independent draw keeps the sigma marginal identical and only
# removes the coupling.  Model shapes are unchanged, so runs resume across this flag.
# See docs/pointflow_per_point_tokens_20260919.md.
model_config["rectified_flow_training_config"]["independent_pointflow_schedule"] = True

model_config["vlm_config"]["tokenizer"] = L(build_processor_lazy)(tokenizer_type="${oc.env:EDGE_DROID_MODEL_PATH}")
model_config["vlm_config"]["pretrained_weights"].update(
    enabled=False,
    backbone_path="",
    credentials_path="",
    enable_gcs_patch_in_boto3=False,
)
action_policy_singlerighthand_edge["model"]["config"] = model_config

# Edge uses an additional K-normalization parameter on the generation pathway.
optimizer_keys = action_policy_singlerighthand_edge["optimizer"]["keys_to_select"]
if "k_norm_und_for_gen" not in optimizer_keys:
    optimizer_keys.append("k_norm_und_for_gen")

# PointFlow trains end to end: the Sonata geometry encoder, the point codec and
# its token/sigma embeddings all live under ``net.pointflow_branch``.  The
# optimizer allowlist freezes any parameter whose name matches no key here, so
# without this entry the whole branch receives no gradient.
if "pointflow_branch" not in optimizer_keys:
    optimizer_keys.append("pointflow_branch")

# The codec is initialized from scratch while every other trained tensor comes
# from a pretrained checkpoint, so the base LR (2e-5) only moves its weights by
# ~lr per step and the branch needs tens of thousands of steps to emit the
# unit-scale velocity.  Give it the same treatment the Nano recipe gives its
# action heads, scaled for that codec being new rather than fine-tuned.
# Lower this to 10.0 if Video Loss starts degrading: the PointFlow gradient also
# flows into the shared trunk.
action_policy_singlerighthand_edge["optimizer"]["lr_multipliers"]["pointflow_branch.codec"] = 25.0

# Edge-DROID already contains trained action heads. Preserve all of them when
# loading the converted DCP; only EMA is warm-started from the regular network.
action_policy_singlerighthand_edge["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
action_policy_singlerighthand_edge["checkpoint"]["enable_gcs_patch_in_boto3"] = False

action_policy_singlerighthand_edge["trainer"]["callbacks"]["gpu_video_augmentation"] = L(GPUVideoAugmentation)(
    enabled="${oc.env:COSMOS_GPU_VIDEO_AUGMENTATION,false}",
    crop_ratio=0.95,
    brightness=0.3,
    contrast=0.4,
    saturation=0.5,
    hue=0.08,
    chunk_size=8,
)
action_policy_singlerighthand_edge["trainer"]["callbacks"]["pointflow_eval"] = L(PointFlowEvalCallback)(
    every_n="${oc.env:POINTFLOW_EVAL_EVERY,1}",
    max_points=512,
    # 4 steps, the same solver (UniPC) and the same shift as the video/action path --
    # the point tokens are denoised inside that loop in the deployed system, so the
    # diagnostic must not integrate them differently.  comparison_steps gives a second
    # UniPC trajectory to see whether the step count is even the binding constraint.
    max_cases=2,
    sampling_steps=4,
    comparison_steps=16,
    seed=42,
    sampler="${oc.env:POINTFLOW_EVAL_SAMPLER,unipc}",
    comparison_sampler="${oc.env:POINTFLOW_EVAL_COMPARISON_SAMPLER,euler}",
    # Diagnostic: sample each case again with one conditioning channel neutralized.
    # The eval hands the model the ground-truth future video/action, so a prediction
    # that does not move when a channel is blanked is not reading it.  Modes:
    # action (zero GT action), video (black out all frames), first_frame (freeze the
    # window to frame 0 -- tests whether FUTURE video frames are read at all; in wam
    # training they are noisy targets, never clean context).
    ablate_action="${oc.env:POINTFLOW_ABLATE_ACTION,false}",
    ablate_modes="${oc.env:POINTFLOW_ABLATE_MODES,}",
    # Route B: also evaluate the deployment-shaped joint rollout (point tokens
    # denoise inside the main video/action loop; first frame + state action clean,
    # future video/action/point jointly sampled).  Writes <case>_joint/ outputs.
    joint_rollout="${oc.env:POINTFLOW_EVAL_JOINT,false}",
    # Diagnostic: sweep the one-step clean estimate x0_hat = x_sigma - sigma*v_hat over
    # a fixed sigma grid and log the ADE at each.  The sampled trajectory ends at
    # sigma = 0, so this shows where the direct estimate breaks -- one number at the
    # training sigma cannot.
    sigma_scan="${oc.env:POINTFLOW_SIGMA_SCAN,false}",
)
action_policy_singlerighthand_edge["trainer"].update(
    run_validation=True,
    # To validate at (re)start — including on resume — pass a tail override:
    # EXTRA_TAIL_OVERRIDES="trainer.run_validation_on_start=true"
    run_validation_on_start=False,
    validation_iter=100,
    max_val_iter=1,
)

action_policy_singlerighthand_edge["dataloader_train"]["dataset_name"] = "action_singlerighthand"
dataloader = action_policy_singlerighthand_edge["dataloader_train"]["dataloader"]
dataloader.update(
    batch_size=2,
    # Keep video/PointFlow decoding off the training critical path.  Two
    # persistent workers overlap CPU decode and host-to-device copies with
    # the GPU step; validation remains single-process below.
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    # Two batches in flight per worker (16 samples) already covers ~70 steps of
    # consumption.  The earlier value of 4 kept 32 point-tracking windows
    # resident, and the trainer's per-step allocation cost on this node is
    # dominated by page reclaim rather than by arithmetic: a 6 MB tensor
    # concatenation measures 2.6 ms on an idle host and 230 ms there, with the
    # process on-CPU the whole time (wall == cpu).
    prefetch_factor=2,
    # Each worker otherwise opens its own OpenCV/FFmpeg/torch thread pools and
    # oversubscribes this rank's CPU partition, which deschedules the trainer
    # during its per-step data marshalling (measured: 2.5 ms of work stretching
    # to 265 ms of wall clock, with the GPU idle).
    worker_init_fn=limit_dataloader_worker_threads,
)
dataloader["datasets"] = {
    "singlerighthand": {
        "ratio": 1,
        "dataset": L(get_action_singlerighthand_raw_sft_dataset)(
            root="${oc.env:SINGLERIGHTHAND_RAW_ROOT}",
            cache_root="${oc.env:SINGLERIGHTHAND_CACHE_ROOT}",
            fps=15.0,
            chunk_length=32,
            split="train",
            split_seed=42,
            split_val_ratio=0.2,
            sample_stride=1,
            mode="wam",
            use_state=True,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            shuffle_block_size=256,
            use_image_augmentation=False,
            use_precomputed_video="${oc.env:SINGLERIGHTHAND_USE_VIDEO_CACHE,false}",
            # Per-window VAE latents: each window's own encode, bit-identical to a
            # cache-less run. Preferred over vae_latent_root below, which stores a
            # whole-episode encode and resamples onto the 15 Hz lattice (measured to
            # differ by about one frame of window shift).
            vae_window_latent_root="${oc.env:SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT,}",
            vae_latent_root="${oc.env:SINGLERIGHTHAND_VAE_LATENT_ROOT,}",
            video_decoder="${oc.env:SINGLERIGHTHAND_VIDEO_DECODER,opencv}",
            episode_allowlist="${oc.env:POINTFLOW_EPISODE_ALLOWLIST,}",
            pointflow_manifest="${oc.env:POINTFLOW_MANIFEST,/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft/pointflow_outputs/task5/mixed_manifest.json}",
            pointflow_max_points=8192,
            pointflow_voxel_size=0.02,
            pointflow_seed=42,
            # GT-ranked restriction to the most-moving voxels. Diagnostic only: the
            # ranking reads future labels, so it cannot be reproduced at inference.
            # 0 keeps the full cloud.
            pointflow_select_motion_fraction="${oc.env:POINTFLOW_SELECT_MOTION_FRACTION,0}",
            # Keep exactly N points by the same ground-truth motion ranking, with NO
            # floor (the fraction above floors at 3).  Set to 1 to collapse the cloud to
            # a single point, which makes "one cluster token has to describe many points"
            # disappear and leaves the rest of the pipeline testable on its own.  Takes
            # precedence over the fraction.  0 disables it.
            pointflow_select_top_n="${oc.env:POINTFLOW_SELECT_TOP_N,0}",
            # Guard for every motion ranking: a point whose 2 cm voxel holds fewer than
            # this many points is ranked last.  A lost track is always alone -- measured
            # on this data, the ten fastest points of a window all sit in single-member
            # voxels and "move" up to 1.6 m, while the fastest point in a voxel of three
            # or more moves 0.15 m.  0 = off.
            pointflow_min_voxel_members="${oc.env:POINTFLOW_MIN_VOXEL_MEMBERS,0}",
            # Supervise only N points, taken from the most-moving voxel that satisfies
            # the guard, and mask every other point out of the loss.  The cloud is NOT
            # reduced, so the geometry encoder still sees a full surface: this separates
            # "the decode path cannot emit one point's motion" from "the geometry input
            # was degenerate".  0 supervises everything.
            pointflow_supervise_cluster_n="${oc.env:POINTFLOW_SUPERVISE_CLUSTER_N,0}",
            # Semantic region restriction for anchor candidates, read from the labeled
            # delivery's region_labels.npy (1=fingertip, 2=hand, 3+=objects; sandwich
            # uses 3=lettuce, 4=bread/cheese).  Anchor-time information only, so unlike
            # the motion ranking this stays a deployable conditioning rule.  Comma-
            # separated; empty keeps every region.
            pointflow_select_regions="${oc.env:POINTFLOW_SELECT_REGIONS,}",
            # Stratified semantic quotas for the top-N budget: "2:0.40,3:0.45,4:0.15"
            # splits N across hand/object/surface (best-ranked within each region,
            # shortfall refilled globally).  Requires POINTFLOW_SELECT_TOP_N > 0;
            # empty keeps the flat global ranking.
            pointflow_select_region_quotas="${oc.env:POINTFLOW_SELECT_REGION_QUOTAS,}",
            # Demote points whose future label is valid for fewer than this many steps
            # in motion-ranked selection: a track that dies mid-window (late-stage
            # points leave the frame) cannot anchor a full-window prediction.  Reads
            # labels, so it is training-time only.  0 = off.
            pointflow_select_min_valid_steps="${oc.env:POINTFLOW_SELECT_MIN_VALID_STEPS,0}",
            # Demote phantom-drift points (uv frozen but 3D drifts; a depth-estimation
            # failure, not real motion) to the bottom of the motion ranking.
            # true/false; default off.
            pointflow_select_phantom_guard="${oc.env:POINTFLOW_SELECT_PHANTOM_GUARD,false}",
            # Precomputed per-window selection cache (tools/build_pointflow_window_cache.py):
            # replaces the ~200 MB full-track reads and the whole selection pipeline
            # per window with a few-hundred-KB npz entry.  Empty = compute online.
            pointflow_window_cache_root="${oc.env:POINTFLOW_WINDOW_CACHE_ROOT,}",
            video_temporal_downsample=4,
            viewpoint="concat_view",
            resolution="480",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            format_prompt_as_json=True,
        ),
    }
}
val_dataloader = copy.deepcopy(action_policy_singlerighthand_edge["dataloader_train"])
val_dataloader["dataset_name"] = "action_singlerighthand_val"
val_dataloader["max_samples_per_batch"] = 2
val_dataloader["dataloader"].update(batch_size=1, num_workers=0, persistent_workers=False, prefetch_factor=None)
val_dataloader["dataloader"]["datasets"]["singlerighthand"]["dataset"]["split"] = "val"
val_dataloader["dataloader"]["datasets"]["singlerighthand"]["dataset"]["iterable_shuffle"] = False
action_policy_singlerighthand_edge["dataloader_val"] = val_dataloader

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_singlerighthand_edge",
    node=action_policy_singlerighthand_edge,
)


__all__ = ["action_policy_singlerighthand_edge"]
