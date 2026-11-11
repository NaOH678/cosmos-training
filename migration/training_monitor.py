"""TensorBoard curves and performance measurements for the WAM task launcher."""

import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from litegen import device as accelerator

from cosmos_framework.utils import distributed, log, misc
from cosmos_framework.utils.callback import Callback


def summarize_windows(windows):
    """Weight windows by steps/time; omit warmup and interrupted windows."""
    valid = [row for row in windows if not row["excluded_reason"]]
    steps = sum(row["steps"] for row in valid)
    seconds = sum(row["seconds"] for row in valid)
    samples = sum(row["samples"] for row in valid)
    return {
        "steady_windows": len(valid),
        "measured_steps": steps,
        "mean_step_seconds": seconds / steps if steps else None,
        "median_window_step_seconds": statistics.median(row["seconds"] / row["steps"] for row in valid)
        if valid
        else None,
        "samples_per_second": samples / seconds if seconds else None,
        "peak_allocated_gib": max((row["peak_allocated_gib"] for row in windows), default=None),
        "peak_reserved_gib": max((row["peak_reserved_gib"] for row in windows), default=None),
    }


class TrainingMonitor(Callback):
    """An optional callback, registered only by migration/launch_wam_task.sh.

    Wall time is synchronized at logging boundaries and reduced with MAX across
    ranks. Windows include data loading, communication and logging overhead.
    Windows containing warmup, validation or checkpoint saving are excluded from
    the steady-state summary. The original W&B callbacks remain active.
    """

    def __init__(self, log_dir, warmup_steps=50):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.writer = None
        self._windows = []
        self._events = set()
        self._window_samples = 0
        self._peak_allocated = 0.0
        self._peak_reserved = 0.0
        self._clock = time.perf_counter

    @staticmethod
    def _sync():
        if accelerator.accelerator_available():
            accelerator.synchronize()

    @staticmethod
    def _reduce(values, operation):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=operation)
        return values

    def on_train_start(self, model, iteration=0):
        self._start_iteration = iteration
        self._last_iteration = iteration
        self._logging_iter = self.config.trainer.logging_iter
        self._output_root = Path(os.environ["IMAGINAIRE_OUTPUT_ROOT"])
        if distributed.is_rank0():
            from torch.utils.tensorboard import SummaryWriter

            self._output_root.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=10,
                purge_step=iteration + 1 if iteration else None,
            )
            log.info(f"TensorBoard log directory: {self.log_dir}")
        self._sync()
        self._window_start = self._clock()

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        if self.writer is not None and (iteration + 1) % self._logging_iter == 0:
            self.writer.add_scalar("optim/lr", scheduler.get_last_lr()[0], iteration + 1)

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        self._window_samples += misc.get_data_batch_size(data_batch)
        if accelerator.accelerator_available():
            # Capture before DeviceMonitor resets PyTorch's peak counters.
            self._peak_allocated = max(self._peak_allocated, accelerator.memory_module().max_memory_allocated() / 2**30)
            self._peak_reserved = max(self._peak_reserved, accelerator.memory_module().max_memory_reserved() / 2**30)

    def on_save_checkpoint_start(self, model, iteration=0):
        self._events.add("checkpoint")

    def on_validation_start(self, model, dataloader_val, iteration=0):
        self._events.add("validation")
        self._val_loss = None

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        count = misc.get_data_batch_size(data_batch)
        values = torch.stack((loss.detach().float().mean() * count, loss.new_tensor(float(count), dtype=torch.float32)))
        self._val_loss = values if self._val_loss is None else self._val_loss + values

    def on_validation_end(self, model, iteration=0):
        if self._val_loss is not None:
            total, count = self._reduce(self._val_loss, dist.ReduceOp.SUM).tolist()
            if self.writer is not None:
                self.writer.add_scalar("val/loss", total / count, iteration)
                self.writer.flush()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration % self._logging_iter and iteration != self.config.trainer.max_iter:
            return
        self._sync()
        now = self._clock()
        seconds = now - self._window_start
        count = misc.get_data_batch_size(data_batch)
        metrics = {"train/loss": loss.detach().float().mean()}
        for key, tag in (
            ("flow_matching_loss_vision", "train/video_loss"),
            ("flow_matching_loss_action", "train/action_loss"),
        ):
            if key in output_batch:
                metrics[tag] = output_batch[key].detach().float().mean()
        totals = torch.stack(
            [value * count for value in metrics.values()]
            + [
                loss.new_tensor(float(count), dtype=torch.float32),
                loss.new_tensor(float(self._window_samples), dtype=torch.float32),
            ]
        )
        totals = self._reduce(totals, dist.ReduceOp.SUM).tolist()
        values = {key: total / totals[-2] for key, total in zip(metrics, totals[:-2])}
        timing = loss.new_tensor(
            [seconds, self._peak_allocated, self._peak_reserved],
            dtype=torch.float64
            if accelerator.get_backend(accelerator.device_type()).reduces_float64
            else torch.float32,
        )
        seconds, allocated, reserved = self._reduce(timing, dist.ReduceOp.MAX).tolist()
        events = set(self._events)
        if self._last_iteration < self._start_iteration + self.warmup_steps:
            events.add("warmup")
        row = {
            "first_step": self._last_iteration + 1,
            "last_step": iteration,
            "steps": iteration - self._last_iteration,
            "seconds": seconds,
            "samples": int(totals[-1]),
            "peak_allocated_gib": allocated,
            "peak_reserved_gib": reserved,
            "excluded_reason": ",".join(sorted(events)),
        }
        if self.writer is not None:
            for key, value in values.items():
                self.writer.add_scalar(key, value, iteration)
            self.writer.add_scalar("perf/observed_step_seconds", seconds / row["steps"], iteration)
            self.writer.add_scalar("perf/peak_allocated_gib", allocated, iteration)
            self.writer.add_scalar("perf/peak_reserved_gib", reserved, iteration)
            if not events:
                self.writer.add_scalar("perf/step_seconds", seconds / row["steps"], iteration)
                self.writer.add_scalar("perf/samples_per_second", row["samples"] / seconds, iteration)
            self.writer.flush()
            self._append_csv("metrics.csv", {"iteration": iteration, **values})
            self._append_csv("performance_windows.csv", row)
            self._windows.append(row)
            self._save_summary()
        self._last_iteration = iteration
        self._window_samples = 0
        self._events.clear()
        self._sync()
        self._window_start = self._clock()

    def _append_csv(self, name, row):
        path = self._output_root / name
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_summary(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_per_gpu = self.config.dataloader_train.dataloader.batch_size
        grad_accum_iter = self.config.trainer.grad_accum_iter
        summary = {
            "segment_start_iteration": self._start_iteration,
            "warmup_steps": self.warmup_steps,
            "window_steps": self._logging_iter,
            "world_size": world_size,
            "precision": self.config.model.config.precision,
            "device": accelerator.device_type(),
            "compile": self.config.model.config.compile.enabled,
            "batch_per_gpu": batch_per_gpu,
            "global_batch": world_size * batch_per_gpu * grad_accum_iter,
            "grad_accum_iter": grad_accum_iter,
            "seed": self.config.trainer.seed,
            "base_checkpoint": os.environ.get("BASE_CHECKPOINT_PATH"),
            "dataset_bundle": os.environ.get("SIM_POINTFK_BUNDLE"),
            "measurement": "max rank wall time; includes I/O, communication and logging; excludes warmup/checkpoint/validation windows",
            **summarize_windows(self._windows),
        }
        summary["status"] = "measured" if summary["measured_steps"] else "insufficient_steady_samples"
        # Non-finite loss is retained in CSV/TensorBoard; benchmark JSON stays valid.
        summary = {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in summary.items()
        }
        target = self._output_root / "performance.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(target)

    def on_train_end(self, model, iteration=0):
        if self.writer is not None:
            self._save_summary()
            self.writer.close()

    def on_app_end(self):
        if self.writer is not None:
            self.writer.close()
