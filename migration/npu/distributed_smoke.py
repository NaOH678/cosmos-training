"""torchrun --nproc_per_node=2 this_file --device npu --output-dir NEW_DIR.

Exercises real LiteGen deferred-meta FSDP2, FP32 master AdamW, RNG and DCP.
"""

import argparse
import json
import os
from pathlib import Path

import litegen
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from litegen.device import get_device
from litegen.optim.master_shadow import MasterShadowAdamW
from litegen.utils.config import MainConfig
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).tanh()
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "npu"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["LITEGEN_DEVICE"] = args.device
    cfg = MainConfig.from_dict(
        {
            "runtime": {"device": args.device},
            "global_batch_size": int(os.environ["WORLD_SIZE"]) * 2,
            "optimization": {
                "parallel": {
                    "strategy": "fsdp2",
                    "fsdp2": {"wrap_strategy": "manual", "wrap_paths": ["layers.0", "layers.1"]},
                }
            },
        }
    )
    gen = litegen.initialize(cfg)
    try:
        with torch.device("meta"):
            model = Network()
        model = gen.prepare(model, materialize_meta=False)
        model.to(dtype=torch.bfloat16).to_empty(device=get_device())
        torch.manual_seed(42)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        opt = MasterShadowAdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
        gen.register(opt, scheduler)
        for step in range(2):
            torch.manual_seed(1000 + step + dist.get_rank())
            x = torch.randn(2, 16, device=get_device(), dtype=torch.bfloat16)
            loss = model(x).float().square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
        from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict

        state = {
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            f"rng_{dist.get_rank()}": get_rand_state_dict(),
        }
        expected = {k: v.to_local().clone() if hasattr(v, "to_local") else v.clone() for k, v in state["model"].items()}
        if dist.get_rank() == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        dcp.save(state, checkpoint_id=args.output_dir / "checkpoint")
        expected_rng = torch.randn(8, device=get_device())
        expected_optim = {
            id(parameter): {
                name: tensor.to_local().clone() if hasattr(tensor, "to_local") else tensor.clone()
                for name, tensor in opt.state[parameter].items()
            }
            for parameter in model.parameters()
        }
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
                for tensor in opt.state[parameter].values():
                    tensor.zero_()
        scheduler.last_epoch = 99
        dcp.load(state, checkpoint_id=args.output_dir / "checkpoint")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optim"])
        scheduler.load_state_dict(state["scheduler"])
        set_rand_state_dict(state[f"rng_{dist.get_rank()}"])
        torch.testing.assert_close(torch.randn(8, device=get_device()), expected_rng, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value.to_local() if hasattr(value, "to_local") else value, expected[key], rtol=0, atol=0
            )
        for parameter in model.parameters():
            assert opt.state[parameter]["master"].dtype == torch.float32
            for name, tensor in opt.state[parameter].items():
                actual = tensor.to_local() if hasattr(tensor, "to_local") else tensor
                torch.testing.assert_close(actual.cpu(), expected_optim[id(parameter)][name].cpu(), rtol=0, atol=0)
        assert scheduler.last_epoch == 2
        # A step after restore must exercise the recovered optimizer state.
        loss = model(torch.ones(2, 16, device=get_device(), dtype=torch.bfloat16)).float().square().mean()
        loss.backward()
        opt.step()
        if dist.get_rank() == 0:
            (args.output_dir / "result.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "device": args.device,
                        "world_size": dist.get_world_size(),
                        "restored_step": 2,
                        "step_after_restore": True,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        gen.shutdown()


if __name__ == "__main__":
    main()
