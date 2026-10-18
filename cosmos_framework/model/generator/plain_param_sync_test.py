"""2-process gloo check of OmniMoTModel._sync_plain_param_grads (broadcast once, then grad averaging)."""
import os
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(3, 4)
        self.b = torch.nn.Linear(4, 2)
        self.unused = torch.nn.Linear(2, 2)  # never receives a gradient on any rank
        self.rank0_only = torch.nn.Linear(3, 2)  # receives a gradient on rank 0 only


def _worker(rank, world, init_file, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(100 + rank)  # different init per rank, like an unsynchronised branch
    net, ema = Net(), Net()
    stub = SimpleNamespace(config=SimpleNamespace(sync_plain_param_grads=True), net=net, net_ema=ema, parallel_dims=None)
    stub._plain_param_group = lambda: None
    for step in range(2):
        net.zero_grad(set_to_none=True)
        x = torch.full((5, 3), float(rank + 1 + step))
        loss = net.b(net.a(x)).sum()
        if rank == 0:
            loss = loss + net.rank0_only(x).sum()
        loss.backward()
        local = net.a.weight.grad.clone()
        local_r0 = net.rank0_only.weight.grad.clone() if rank == 0 else torch.zeros_like(net.rank0_only.weight)
        OmniMoTModel._sync_plain_param_grads(stub)
        if step == 1:
            gathered = [torch.zeros_like(local) for _ in range(world)]
            dist.all_gather(gathered, local)
            g_r0 = [torch.zeros_like(local_r0) for _ in range(world)]
            dist.all_gather(g_r0, local_r0)
            out[rank] = {
                "w": net.a.weight.detach().clone(),
                "ema": ema.a.weight.detach().clone(),
                "g": net.a.weight.grad.clone(),
                "mean": torch.stack(gathered).mean(0),
                "local": local,
                "unused_none": net.unused.weight.grad is None,
                "r0": net.rank0_only.weight.grad.clone(),
                "r0_mean": torch.stack(g_r0).mean(0),
            }
    dist.destroy_process_group()


def test_sync_plain_param_grads_two_ranks():
    with tempfile.TemporaryDirectory() as d:
        manager = mp.Manager()
        out = manager.dict()
        mp.spawn(_worker, args=(2, os.path.join(d, "init"), out), nprocs=2, join=True)
        a, b = out[0], out[1]
        torch.testing.assert_close(a["w"], b["w"])
        torch.testing.assert_close(a["ema"], b["ema"])
        torch.testing.assert_close(a["g"], b["g"])
        torch.testing.assert_close(a["g"], a["mean"])
        assert not torch.allclose(a["local"], b["local"])  # the ranks really saw different data
        assert a["unused_none"] and b["unused_none"]  # no grad anywhere -> left untouched (no weight decay)
        torch.testing.assert_close(a["r0"], b["r0"])
        torch.testing.assert_close(a["r0"], a["r0_mean"])  # grad on one rank only: averaged with zeros
