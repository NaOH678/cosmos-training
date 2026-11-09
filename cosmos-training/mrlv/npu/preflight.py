"""Record the target stack and exercise the selected NPU without CPU fallback."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-devices", type=int)
    args = parser.parse_args()
    report = {
        "status": "failed",
        "target_chip": "Ascend 910B",
        "target_cpu": "Kunpeng 920-5250",
        "requested_image": os.environ.get("NPU_JOB_IMAGE"),
        "architecture": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch
        from litegen import device

        report["torch"] = torch.__version__
        report["device"] = device.device_type()
        if device.device_type() != "npu":
            raise RuntimeError("Set LITEGEN_DEVICE=npu; this is an NPU acceptance command")
        reason = device.get_backend("npu").unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        report["torch_npu"] = importlib.metadata.version("torch-npu")
        report["device_count"] = device.get_backend("npu").device_count()
        if args.expected_devices is not None and report["device_count"] != args.expected_devices:
            raise RuntimeError(f"Expected {args.expected_devices} visible NPUs; found {report['device_count']}")
        report["devices"] = []
        for index in range(report["device_count"]):
            device.set_device(index)
            item = {"index": index, "name": device.memory_module().get_device_name(), "status": "running"}
            report["devices"].append(item)
            if "910b" not in item["name"].lower():
                raise RuntimeError(f"Expected a real 910B at device {index}: {item['name']}")
            item["bf16"] = device.get_backend("npu").supports_bf16()
            if not item["bf16"]:
                raise RuntimeError(f"WAM requires BF16 on device {index}")
            sample = torch.ones(16, 16, device=device.get_device(), dtype=torch.bfloat16)
            assert sample.sum().item() == 256
            product = sample @ sample
            assert torch.all(product == 16).item()
            device.synchronize()
            item.update(status="passed", bf16_sum=256, bf16_matmul=True, synchronized=True)
        device.set_device(0)
        report["device_name"] = report["devices"][0]["name"]
        report["bf16"] = all(item["bf16"] for item in report["devices"])
        for command in (["npu-smi", "info"], ["uname", "-a"]):
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
            report[" ".join(command)] = result.stdout
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
