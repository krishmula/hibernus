"""The fingerprint module reads the GPU and driver from the NVIDIA driver's files, and the rest from
PyTorch and the kernel. A fake driver folder stands in for /proc/driver/nvidia."""

import json
import platform
import subprocess
import sys

import pytest
import torch

from hibernus import fingerprint

KEYS = {"gpu", "driver", "cuda", "cudnn", "pytorch", "kernel"}


def fake_driver(tmp_path, version_line, models):
    (tmp_path / "version").write_text(version_line + "\nGCC version:  gcc version 12.3.0\n")
    for i, model in enumerate(models):
        gpu = tmp_path / "gpus" / f"0000:00:1{i}.0"
        gpu.mkdir(parents=True)
        (gpu / "information").write_text(f"Model: \t\t {model}\nIRQ:   \t\t 33\n")
    return str(tmp_path)


@pytest.mark.parametrize(
    "version_line",
    [
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  550.54.15  Release Build  (dvs-builder@U16-I2-C03-12-4)",
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.15  Tue Mar  5 22:23:56 UTC 2024",
    ],
)
def test_a_gpu_host_reports_its_model_and_driver(tmp_path, version_line):
    found = fingerprint.collect(fake_driver(tmp_path, version_line, ["NVIDIA A10G"]))

    assert found["gpu"] == "NVIDIA A10G"
    assert found["driver"] == "550.54.15"


def test_a_host_without_the_nvidia_driver_reports_none(tmp_path):
    found = fingerprint.collect(str(tmp_path / "missing"))

    assert found["gpu"] == "none"
    assert found["driver"] == "none"


def test_the_rest_comes_from_pytorch_and_the_kernel(tmp_path):
    found = fingerprint.collect(str(tmp_path))

    assert set(found) == KEYS
    assert all(isinstance(value, str) and value for value in found.values())
    assert found["pytorch"] == torch.__version__
    assert found["cuda"] == (torch.version.cuda or "none")
    assert found["kernel"] == platform.release()


def test_running_the_module_prints_the_same_fingerprint():
    out = subprocess.run([sys.executable, "-m", "hibernus.fingerprint"], capture_output=True, text=True, check=True)

    assert json.loads(out.stdout) == fingerprint.collect()
