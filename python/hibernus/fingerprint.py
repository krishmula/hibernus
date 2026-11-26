"""The fingerprint of this host and of the Python environment that runs this module.

The training hook sends it at initialize, and the Supervisor writes it into each manifest. On the
restore host, `hib restore` runs `python -m hibernus.fingerprint` with the same interpreter and
refuses the restore if any value differs. One collector on both hosts means the two fingerprints
always come out in the same format.

A value the host lacks is "none", such as the driver on a host without an NVIDIA GPU.
"""

import glob
import json
import os
import platform
import re

NONE = "none"


def collect(proc="/proc/driver/nvidia"):
    """Returns the six values as strings. It reads the GPU from the driver's files, not through
    CUDA, so it never starts CUDA in the training process."""
    fingerprint = {"gpu": _gpu(proc), "driver": _driver(proc)}
    fingerprint.update(_torch())
    fingerprint["kernel"] = platform.release() or NONE
    return fingerprint


def _gpu(proc):
    models = []
    for path in sorted(glob.glob(os.path.join(proc, "gpus", "*", "information"))):
        with open(path) as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "Model":
                    models.append(value.strip())
    return ", ".join(models) or NONE


def _driver(proc):
    # Such as "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  550.54.15  Release Build ...".
    try:
        with open(os.path.join(proc, "version")) as f:
            text = f.read()
    except FileNotFoundError:
        return NONE
    match = re.search(r"Kernel Module.*?\s(\d+\.\d+(?:\.\d+)?)\s", text)
    if match is None:
        raise RuntimeError(f"no driver version in {os.path.join(proc, 'version')}: {text!r}")
    return match.group(1)


def _torch():
    try:
        import torch
    except ImportError:
        return {"cuda": NONE, "cudnn": NONE, "pytorch": NONE}
    # Ask the cuDNN library directly. torch.backends.cudnn.version() and is_available() open the
    # GPU in the training process, and a process with the GPU open needs cuda-checkpoint to dump.
    try:
        from torch._C import _cudnn

        cudnn = _cudnn.getVersionInt()
    except (ImportError, AttributeError):  # a PyTorch built without cuDNN
        cudnn = None
    return {
        "cuda": torch.version.cuda or NONE,
        "cudnn": str(cudnn) if cudnn else NONE,
        "pytorch": torch.__version__,
    }


if __name__ == "__main__":
    print(json.dumps(collect()))
