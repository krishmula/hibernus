"""The application checkpoint: the state the training loop owns, written with torch.save.

`save` runs at an update boundary, so there are no partial gradients and the next batch has not
been read. `load` puts the loop's objects back in that state. The loop names its objects in a
dict, such as {"model": model, "optimizer": optimizer, "scheduler": scheduler}. Each one needs
state_dict() and load_state_dict(), as every torch module, optimizer, and scheduler has. The file
also holds the step and the Python, NumPy, and torch RNG states, so the loop never passes those.
An optional config holds plain values that must match on restore, such as the shuffle seed.

The file is a pickle, so load only files the Supervisor wrote. The RNG states are not plain
tensors, which is why `torch.load` runs with `weights_only=False`.
"""

import os
import random
import tempfile

import numpy as np
import torch

SCHEMA_VERSION = 2


def save(path, step, state, config=None):
    """Writes the checkpoint to `path`. It appears whole or not at all."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "config": config or {},
        "state": {name: obj.state_dict() for name, obj in state.items()},
        "rng": _capture_rng(),
    }
    folder = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(dir=folder, prefix=".app-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def load(path, state, config=None, *, expected_step=None):
    """Loads `path` into the objects in `state` and returns the step it holds.

    It checks the schema, the step, the config, and the names before it changes anything. Each
    object's own load_state_dict then makes its own checks.
    """
    payload = torch.load(path, weights_only=False)
    _check(path, payload, state, config or {}, expected_step)
    for name, obj in state.items():
        obj.load_state_dict(payload["state"][name])
    _restore_rng(payload["rng"])
    return payload["step"]


def _check(path, payload, state, config, expected_step):
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unknown checkpoint schema version {payload.get('schema_version')}, "
            f"want {SCHEMA_VERSION}"
        )
    if expected_step is not None and payload["step"] != expected_step:
        raise ValueError(
            f"{path} holds step {payload['step']}, but the manifest says step {expected_step}"
        )
    if payload["config"] != config:
        changed = sorted(
            key for key in set(payload["config"]) | set(config)
            if payload["config"].get(key) != config.get(key)
        )
        details = ", ".join(
            f"{key} was {payload['config'].get(key)!r}, now {config.get(key)!r}" for key in changed
        )
        raise ValueError(f"{path} was written with a different config: {details}")
    if set(payload["state"]) != set(state):
        raise ValueError(
            f"{path} holds {sorted(payload['state'])}, but the training loop passed {sorted(state)}"
        )


def _capture_rng():
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng


def _restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
