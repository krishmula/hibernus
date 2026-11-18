"""The application checkpoint: the state the training loop owns, saved with torch.save.

`save` runs at an update boundary, so there are no partial gradients and the next batch has not
been read. `load` puts a fresh trainer back in that state. A trainer here is anything with the
attributes of `hibernus.workload.Trainer`.

The file is a pickle, so load only files the Supervisor wrote. The RNG states are not plain
tensors, which is why `torch.load` runs with `weights_only=False`.
"""

import os
import random
import tempfile

import numpy as np
import torch

from hibernus.workload import Progress

SCHEMA_VERSION = 1


def save(path, trainer):
    """Writes the checkpoint to `path`. It appears whole or not at all."""
    state = {
        "schema_version": SCHEMA_VERSION,
        "model": trainer.model.state_dict(),  # weights and buffers
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict() if trainer.scaler is not None else None,
        "progress": {
            "completed_updates": trainer.progress.completed_updates,
            "epoch": trainer.progress.epoch,
            "position": trainer.progress.position,
        },
        "sampler": {"seed": trainer.sampler.seed, "batch_size": trainer.sampler.batch_size},
        "rng": _capture_rng(),
        "dataset_version": trainer.dataset.version,
    }
    folder = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(dir=folder, prefix=".app-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def load(path, trainer, *, expected_step=None):
    """Restores `trainer` from `path`. It checks everything first, so a refusal changes nothing."""
    state = torch.load(path, weights_only=False)
    _check(state, trainer)
    if expected_step is not None and state["progress"]["completed_updates"] != expected_step:
        raise ValueError(
            f"{path} holds step {state['progress']['completed_updates']}, "
            f"but the manifest says step {expected_step}"
        )

    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    if trainer.scaler is not None:
        trainer.scaler.load_state_dict(state["scaler"])
    trainer.progress = Progress(**state["progress"])
    trainer.reset_data_position()
    _restore_rng(state["rng"])


def _check(state, trainer):
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unknown checkpoint schema version {state.get('schema_version')}, want {SCHEMA_VERSION}"
        )
    if state["dataset_version"] != trainer.dataset.version:
        raise ValueError("the checkpoint was saved with a different dataset")
    if state["sampler"]["seed"] != trainer.sampler.seed:
        raise ValueError(
            f"the checkpoint has shuffle seed {state['sampler']['seed']}, "
            f"but the trainer has {trainer.sampler.seed}"
        )
    if state["sampler"].get("batch_size") != trainer.sampler.batch_size:
        raise ValueError("the checkpoint and the trainer have a different batch size")
    if (state["scaler"] is None) != (trainer.scaler is None):
        raise ValueError("the checkpoint and the trainer disagree about using a GradScaler")


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
