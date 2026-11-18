"""The application checkpoint: save at an update boundary, restore, and continue exactly."""

import os
import random

import numpy as np
import pytest
import torch

from hibernus import app_checkpoint
from hibernus.workload import build_trainer


def run(trainer, updates):
    return [trainer.run_update() for _ in range(updates)]


def fail_save(*args, **kwargs):
    raise OSError("disk full")


def assert_same_state(a, b):
    for (name, pa), (_, pb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.equal(pa, pb), name
    sa, sb = a.optimizer.state_dict(), b.optimizer.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    for key in sa["state"]:
        for field in sa["state"][key]:
            assert torch.equal(
                torch.as_tensor(sa["state"][key][field]),
                torch.as_tensor(sb["state"][key][field]),
            ), (key, field)
    assert a.scheduler.state_dict() == b.scheduler.state_dict()
    assert a.progress == b.progress


def resume_matches(tmp_path, save_after, then, **kwargs):
    """Runs straight through, and again with a save, a fresh trainer, and a load. Compares."""
    path = tmp_path / "app.pt"

    reference = build_trainer(**kwargs)
    run(reference, save_after)
    app_checkpoint.save(path, reference)
    expected = run(reference, then)

    # A fresh trainer starts from different weights and a different RNG stream, so only the
    # checkpoint can make it match.
    restored = build_trainer(init_seed=999, **kwargs)
    torch.manual_seed(12345)
    app_checkpoint.load(path, restored)
    actual = run(restored, then)

    assert actual == expected  # same next batches, same sample IDs, same losses
    assert_same_state(restored, reference)


def test_save_writes_one_file_and_no_temporary_file(tmp_path):
    trainer = build_trainer()
    run(trainer, 2)

    app_checkpoint.save(tmp_path / "app.pt", trainer)

    assert os.listdir(tmp_path) == ["app.pt"]


def test_the_file_holds_every_piece_of_state(tmp_path):
    trainer = build_trainer(use_scaler=True)
    run(trainer, 2)
    app_checkpoint.save(tmp_path / "app.pt", trainer)

    saved = torch.load(tmp_path / "app.pt", weights_only=False)

    assert set(saved) >= {
        "schema_version", "model", "optimizer", "scheduler", "scaler", "progress", "sampler",
        "rng", "dataset_version",
    }
    assert saved["progress"] == {"completed_updates": 2, "epoch": 0, "position": 8}
    assert saved["sampler"]["seed"] == 0
    assert set(saved["rng"]) >= {"python", "numpy", "torch_cpu"}


def test_a_failed_save_leaves_no_file_behind(tmp_path, monkeypatch):
    trainer = build_trainer()

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError):
        app_checkpoint.save(tmp_path / "app.pt", trainer)

    assert os.listdir(tmp_path) == []


def test_a_failed_save_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    trainer = build_trainer()
    app_checkpoint.save(tmp_path / "app.pt", trainer)
    before = (tmp_path / "app.pt").read_bytes()
    run(trainer, 1)

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError):
        app_checkpoint.save(tmp_path / "app.pt", trainer)

    assert (tmp_path / "app.pt").read_bytes() == before


def test_resume_mid_epoch_matches_a_run_that_never_stopped(tmp_path):
    resume_matches(tmp_path, save_after=2, then=3)


def test_resume_at_an_epoch_boundary_starts_the_next_epoch_at_position_zero(tmp_path):
    # 64 samples / batch 4 = 16 batches = 4 updates per epoch, so update 4 ends epoch 0.
    resume_matches(tmp_path, save_after=4, then=5)


def test_resume_with_a_grad_scaler(tmp_path):
    resume_matches(tmp_path, save_after=3, then=3, use_scaler=True)


def test_resume_restores_the_scheduler_learning_rate(tmp_path):
    reference = build_trainer()
    run(reference, 3)
    app_checkpoint.save(tmp_path / "app.pt", reference)
    restored = build_trainer(init_seed=7)
    fresh_lr = restored.optimizer.param_groups[0]["lr"]

    app_checkpoint.load(tmp_path / "app.pt", restored)

    assert restored.scheduler.get_last_lr() == reference.scheduler.get_last_lr()
    assert restored.optimizer.param_groups[0]["lr"] != fresh_lr


def test_resume_restores_python_numpy_and_torch_rng(tmp_path):
    trainer = build_trainer()
    run(trainer, 1)
    app_checkpoint.save(tmp_path / "app.pt", trainer)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    app_checkpoint.load(tmp_path / "app.pt", build_trainer())
    actual = (random.random(), np.random.rand(), torch.rand(1).item())

    assert actual == expected


def test_load_reads_the_same_next_batch_as_the_original_would(tmp_path):
    original = build_trainer()
    run(original, 2)
    app_checkpoint.save(tmp_path / "app.pt", original)
    restored = build_trainer(init_seed=5)
    app_checkpoint.load(tmp_path / "app.pt", restored)

    assert restored.run_update().sample_ids == original.run_update().sample_ids


def test_load_refuses_a_different_dataset(tmp_path):
    app_checkpoint.save(tmp_path / "app.pt", build_trainer(data_seed=0))

    with pytest.raises(ValueError, match="dataset"):
        app_checkpoint.load(tmp_path / "app.pt", build_trainer(data_seed=1))


def test_load_refuses_a_different_shuffle_seed(tmp_path):
    app_checkpoint.save(tmp_path / "app.pt", build_trainer(shuffle_seed=0))

    with pytest.raises(ValueError, match="seed"):
        app_checkpoint.load(tmp_path / "app.pt", build_trainer(shuffle_seed=1))


def test_load_refuses_an_unknown_schema_version(tmp_path):
    app_checkpoint.save(tmp_path / "app.pt", build_trainer())
    saved = torch.load(tmp_path / "app.pt", weights_only=False)
    saved["schema_version"] = 999
    torch.save(saved, tmp_path / "app.pt")

    with pytest.raises(ValueError, match="schema"):
        app_checkpoint.load(tmp_path / "app.pt", build_trainer())


def test_a_failed_load_leaves_the_trainer_untouched(tmp_path):
    app_checkpoint.save(tmp_path / "app.pt", build_trainer(data_seed=0))
    trainer = build_trainer(data_seed=1)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}

    with pytest.raises(ValueError):
        app_checkpoint.load(tmp_path / "app.pt", trainer)

    assert all(torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())
    assert trainer.progress.completed_updates == 0


def test_load_of_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        app_checkpoint.load(tmp_path / "nope.pt", build_trainer())


def test_load_refuses_a_different_batch_size_before_changing_training_state(tmp_path):
    original = build_trainer(batch_size=4)
    run(original, 1)
    app_checkpoint.save(tmp_path / "app.pt", original)
    trainer = build_trainer(init_seed=9, batch_size=8)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="batch size"):
        app_checkpoint.load(tmp_path / "app.pt", trainer)

    assert trainer.progress.completed_updates == 0
    assert all(torch.equal(before[name], value) for name, value in trainer.model.state_dict().items())
    assert torch.equal(rng_before, torch.get_rng_state())
