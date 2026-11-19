"""DataLoader worker processes (num_workers > 0) must not change what training sees.

Opening a DataLoader iterator starts fresh workers, and each worker seeds its own RNG. A restored
run opens a new iterator, so random augmentation drawn from a worker's RNG would differ from a run
that never stopped. The workload derives each sample's augmentation from (seed, epoch, index)
instead, so the workers hold no RNG state that a checkpoint would have to save.

Each iterator that starts workers costs a few seconds, so these tests keep their runs short.
"""

import multiprocessing
import socket
import sys
import time

import pytest
import torch

from hibernus import app_checkpoint
from hibernus.workload import build_trainer

WORKERS = 2


def run(trainer, updates):
    return [trainer.run_update() for _ in range(updates)]


def new_children(before):
    """Child processes started since `before`. DataLoader workers are children of this process."""
    return [p for p in multiprocessing.active_children() if p not in before]


def wait_until_none(before, seconds=10):
    deadline = time.time() + seconds
    while new_children(before) and time.time() < deadline:
        time.sleep(0.05)
    return new_children(before)


def test_a_sample_is_augmented_the_same_way_every_time_it_is_read_in_an_epoch():
    dataset = build_trainer(augment=True).dataset

    _, first, _ = dataset[(0, 5)]
    _, again, _ = dataset[(0, 5)]

    assert torch.equal(first, again)


def test_a_sample_is_augmented_differently_in_the_next_epoch():
    dataset = build_trainer(augment=True).dataset

    _, epoch_0, _ = dataset[(0, 5)]
    _, epoch_1, _ = dataset[(1, 5)]

    assert not torch.equal(epoch_0, epoch_1)


def test_augmentation_changes_the_image_but_not_the_label():
    plain, augmented = build_trainer().dataset, build_trainer(augment=True).dataset

    _, clean_image, clean_label = plain[(0, 5)]
    _, noisy_image, noisy_label = augmented[(0, 5)]

    assert not torch.equal(clean_image, noisy_image)
    assert clean_label == noisy_label


def test_augmentation_is_part_of_the_dataset_version():
    # A checkpoint saved with augmentation must refuse to load into a trainer without it.
    assert build_trainer(augment=True).dataset.version != build_trainer().dataset.version


def test_workers_give_the_same_run_as_no_workers():
    without = run(build_trainer(augment=True), 5)  # 5 updates cross an epoch boundary
    with_workers = run(build_trainer(augment=True, num_workers=WORKERS), 5)

    assert with_workers == without  # same batches, same augmentation, same losses


def test_resume_with_workers_matches_a_run_that_never_stopped(tmp_path):
    path = tmp_path / "app.pt"

    reference = build_trainer(augment=True, num_workers=WORKERS)
    run(reference, 2)
    app_checkpoint.save(path, reference)
    expected = run(reference, 3)  # the reference keeps its open iterator and its warm workers

    # The restored trainer starts new workers from the saved cursor.
    restored = build_trainer(init_seed=999, augment=True, num_workers=WORKERS)
    app_checkpoint.load(path, restored)
    actual = run(restored, 3)

    assert actual == expected


def test_a_checkpoint_made_with_workers_resumes_without_them(tmp_path):
    path = tmp_path / "app.pt"

    reference = build_trainer(augment=True, num_workers=WORKERS)
    run(reference, 2)
    app_checkpoint.save(path, reference)
    expected = run(reference, 3)

    restored = build_trainer(init_seed=999, augment=True, num_workers=0)
    app_checkpoint.load(path, restored)

    assert run(restored, 3) == expected


# A process snapshot (CRIU) dumps the training process. The workers are not part of what we dump:
# the hook closes them first, so the dump holds one process, and the restored trainer opens new
# workers from its saved cursor.


def test_close_data_stops_every_worker_process():
    before = set(multiprocessing.active_children())
    trainer = build_trainer(augment=True, num_workers=WORKERS)
    run(trainer, 1)
    assert len(new_children(before)) == WORKERS  # the iterator started the workers

    trainer.close_data()

    assert wait_until_none(before) == []


def test_close_data_without_an_open_iterator_does_nothing():
    trainer = build_trainer(num_workers=WORKERS)

    trainer.close_data()  # nothing is open yet
    trainer.close_data()  # and closing twice is fine

    assert run(trainer, 1)[0].update == 1


def test_loading_a_checkpoint_stops_the_workers_of_the_trainer_it_loads_into(tmp_path):
    path = tmp_path / "app.pt"
    app_checkpoint.save(path, build_trainer(augment=True))
    before = set(multiprocessing.active_children())
    trainer = build_trainer(augment=True, num_workers=WORKERS)
    run(trainer, 1)

    app_checkpoint.load(path, trainer)

    assert wait_until_none(before) == []


def test_training_continues_exactly_after_the_workers_were_closed():
    straight = run(build_trainer(augment=True, num_workers=WORKERS), 5)

    trainer = build_trainer(augment=True, num_workers=WORKERS)
    first = run(trainer, 2)
    trainer.close_data()  # what the hook does before a process snapshot
    rest = run(trainer, 3)  # new workers, started from the cursor

    assert first + rest == straight


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="workers inherit the socket by fork, on Linux")
def test_forked_workers_hold_the_hook_socket_open_until_they_are_closed():
    # The Supervisor starts the dump when the hook's connection closes. A worker forked after the
    # connection opened holds a copy of it, so closing the hook's end is not enough while workers
    # run. This is why the hook stops the workers before a process snapshot.
    ours, supervisor_end = socket.socketpair()
    trainer = build_trainer(num_workers=WORKERS)
    run(trainer, 1)  # starts the workers, after the socket exists

    ours.close()
    supervisor_end.settimeout(0.5)
    with pytest.raises(socket.timeout):
        supervisor_end.recv(1)  # no end-of-file: a worker still holds the other end

    trainer.close_data()
    supervisor_end.settimeout(5)
    assert supervisor_end.recv(1) == b""  # now the close reaches the Supervisor
