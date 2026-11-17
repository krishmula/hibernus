"""The test workload: a small image classifier trained with four microbatches per update."""

import torch

from hibernus.workload import MICROBATCHES_PER_UPDATE, build_trainer


def run(trainer, updates):
    return [trainer.run_update() for _ in range(updates)]


def test_one_update_reads_four_microbatches_and_steps_the_optimizer_once():
    trainer = build_trainer()

    record = run(trainer, 1)[0]

    assert MICROBATCHES_PER_UPDATE == 4
    assert len(record.sample_ids) == 4
    assert all(len(batch) == 4 for batch in record.sample_ids)
    assert trainer.progress.completed_updates == 1
    assert trainer.progress.position == 4
    # AdamW counts one step per optimizer update, not one per microbatch.
    steps = [s["step"] for s in trainer.optimizer.state.values()]
    assert steps and all(int(s) == 1 for s in steps)


def test_gradients_are_cleared_at_the_update_boundary():
    trainer = build_trainer()

    run(trainer, 1)

    assert all(p.grad is None or not p.grad.any() for p in trainer.model.parameters())


def test_same_seeds_give_the_same_run():
    a = run(build_trainer(), 6)
    b = run(build_trainer(), 6)

    assert a == b


def test_different_shuffle_seed_gives_a_different_order():
    a = run(build_trainer(shuffle_seed=0), 2)
    b = run(build_trainer(shuffle_seed=1), 2)

    assert [r.sample_ids for r in a] != [r.sample_ids for r in b]


def test_an_epoch_visits_every_sample_once_then_normalizes_to_the_next_epoch():
    trainer = build_trainer(n_samples=64, batch_size=4)  # 16 batches = 4 updates per epoch

    records = run(trainer, 4)

    seen = sorted(i for r in records for batch in r.sample_ids for i in batch)
    assert seen == list(range(64))
    assert trainer.progress.epoch == 1
    assert trainer.progress.position == 0
    assert records[-1].epoch == 0  # the record names the epoch its batches came from


def test_the_second_epoch_uses_a_new_order():
    trainer = build_trainer()

    first = run(trainer, 4)
    second = run(trainer, 4)

    assert [r.sample_ids for r in first] != [r.sample_ids for r in second]


def test_the_model_has_dropout_and_buffers():
    model = build_trainer().model

    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert len(list(model.buffers())) > 0
