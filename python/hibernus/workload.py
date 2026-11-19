"""A small image classifier that Hibernus can checkpoint.

The model has dropout and batch norm, the data is a map-style dataset read with
`num_workers=0` by default (or more workers, see `build_trainer`), and one optimizer update takes
four microbatches. Each call to
`Trainer.run_update` ends at an update boundary: the optimizer, scheduler and gradient reset have
finished, and the next batch has not been read. That is where the application checkpoint is saved.

Needs torch and numpy, so the training hook in `hibernus/__init__.py` does not import it.
"""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

MICROBATCHES_PER_UPDATE = 4
IMAGE_SIZE = 8
NUM_CLASSES = 4


@dataclass
class Progress:
    """Where training stands. `position` is the next batch to read within `epoch`."""

    completed_updates: int = 0
    epoch: int = 0
    position: int = 0


@dataclass(frozen=True)
class UpdateRecord:
    """What one update did. A resumed run must produce the same records as one that never stopped."""

    update: int
    epoch: int
    sample_ids: list
    loss: float


AUGMENT_NOISE = 0.1
_SEED_STRIDE = 1_000_003  # larger than any sample index here, so (epoch, index) pairs never collide


class SyntheticImages(Dataset):
    """Fixed random images, labeled by which quadrant is brightest. Needs no download.

    A sample is named by `(epoch, index)`. With `augment`, the image gets noise that depends only on
    the dataset seed, the epoch and the index. That makes the dataset stateless: it gives the same
    sample no matter which worker reads it, how many workers there are, or whether the DataLoader
    was rebuilt after a restore. Noise drawn from a worker's own RNG would not have that property.
    """

    def __init__(self, n_samples, seed, augment=False):
        self.augment = augment
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n_samples, 1, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
        half = IMAGE_SIZE // 2
        quadrants = torch.stack(
            [
                self.images[:, 0, :half, :half].sum(dim=(1, 2)),
                self.images[:, 0, :half, half:].sum(dim=(1, 2)),
                self.images[:, 0, half:, :half].sum(dim=(1, 2)),
                self.images[:, 0, half:, half:].sum(dim=(1, 2)),
            ],
            dim=1,
        )
        self.labels = quadrants.argmax(dim=1)
        content = self.images.numpy().tobytes() + self.labels.numpy().tobytes()
        if augment:
            content += b"augment"  # a checkpoint saved with augmentation must not load without it
        self.version = hashlib.sha256(content).hexdigest()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        epoch, index = key
        image = self.images[index]
        if self.augment:
            generator = torch.Generator().manual_seed(
                (self.seed * _SEED_STRIDE + epoch) * _SEED_STRIDE + index
            )
            image = image + AUGMENT_NOISE * torch.randn(image.shape, generator=generator)
        return index, image, self.labels[index]


class ResumableBatchSampler:
    """Yields batches of `(epoch, index)` samples in a shuffled order that depends only on seed and epoch.

    `set_state` makes the next iteration start at a given epoch and batch position, which is how a
    restored run reads the same next batch. The epoch order is recreated from the seed, so the
    sampler never needs its own RNG state.
    """

    def __init__(self, n_samples, batch_size, seed):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seed = seed
        self.batches_per_epoch = n_samples // batch_size
        self.epoch = 0
        self.position = 0

    def set_state(self, epoch, position):
        self.epoch = epoch
        self.position = position

    def __len__(self):
        return self.batches_per_epoch - self.position

    def __iter__(self):
        # Read the state now, not when the first batch is requested.
        return self._batches(self.epoch, self.position)

    def _batches(self, epoch, position):
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(self.n_samples, generator=generator)
        for batch in range(position, self.batches_per_epoch):
            start = batch * self.batch_size
            yield [(epoch, index) for index in order[start : start + self.batch_size].tolist()]


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(8 * IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.norm(self.conv(x)))
        return self.fc(self.dropout(x.flatten(1)))


class Trainer:
    """Trains in whole optimizer updates. Between updates it is at an update boundary."""

    def __init__(self, model, optimizer, scheduler, scaler, dataset, sampler, num_workers=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.dataset = dataset
        self.sampler = sampler
        self.progress = Progress()
        # Opening a DataLoader iterator draws a number from the loader's generator. With the
        # default, that is the global torch RNG, so a restored run (which opens an iterator after
        # the load) would shift the dropout stream by one draw. A private generator avoids that.
        #
        # With num_workers > 0, an open iterator holds batches that workers already prepared. The
        # checkpoint never records them: `progress.position` counts batches the loop consumed, and
        # `reset_data_position` drops the iterator, so a restored run rebuilds the workers from that
        # cursor and discards whatever was in flight. Each worker seeds its own RNG when an
        # iterator opens, which a rebuilt iterator cannot reproduce, so the dataset must not draw
        # randomness from a worker's RNG (see SyntheticImages).
        self._loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            generator=torch.Generator(),
        )
        self._batches = None

    def close_data(self):
        """Closes the open data iterator and stops its DataLoader workers now.

        A process snapshot dumps one process, so the hook calls this first. The next
        `run_update` opens a new iterator, with new workers, at the cursor in `progress`. Closing
        twice, or with nothing open, does nothing.
        """
        iterator, self._batches = self._batches, None
        shutdown = getattr(iterator, "_shutdown_workers", None)  # only a multi-process iterator has it
        if shutdown is not None:
            shutdown()

    def reset_data_position(self):
        """Drops the open data iterator so the next update starts from `progress`."""
        self.close_data()

    def run_update(self):
        """Reads four microbatches, takes one optimizer step, and stops at the update boundary."""
        if self._batches is None:
            self.sampler.set_state(self.progress.epoch, self.progress.position)
            self._batches = iter(self._loader)

        self.model.train()
        sample_ids, total_loss = [], 0.0
        for _ in range(MICROBATCHES_PER_UPDATE):
            ids, images, labels = next(self._batches)
            loss = F.cross_entropy(self.model(images), labels) / MICROBATCHES_PER_UPDATE
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            sample_ids.append(ids.tolist())
            total_loss += loss.item()

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        record = UpdateRecord(
            update=self.progress.completed_updates + 1,
            epoch=self.progress.epoch,
            sample_ids=sample_ids,
            loss=total_loss,
        )
        self.progress.completed_updates += 1
        self.progress.position += MICROBATCHES_PER_UPDATE
        if self.progress.position >= self.sampler.batches_per_epoch:
            self.progress.epoch += 1
            self.progress.position = 0
            self.close_data()
        return record


def build_trainer(
    init_seed=0,
    data_seed=0,
    shuffle_seed=0,
    n_samples=64,
    batch_size=4,
    total_updates=100,
    use_scaler=False,
    num_workers=0,
    augment=False,
):
    """Builds the workload. `init_seed` sets the starting weights and the dropout RNG stream.

    `num_workers` is the DataLoader's worker process count. `augment` adds per-sample noise that
    depends only on the seed, epoch and index, so it is the same for any number of workers.
    """
    batches_per_epoch = n_samples // batch_size
    if batches_per_epoch % MICROBATCHES_PER_UPDATE:
        raise ValueError(
            f"{batches_per_epoch} batches per epoch is not a multiple of {MICROBATCHES_PER_UPDATE}, "
            "so an epoch would end inside an update"
        )
    torch.manual_seed(init_seed)
    model = SmallClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8) if use_scaler else None
    dataset = SyntheticImages(n_samples, data_seed, augment=augment)
    sampler = ResumableBatchSampler(n_samples, batch_size, shuffle_seed)
    return Trainer(model, optimizer, scheduler, scaler, dataset, sampler, num_workers=num_workers)
