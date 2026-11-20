"""A small image classifier that Hibernus can checkpoint.

The model has dropout and batch norm, the data is a map-style dataset read with
`num_workers=0`, and one optimizer update takes four microbatches. Each call to
`Trainer.run_update` ends at an update boundary: the optimizer, scheduler and gradient reset have
finished, and the next batch has not been read. That is where the application checkpoint is written.
`Trainer.state_dict` and `Trainer.load_state_dict` let the hook treat the whole trainer as one
object, the way it treats a plain model or optimizer.

Needs torch and numpy, so the training hook in `hibernus/__init__.py` does not import it.
"""

import hashlib
from dataclasses import asdict, dataclass

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


class SyntheticImages(Dataset):
    """Fixed random images, labeled by which quadrant is brightest. Needs no download."""

    def __init__(self, n_samples, seed):
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
        self.version = hashlib.sha256(
            self.images.numpy().tobytes() + self.labels.numpy().tobytes()
        ).hexdigest()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, self.images[index], self.labels[index]


class ResumableBatchSampler:
    """Yields batches of sample indices in a shuffled order that depends only on seed and epoch.

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
            yield order[start : start + self.batch_size].tolist()


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

    def __init__(self, model, optimizer, scheduler, scaler, dataset, sampler):
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
        self._loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=0, generator=torch.Generator()
        )
        self._batches = None

    def state_dict(self):
        return {
            "model": self.model.state_dict(),  # weights and buffers
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "progress": asdict(self.progress),
            "sampler": {"seed": self.sampler.seed, "batch_size": self.sampler.batch_size},
            "dataset_version": self.dataset.version,
        }

    def load_state_dict(self, state):
        """Checks everything first, so a refusal changes nothing."""
        self._check(state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.progress = Progress(**state["progress"])
        self.reset_data_position()

    def _check(self, state):
        if state["dataset_version"] != self.dataset.version:
            raise ValueError("the checkpoint was written with a different dataset")
        if state["sampler"]["seed"] != self.sampler.seed:
            raise ValueError(
                f"the checkpoint has shuffle seed {state['sampler']['seed']}, "
                f"but the trainer has {self.sampler.seed}"
            )
        if state["sampler"]["batch_size"] != self.sampler.batch_size:
            raise ValueError("the checkpoint and the trainer have a different batch size")
        if (state["scaler"] is None) != (self.scaler is None):
            raise ValueError("the checkpoint and the trainer disagree about using a GradScaler")

    def reset_data_position(self):
        """Drops the open data iterator so the next update starts from `progress`."""
        self._batches = None

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
            self._batches = None
        return record


def build_trainer(
    init_seed=0,
    data_seed=0,
    shuffle_seed=0,
    n_samples=64,
    batch_size=4,
    total_updates=100,
    use_scaler=False,
):
    """Builds the workload. `init_seed` sets the starting weights and the dropout RNG stream."""
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
    dataset = SyntheticImages(n_samples, data_seed)
    sampler = ResumableBatchSampler(n_samples, batch_size, shuffle_seed)
    return Trainer(model, optimizer, scheduler, scaler, dataset, sampler)
