"""A plain PyTorch training loop, trained under Hibernus.

Hibernus adds connect, initialize, and boundary, plus the state and config they take. The loop also
has the two habits that any training loop needs to resume exactly where it stopped. It seeds the
shuffle from the epoch, and it counts its updates. It trains on the GPU when there is one. Run it alone to train, or under `hib run`. Needs
torch and numpy: `pip install -e "python[train]"`.
"""

import argparse
import itertools
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import hibernus


def make_dataset(seed, n_samples=512, n_features=20, n_classes=3):
    """Random points, labeled by a fixed random linear rule. Needs no download."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, n_features, generator=generator)
    rule = torch.randn(n_features, n_classes, generator=generator)
    return TensorDataset(x, (x @ rule).argmax(dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # A restore must continue exactly like a run that never stopped, so the GPU must give the
        # same results on every run. cuBLAS reads its setting before the first tensor reaches the GPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    print(f"training on {device}", file=sys.stderr, flush=True)

    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates)
    shuffle = torch.Generator()
    loader = DataLoader(
        make_dataset(args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=shuffle,
    )

    hib = hibernus.connect()
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
    # A restore with other values here would resume on the wrong data, so Hibernus refuses it.
    config = {"seed": args.seed, "batch_size": args.batch_size, "samples": len(loader.dataset)}
    step = hib.initialize(state, config)  # 0 when fresh; after a restore, the step it stopped at

    epoch, skip = divmod(step, len(loader))
    while step < args.updates:
        shuffle.manual_seed(args.seed + epoch)  # the same order for this epoch on every run
        for x, y in itertools.islice(loader, skip, None):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            print(f"step {step} epoch {epoch} loss {loss.item():.6f}", flush=True)
            hib.boundary(step, state, config)  # the update boundary
            if step == args.updates:
                break
        epoch, skip = epoch + 1, 0


if __name__ == "__main__":
    main()
