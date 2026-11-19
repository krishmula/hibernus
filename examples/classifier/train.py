"""The small image classifier, trained under Hibernus. It saves and restores the full state.

Run it alone to train, or under `hib run`, which checkpoints it at an update boundary and restores
it on another host. Needs torch and numpy: `pip install -e "python[train]"`.
"""

import argparse
import time

import hibernus
from hibernus.workload import build_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-sleep", type=float, default=0.0, help="seconds to wait after each update")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--augment", action="store_true", help="add per-sample noise (same for any worker count)")
    args = parser.parse_args()

    trainer = build_trainer(
        init_seed=args.seed,
        shuffle_seed=args.seed,
        total_updates=args.updates,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    hib = hibernus.connect()
    hib.start(trainer=trainer)  # after a restore, the trainer is back at the saved update
    while trainer.progress.completed_updates < args.updates:
        record = trainer.run_update()
        print(f"step {record.update} loss {record.loss:.4f}", flush=True)
        time.sleep(args.step_sleep)
        hib.boundary(record.update, trainer=trainer)


if __name__ == "__main__":
    main()
