"""The small image classifier, trained under Hibernus.

Run it alone to train, or under `hib run`, which checkpoints it at an update boundary and restores
it on another host. Needs torch and numpy: `pip install -e "python[train]"`.
"""

import argparse
import sys
import time

import hibernus
from hibernus.workload import build_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-seconds", type=float, default=0, help="pause after each step, to pace a demo")
    args = parser.parse_args()

    trainer = build_trainer(init_seed=args.seed, shuffle_seed=args.seed, total_updates=args.updates)
    print(f"training on {trainer.device}", file=sys.stderr, flush=True)
    hib = hibernus.connect()
    state = {"trainer": trainer}
    hib.initialize(state)  # after a restore, the trainer is back at the update it stopped at
    while trainer.progress.completed_updates < args.updates:
        record = trainer.run_update()
        print(f"step {record.update} loss {record.loss:.4f}", flush=True)
        time.sleep(args.step_seconds)
        hib.boundary(record.update, state)


if __name__ == "__main__":
    main()
