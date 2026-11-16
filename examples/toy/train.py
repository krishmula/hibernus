"""A toy training loop. Its only state is the step number."""

import argparse
import time

import hibernus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--step-seconds", type=float, default=0.5)
    args = parser.parse_args()

    hib = hibernus.connect()
    step = hib.start()
    while step < args.steps:
        step += 1
        time.sleep(args.step_seconds)
        print(f"step {step}", flush=True)
        hib.boundary(step)


if __name__ == "__main__":
    main()
