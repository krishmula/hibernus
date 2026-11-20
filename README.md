# Hibernus

Hibernus lets a training workload resume after a disruption. A Supervisor runs the training process. When a disruption handler runs `hib checkpoint`, the training hook finishes the current update and pauses before reading the next batch. The Supervisor prints the path to the resulting restore point. A launcher can then run `hib restore` to continue from the next update.

The current implementation writes an application checkpoint to a filesystem directory. Restore verifies the files against the manifest and starts a fresh training process with the recorded command. The design also describes a GPU process snapshot and host fingerprint checks. Both are planned.

## Try a checkpoint and restore

From the repository root, with Go 1.27.4, Python 3.10 or later, uv, Make, and Bash installed, run the local example. It uses a toy training loop and requires no GPU.

```sh
make e2e
```

The example builds `bin/hib`, installs the training hook in `.venv`, and starts a 12-step workload. It checkpoints the workload, checks that it stays paused, then stops the training process to simulate losing the checkpoint host. It also checks that restore rejects a damaged application checkpoint. On success it prints the last completed step and the first step after restore.

## Run a workload

Build with `make build venv`, then start the Supervisor in one terminal:

```sh
./bin/hib run --workload toy --store ./store .venv/bin/python examples/toy/train.py
```

In a second terminal in the same directory, inspect the workload and checkpoint it:

```sh
./bin/hib status --workload toy
location=$(./bin/hib checkpoint --workload toy --deadline 10s)
```

Once the checkpoint completes, press Ctrl-C in the first terminal to stop the paused workload. In the second terminal, restore it:

```sh
./bin/hib restore --store ./store "$location"
```

For a restore on another host, the launcher must make the restore point, recorded working directory, training command, and its dependencies available there.

## Connect a training loop

Call `hibernus.connect()` and then `initialize()` before the first update. Call `boundary()` after the optimizer update, scheduler update, and gradient reset, before reading the next batch.

The [toy loop](examples/toy/train.py) tracks only a step number. The [PyTorch classifier](examples/classifier/train.py) passes its trainer to both hook calls so the application checkpoint also captures the model, optimizer, and other training state. Install its dependencies with `uv pip install --python .venv -e 'python[train]'`.

## Further reading

- [System design](docs/design/system-design.md)
- [Restore point and protocol formats](docs/design/data-shapes.md)
- [Implementation plan](docs/design/v1-plan.md)
- [Apache 2.0 license](LICENSE)
