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

To include the training state in the application checkpoint, pass both calls a dict of the objects the loop owns. Each object needs `state_dict()` and `load_state_dict()`, as every PyTorch module, optimizer, and scheduler has. The hook also records the step and the Python, NumPy, and torch RNG states. An optional `config` dict holds plain values that must stay the same across a restore, such as the shuffle seed and batch size. A restore whose config differs fails before it changes anything.

```python
hib = hibernus.connect()
state = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
config = {"seed": seed, "batch_size": batch_size}
step = hib.initialize(state, config)  # 0 on a fresh run; after a restore, the step it stopped at
...
hib.boundary(step, state, config)     # after each update
```

### Resume the data at the right batch

A restore from the application checkpoint starts the script again from the top. The hook restores the training state, but it can't move the loop back to its place in the data. To continue with exactly the batches an uninterrupted run would use, the loop must do two things:

1. Make each epoch's shuffle order depend only on a seed and the epoch number. With a plain `DataLoader(shuffle=True)`, pass `generator=g` and call `g.manual_seed(seed + epoch)` at the start of each epoch.
2. Skip the batches the epoch already used. With a fixed number of batches per epoch (`drop_last=True`), `epoch, skip = divmod(step, len(loader))` gives the place, and `itertools.islice(loader, skip, None)` skips that many batches.

`islice` reads the skipped batches and discards them. On a large dataset, a restore late in an epoch can spend minutes loading data before training continues. To avoid that, use a sampler that starts at a given batch, such as Accelerate's `skip_first_batches`.

A loop that does neither still restores. For the rest of that epoch, it repeats some samples and misses others. The planned process snapshot will not have this problem, because a restore from it continues the running DataLoader.

The [toy loop](examples/toy/train.py) tracks only a step number. The [plain PyTorch loop](examples/plain/train.py) is an ordinary training loop that follows the steps above. The [classifier](examples/classifier/train.py) is a harder test workload, with microbatches, batch norm, and an optional GradScaler, that passes one trainer object. Install their dependencies with `uv pip install --python .venv -e 'python[train]'`.

## Further reading

- [System design](docs/design/system-design.md)
- [Restore point and protocol formats](docs/design/data-shapes.md)
- [Implementation plan](docs/design/v1-plan.md)
- [Apache 2.0 license](LICENSE)
