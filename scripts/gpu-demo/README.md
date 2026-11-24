# GPU disruption drill

`run.sh` trains, runs `hib checkpoint` after a set time, kills the training process, waits, runs
`hib restore` from S3, and lets the run finish. A second, uninterrupted run trains alongside it, so
the restored losses are compared with the reference update by update.

```sh
make build venv
uv pip install --python .venv -e 'python[train]'
scripts/gpu-demo/run.sh --checkpoint-after 20 --down-for 10 --serve 8000
```

| Option | Meaning |
|---|---|
| `--checkpoint-after S` | seconds of training before `hib checkpoint` runs |
| `--down-for S` | seconds between losing the host and `hib restore` |
| `--updates N`, `--step-sleep S` | length of the run; the sleep makes it last long enough to watch |
| `--bucket NAME` | S3 bucket for restore points (default `$HIB_BUCKET`) |
| `--serve PORT` | serve `dashboard/index.html` and `live.json` on 127.0.0.1 |

The script needs AWS credentials allowed to use the bucket (an instance role with the policy from
`deploy/terraform` is enough) and a region.

## Live metrics

While the drill runs, `live.json` in the output folder (`/var/tmp/hib-demo` by default) is rewritten every
2 seconds by `bridge.py`, from the files the drill leaves behind: the event log, the loss logs, and two
snapshots read by `probe.py` (the checkpoint before the disruption and the restored run after it).
`dashboard/index.html` polls that file and redraws, so the status, the before and after panels, the loss
chart and the log update by themselves. On a remote host, forward the port first:

```sh
ssh -L 8000:127.0.0.1:8000 ubuntu@<host>
```

then open http://127.0.0.1:8000/.

## What it does and does not prove

This drill uses the application checkpoint. A restore starts a new process, so the PID changes; the
comparison covers the model, optimizer and RNG state, the data position, and every loss after the
restore. The GPU process snapshot (CRIU with CUDA) is not built yet, and the manifest says so.
