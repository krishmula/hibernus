# v1 plan

Drafted on 2026-11-16. v1 is done when every requirement in section 4 of the system design, F1 to F5 and N1 to N9, passes its test on an EC2 G5 host, and the demo runs end to end.

## How the work runs

- Each phase ends with a check that someone can run. The next phase starts only after the check passes.
- Every phase reruns the same end-to-end check: `hib run`, then `hib checkpoint`, then `hib restore`, then the workload continues at step K+1.
- Claude writes the code. Krishna reviews each step and must understand it before the next step starts.
- All work happens on branches. Nothing goes on `main` directly.

## Phases

| Phase | What it adds | Where it runs | Owner | Check | Requirements |
|---|---|---|---|---|---|
| 0. Setup | A branch, the Go module, the Python package, and the toy example | macOS | Claude | `go build ./...` passes, and `pip install -e python/` works | none |
| 1. First build | `hib run`, `hib status`, `hib checkpoint`, and `hib restore`, with a fake checkpoint that holds only the step | macOS | Claude | The training process continues at step K+1 after a restore | F1, F5 |
| 2. Application checkpoint | The training hook saves and loads the model, optimizer, scheduler, GradScaler, RNG states, and data position | macOS, CPU | Nitish | A restored run matches a run that never stopped: same next batch, loss, and parameters | F3, N1, N2 |
| 3. S3 and verification | Upload to S3, checksums, the manifest written last | macOS or Linux | Claude | Kill the Supervisor mid-upload. `hib restore` refuses the partial restore point | N4 |
| 4. Move to a G5 | Run phases 1 to 3 on a G5 with GPU training, and add the fingerprint check | G5 | Claude | The end-to-end check passes on a G5. A host with a different driver is refused | F2, N5 |
| 5. CRIU process snapshot | CRIU captures the training process on the CPU side. The new Supervisor restores it, and the training hook reconnects | Linux | Krishna | The restore uses the process snapshot. With the snapshot corrupted, it falls back to the application checkpoint | N3 |
| 6. cuda-checkpoint | GPU memory goes into the process snapshot | G5 | Krishna | The end-to-end check passes on a G5 through the process snapshot | F3 on GPU |
| 7. DataLoader child processes | The process snapshot includes the DataLoader child processes | G5 | Krishna | The end-to-end check passes with `num_workers > 0` | out of v1 |
| 8. Measure and harden | Timing reports, the idle overhead test, the RAM and disk checks, and install on a fresh host | G5 | Claude | Each test in section 4 passes | F4, N6, N7, N8, N9 |
| 9. Demo | An emulated disruption on one G5, and a restore on a second G5 | 2 x G5 | Krishna and Nitish | The demo runs start to finish | all |

## Results

### Phase 4, 2026-11-21

Passed on EC2 g5.xlarge hosts (NVIDIA A10G) in Nitish's AWS account, `us-east-1`, because the Hibernus account had no G quota yet. Hibernus `0603a8c`, PyTorch 2.14.1 with CUDA 13.0, the classifier example, and a real S3 bucket. Only the application checkpoint, as planned.

| Check | Result |
|---|---|
| The Python tests on the GPU | 45 passed |
| Checkpoint to S3 | Step 540 in 0.1 s. The manifest holds the real fingerprint: driver 595.112.07, cuDNN 92400, kernel 6.8.0-1069-aws |
| Restore on a second G5 with the same fingerprint (F2) | Training continued at step 541. Steps 541 to 590 matched a run that never stopped, line for line, on another physical GPU |
| Restore on a host with another driver (N5) | The checkpoint host, after its driver changed to 595.91.07. `hib restore` exited 1 with `driver: checkpoint host 595.112.07, this host 595.91.07` and started no training process |

Only one G5 ran at a time, because AWS had no second G5 free in any zone that day. The checkpoint host went away before the restore host started, as in a real disruption.

## Runs in parallel with phase 1

A CRIU test on Linux, owned by Krishna, in `scripts/criu-test`. It checks the phase 5 design on one host before any of it goes into `hib`. A small Go stand-in Supervisor starts a Python stand-in training process as PID 1 of its own PID namespace. At step 5 the training process pauses: it installs a SIGCONT handler, closes its connection, and waits. The stand-in Supervisor stops it with SIGSTOP and runs `criu dump`. A second run of the stand-in Supervisor becomes a subreaper, runs `criu restore --restore-detached`, sends SIGCONT, and accepts the reconnect. The test answers four questions:

- Does CRIU dump a stopped PID 1 of a namespace, and restore it still stopped in a new namespace?
- Does the stand-in Supervisor adopt the restored training process?
- Does SIGCONT end the wait, including with torch's threads running, and does the reconnect work?
- Does anything break without a `/proc` remount, which would need a mount namespace?

It runs three cases: plain Python, Python with torch loaded, and `--restore-sibling` in place of the subreaper. On a fresh Ubuntu host, as root: `setup.sh`, then `run.sh`.

## Decisions needed, and by when

| Decision | Needed before |
|---|---|
| Does v1 cover inference workloads? | Phase 2 |
| Which environment variables the manifest records, since some hold secrets | Phase 3 |

## Decisions made

| Decision | Made on |
|---|---|
| v1 uses DataLoader `num_workers = 0`. Phase 7 is out of v1 | 2026-11-21 |
| The CRIU test runs on a c6i.large in Nitish's account | 2026-11-21 |
