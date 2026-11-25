# Process snapshot

How phase 5 makes the process snapshot on the checkpoint host and restores it on the restore host. The CRIU test checked this design on 2026-11-21. Its results are in [v1-plan.md](v1-plan.md#criu-test-2026-11-21). This page covers the CPU side only. Phase 6 adds `cuda-checkpoint` for GPU memory.

## Checkpoint host

1. `hib run` starts the training process as PID 1 of a new PID namespace, in a new session. It stays the Supervisor's direct child. Its stdin reads from `/dev/null`, and its stdout and stderr go to a file in the run directory. `hib run` copies new lines from that file to its own output.
2. At the update boundary, the training hook writes the application checkpoint and sends `app_checkpoint`, as today.
3. The training hook installs a SIGCONT handler that sets a flag, closes its connection, and waits for the flag in 100 ms slices.
4. The Supervisor sees the connection close and sends SIGSTOP. The signal must come from outside the namespace, because PID 1 of a namespace ignores SIGSTOP sent from inside it.
5. The Supervisor confirms the stop, then runs `criu dump`. CRIU writes the images and ends the training process.
6. The Supervisor uploads both layers and the output file, and writes the manifest last.

## Restore host

1. `hib restore` verifies the restore point and the fingerprint.
2. The Supervisor declares itself a subreaper and listens on the same socket path as the checkpoint host. It puts the output file back at its old path.
3. The Supervisor runs `criu restore --restore-detached --pidfile`. CRIU brings the training process back, still stopped, as PID 1 of a new namespace, and exits. The kernel hands the orphaned training process to the Supervisor.
4. The Supervisor sends SIGCONT and starts a 30 s limit. The handler sets the flag, and the wait ends.
5. The training hook opens a new connection and sends `restored` with step K, the fingerprint, and python. The Supervisor checks that K matches the restore point, keeps the fingerprint and python for later manifests, moves from Restoring to Running, and replies. The training hook waits for that reply before it trains.
6. Training continues at step K+1. `hib restore` reports that it used the process snapshot.

## Fallback to the application checkpoint

`hib restore` uses the application checkpoint when any of these happen:

- The process snapshot fails its checksums.
- CRIU is missing on the restore host, or older than 3.17.
- `criu restore` fails.
- `restored` doesn't arrive within 30 s. The Supervisor kills the training process first. That is safe, because the training hook doesn't train until the Supervisor replies.

The fallback starts the training script fresh, and the training hook loads the application checkpoint through `initialize`, as today. `hib restore` reports that it used the application checkpoint.

## Decisions

Settled on 2026-11-24.

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | How does a restored training process tell the Supervisor it's back? | A new `restored` request with the step, the fingerprint, and python. It doesn't reuse `initialize` | A restored process wakes up past `initialize`. The Supervisor needs to know the wake worked, check the step, and get the fingerprint and python for later manifests. `initialize` asks where to begin, while `restored` says where the process already is |
| 2 | How long does the Supervisor wait for `restored`? | 30 s, then kill the training process and fall back | The CRIU test reconnected in well under 1 s |
| 3 | Which layer does `hib restore` use? | The process snapshot first, with the fallbacks above. It reports which layer it used | The process snapshot is faster and exact. The phase 5 check needs the report |
| 4 | How does the restore host's Supervisor become the training process's parent? | A subreaper, with `criu restore --restore-detached` | `--restore-sibling` also passed the test, but CRIU warns that some kernels don't support it |
| 5 | Where do the training process's input and output go? | Input from `/dev/null`. Output to a file in the run directory, which goes to the restore host with the process snapshot. `hib run` and `hib restore` copy its new lines to their output | CRIU can't restore a terminal that belongs to a process outside the snapshot. CRIU reopens a file by its path, and the test confirmed it. CRIU doesn't keep the file's contents, so `hib` copies the file |
| 6 | What do Ctrl+C and SIGTERM do? | The Supervisor passes SIGINT and SIGTERM to the training process. The training hook installs a SIGTERM handler. After a grace period, the Supervisor sends SIGKILL | In its own session, the training process no longer gets Ctrl+C from the terminal. The kernel drops a signal to PID 1 of a namespace unless the process handles it. Python handles SIGINT by default but not SIGTERM |
| 7 | What if CRIU is missing or too old? | `hib` requires CRIU 3.17 or later. Without it, the checkpoint host writes only the application checkpoint, with a warning, and the restore host uses the application checkpoint | CRIU 3.16 can't restore rseq, which glibc 2.35 turns on, so the restored process crashes. Training never depends on CRIU |

## Open questions

- If the Supervisor dies while the training process waits for SIGCONT, nothing ends the wait. Decide whether that matters for v1, since the host is about to go away anyway.
- The grace period before SIGKILL in decision 6.
