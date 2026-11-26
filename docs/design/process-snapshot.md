# Process snapshot

How phase 5 makes the process snapshot on the checkpoint host and restores it on the restore host. The CRIU test checked this design on 2026-11-21. Its results are in [v1-plan.md](v1-plan.md#criu-test-2026-11-21). CRIU 4.3 runs `cuda-checkpoint` through its CUDA plugin, so a training process with the GPU open also works, as the two-host test on 2026-11-26 showed. Phase 6 still owns the GPU work: timing, larger models, and its own checks.

## Checkpoint host

1. `hib run` starts the training process in a new session, with no PID namespace (see decision 8). It stays the Supervisor's direct child. Its stdin reads from `/dev/null`, and its stdout and stderr go to `<workload>.log` in the run directory. `hib run` copies new lines from that file to its own output. CRIU needs root, so without root, or without CRIU 3.17 or later, `hib run` warns that restore points hold only the application checkpoint. On a host with an NVIDIA GPU, a training process with the GPU open also needs CRIU 4.0 or later, for its CUDA plugin, and `cuda-checkpoint` on PATH. Without them, `hib run` warns, and the Supervisor skips the dump of such a process, so its restore points hold only the application checkpoint.
2. At the update boundary, the training hook writes the application checkpoint and sends `app_checkpoint`.
3. The training hook installs a SIGCONT handler that sets a flag, closes its connection, and waits for the flag in 100 ms slices.
4. The Supervisor sees the connection close and runs `criu dump`. It sends no SIGSTOP first (decision 9). CRIU freezes the training process, writes the images, and ends the process. With the GPU open, CRIU's CUDA plugin runs `cuda-checkpoint`, so the images hold GPU memory too.
5. The Supervisor copies the output file into the restore point, uploads both layers, and writes the manifest last. The manifest records the output file's path. `hib run` exits 0, because the workload is checkpointed.

## Restore host

1. `hib restore` verifies the restore point and the fingerprint.
2. It puts the output file back at its recorded path, declares itself a subreaper, and listens on the same socket path as the checkpoint host.
3. It runs `criu restore --restore-detached --pidfile`. CRIU brings the training process back with its old PID and thread IDs, still in the training hook's wait, and exits. The kernel hands the orphaned training process to the Supervisor.
4. The Supervisor sends SIGCONT and starts a 30 s limit. The handler sets the flag, and the wait ends.
5. The training hook opens a new connection and sends `restored` with step K, the fingerprint, and python. The Supervisor checks that K matches the restore point, keeps the fingerprint and python for later manifests, moves from Restoring to Running, and replies. The training hook waits for that reply before it trains.
6. Training continues at step K+1. `hib restore` prints `restored the process snapshot at step K`.

## What the restore host must match

CRIU restores the training process exactly, so the restore host must hold what the process had open, byte for byte:

- **The same PID and thread IDs, free.** Without a PID namespace, another process may hold one. Then `criu restore` fails, and `hib restore` falls back.
- **The same files at the same paths:** python, the venv, the training script, and every shared library the process loaded. CRIU refuses a library whose size differs. On 2026-11-26, `unattended-upgrades` upgraded `libexpat` on one host and not the other, and the restore fell back. Hosts for a demo need the same packages, with automatic upgrades stopped.
- **The same run directory** (`HIB_RUN_DIR`), because the training hook reconnects to the socket path it had.

## Fallback to the application checkpoint

`hib restore` uses the application checkpoint when any of these happen:

- The process snapshot fails its checksums.
- CRIU is missing on the restore host, or older than 3.17.
- `criu restore` fails, for example because a PID is taken or a library differs.
- `restored` doesn't arrive within 30 s. The Supervisor kills the training process first. That is safe, because the training hook doesn't train until the Supervisor replies.

The fallback starts the training script fresh, and the training hook loads the application checkpoint through `initialize`, as today. `hib restore` reports that it used the application checkpoint.

## Decisions

Decisions 1 to 7 settled on 2026-11-24, and 8 to 10 on 2026-11-25.

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | How does a restored training process tell the Supervisor it's back? | A new `restored` request with the step, the fingerprint, and python. It doesn't reuse `initialize` | A restored process wakes up past `initialize`. The Supervisor needs to know the wake worked, check the step, and get the fingerprint and python for later manifests. `initialize` asks where to begin, while `restored` says where the process already is |
| 2 | How long does the Supervisor wait for `restored`? | 30 s, then kill the training process and fall back | The CRIU test reconnected in well under 1 s |
| 3 | Which layer does `hib restore` use? | The process snapshot first, with the fallbacks above. It reports which layer it used | The process snapshot is faster and exact. The phase 5 check needs the report |
| 4 | How does the restore host's Supervisor become the training process's parent? | A subreaper, with `criu restore --restore-detached` | `--restore-sibling` also passed the test, but CRIU warns that some kernels don't support it |
| 5 | Where do the training process's input and output go? | Input from `/dev/null`. Output to a file in the run directory, which goes to the restore host with the process snapshot. `hib run` and `hib restore` copy its new lines to their output | CRIU can't restore a terminal that belongs to a process outside the snapshot. CRIU reopens a file by its path, and the test confirmed it. CRIU doesn't keep the file's contents, so `hib` copies the file |
| 6 | What do Ctrl+C and SIGTERM do? | The Supervisor passes SIGINT and SIGTERM to the training process. The training hook installs a SIGTERM handler. After 10 s, the Supervisor sends SIGKILL | In its own session, the training process no longer gets Ctrl+C from the terminal. The SIGTERM handler ends training cleanly, and it would matter again with a PID namespace, where the kernel drops a signal to PID 1 unless the process handles it |
| 7 | What if CRIU is missing or too old? | `hib` requires CRIU 3.17 or later. Without it, the checkpoint host writes only the application checkpoint, with a warning, and the restore host uses the application checkpoint | CRIU 3.16 can't restore rseq, which glibc 2.35 turns on, so the restored process crashes. Training never depends on CRIU |
| 8 | Does the training process get its own PID namespace? | No, for now. It gets its own session only | Settled on 2026-11-25. In a PID namespace that shares the host's `/proc`, CUDA fails to start. With its own mount namespace and `/proc`, CUDA works, but CRIU 4.3 can't restore the mount tree it copies from the host. The cost: a taken PID makes the restore fall back. A container-style root, with a short list of mounts, may bring the namespace back later |
| 9 | Does the Supervisor stop the training process before `criu dump`? | No | CRIU's CUDA plugin has to talk to the running process to pause CUDA, so a stopped process hangs the dump. CRIU freezes the process itself, and the training hook wakes on SIGCONT whether or not it was stopped |
| 10 | Does the fingerprint open the GPU? | No. It reads cuDNN's version from the library | `torch.backends.cudnn.version()` opens the GPU, and a training process with the GPU open needs `cuda-checkpoint` to dump |

## Parked

Not an immediate problem, so it waits until after phase 5.

- **The Supervisor dies while the training process waits for SIGCONT.** The training hook closed its connection, so it can't see that the Supervisor died.
  - If the Supervisor died before SIGSTOP, the training process keeps waiting. A fix: the Supervisor always sends SIGSTOP, even without CRIU, and the training hook exits after 600 waits of 100 ms. A count works where a clock time limit wouldn't, because the count doesn't advance while the process is stopped or across a restore.
  - If the Supervisor died after SIGSTOP, the training process stays stopped, and nothing inside it can run. It uses memory but no CPU. In a real disruption, the host goes away soon after. A parent-death signal would end it, but CRIU restores that setting, so `criu`'s own exit in detached mode would kill the restored training process.
