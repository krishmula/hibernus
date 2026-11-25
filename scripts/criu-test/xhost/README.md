# Cross-host CRIU test

Dumps a process on host A, copies the images to host B, and restores it there. `counter.py` keeps a
SHA-256 chain and 64 MB of memory, so a correct restore must continue the exact chain from where
the dump stopped.

Both hosts need Ubuntu, CRIU 3.17 or newer (`setup.sh` in the parent directory installs it), and
the same CPU family and kernel. The process runs as PID 1 of its own PID namespace, so the restore
does not need its old host PID to be free.

1. Copy `counter.py` and `dump-on-host-a.sh` to host A, and run `bash dump-on-host-a.sh`.
   It writes `~/xhost.tgz` (the images and `out.log`).
2. Copy `~/xhost.tgz` to host B, with `counter.py` and `restore-on-host-b.sh`, and run
   `bash restore-on-host-b.sh`.

Check the result by recomputing the chain: step N must print `sha256` applied N times to
`b"hibernus"`, first 16 hex characters.

Do not start the process with `unshare --kill-child`. That sets a parent-death signal, which CRIU
restores, and the restored process is killed as soon as CRIU exits.

Result on two t3.large hosts (same CPU, different availability zones): dumped at step 7 on A,
restored on B as PID 1 of a new namespace (host PID 3049), and every later step matched the
recomputed chain. Not tested: different CPU types or kernels, and a GPU process.
