"""The stand-in training process for the CRIU test.

It talks to the stand-in Supervisor the way the training hook will in phase 5: it reports each
update boundary, and when the Supervisor asks it to pause, it waits for SIGCONT, reconnects, and
continues at the next step.
"""

import argparse
import json
import os
import signal
import socket
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-after-restore", type=int, default=5)
    parser.add_argument("--torch", action="store_true", help="run a torch matmul each step, so torch's thread pool exists")
    args = parser.parse_args()

    matmul = start_torch() if args.torch else None
    # A second Python thread, so the process is never single-threaded when the signal arrives.
    threading.Thread(target=idle, daemon=True).start()

    sock = connect()
    step, stop_at = 0, None
    while stop_at is None or step < stop_at:
        step += 1
        if matmul:
            matmul()
        time.sleep(0.1)
        print(f"step {step}", flush=True)
        reply = send(sock, {"op": "boundary", "step": step, "threads": thread_count()})
        if reply["pause"]:
            sock = pause(sock, step)
            stop_at = step + args.steps_after_restore
    sock.close()


def pause(sock, step):
    """The new wait: install the SIGCONT handler, close the connection, and wait for the flag."""
    woke = threading.Event()
    signal.signal(signal.SIGCONT, lambda signum, frame: woke.set())
    write(sock, {"op": "app_checkpoint"})
    sock.close()
    print(f"paused after step {step}", flush=True)
    woke.wait()
    print(f"woke after step {step}", flush=True)

    sock = connect()
    pid = os.getpid()
    send(
        sock,
        {
            "op": "restored",
            "step": step,
            "threads": thread_count(),
            # Without a mount namespace, /proc belongs to the host's PID namespace, so these show
            # what a library sees when it reads /proc after the restore.
            "getpid": pid,
            "proc_self_pid": int(read("/proc/self/stat").split()[0]),
            "proc_getpid_comm": read(f"/proc/{pid}/comm").strip(),
        },
    )
    return sock


def start_torch():
    import torch

    torch.set_num_threads(4)
    a = torch.randn(512, 512)
    return lambda: a @ a


def idle():
    while True:
        time.sleep(1)


def thread_count():
    return len(os.listdir("/proc/self/task"))


def read(path):
    with open(path) as f:
        return f.read()


def connect():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(os.environ["HIB_SOCKET"])
    return sock


def send(sock, message):
    write(sock, message)
    line = sock.makefile("rb").readline()
    if not line:
        raise ConnectionError("the Supervisor closed the connection")
    return json.loads(line)


def write(sock, message):
    sock.sendall(json.dumps(message).encode() + b"\n")


if __name__ == "__main__":
    main()
