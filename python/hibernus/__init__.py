"""The Hibernus training hook. It runs inside the training process and talks to the Supervisor."""

import json
import os
import signal
import socket
import sys
import threading

SOCKET_ENV = "HIB_SOCKET"

# The stand-in application checkpoint of the first build: one file that holds only the step.
STEP_FILE = "step.json"

# The application checkpoint of a training loop that passes its state to the hook.
APP_FILE = "app.pt"


class Hook:
    """Talks to the Supervisor over its socket."""

    def __init__(self, socket_path):
        self._socket_path = socket_path
        self._attach(_connect(socket_path))
        _exit_on_sigterm()

    def _attach(self, sock):
        self._sock = sock
        self._file = sock.makefile("rwb")

    def initialize(self, state=None, config=None):
        """Returns the last finished step: 0 for a fresh run, or K after a restore from step K.

        `state` names what the training loop owns, such as {"model": model, "optimizer": opt}.
        Each value needs state_dict() and load_state_dict(), as every torch module, optimizer,
        and scheduler has. A restore loads the application checkpoint into them. Without state,
        a restore reads the stand-in step file.

        `config` holds plain values that must not change across a restore, such as the shuffle
        seed and batch size. A restore whose config differs is refused before anything changes.
        """
        reply = self._send({"op": "initialize", **_host()})
        resume = reply["resume"]
        if resume is None:
            return 0
        if state is None:
            return _load_step(resume["files"], resume["step"])
        return _load_state(resume["files"], resume["step"], state, config)

    def boundary(self, step, state=None, config=None):
        """Reports that step finished. The training loop calls it at each update boundary.

        With state, a pause writes its application checkpoint, with config.
        """
        reply = self._send({"op": "boundary", "step": step})
        if reply["pause"]:
            self._checkpoint(step, reply["dir"], state, config)

    def _checkpoint(self, step, folder, state, config):
        try:
            os.makedirs(folder, exist_ok=True)
            if state is None:
                path = os.path.join(folder, STEP_FILE)
                with open(path, "w") as f:
                    json.dump({"step": step}, f)
            else:
                from hibernus import app_checkpoint  # needs torch, so only state pulls it in

                path = os.path.join(folder, APP_FILE)
                app_checkpoint.save(path, step, state, config)
            self._write({"op": "app_checkpoint", "files": [path]})
        except Exception as e:  # tell the Supervisor why, whatever went wrong
            self._write({"op": "app_checkpoint", "error": str(e)})
        self._pause()
        # Only a restore from the process snapshot gets here. The training process woke up at
        # this step, so the training loop continues at the next one.
        self._attach(_connect(self._socket_path))
        reply = self._send({"op": "restored", "step": step, **_host()})
        if reply.get("continue") is not True:
            raise RuntimeError(f"the Supervisor answered restored with {reply}")

    def _pause(self):
        """Closes the connection, and waits until SIGCONT arrives.

        Once the connection closes, the Supervisor stops this process, and CRIU dumps it into the
        process snapshot. On the checkpoint host, the wait never ends: CRIU ends the process, or
        the host goes away. On the restore host, the Supervisor sends SIGCONT after criu restore.
        """
        woke = threading.Event()
        previous = signal.signal(signal.SIGCONT, lambda signum, frame: woke.set())
        try:
            self._file.close()
            self._sock.close()
            # Wait in slices. The kernel may hand SIGCONT to any thread, and CPython runs the
            # handler only when the main thread runs Python, so a bare woke.wait() can sleep forever.
            while not woke.wait(0.1):
                pass
        finally:
            signal.signal(signal.SIGCONT, previous)

    def _send(self, message):
        self._write(message)
        line = self._file.readline()
        if not line:
            raise ConnectionError("the Supervisor closed the connection")
        reply = json.loads(line)
        if "refused" in reply:
            raise RuntimeError(f"the Supervisor refused {message['op']}: {reply['refused']}")
        return reply

    def _write(self, message):
        self._file.write(json.dumps(message).encode() + b"\n")
        self._file.flush()


def _exit_on_sigterm():
    """Makes SIGTERM end training, as Ctrl+C does. hib passes SIGTERM on to the training process,
    which runs on Linux as PID 1 of its own PID namespace. The kernel drops a signal to PID 1 unless
    the process handles it, so without a handler SIGTERM would do nothing. A SIGTERM handler that the
    training script installed stays."""
    if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
        signal.signal(signal.SIGTERM, _sigterm)


def _sigterm(signum, frame):
    sys.exit(128 + signum)  # the shell's convention: 143 for SIGTERM


def _connect(socket_path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    return sock


def _host():
    """The fingerprint of the host the training process runs on, and the interpreter. The
    interpreter lets hib restore collect the restore host's fingerprint the same way."""
    # Imported here, so `python -m hibernus.fingerprint` doesn't find it already loaded and warn.
    from hibernus import fingerprint

    return {"fingerprint": fingerprint.collect(), "python": sys.executable}


def _load_step(files, step):
    """Reads the step back from the stand-in application checkpoint."""
    for path in files:
        if os.path.basename(path) == STEP_FILE:
            with open(path) as f:
                saved = json.load(f)["step"]
            if saved != step:
                raise RuntimeError(f"{path} holds step {saved}, but the manifest says step {step}")
            return saved
    raise RuntimeError(f"the restore point has no {STEP_FILE}")


def _load_state(files, step, state, config):
    """Loads the application checkpoint into state, and checks it holds the manifest's step."""
    from hibernus import app_checkpoint

    for path in files:
        if os.path.basename(path) == APP_FILE:
            return app_checkpoint.load(path, state, config, expected_step=step)
    raise RuntimeError(f"the restore point has no {APP_FILE}")


class NoSupervisor:
    """Stands in for Hook when the training process runs without hib, so training runs as usual."""

    def initialize(self, state=None, config=None):
        return 0

    def boundary(self, step, state=None, config=None):
        pass


def connect():
    socket_path = os.environ.get(SOCKET_ENV)
    if socket_path:
        return Hook(socket_path)
    return NoSupervisor()
