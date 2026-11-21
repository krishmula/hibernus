"""The Hibernus training hook. It runs inside the training process and talks to the Supervisor."""

import json
import os
import socket
import sys

SOCKET_ENV = "HIB_SOCKET"

# The stand-in application checkpoint of the first build: one file that holds only the step.
STEP_FILE = "step.json"

# The application checkpoint of a training loop that passes its state to the hook.
APP_FILE = "app.pt"


class Hook:
    """Talks to the Supervisor over its socket."""

    def __init__(self, socket_path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        self._attach(sock)

    @classmethod
    def from_socket(cls, sock):
        """A hook on a connection that is already open. Tests use it with a fake Supervisor."""
        hook = cls.__new__(cls)
        hook._attach(sock)
        return hook

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
        # Imported here, so `python -m hibernus.fingerprint` doesn't find it already loaded and warn.
        from hibernus import fingerprint

        # The interpreter lets hib restore collect the restore host's fingerprint the same way.
        reply = self._send({"op": "initialize", "fingerprint": fingerprint.collect(), "python": sys.executable})
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
        # Stay paused. The Supervisor never answers, so this wait ends only when the connection
        # breaks, which means the host went away.
        self._file.readline()
        raise ConnectionError("the Supervisor went away while training was paused")

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
