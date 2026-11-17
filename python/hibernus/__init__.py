"""The Hibernus training hook. It runs inside the training process and talks to the Supervisor."""

import json
import os
import socket

SOCKET_ENV = "HIB_SOCKET"

# The stand-in application checkpoint of the first build: one file that holds only the step.
STEP_FILE = "step.json"

# The application checkpoint of a training loop that passes its trainer to the hook.
APP_FILE = "app.pt"


class Hook:
    """Talks to the Supervisor over its socket. Without a Supervisor, every call does nothing."""

    def __init__(self, socket_path=None):
        self._sock = None
        self._file = None
        if socket_path:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)
            self._attach(sock)

    @classmethod
    def from_socket(cls, sock):
        """A hook on a connection that is already open. Tests use it with a fake Supervisor."""
        hook = cls()
        hook._attach(sock)
        return hook

    def _attach(self, sock):
        self._sock = sock
        self._file = sock.makefile("rwb")

    def start(self, trainer=None):
        """Returns the last finished step: 0 for a fresh run, or K after a restore from step K.

        With a trainer (see hibernus.workload), a restore loads the application checkpoint into
        it. Without one, a restore reads the stand-in step file.
        """
        if self._file is None:
            return 0
        reply = self._send({"op": "start"})
        resume = reply["resume"]
        if resume is None:
            return 0
        if trainer is None:
            return _load_step(resume["files"], resume["step"])
        return _load_trainer(resume["files"], resume["step"], trainer)

    def boundary(self, step, trainer=None):
        """Reports that step finished. The training loop calls it at each update boundary.

        With a trainer, a pause saves its application checkpoint. The step must be the number of
        updates the trainer has completed.
        """
        if trainer is not None and step != trainer.progress.completed_updates:
            raise ValueError(
                f"boundary({step}), but the trainer has completed "
                f"{trainer.progress.completed_updates} updates; the step must match"
            )
        if self._file is None:
            return
        reply = self._send({"op": "boundary", "step": step})
        if reply["pause"]:
            self._checkpoint(step, reply["dir"], trainer)

    def _checkpoint(self, step, folder, trainer):
        try:
            os.makedirs(folder, exist_ok=True)
            if trainer is None:
                path = os.path.join(folder, STEP_FILE)
                with open(path, "w") as f:
                    json.dump({"step": step}, f)
            else:
                from hibernus import app_checkpoint  # needs torch, so only a trainer pulls it in

                path = os.path.join(folder, APP_FILE)
                app_checkpoint.save(path, trainer)
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


def _load_trainer(files, step, trainer):
    """Loads the application checkpoint into the trainer, and checks it holds the manifest's step."""
    from hibernus import app_checkpoint

    for path in files:
        if os.path.basename(path) == APP_FILE:
            app_checkpoint.load(path, trainer)
            saved = trainer.progress.completed_updates
            if saved != step:
                raise RuntimeError(f"{path} holds step {saved}, but the manifest says step {step}")
            return saved
    raise RuntimeError(f"the restore point has no {APP_FILE}")


def connect():
    return Hook(os.environ.get(SOCKET_ENV))
