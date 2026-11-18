"""The Hibernus training hook. It runs inside the training process and talks to the Supervisor."""

import json
import os
import socket

SOCKET_ENV = "HIB_SOCKET"

# The stand-in application checkpoint of the first build: one file that holds only the step.
STEP_FILE = "step.json"


class Hook:
    """Talks to the Supervisor over its socket."""

    def __init__(self, socket_path):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._file = self._sock.makefile("rwb")

    def initialize(self):
        """Returns the last finished step: 0 for a fresh run, or K after a restore from step K."""
        reply = self._send({"op": "initialize"})
        resume = reply["resume"]
        if resume is None:
            return 0
        return _load_step(resume["files"], resume["step"])

    def boundary(self, step):
        """Reports that step finished. The training loop calls it at each update boundary."""
        reply = self._send({"op": "boundary", "step": step})
        if reply["pause"]:
            self._checkpoint(step, reply["dir"])

    def _checkpoint(self, step, folder):
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, STEP_FILE)
            with open(path, "w") as f:
                json.dump({"step": step}, f)
            self._write({"op": "app_checkpoint", "files": [path]})
        except OSError as e:
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


class NoSupervisor:
    """Stands in for Hook when the training process runs without hib, so training runs as usual."""

    def initialize(self):
        return 0

    def boundary(self, step):
        pass


def connect():
    socket_path = os.environ.get(SOCKET_ENV)
    if socket_path:
        return Hook(socket_path)
    return NoSupervisor()
