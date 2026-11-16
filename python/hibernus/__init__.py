"""The Hibernus training hook. It runs inside the training process and talks to the Supervisor."""

import json
import os
import socket

SOCKET_ENV = "HIB_SOCKET"


class Hook:
    """Talks to the Supervisor over its socket. Without a Supervisor, every call does nothing."""

    def __init__(self, socket_path=None):
        self._sock = None
        self._file = None
        if socket_path:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(socket_path)
            self._file = self._sock.makefile("rwb")

    def start(self):
        """Returns the last finished step: 0 for a fresh run."""
        if self._file is None:
            return 0
        reply = self._send({"op": "start"})
        if reply["resume"] is not None:
            raise NotImplementedError("resuming from a restore point is not built yet")
        return 0

    def boundary(self, step):
        """Reports that step finished. The training loop calls it at each update boundary."""
        if self._file is None:
            return
        reply = self._send({"op": "boundary", "step": step})
        if reply["pause"]:
            raise NotImplementedError("pausing for a checkpoint is not built yet")

    def _send(self, message):
        self._file.write(json.dumps(message).encode() + b"\n")
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise ConnectionError("the Supervisor closed the connection")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"the Supervisor refused {message['op']}: {reply['error']}")
        return reply


def connect():
    return Hook(os.environ.get(SOCKET_ENV))
