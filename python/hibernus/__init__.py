"""The Hibernus training hook. It runs inside the training process and talks to the Supervisor."""

import os

SOCKET_ENV = "HIB_SOCKET"


class Hook:
    def __init__(self, socket_path):
        self.socket_path = socket_path

    def start(self):
        return 0

    def boundary(self, step):
        pass


def connect():
    return Hook(os.environ.get(SOCKET_ENV))
