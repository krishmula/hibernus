"""The hook's part of a process snapshot: close the socket, wait for the restore, reconnect.

CRIU can't restore a connection to a process outside the dump, so a hook told to snapshot closes
its socket after it reports the application checkpoint, and waits for the SIGUSR1 that a restore
sends. Then it opens a new connection and says "resumed". A fake Supervisor on socket pairs plays
the Supervisor's side. These tests need SIGUSR1, so they run on Linux.
"""

import json
import os
import signal
import socket
import threading
import time

import pytest

import hibernus

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="needs SIGUSR1 (Linux)")


class SnapshotSupervisor:
    """Answers boundary with a snapshot pause, then plays the restore.

    `wake_on` says when it sends SIGUSR1 to the hook: "closed" after the hook closes its first
    connection (the real order), or "reported" right after app_checkpoint arrives, before the hook
    could have closed anything (the hook must survive a signal that early).
    """

    def __init__(self, app_dir, wake_on="closed"):
        self.app_dir = app_dir
        self.wake_on = wake_on
        self.events = []  # what happened, in order
        self.requests = []
        self.second_connection_ok = threading.Event()
        ours, theirs = socket.socketpair()
        self._first = theirs
        self.hook = hibernus.Hook.from_socket(ours, reconnect=self._reconnect)
        threading.Thread(target=self._serve_first, daemon=True).start()

    def _serve_first(self):
        f = self._first.makefile("rwb")
        for line in f:
            request = json.loads(line)
            self.requests.append(request)
            if request["op"] == "boundary":
                reply = {"pause": True, "dir": str(self.app_dir), "snapshot": True}
                f.write(json.dumps(reply).encode() + b"\n")
                f.flush()
            elif request["op"] == "app_checkpoint":
                self.events.append("reported")
                if request.get("error"):
                    # The Supervisor fails the checkpoint and the host goes away, which the
                    # hook sees as a broken connection.
                    self._first.shutdown(socket.SHUT_RDWR)
                    return
                if self.wake_on == "reported":
                    self._wake()
        # The loop ends when the hook closes its end.
        self.events.append("closed")
        if self.wake_on == "closed":
            self._wake()

    def _wake(self):
        self.events.append("wake")
        os.kill(os.getpid(), signal.SIGUSR1)

    def _reconnect(self):
        self.events.append("reconnect")
        ours, theirs = socket.socketpair()
        threading.Thread(target=self._serve_second, args=(theirs,), daemon=True).start()
        return ours

    def _serve_second(self, sock):
        f = sock.makefile("rwb")
        for line in f:
            request = json.loads(line)
            self.requests.append(request)
            f.write(json.dumps({"ok": True}).encode() + b"\n")
            f.flush()
            if request["op"] == "resumed":
                self.second_connection_ok.set()


def test_the_hook_reports_closes_waits_then_reconnects_and_says_resumed(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    sup = SnapshotSupervisor(app)

    sup.hook.boundary(2)  # returns only after the wake and the reconnect

    assert sup.second_connection_ok.wait(timeout=5)
    assert [r["op"] for r in sup.requests] == ["boundary", "app_checkpoint", "resumed"]
    # The order is the point: the socket is closed before anything can dump the process.
    assert sup.events == ["reported", "closed", "wake", "reconnect"]
    assert (app / "step.json").exists()


def test_a_wake_signal_that_arrives_early_does_not_kill_the_hook(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    sup = SnapshotSupervisor(app, wake_on="reported")

    sup.hook.boundary(2)  # SIGUSR1 arrives before the hook has closed its socket

    assert sup.second_connection_ok.wait(timeout=5)


def test_the_hook_holds_no_connection_while_it_waits(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    sup = SnapshotSupervisor(app, wake_on="never")
    done = threading.Event()

    def pause():
        sup.hook.boundary(2)
        done.set()

    # The signal handler runs in the main thread, so the hook has to run there. Send the wake from a
    # helper once the hook has closed its connection.
    def waker():
        deadline = time.time() + 5
        while ("closed" not in sup.events or sup.hook._file is not None) and time.time() < deadline:
            time.sleep(0.01)
        # The hook is waiting now. Its connection is closed, so there is no fd for CRIU to dump.
        assert sup.hook._file is None and sup.hook._sock is None
        sup._wake()

    threading.Thread(target=waker, daemon=True).start()
    pause()

    assert done.is_set()


def test_the_wake_handler_is_put_back_after_the_wait(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    marker = []
    previous = signal.signal(signal.SIGUSR1, lambda *_: marker.append("user handler"))
    try:
        sup = SnapshotSupervisor(app)
        sup.hook.boundary(2)
        assert signal.getsignal(signal.SIGUSR1) is not signal.SIG_DFL
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.1)
        assert marker == ["user handler"], "the user's own SIGUSR1 handler was not restored"
    finally:
        signal.signal(signal.SIGUSR1, previous)


def test_a_failed_save_falls_back_to_waiting_on_the_socket(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the folder should go")
    sup = SnapshotSupervisor(blocker / "app", wake_on="never")  # makedirs fails under a file
    before = signal.getsignal(signal.SIGUSR1)

    with pytest.raises(ConnectionError):  # nothing will dump, so nothing wakes the hook
        sup.hook.boundary(2)

    assert [r["op"] for r in sup.requests] == ["boundary", "app_checkpoint"]
    assert sup.requests[1].get("error"), "the Supervisor was not told why the save failed"
    assert signal.getsignal(signal.SIGUSR1) is before, "the wake handler was left installed"


def test_without_snapshot_the_hook_still_waits_on_the_socket_as_before(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    ours, theirs = socket.socketpair()
    hook = hibernus.Hook.from_socket(ours)

    def supervisor():
        f = theirs.makefile("rwb")
        f.readline()
        f.write(json.dumps({"pause": True, "dir": str(app)}).encode() + b"\n")
        f.flush()
        f.readline()  # app_checkpoint
        f.close()
        theirs.shutdown(socket.SHUT_RDWR)
        theirs.close()

    threading.Thread(target=supervisor, daemon=True).start()

    with pytest.raises(ConnectionError):  # the first build's behavior
        hook.boundary(2)


def _trainer_that_logs_close(events, fail=False):
    """A real trainer whose close_data also writes to the fake Supervisor's event log."""
    from hibernus.workload import build_trainer

    trainer = build_trainer()
    trainer.run_update()  # so boundary(1) matches the trainer's completed updates
    original = trainer.close_data

    def close_data():
        events.append("workers closed")
        if fail:
            raise RuntimeError("could not stop the workers")
        original()

    trainer.close_data = close_data
    return trainer


def test_a_snapshot_pause_closes_the_workers_after_the_save_and_before_the_report(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    sup = SnapshotSupervisor(app)
    trainer = _trainer_that_logs_close(sup.events)

    sup.hook.boundary(1, trainer=trainer)

    assert sup.second_connection_ok.wait(timeout=5)
    # The Supervisor dumps only after it hears "reported", so the workers must be gone by then.
    assert sup.events == ["workers closed", "reported", "closed", "wake", "reconnect"]
    assert (app / "app.pt").exists()  # the save came first: the checkpoint holds the open cursor


def test_a_failed_worker_shutdown_is_reported_like_a_failed_save(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    sup = SnapshotSupervisor(app, wake_on="never")
    trainer = _trainer_that_logs_close(sup.events, fail=True)

    with pytest.raises(ConnectionError):  # nothing will dump, so nothing wakes the hook
        sup.hook.boundary(1, trainer=trainer)

    assert sup.requests[1]["op"] == "app_checkpoint"
    assert "could not stop the workers" in sup.requests[1]["error"]


def test_a_pause_without_a_snapshot_leaves_the_workers_alone(tmp_path):
    from hibernus.workload import build_trainer

    app = tmp_path / "app"
    app.mkdir()
    trainer = build_trainer()
    trainer.run_update()
    closed = []
    trainer.close_data = lambda: closed.append(True)
    ours, theirs = socket.socketpair()
    hook = hibernus.Hook.from_socket(ours)

    def supervisor():
        f = theirs.makefile("rwb")
        f.readline()
        f.write(json.dumps({"pause": True, "dir": str(app)}).encode() + b"\n")
        f.flush()
        f.readline()  # app_checkpoint
        f.close()
        theirs.shutdown(socket.SHUT_RDWR)
        theirs.close()

    threading.Thread(target=supervisor, daemon=True).start()

    with pytest.raises(ConnectionError):
        hook.boundary(1, trainer=trainer)

    assert closed == []  # the process is paused until the host goes away, so nothing is dumped
