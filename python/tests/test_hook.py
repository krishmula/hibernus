"""The training hook saves and loads the application checkpoint when the Supervisor asks.

A fake Supervisor on a socket pair stands in for the real one. It speaks the same one-line JSON
messages (see docs/design/data-shapes.md).
"""

import json
import socket
import threading

import pytest
import torch

import hibernus
from hibernus import app_checkpoint
from hibernus.workload import build_trainer


class FakeSupervisor:
    """Answers each message from a table. After `app_checkpoint` it closes the connection, which
    is how a checkpoint host going away looks to the hook."""

    def __init__(self, replies):
        self.replies = replies
        self.requests = []
        ours, theirs = socket.socketpair()
        self.hook = hibernus.Hook.from_socket(ours)
        self._sock = theirs
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        f = self._sock.makefile("rwb")
        for line in f:
            request = json.loads(line)
            self.requests.append(request)
            if request["op"] == "app_checkpoint":
                break
            f.write(json.dumps(self.replies[request["op"]]).encode() + b"\n")
            f.flush()
        f.close()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()

    def finish(self):
        self._thread.join(timeout=5)


def run(trainer, updates):
    return [trainer.run_update() for _ in range(updates)]


def test_a_pause_writes_app_pt_and_reports_it(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(app)}})
    trainer = build_trainer()
    run(trainer, 2)

    with pytest.raises(ConnectionError):  # the hook waits until the host goes away
        sup.hook.boundary(2, trainer=trainer)
    sup.finish()

    assert sup.requests == [
        {"op": "boundary", "step": 2},
        {"op": "app_checkpoint", "files": [str(app / "app.pt")]},
    ]
    # Both trainers draw from the one global RNG, so run the original first, then load.
    expected = trainer.run_update()
    restored = build_trainer(init_seed=9)
    app_checkpoint.load(app / "app.pt", restored)
    assert restored.run_update() == expected


def test_no_pause_means_no_file(tmp_path):
    sup = FakeSupervisor({"boundary": {"pause": False}})
    trainer = build_trainer()
    run(trainer, 1)

    sup.hook.boundary(1, trainer=trainer)

    assert list(tmp_path.iterdir()) == []


def test_a_failed_save_is_reported_to_the_supervisor(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(app)}})
    trainer = build_trainer()
    run(trainer, 1)

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", disk_full)
    with pytest.raises(ConnectionError):
        sup.hook.boundary(1, trainer=trainer)
    sup.finish()

    assert sup.requests[1] == {"op": "app_checkpoint", "error": "disk full"}


def test_a_boundary_step_that_disagrees_with_the_trainer_is_refused():
    sup = FakeSupervisor({"boundary": {"pause": False}})
    trainer = build_trainer()
    run(trainer, 2)

    with pytest.raises(ValueError, match="step"):
        sup.hook.boundary(5, trainer=trainer)

    assert sup.requests == []  # nothing was reported


def test_initialize_after_a_restore_loads_the_trainer_and_returns_the_step(tmp_path):
    original = build_trainer()
    run(original, 3)
    app_checkpoint.save(tmp_path / "app.pt", original)
    expected = original.run_update()  # before the load, because both share the global RNG
    sup = FakeSupervisor({"initialize": {"resume": {"step": 3, "files": [str(tmp_path / "app.pt")]}}})
    fresh = build_trainer(init_seed=9)

    step = sup.hook.initialize(trainer=fresh)

    assert step == 3
    assert fresh.progress.completed_updates == 3
    assert fresh.run_update() == expected


def test_initialize_refuses_a_checkpoint_that_disagrees_with_the_manifest_step(tmp_path):
    original = build_trainer()
    run(original, 3)
    app_checkpoint.save(tmp_path / "app.pt", original)
    sup = FakeSupervisor({"initialize": {"resume": {"step": 4, "files": [str(tmp_path / "app.pt")]}}})

    with pytest.raises(RuntimeError, match="step"):
        sup.hook.initialize(trainer=build_trainer())


def test_initialize_refuses_a_restore_point_without_app_pt(tmp_path):
    other = tmp_path / "step.json"
    other.write_text('{"step": 3}')
    sup = FakeSupervisor({"initialize": {"resume": {"step": 3, "files": [str(other)]}}})

    with pytest.raises(RuntimeError, match="app.pt"):
        sup.hook.initialize(trainer=build_trainer())


def test_a_fresh_initialize_leaves_the_trainer_alone():
    sup = FakeSupervisor({"initialize": {"resume": None}})
    trainer = build_trainer()

    assert sup.hook.initialize(trainer=trainer) == 0
    assert trainer.progress.completed_updates == 0


def test_without_a_supervisor_a_trainer_trains_normally(monkeypatch):
    monkeypatch.delenv("HIB_SOCKET", raising=False)
    hook = hibernus.connect()
    assert isinstance(hook, hibernus.NoSupervisor)
    trainer = build_trainer()

    assert hook.initialize(trainer=trainer) == 0
    for _ in range(3):
        trainer.run_update()
        hook.boundary(trainer.progress.completed_updates, trainer=trainer)

    assert trainer.progress.completed_updates == 3


def test_without_a_trainer_the_stand_in_step_file_still_works(tmp_path):
    app = tmp_path / "app"
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(app)}})

    with pytest.raises(ConnectionError):
        sup.hook.boundary(7)
    sup.finish()

    assert json.loads((app / "step.json").read_text()) == {"step": 7}
    assert sup.requests[1]["files"] == [str(app / "step.json")]
