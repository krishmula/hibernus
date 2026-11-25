"""The training hook saves and loads the application checkpoint when the Supervisor asks.

A fake Supervisor on a socket pair stands in for the real one. It speaks the same one-line JSON
messages (see docs/design/data-shapes.md).
"""

import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time

import pytest
import torch

import hibernus
from hibernus import app_checkpoint, fingerprint
from hibernus.workload import build_trainer


class FakeSupervisor:
    """Answers each message from a table, on a real socket. A list in the table gives one answer
    per request, in order.

    When the hook closes its connection after `app_checkpoint`, the fake sends SIGCONT to this
    process, as the restore host's Supervisor does after a restore from the process snapshot. Then
    it serves the hook's new connection, which starts with `restored`.
    """

    def __init__(self, replies):
        self.replies = {"restored": {"continue": True}, **replies}
        self.requests = []
        self._dir = tempfile.mkdtemp(dir="/tmp")  # short, because a socket path must stay under ~100 bytes
        path = os.path.join(self._dir, "w.sock")
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(path)
        self._listener.listen()
        self._listener.settimeout(0.1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.hook = hibernus.Hook(path)

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            conn.settimeout(None)
            with conn, conn.makefile("rwb") as f:
                for line in f:
                    request = json.loads(line)
                    self.requests.append(request)
                    if request["op"] != "app_checkpoint":
                        reply = self.replies[request["op"]]
                        if isinstance(reply, list):  # one answer per request, in order
                            reply = reply.pop(0)
                        f.write(json.dumps(reply).encode() + b"\n")
                        f.flush()
            if self.requests and self.requests[-1]["op"] == "app_checkpoint":
                os.kill(os.getpid(), signal.SIGCONT)

    def finish(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._listener.close()
        shutil.rmtree(self._dir)


def run(trainer, updates):
    return [trainer.run_update() for _ in range(updates)]


def test_a_pause_writes_app_pt_and_reports_it(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(app)}})
    trainer = build_trainer()
    run(trainer, 2)

    sup.hook.boundary(2, {"trainer": trainer})
    sup.finish()

    assert sup.requests[:2] == [
        {"op": "boundary", "step": 2},
        {"op": "app_checkpoint", "files": [str(app / "app.pt")]},
    ]
    # Both trainers draw from the one global RNG, so run the original first, then load.
    expected = trainer.run_update()
    restored = build_trainer(init_seed=9)
    app_checkpoint.load(app / "app.pt", {"trainer": restored})
    assert restored.run_update() == expected


def test_no_pause_means_no_file(tmp_path):
    sup = FakeSupervisor({"boundary": {"pause": False}})
    trainer = build_trainer()
    run(trainer, 1)

    sup.hook.boundary(1, {"trainer": trainer})

    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_is_reported_to_the_supervisor(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(app)}})
    trainer = build_trainer()
    run(trainer, 1)

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", disk_full)
    sup.hook.boundary(1, {"trainer": trainer})
    sup.finish()

    assert sup.requests[1] == {"op": "app_checkpoint", "error": "disk full"}


def test_after_sigcont_the_hook_sends_restored_and_training_continues(tmp_path):
    sup = FakeSupervisor({"boundary": [{"pause": True, "dir": str(tmp_path)}, {"pause": False}]})

    sup.hook.boundary(7)  # returns only after SIGCONT and the Supervisor's answer to restored
    sup.hook.boundary(8)
    sup.finish()

    assert [r["op"] for r in sup.requests] == ["boundary", "app_checkpoint", "restored", "boundary"]
    restored = sup.requests[2]
    assert restored["step"] == 7
    assert restored["fingerprint"] == fingerprint.collect()
    assert restored["python"] == sys.executable
    assert sup.requests[3] == {"op": "boundary", "step": 8}


def test_the_pause_lasts_until_sigcont(tmp_path, monkeypatch):
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(tmp_path)}})
    # This fake never sends SIGCONT itself. After 0.5 s, raise_signal sends it to the timer's own
    # thread, not the main thread, as the kernel may do. A bare woke.wait() would never see it.
    monkeypatch.setattr(os, "kill", lambda pid, signum: None)
    timer = threading.Timer(0.5, signal.raise_signal, [signal.SIGCONT])
    start = time.monotonic()
    timer.start()

    sup.hook.boundary(7)
    sup.finish()

    assert time.monotonic() - start >= 0.5
    assert sup.requests[-1]["op"] == "restored"
    assert signal.getsignal(signal.SIGCONT) == signal.SIG_DFL  # the hook put the old handler back


def test_a_refused_restored_stops_training(tmp_path):
    sup = FakeSupervisor(
        {
            "boundary": {"pause": True, "dir": str(tmp_path)},
            "restored": {"refused": "the training process is at step 7, but the restore point holds step 6"},
        }
    )

    with pytest.raises(RuntimeError, match="refused restored"):
        sup.hook.boundary(7)
    sup.finish()


@pytest.fixture
def sigterm_handler():
    previous = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    yield
    signal.signal(signal.SIGTERM, previous)


def test_under_hib_sigterm_ends_training_with_143(sigterm_handler):
    sup = FakeSupervisor({})
    sup.finish()

    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL  # else raise_signal ends pytest
    with pytest.raises(SystemExit) as stopped:
        signal.raise_signal(signal.SIGTERM)
    assert stopped.value.code == 143


def test_the_training_scripts_own_sigterm_handler_stays(sigterm_handler):
    def own(signum, frame):
        pass

    signal.signal(signal.SIGTERM, own)
    sup = FakeSupervisor({})
    sup.finish()

    assert signal.getsignal(signal.SIGTERM) is own


def test_initialize_after_a_restore_loads_the_trainer_and_returns_the_step(tmp_path):
    original = build_trainer()
    run(original, 3)
    app_checkpoint.save(tmp_path / "app.pt", 3, {"trainer": original})
    expected = original.run_update()  # before the load, because both share the global RNG
    sup = FakeSupervisor({"initialize": {"resume": {"step": 3, "files": [str(tmp_path / "app.pt")]}}})
    fresh = build_trainer(init_seed=9)

    step = sup.hook.initialize({"trainer": fresh})

    assert step == 3
    assert fresh.progress.completed_updates == 3
    assert fresh.run_update() == expected


def test_initialize_refuses_a_checkpoint_that_disagrees_with_the_manifest_step(tmp_path):
    original = build_trainer()
    run(original, 3)
    app_checkpoint.save(tmp_path / "app.pt", 3, {"trainer": original})
    sup = FakeSupervisor({"initialize": {"resume": {"step": 4, "files": [str(tmp_path / "app.pt")]}}})

    fresh = build_trainer(init_seed=9)
    before = {name: value.clone() for name, value in fresh.model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="step"):
        sup.hook.initialize({"trainer": fresh})

    assert fresh.progress.completed_updates == 0
    assert all(torch.equal(before[name], value) for name, value in fresh.model.state_dict().items())
    assert torch.equal(rng_before, torch.get_rng_state())


def test_initialize_refuses_a_restore_point_without_app_pt(tmp_path):
    other = tmp_path / "step.json"
    other.write_text('{"step": 3}')
    sup = FakeSupervisor({"initialize": {"resume": {"step": 3, "files": [str(other)]}}})

    with pytest.raises(RuntimeError, match="app.pt"):
        sup.hook.initialize({"trainer": build_trainer()})


def test_a_fresh_initialize_leaves_the_trainer_alone():
    sup = FakeSupervisor({"initialize": {"resume": None}})
    trainer = build_trainer()

    assert sup.hook.initialize({"trainer": trainer}) == 0
    assert trainer.progress.completed_updates == 0


def test_initialize_sends_the_fingerprint_and_the_interpreter():
    sup = FakeSupervisor({"initialize": {"resume": None}})
    sup.hook.initialize()

    request = sup.requests[0]
    assert request["fingerprint"] == fingerprint.collect()
    assert request["python"] == sys.executable


def test_without_a_supervisor_a_trainer_trains_normally(monkeypatch):
    monkeypatch.delenv("HIB_SOCKET", raising=False)
    hook = hibernus.connect()
    assert isinstance(hook, hibernus.NoSupervisor)
    trainer = build_trainer()

    assert hook.initialize({"trainer": trainer}) == 0
    for _ in range(3):
        trainer.run_update()
        hook.boundary(trainer.progress.completed_updates, {"trainer": trainer})

    assert trainer.progress.completed_updates == 3


def test_without_state_the_stand_in_step_file_still_works(tmp_path):
    app = tmp_path / "app"
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(app)}})

    sup.hook.boundary(7)
    sup.finish()

    assert json.loads((app / "step.json").read_text()) == {"step": 7}
    assert sup.requests[1]["files"] == [str(app / "step.json")]


def plain_loop(seed):
    """A model, optimizer, and scheduler held in plain variables, with no Trainer."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    data = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))

    def update(step):
        loss = model(data[step % len(data)]).pow(2).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return loss.item()

    return {"model": model, "optimizer": optimizer, "scheduler": scheduler}, update


def test_a_plain_loop_pauses_and_restores_to_the_same_losses(tmp_path):
    state, update = plain_loop(seed=0)
    expected = [update(step) for step in range(6)]

    app = tmp_path / "app"
    sup = FakeSupervisor({"boundary": {"pause": True, "dir": str(app)}})
    state, update = plain_loop(seed=0)
    before = [update(step) for step in range(3)]
    sup.hook.boundary(3, state, {"batch_size": 1})
    sup.finish()

    files = sup.requests[1]["files"]
    sup = FakeSupervisor({"initialize": {"resume": {"step": 3, "files": files}}})
    state, update = plain_loop(seed=9)  # other weights and RNG stream, so only the load can match
    step = sup.hook.initialize(state, {"batch_size": 1})
    after = [update(s) for s in range(step, 6)]

    assert step == 3
    assert before + after == expected


def test_initialize_refuses_a_restore_with_a_different_config(tmp_path):
    state, _ = plain_loop(seed=0)
    app_checkpoint.save(tmp_path / "app.pt", 3, state, {"batch_size": 1})
    sup = FakeSupervisor({"initialize": {"resume": {"step": 3, "files": [str(tmp_path / "app.pt")]}}})

    with pytest.raises(ValueError, match="batch_size was 1, now 2"):
        sup.hook.initialize(plain_loop(seed=9)[0], {"batch_size": 2})
