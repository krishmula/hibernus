package supervisor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
)

// shortSocketPath avoids t.TempDir, whose paths on macOS are too long for a socket.
func shortSocketPath(t *testing.T) string {
	dir, err := os.MkdirTemp("", "hib")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return filepath.Join(dir, "w.sock")
}

// checkpointHost is the fingerprint the tests' training hook reports.
var checkpointHost = manifest.Fingerprint{
	GPU: "NVIDIA A10G", Driver: "550.54.15", CUDA: "12.4", CuDNN: "90100", PyTorch: "2.5.1", Kernel: "6.8.0-1015-aws",
}

const testPython = "/venv/bin/python"

func initializeRequest() protocol.Request {
	fingerprint := checkpointHost
	return protocol.Request{Op: "initialize", Fingerprint: &fingerprint, Python: testPython}
}

func initialize(t *testing.T, path string) protocol.InitializeReply {
	t.Helper()
	var reply protocol.InitializeReply
	if err := protocol.Send(path, initializeRequest(), &reply); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	return reply
}

// serve starts a Supervisor on a fresh socket and returns the socket path. For a fresh run, it
// sends initialize as the training hook would, so the Supervisor knows the fingerprint.
func serve(t *testing.T, sup *Supervisor) string {
	t.Helper()
	path := serveWithoutInitialize(t, sup)
	if sup.resume == nil {
		initialize(t, path)
	}
	return path
}

func serveWithoutInitialize(t *testing.T, sup *Supervisor) string {
	t.Helper()
	path := shortSocketPath(t)
	listener, err := Listen(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { listener.Close() })
	go sup.Serve(listener)
	return path
}

func status(t *testing.T, path string) protocol.StatusReply {
	t.Helper()
	var reply protocol.StatusReply
	if err := protocol.Send(path, protocol.Request{Op: "status"}, &reply); err != nil {
		t.Fatalf("status: %v", err)
	}
	return reply
}

func TestANewWorkloadIsRunningAtStepZero(t *testing.T) {
	path := serve(t, New(Config{Workload: "w"}, nil))
	if got := status(t, path); got.State != "Running" || got.Step != 0 {
		t.Fatalf("status = %+v, want Running at step 0", got)
	}
}

func TestInitializeOnAFreshRunHasNothingToResume(t *testing.T) {
	path := serveWithoutInitialize(t, New(Config{Workload: "w"}, nil))
	if reply := initialize(t, path); reply.Resume != nil {
		t.Fatalf("initialize = %+v, want no resume", reply)
	}
}

func TestInitializeWithoutAFingerprintIsRefused(t *testing.T) {
	path := serveWithoutInitialize(t, New(Config{Workload: "w"}, nil))
	var reply protocol.InitializeReply
	err := protocol.Send(path, protocol.Request{Op: "initialize"}, &reply)
	if err == nil || !strings.Contains(err.Error(), "no fingerprint") {
		t.Fatalf("initialize without a fingerprint = %v, want a refusal", err)
	}
}

func TestInitializeWithAnIncompleteFingerprintIsRefused(t *testing.T) {
	path := serveWithoutInitialize(t, New(Config{Workload: "w"}, nil))
	req := initializeRequest()
	req.Fingerprint.Driver = ""
	var reply protocol.InitializeReply
	err := protocol.Send(path, req, &reply)
	if err == nil || !strings.Contains(err.Error(), "driver") {
		t.Fatalf("initialize without a driver = %v, want a refusal that names the driver", err)
	}
}

func TestBoundaryRecordsTheStepWithoutPausing(t *testing.T) {
	path := serve(t, New(Config{Workload: "w"}, nil))
	if reply := boundary(t, path, 5); reply.Pause {
		t.Fatalf("boundary with no checkpoint requested = %+v, want no pause", reply)
	}
	if got := status(t, path); got.Step != 5 {
		t.Fatalf("step after boundary = %d, want 5", got.Step)
	}
}

func TestUnknownOpIsRefused(t *testing.T) {
	path := serve(t, New(Config{Workload: "w"}, nil))
	var reply protocol.StatusReply
	err := protocol.Send(path, protocol.Request{Op: "dance"}, &reply)
	if err == nil || err.Error() != `unknown op "dance"` {
		t.Fatalf("unknown op = %v, want a refusal", err)
	}
}

func TestCleanExitFinishesTheWorkload(t *testing.T) {
	sup := New(Config{Workload: "w"}, nil)
	path := serve(t, sup)
	sup.ChildExited(0)
	if got := status(t, path); got.State != "Finished" {
		t.Fatalf("state after a clean exit = %s, want Finished", got.State)
	}
}

func TestListenRefusesALiveSupervisor(t *testing.T) {
	path := serve(t, New(Config{Workload: "w"}, nil))
	second, err := Listen(path)
	if err == nil {
		second.Close()
	}
	if err == nil || !strings.Contains(err.Error(), "already runs") {
		t.Fatalf("a second Listen on a live socket = %v, want a refusal", err)
	}
}

func TestListenReplacesALeftoverSocketFile(t *testing.T) {
	path := shortSocketPath(t)
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	listener, err := Listen(path)
	if err != nil {
		t.Fatalf("Listen over a leftover file: %v", err)
	}
	listener.Close()
}

func restoredRequest(step int) protocol.Request {
	req := initializeRequest()
	req.Op = protocol.OpRestored
	req.Step = step
	return req
}

func restored(path string, step int) (protocol.RestoredReply, error) {
	var reply protocol.RestoredReply
	err := protocol.Send(path, restoredRequest(step), &reply)
	return reply, err
}

// restoreFromSnapshot stands in for hib restore after criu restore: a Supervisor in Restoring at
// step 7, which waits for the training process to send restored.
func restoreFromSnapshot(t *testing.T) (*Supervisor, string) {
	t.Helper()
	sup := New(Config{Workload: "w", Store: t.TempDir()}, &protocol.Resume{Step: 7})
	return sup, serveWithoutInitialize(t, sup)
}

func TestRestoredAtTheRestorePointStepContinuesTraining(t *testing.T) {
	sup, path := restoreFromSnapshot(t)

	reply, err := restored(path, 7)
	if err != nil || !reply.Continue {
		t.Fatalf("restored at step 7 = %+v, %v, want continue", reply, err)
	}
	if got := status(t, path); got.State != "Running" || got.Step != 7 {
		t.Fatalf("status = %+v, want Running at step 7", got)
	}
	if sup.host == nil || sup.host.python != testPython || sup.host.fingerprint != checkpointHost {
		t.Fatalf("host = %+v, want the fingerprint and python that restored carried", sup.host)
	}
	if got := boundary(t, path, 8); got.Pause {
		t.Fatal("step 8 paused, want training to continue")
	}
}

func TestRestoredAtAnotherStepIsRefused(t *testing.T) {
	_, path := restoreFromSnapshot(t)

	_, err := restored(path, 6)
	if err == nil || !strings.Contains(err.Error(), "at step 6, but the restore point holds step 7") {
		t.Fatalf("restored at step 6 = %v, want a refusal that names both steps", err)
	}
	if got := status(t, path); got.State != "Restoring" {
		t.Fatalf("state after the refusal = %s, want Restoring", got.State)
	}
}

func TestRestoredWithoutAFingerprintIsRefused(t *testing.T) {
	_, path := restoreFromSnapshot(t)

	var reply protocol.RestoredReply
	err := protocol.Send(path, protocol.Request{Op: protocol.OpRestored, Step: 7}, &reply)
	if err == nil || !strings.Contains(err.Error(), "restored carries no fingerprint") {
		t.Fatalf("restored without a fingerprint = %v, want a refusal", err)
	}
}

func TestRestoredOutsideARestoreIsRefused(t *testing.T) {
	path := serve(t, New(Config{Workload: "w"}, nil))

	_, err := restored(path, 0)
	if err == nil || !strings.Contains(err.Error(), "the workload is Running") {
		t.Fatalf("restored on a fresh run = %v, want a refusal", err)
	}
}
