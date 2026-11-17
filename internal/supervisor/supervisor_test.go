package supervisor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

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

// serve starts a Supervisor on a fresh socket and returns the socket path.
func serve(t *testing.T, sup *Supervisor) string {
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

func TestStartOnAFreshRunHasNothingToResume(t *testing.T) {
	path := serve(t, New(Config{Workload: "w"}, nil))
	var start protocol.StartReply
	if err := protocol.Send(path, protocol.Request{Op: "start"}, &start); err != nil || start.Resume != nil {
		t.Fatalf("start = %+v, %v; want no resume", start, err)
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
