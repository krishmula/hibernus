package supervisor

import (
	"os"
	"path/filepath"
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

func status(t *testing.T, path string) protocol.StatusReply {
	t.Helper()
	var reply protocol.StatusReply
	if err := protocol.Send(path, protocol.Request{Op: "status"}, &reply); err != nil {
		t.Fatalf("status: %v", err)
	}
	return reply
}

func TestSocketConversation(t *testing.T) {
	path := shortSocketPath(t)
	listener, err := Listen(path)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	sup := New(Config{Workload: "w"}, nil)
	go sup.Serve(listener)

	if got := status(t, path); got.State != "Running" || got.Step != 0 {
		t.Fatalf("first status = %+v, want Running at step 0", got)
	}

	var start protocol.StartReply
	if err := protocol.Send(path, protocol.Request{Op: "start"}, &start); err != nil || start.Resume != nil {
		t.Fatalf("start = %+v, %v; want no resume", start, err)
	}

	var boundary protocol.BoundaryReply
	if err := protocol.Send(path, protocol.Request{Op: "boundary", Step: 5}, &boundary); err != nil || boundary.Pause {
		t.Fatalf("boundary = %+v, %v; want no pause", boundary, err)
	}
	if got := status(t, path); got.Step != 5 {
		t.Fatalf("step after boundary = %d, want 5", got.Step)
	}

	if err := protocol.Send(path, protocol.Request{Op: "dance"}, &boundary); err == nil {
		t.Fatal("unknown op succeeded, want an error")
	}

	sup.ChildExited(0)
	if got := status(t, path); got.State != "Finished" {
		t.Fatalf("state after a clean exit = %s, want Finished", got.State)
	}
}

func TestListenRefusesALiveSupervisor(t *testing.T) {
	path := shortSocketPath(t)
	first, err := Listen(path)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	go New(Config{Workload: "w"}, nil).Serve(first)

	if second, err := Listen(path); err == nil {
		second.Close()
		t.Fatal("a second Listen on a live socket succeeded")
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
