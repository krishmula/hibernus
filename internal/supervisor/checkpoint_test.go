package supervisor

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
)

// requestCheckpoint sends hib checkpoint's request in the background and returns where its answer arrives.
func requestCheckpoint(t *testing.T, path string, deadline time.Duration) <-chan protocol.CheckpointReply {
	t.Helper()
	answer := make(chan protocol.CheckpointReply, 1)
	go func() {
		var reply protocol.CheckpointReply
		if err := protocol.Send(path, protocol.Request{Op: "checkpoint", Deadline: time.Now().Add(deadline)}, &reply); err != nil {
			t.Errorf("hib checkpoint: %v", err)
		}
		answer <- reply
	}()
	return answer
}

func waitForState(t *testing.T, path string, want string) {
	t.Helper()
	for range 200 {
		if status(t, path).State == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("state never became %s; it is %s", want, status(t, path).State)
}

func receive(t *testing.T, answer <-chan protocol.CheckpointReply) protocol.CheckpointReply {
	t.Helper()
	select {
	case reply := <-answer:
		return reply
	case <-time.After(5 * time.Second):
		t.Fatal("hib checkpoint got no answer")
		return protocol.CheckpointReply{}
	}
}

func boundary(t *testing.T, path string, step int) protocol.BoundaryReply {
	t.Helper()
	var reply protocol.BoundaryReply
	if err := protocol.Send(path, protocol.Request{Op: "boundary", Step: step}, &reply); err != nil {
		t.Fatalf("boundary %d: %v", step, err)
	}
	return reply
}

// tell sends a message that gets no reply, such as app_checkpoint.
func tell(t *testing.T, path string, req protocol.Request) {
	t.Helper()
	conn, err := net.Dial("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := json.NewEncoder(conn).Encode(req); err != nil {
		t.Fatal(err)
	}
}

// writeStepFile does what the training hook does after "pause": it writes the stand-in
// application checkpoint and reports it.
func writeStepFile(t *testing.T, path, dir string, step int) {
	t.Helper()
	file := filepath.Join(dir, "step.json")
	if err := os.WriteFile(file, fmt.Appendf(nil, `{"step": %d}`, step), 0o644); err != nil {
		t.Fatal(err)
	}
	tell(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{file}})
}

// finishCheckpoint does what the training hook does at the first update boundary after a checkpoint
// request: it pauses and writes the application checkpoint.
func finishCheckpoint(t *testing.T, path string, step int) {
	t.Helper()
	waitForState(t, path, "Checkpointing")
	reply := boundary(t, path, step)
	if !reply.Pause {
		t.Fatalf("boundary in Checkpointing = %+v, want a pause", reply)
	}
	writeStepFile(t, path, reply.Dir, step)
}

// checkpointed returns a Supervisor's socket path after a checkpoint at step 2, and the restore point folder.
func checkpointed(t *testing.T) (path, root string) {
	t.Helper()
	store := t.TempDir()
	path = serve(t, New(Config{Workload: "w", Store: store}, nil))
	answer := requestCheckpoint(t, path, 5*time.Second)
	finishCheckpoint(t, path, 2)
	receive(t, answer)
	return path, filepath.Join(store, "w", "step-2")
}

func TestCheckpointWritesARestorePoint(t *testing.T) {
	store := t.TempDir()
	path := serve(t, New(Config{Workload: "w", Store: store}, nil))
	answer := requestCheckpoint(t, path, 5*time.Second)
	waitForState(t, path, "Checkpointing")

	reply := boundary(t, path, 2)
	if !reply.Pause || reply.Dir != filepath.Join(store, "w", "step-2", "app") {
		t.Fatalf("boundary in Checkpointing = %+v, want a pause with the app folder", reply)
	}
	writeStepFile(t, path, reply.Dir, 2)

	root := filepath.Join(store, "w", "step-2")
	if got := receive(t, answer); !got.OK || got.Location != root || got.Manifest == nil || got.Manifest.Step != 2 {
		t.Fatalf("checkpoint = %+v, want ok at %s for step 2", got, root)
	}
	m, err := manifest.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Layers.ApplicationCheckpoint.Verify(root); err != nil {
		t.Fatal(err)
	}
	if m.Layers.ProcessSnapshot.Status != manifest.StatusFailed {
		t.Fatalf("process snapshot status = %s, want failed", m.Layers.ProcessSnapshot.Status)
	}
}

func TestRequestsDuringACheckpointShareItsAnswer(t *testing.T) {
	path := serve(t, New(Config{Workload: "w", Store: t.TempDir()}, nil))
	first := requestCheckpoint(t, path, 5*time.Second)
	waitForState(t, path, "Checkpointing")
	second := requestCheckpoint(t, path, 5*time.Second)
	finishCheckpoint(t, path, 2)

	a, b := receive(t, first), receive(t, second)
	if !a.OK || b.Location != a.Location {
		t.Fatalf("the two requests got %+v and %+v, want the same restore point", a, b)
	}
}

func TestStatusAfterACheckpointReportsTheRestorePoint(t *testing.T) {
	path, root := checkpointed(t)
	if got := status(t, path); got.State != "Checkpointed" || got.Location == nil || *got.Location != root {
		t.Fatalf("status after the checkpoint = %+v, want Checkpointed at %s", got, root)
	}
}

func TestCheckpointAfterACheckpointReturnsTheSameRestorePoint(t *testing.T) {
	path, root := checkpointed(t)
	if late := receive(t, requestCheckpoint(t, path, 5*time.Second)); !late.OK || late.Location != root {
		t.Fatalf("a request after Checkpointed got %+v, want the restore point at %s", late, root)
	}
}

func TestCheckpointFailsAtTheDeadline(t *testing.T) {
	path := serve(t, New(Config{Workload: "w", Store: t.TempDir()}, nil))
	answer := receive(t, requestCheckpoint(t, path, 300*time.Millisecond))
	if answer.OK || answer.State != "Failed" {
		t.Fatalf("checkpoint with no update boundary = %+v, want Failed", answer)
	}
	if got := status(t, path); got.State != "Failed" || got.Error == nil {
		t.Fatalf("status after the deadline = %+v, want Failed with an error", got)
	}
}

func TestCheckpointFailsWhenTrainingDies(t *testing.T) {
	sup := New(Config{Workload: "w", Store: t.TempDir()}, nil)
	path := serve(t, sup)
	answer := requestCheckpoint(t, path, 5*time.Second)
	waitForState(t, path, "Checkpointing")
	sup.ChildExited(1)
	if got := receive(t, answer); got.OK || got.State != "Failed" {
		t.Fatalf("checkpoint after the training process died = %+v, want Failed", got)
	}
}

func TestCheckpointAfterTrainingFinished(t *testing.T) {
	sup := New(Config{Workload: "w", Store: t.TempDir()}, nil)
	path := serve(t, sup)
	sup.ChildExited(0)
	if got := receive(t, requestCheckpoint(t, path, 5*time.Second)); got.OK || got.State != "Finished" {
		t.Fatalf("checkpoint after training finished = %+v, want Finished", got)
	}
}

func TestRestoringHandsBackTheStepAndQueuesACheckpoint(t *testing.T) {
	resume := &protocol.Resume{Step: 7, Files: []string{"/restore/app/step.json"}}
	path := serve(t, New(Config{Workload: "w", Store: t.TempDir()}, resume))

	if got := status(t, path); got.State != "Restoring" || got.Step != 7 {
		t.Fatalf("status = %+v, want Restoring at step 7", got)
	}
	answer := requestCheckpoint(t, path, 5*time.Second)
	time.Sleep(50 * time.Millisecond)
	if got := status(t, path); got.State != "Restoring" {
		t.Fatalf("a checkpoint during Restoring moved the state to %s", got.State)
	}

	var initialize protocol.InitializeReply
	if err := protocol.Send(path, protocol.Request{Op: "initialize"}, &initialize); err != nil {
		t.Fatal(err)
	}
	if initialize.Resume == nil || initialize.Resume.Step != 7 {
		t.Fatalf("initialize = %+v, want resume at step 7", initialize)
	}
	if got := status(t, path); got.State != "Checkpointing" {
		t.Fatalf("state after initialize = %s, want the queued checkpoint to begin", got.State)
	}

	reply := boundary(t, path, 8)
	writeStepFile(t, path, reply.Dir, 8)
	if got := receive(t, answer); !got.OK || got.Manifest.Step != 8 {
		t.Fatalf("queued checkpoint = %+v, want a restore point at step 8", got)
	}
}

func TestGiveUpAt(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	tests := []struct {
		left time.Duration
		want time.Duration
	}{
		{90 * time.Second, 85 * time.Second},
		{10 * time.Second, 9 * time.Second},
		{-time.Second, -time.Second},
	}
	for _, tt := range tests {
		if got := giveUpAt(now, now.Add(tt.left)).Sub(now); got != tt.want {
			t.Errorf("giveUpAt with %v left = now + %v, want now + %v", tt.left, got, tt.want)
		}
	}
}
