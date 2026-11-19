package supervisor

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/smithy-go"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/store"
)

// These tests give the Supervisor a remote, an in-memory store in place of S3. A checkpoint is
// done only when its restore point is published there.

var bg = context.Background()

func remoteConfig(t *testing.T, mem *store.Memory) Config {
	t.Helper()
	return Config{
		Workload: "w",
		Store:    t.TempDir(),
		Remote: &Remote{
			Store:    mem,
			Location: func(workload string, step int) string { return layout.S3URL("test-bucket", workload, step) },
		},
	}
}

func remoteSupervisor(t *testing.T, mem *store.Memory) (*Supervisor, string) {
	t.Helper()
	sup := New(remoteConfig(t, mem), nil)
	return sup, serve(t, sup)
}

// checkpointOnce plays a checkpoint up to the point where the hook reports its application
// checkpoint, and returns where hib checkpoint's answer arrives.
func checkpointOnce(t *testing.T, path string, deadline time.Duration) <-chan protocol.CheckpointReply {
	t.Helper()
	answer := requestCheckpoint(t, path, deadline)
	waitForState(t, path, "Checkpointing")
	reply := boundary(t, path, 2)
	if !reply.Pause {
		t.Fatalf("boundary in Checkpointing = %+v, want a pause", reply)
	}
	writeStepFile(t, path, reply.Dir, 2)
	return answer
}

// holdUploads makes every Put wait until release is called. reached gets a value each time a Put starts.
func holdUploads(t *testing.T, mem *store.Memory) (release func(), reached <-chan struct{}) {
	t.Helper()
	gate := make(chan struct{})
	arrived := make(chan struct{}, 16)
	var once sync.Once
	release = func() { once.Do(func() { close(gate) }) }
	t.Cleanup(release)
	mem.PutHook = func(key string) error {
		arrived <- struct{}{}
		<-gate
		return nil
	}
	return release, arrived
}

func waitFor(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func fastRetries(t *testing.T) {
	t.Helper()
	old := retryWait
	retryWait = 5 * time.Millisecond
	t.Cleanup(func() { retryWait = old })
}

func TestACheckpointIsPublishedBeforeItIsDone(t *testing.T) {
	mem := store.NewMemory()
	_, path := remoteSupervisor(t, mem)

	answer := receive(t, checkpointOnce(t, path, 5*time.Second))

	want := layout.S3URL("test-bucket", "w", 2)
	if !answer.OK || answer.Location != want {
		t.Fatalf("checkpoint = %+v, want ok at %s", answer, want)
	}
	puts := mem.Puts()
	if wantPuts := []string{"w/step-2/app/step.json", "w/step-2/manifest.json"}; !slices.Equal(puts, wantPuts) {
		t.Fatalf("stored %v, want %v: the files first, the manifest last", puts, wantPuts)
	}
	got := status(t, path)
	if got.State != "Checkpointed" || got.Location == nil || *got.Location != want {
		t.Fatalf("status = %+v, want Checkpointed at %s", got, want)
	}
}

func TestTheStateStaysCheckpointingWhileTheUploadRuns(t *testing.T) {
	mem := store.NewMemory()
	release, reached := holdUploads(t, mem)
	_, path := remoteSupervisor(t, mem)

	answer := checkpointOnce(t, path, 5*time.Second)
	waitFor(t, reached, "the upload to start")

	// status answers, so the upload is not holding the Supervisor's lock.
	if got := status(t, path); got.State != "Checkpointing" {
		t.Fatalf("state during the upload = %s, want Checkpointing", got.State)
	}
	select {
	case reply := <-answer:
		t.Fatalf("hib checkpoint was answered during the upload: %+v", reply)
	default:
	}

	release()
	if reply := receive(t, answer); !reply.OK {
		t.Fatalf("checkpoint after the upload = %+v, want ok", reply)
	}
}

func TestATemporaryUploadFailureIsRetried(t *testing.T) {
	fastRetries(t)
	mem := store.NewMemory()
	var calls atomic.Int32
	mem.PutHook = func(key string) error {
		if calls.Add(1) <= 2 {
			return errors.New("connection reset")
		}
		return nil
	}
	_, path := remoteSupervisor(t, mem)

	answer := receive(t, checkpointOnce(t, path, 5*time.Second))

	if !answer.OK {
		t.Fatalf("checkpoint = %+v, want ok after the retries", answer)
	}
	if calls.Load() <= 2 {
		t.Fatalf("Put was called %d times, want more than the 2 that failed", calls.Load())
	}
	if _, err := mem.Stat(bg, "w/step-2/manifest.json"); err != nil {
		t.Fatalf("no manifest in the store after the retries: %v", err)
	}
}

func TestAPermanentUploadFailureFailsAtOnce(t *testing.T) {
	fastRetries(t)
	mem := store.NewMemory()
	var calls atomic.Int32
	mem.PutHook = func(key string) error {
		calls.Add(1)
		return &smithy.GenericAPIError{Code: "AccessDenied", Message: "not allowed"}
	}
	_, path := remoteSupervisor(t, mem)
	start := time.Now()

	answer := receive(t, checkpointOnce(t, path, 5*time.Second))

	if answer.OK || answer.State != "Failed" || !strings.Contains(answer.Error, "AccessDenied") {
		t.Fatalf("checkpoint = %+v, want Failed with the S3 error", answer)
	}
	if calls.Load() != 1 {
		t.Fatalf("Put was called %d times, want 1: waiting does not fix a permission error", calls.Load())
	}
	if took := time.Since(start); took > 3*time.Second {
		t.Fatalf("the failure took %s: the Supervisor waited for the deadline instead of giving up at once", took)
	}
	if puts := mem.Puts(); len(puts) != 0 {
		t.Fatalf("stored %v, want nothing", puts)
	}
	if got := status(t, path); got.State != "Failed" {
		t.Fatalf("state = %s, want Failed: a restore point that is not in the store is not a restore point", got.State)
	}
}

func TestTheDeadlineDuringTheUploadFailsTheCheckpointAndLeavesNoManifest(t *testing.T) {
	mem := store.NewMemory()
	release, reached := holdUploads(t, mem)
	sup, path := remoteSupervisor(t, mem)

	answer := checkpointOnce(t, path, 400*time.Millisecond)
	waitFor(t, reached, "the upload to start")

	reply := receive(t, answer)
	if reply.OK || reply.State != "Failed" {
		t.Fatalf("checkpoint = %+v, want Failed at the deadline", reply)
	}

	release()
	sup.Settle() // the upload ends once it is let go
	if _, err := mem.Stat(bg, "w/step-2/manifest.json"); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("a manifest reached the store after the deadline (%v): the restore point would look complete", err)
	}
	if got := status(t, path); got.State != "Failed" {
		t.Fatalf("state after the upload ended = %s, want it to stay Failed", got.State)
	}
}

func TestTheTrainingProcessEndingDuringTheUploadIsNotAFailure(t *testing.T) {
	// A process snapshot ends the training process on purpose, and the upload goes on after it.
	mem := store.NewMemory()
	release, reached := holdUploads(t, mem)
	sup, path := remoteSupervisor(t, mem)
	answer := checkpointOnce(t, path, 5*time.Second)
	waitFor(t, reached, "the upload to start")

	sup.ChildExited(0)
	release()

	if reply := receive(t, answer); !reply.OK {
		t.Fatalf("checkpoint = %+v, want ok: the restore point was complete when the process ended", reply)
	}
}

func TestSettleWaitsForTheUpload(t *testing.T) {
	mem := store.NewMemory()
	release, reached := holdUploads(t, mem)
	sup, path := remoteSupervisor(t, mem)
	answer := checkpointOnce(t, path, 5*time.Second)
	waitFor(t, reached, "the upload to start")
	settled := make(chan struct{})
	go func() {
		sup.Settle()
		close(settled)
	}()

	select {
	case <-settled:
		t.Fatal("Settle returned while the upload was still running: hib run would exit under it")
	case <-time.After(150 * time.Millisecond):
	}

	release()
	waitFor(t, settled, "Settle to return after the upload")
	receive(t, answer)
}

func TestTheProcessSnapshotIsPublishedToo(t *testing.T) {
	mem := store.NewMemory()
	snap := &fakeSnapshotter{}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	cfg.Remote = &Remote{
		Store:    mem,
		Location: func(workload string, step int) string { return layout.S3URL("test-bucket", workload, step) },
	}
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	reply, answer := pauseForSnapshot(t, path, 2)

	sendAndHold(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{writeApp(t, reply.Dir)}}).Close()
	got := receive(t, answer)

	if want := layout.S3URL("test-bucket", "w", 2); !got.OK || got.Location != want {
		t.Fatalf("checkpoint = %+v, want ok at %s", got, want)
	}
	puts := mem.Puts()
	if len(puts) != 4 || puts[len(puts)-1] != "w/step-2/manifest.json" {
		t.Fatalf("stored %v, want the application file, the two images, and the manifest last", puts)
	}
	for _, key := range []string{"w/step-2/app/app.pt", "w/step-2/process/core-200.img", "w/step-2/process/pages-1.img"} {
		if !slices.Contains(puts, key) {
			t.Errorf("%s was not stored; stored %v", key, puts)
		}
	}
}

func TestWithoutARemoteTheLocalFolderIsTheLocation(t *testing.T) {
	// The first build's behavior, unchanged.
	store := t.TempDir()
	path := serve(t, New(Config{Workload: "w", Store: store}, nil))

	answer := receive(t, checkpointOnce(t, path, 5*time.Second))

	if !answer.OK || strings.HasPrefix(answer.Location, "s3://") {
		t.Fatalf("checkpoint = %+v, want a local folder", answer)
	}
}
