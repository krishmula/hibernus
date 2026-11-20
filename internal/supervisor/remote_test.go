package supervisor

import (
	"context"
	"errors"
	"os"
	"path/filepath"
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
		Remote:   &Remote{Store: mem, Bucket: "test-bucket"},
	}
}

// publishedAt reads the restore point that hib checkpoint printed, and checks that it is
// workload w at step in test-bucket.
func publishedAt(t *testing.T, location string, step int) layout.RestorePoint {
	t.Helper()
	rp, err := layout.ParseS3URL(location)
	if err != nil {
		t.Fatalf("location %q: %v", location, err)
	}
	if rp.Bucket != "test-bucket" || rp.Workload != "w" || rp.Step != step {
		t.Fatalf("location %q, want workload w at step %d in test-bucket", location, step)
	}
	return rp
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

	if !answer.OK {
		t.Fatalf("checkpoint = %+v, want ok", answer)
	}
	want := answer.Location
	key := publishedAt(t, want, 2).Key()
	puts := mem.Puts()
	if wantPuts := []string{key + "/app/step.json", key + "/manifest.json"}; !slices.Equal(puts, wantPuts) {
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
	if _, err := mem.Stat(bg, publishedAt(t, answer.Location, 2).Key()+"/manifest.json"); err != nil {
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
	if rp, found, err := store.NewestComplete(bg, mem, "w"); err != nil || found {
		t.Fatalf("a manifest reached the store after the deadline (%s, %v): the restore point would look complete", rp.Key(), err)
	}
	if got := status(t, path); got.State != "Failed" {
		t.Fatalf("state after the upload ended = %s, want it to stay Failed", got.State)
	}
}

func TestTheTrainingProcessEndingDuringTheUploadIsNotAFailure(t *testing.T) {
	// The restore point is complete on the host before the upload starts, so the upload goes on.
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

func TestWithoutARemoteTheLocalFolderIsTheLocation(t *testing.T) {
	// The first build's behavior, unchanged.
	store := t.TempDir()
	path := serve(t, New(Config{Workload: "w", Store: store}, nil))

	answer := receive(t, checkpointOnce(t, path, 5*time.Second))

	if !answer.OK || strings.HasPrefix(answer.Location, "s3://") {
		t.Fatalf("checkpoint = %+v, want a local folder", answer)
	}
}

func TestACheckpointAtTheStepOfACutOffUploadIsPublished(t *testing.T) {
	// The first attempt's upload stops before the manifest, and IAM grants no delete, so its
	// files stay in the store.
	mem := store.NewMemory()
	mem.PutHook = func(key string) error {
		if strings.HasSuffix(key, "manifest.json") {
			return &smithy.GenericAPIError{Code: "AccessDenied", Message: "cut off"}
		}
		return nil
	}
	_, first := remoteSupervisor(t, mem)
	if reply := receive(t, checkpointOnce(t, first, 5*time.Second)); reply.OK {
		t.Fatalf("first checkpoint = %+v, want it to fail before the manifest", reply)
	}
	mem.PutHook = nil

	// A restored workload reaches the same step with different training state.
	_, second := remoteSupervisor(t, mem)
	answer := requestCheckpoint(t, second, 5*time.Second)
	waitForState(t, second, "Checkpointing")
	reply := boundary(t, second, 2)
	file := filepath.Join(reply.Dir, "step.json")
	if err := os.WriteFile(file, []byte(`{"step": 2, "restored": true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	tell(t, second, protocol.Request{Op: "app_checkpoint", Files: []string{file}})

	if got := receive(t, answer); !got.OK {
		t.Fatalf("second checkpoint = %+v, want ok: the cut-off upload must not block it", got)
	}
	latest, err := store.FetchLatest(bg, mem, "w", func(step int) string { return filepath.Join(t.TempDir(), "w") })
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(latest.Root, "app", "step.json"))
	if err != nil || string(data) != `{"step": 2, "restored": true}` {
		t.Fatalf("the newest restore point holds %q, %v; want the second checkpoint's file", data, err)
	}
}
