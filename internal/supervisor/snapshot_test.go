package supervisor

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
)

// fakeSnapshotter stands in for CRIU. It writes one image file and the output file, as the real
// one does, and records which process it was asked to dump.
type fakeSnapshotter struct {
	err    error
	during func() // runs inside Dump, as CRIU ending the training process would
	pid    chan int
}

func newFakeSnapshotter() *fakeSnapshotter { return &fakeSnapshotter{pid: make(chan int, 1)} }

func (f *fakeSnapshotter) Dump(pid int, root string) ([]string, error) {
	f.pid <- pid
	if f.during != nil {
		f.during()
	}
	if f.err != nil {
		return nil, f.err
	}
	if err := os.MkdirAll(filepath.Join(root, "process"), 0o755); err != nil {
		return nil, err
	}
	files := []string{filepath.Join("process", "core-1.img"), "output.log"}
	for _, file := range files {
		if err := os.WriteFile(filepath.Join(root, file), []byte(file), 0o644); err != nil {
			return nil, err
		}
	}
	return files, nil
}

// checkpointWithSnapshot runs one checkpoint at step 2 with snap as the Snapshotter, and returns
// hib checkpoint's answer and the manifest.
func checkpointWithSnapshot(t *testing.T, snap *fakeSnapshotter, configure func(*Supervisor)) (bool, manifest.Manifest) {
	t.Helper()
	store := t.TempDir()
	sup := New(Config{Workload: "w", Store: store, Snapshot: snap, Output: "/run/hib/w.log"}, nil)
	sup.TrainingProcess(4242)
	if configure != nil {
		configure(sup)
	}
	path := serve(t, sup)
	answer := requestCheckpoint(t, path, 5*time.Second)
	finishCheckpoint(t, path, 2)
	reply := receive(t, answer)
	m, err := manifest.Load(filepath.Join(store, "w", "step-2"))
	if err != nil {
		t.Fatal(err)
	}
	return reply.OK, m
}

func TestThePauseIsFollowedByTheProcessSnapshot(t *testing.T) {
	snap := newFakeSnapshotter()
	ok, m := checkpointWithSnapshot(t, snap, nil)

	if !ok {
		t.Fatal("the checkpoint failed")
	}
	if pid := <-snap.pid; pid != 4242 {
		t.Errorf("dumped process %d, want the training process 4242", pid)
	}
	layer := m.Layers.ProcessSnapshot
	var paths []string
	for _, f := range layer.Files {
		paths = append(paths, f.Path)
	}
	if layer.Status != manifest.StatusOK || !slices.Equal(paths, []string{"process/core-1.img", "output.log"}) {
		t.Errorf("process snapshot layer = %+v, want ok with the image and output.log", layer)
	}
	if m.Output != "/run/hib/w.log" {
		t.Errorf("the manifest records output %q, want /run/hib/w.log", m.Output)
	}
	if m.Layers.ApplicationCheckpoint.Status != manifest.StatusOK {
		t.Errorf("application checkpoint status = %s, want ok", m.Layers.ApplicationCheckpoint.Status)
	}
}

func TestAFailedProcessSnapshotStillLeavesTheApplicationCheckpoint(t *testing.T) {
	snap := newFakeSnapshotter()
	snap.err = errors.New("criu dump: mnt: unsupported")
	ok, m := checkpointWithSnapshot(t, snap, nil)

	if !ok {
		t.Fatal("the checkpoint failed, but the application checkpoint was written")
	}
	layer := m.Layers.ProcessSnapshot
	if layer.Status != manifest.StatusFailed || layer.Error == nil || *layer.Error != "criu dump: mnt: unsupported" {
		t.Errorf("process snapshot layer = %+v, want failed with CRIU's reason", layer)
	}
	if m.Output != "" {
		t.Errorf("the manifest records output %q without a process snapshot", m.Output)
	}
}

func TestTheTrainingProcessEndingDuringTheDumpIsNotAFailure(t *testing.T) {
	snap := newFakeSnapshotter()
	var sup *Supervisor
	snap.during = func() { sup.ChildExited(137) } // CRIU kills the training process after the dump
	ok, m := checkpointWithSnapshot(t, snap, func(s *Supervisor) { sup = s })

	if !ok || m.Layers.ProcessSnapshot.Status != manifest.StatusOK {
		t.Fatalf("checkpoint ok = %v, process snapshot %s; want ok with the process snapshot", ok, m.Layers.ProcessSnapshot.Status)
	}
	if !sup.Checkpointed() {
		t.Error("the workload is not Checkpointed")
	}
}

func TestWithoutASnapshotterTheManifestSaysWhy(t *testing.T) {
	store := t.TempDir()
	path := serve(t, New(Config{Workload: "w", Store: store, NoSnapshot: "criu is not installed"}, nil))
	answer := requestCheckpoint(t, path, 5*time.Second)
	finishCheckpoint(t, path, 2)
	receive(t, answer)

	m, err := manifest.Load(filepath.Join(store, "w", "step-2"))
	if err != nil {
		t.Fatal(err)
	}
	if layer := m.Layers.ProcessSnapshot; layer.Status != manifest.StatusFailed || layer.Error == nil || *layer.Error != "criu is not installed" {
		t.Errorf("process snapshot layer = %+v, want failed because criu is not installed", layer)
	}
}
