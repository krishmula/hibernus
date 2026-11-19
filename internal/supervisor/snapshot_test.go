package supervisor

import (
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
)

const (
	supervisorPID = 100 // the fake `hib run`
	trainingPID   = 200 // the fake training process, a child of the Supervisor
)

// fakeSnapshotter stands in for CRIU. It records the dump, and can fail, block, or run a hook.
type fakeSnapshotter struct {
	mu      sync.Mutex
	dumped  []string // "pid:dir" per Dump
	err     error
	onDump  func()        // runs inside Dump, before it returns
	release chan struct{} // when set, Dump waits for it
}

func (f *fakeSnapshotter) Dump(pid int, dir string) error {
	f.mu.Lock()
	f.dumped = append(f.dumped, fmt.Sprintf("%d:%s", pid, dir))
	f.mu.Unlock()
	if f.release != nil {
		<-f.release
	}
	if f.onDump != nil {
		f.onDump()
	}
	if f.err == nil {
		// CRIU writes its images into dir.
		os.MkdirAll(dir, 0o755)
		os.WriteFile(filepath.Join(dir, "core-200.img"), []byte("core"), 0o644)
		os.WriteFile(filepath.Join(dir, "pages-1.img"), []byte("pages"), 0o644)
	}
	return f.err
}

func (f *fakeSnapshotter) dumps() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.dumped...)
}

func (f *fakeSnapshotter) RestoreCommand(dir string) (*exec.Cmd, string) { return nil, "" }
func (f *fakeSnapshotter) Wake(pid int) error                            { return nil }

// fakeProc writes a /proc folder that holds the processes in parents (pid to ppid).
func fakeProc(t *testing.T, parents map[int]int) string {
	t.Helper()
	root := t.TempDir()
	for pid, ppid := range parents {
		dir := filepath.Join(root, fmt.Sprint(pid))
		if err := os.MkdirAll(filepath.Join(dir, "fd"), 0o755); err != nil {
			t.Fatal(err)
		}
		comm := "hib"
		if pid == trainingPID {
			comm = "python3"
		}
		stat := fmt.Sprintf("%d (%s) S %d %d %d", pid, comm, ppid, pid, pid) + strings.Repeat(" 0", 13) + " 1 0 0 0 100\n"
		os.WriteFile(filepath.Join(dir, "stat"), []byte(stat), 0o644)
		os.WriteFile(filepath.Join(dir, "cmdline"), []byte(comm+"\x00train.py\x00"), 0o644)
	}
	return root
}

func snapshotConfig(t *testing.T, snap *fakeSnapshotter, proc string) (Config, string) {
	logPath := filepath.Join(t.TempDir(), "tree.log")
	return Config{
		Workload: "w", Store: t.TempDir(), Snapshotter: snap,
		ProcRoot: proc, SelfPID: supervisorPID, TreeLog: logPath,
	}, logPath
}

// closeAfter dials the socket, sends one message, and returns the open connection. The hook does
// this with app_checkpoint, and then closes its socket so that CRIU has no connection to dump.
func sendAndHold(t *testing.T, path string, req protocol.Request) net.Conn {
	t.Helper()
	conn, err := net.Dial("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fmt.Fprintf(conn, `{"op":%q,"files":[%q]}`+"\n", req.Op, req.Files[0]); err != nil {
		t.Fatal(err)
	}
	return conn
}

// pauseForSnapshot runs the first half of a checkpoint and returns the pause reply.
func pauseForSnapshot(t *testing.T, path string, step int) (protocol.BoundaryReply, <-chan protocol.CheckpointReply) {
	t.Helper()
	answer := requestCheckpoint(t, path, 5*time.Second)
	waitForState(t, path, "Checkpointing")
	return boundary(t, path, step), answer
}

func writeApp(t *testing.T, dir string) string {
	t.Helper()
	file := filepath.Join(dir, "app.pt")
	if err := os.WriteFile(file, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	return file
}

func TestPauseTellsTheHookToSnapshotOnlyWhenThereIsSomethingToDump(t *testing.T) {
	snap := &fakeSnapshotter{}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)

	reply, _ := pauseForSnapshot(t, path, 2)

	if !reply.Pause || !reply.Snapshot {
		t.Fatalf("boundary reply = %+v, want a pause with Snapshot set", reply)
	}

	// No snapshotter: the first build's behavior stays.
	plain := serve(t, New(Config{Workload: "p", Store: t.TempDir()}, nil))
	if reply, _ := pauseForSnapshot(t, plain, 2); !reply.Pause || reply.Snapshot {
		t.Fatalf("boundary reply without a snapshotter = %+v, want a pause without Snapshot", reply)
	}
}

func TestTheDumpWaitsUntilTheHookClosesItsSocket(t *testing.T) {
	snap := &fakeSnapshotter{}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	reply, answer := pauseForSnapshot(t, path, 2)

	conn := sendAndHold(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{writeApp(t, reply.Dir)}})
	time.Sleep(200 * time.Millisecond)
	if got := snap.dumps(); len(got) != 0 {
		t.Fatalf("dumped while the hook's socket was still open: %v", got)
	}

	conn.Close() // the hook closes its socket
	got := receive(t, answer)

	root := filepath.Join(cfg.Store, "w", "step-2")
	if !got.OK || got.Location != root {
		t.Fatalf("checkpoint = %+v, want ok at %s", got, root)
	}
	if dumps := snap.dumps(); len(dumps) != 1 || dumps[0] != fmt.Sprintf("%d:%s", trainingPID, filepath.Join(root, "process")) {
		t.Fatalf("dumps = %v, want one dump of pid %d into %s", dumps, trainingPID, filepath.Join(root, "process"))
	}
}

func TestTheManifestRecordsBothLayersWithTheirFiles(t *testing.T) {
	snap := &fakeSnapshotter{}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	reply, answer := pauseForSnapshot(t, path, 2)
	sendAndHold(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{writeApp(t, reply.Dir)}}).Close()
	got := receive(t, answer)

	root := got.Location
	m, err := manifest.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	process := m.Layers.ProcessSnapshot
	if process.Status != manifest.StatusOK || process.Error != nil {
		t.Fatalf("process snapshot = %+v, want ok", process)
	}
	if len(process.Files) != 2 {
		t.Fatalf("process snapshot lists %d files, want the 2 images", len(process.Files))
	}
	for _, layer := range []manifest.Layer{process, m.Layers.ApplicationCheckpoint} {
		if err := layer.Verify(root); err != nil {
			t.Fatal(err)
		}
	}
	if status(t, path).State != "Checkpointed" {
		t.Fatalf("state = %s, want Checkpointed", status(t, path).State)
	}
}

func TestADumpFailureStillGivesARestorePointFromTheApplicationCheckpoint(t *testing.T) {
	snap := &fakeSnapshotter{err: errors.New("criu dump failed: tty: No task found with sid 285")}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	reply, answer := pauseForSnapshot(t, path, 2)
	sendAndHold(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{writeApp(t, reply.Dir)}}).Close()

	got := receive(t, answer)

	if !got.OK {
		t.Fatalf("checkpoint = %+v, want ok: the application checkpoint is good", got)
	}
	m, err := manifest.Load(got.Location)
	if err != nil {
		t.Fatal(err)
	}
	process := m.Layers.ProcessSnapshot
	if process.Status != manifest.StatusFailed || process.Error == nil || !strings.Contains(*process.Error, "No task found with sid 285") {
		t.Fatalf("process snapshot = %+v, want failed with CRIU's reason", process)
	}
	if err := m.Layers.ApplicationCheckpoint.Verify(got.Location); err != nil {
		t.Fatal(err)
	}
}

func TestAHookErrorFailsTheCheckpointAndSkipsTheDump(t *testing.T) {
	snap := &fakeSnapshotter{}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	_, answer := pauseForSnapshot(t, path, 2)
	conn, _ := net.Dial("unix", path)
	fmt.Fprintln(conn, `{"op":"app_checkpoint","error":"disk full"}`)
	conn.Close()

	got := receive(t, answer)

	if got.OK || !strings.Contains(got.Error, "disk full") {
		t.Fatalf("checkpoint = %+v, want a failure that says disk full", got)
	}
	if dumps := snap.dumps(); len(dumps) != 0 {
		t.Fatalf("dumped after a hook error: %v", dumps)
	}
}

func TestTheChildEndingDuringTheDumpIsNotAFailure(t *testing.T) {
	snap := &fakeSnapshotter{}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	snap.onDump = func() { sup.ChildExited(137) } // CRIU ends the tree, so `hib run` sees it exit
	path := serve(t, sup)
	reply, answer := pauseForSnapshot(t, path, 2)
	sendAndHold(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{writeApp(t, reply.Dir)}}).Close()

	got := receive(t, answer)

	if !got.OK {
		t.Fatalf("checkpoint = %+v, want ok", got)
	}
	if state := status(t, path).State; state != "Checkpointed" {
		t.Fatalf("state = %s, want Checkpointed: the dump ended the child on purpose", state)
	}
	if !sup.Checkpointed() {
		t.Fatal("Checkpointed() is false after a good checkpoint")
	}
}

func TestSettleWaitsForADumpInFlight(t *testing.T) {
	snap := &fakeSnapshotter{release: make(chan struct{})}
	cfg, _ := snapshotConfig(t, snap, fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID}))
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	reply, _ := pauseForSnapshot(t, path, 2)
	sendAndHold(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{writeApp(t, reply.Dir)}}).Close()
	for len(snap.dumps()) == 0 {
		time.Sleep(5 * time.Millisecond)
	}

	settled := make(chan struct{})
	go func() { sup.Settle(); close(settled) }()
	select {
	case <-settled:
		t.Fatal("Settle returned while the dump was still running")
	case <-time.After(150 * time.Millisecond):
	}

	close(snap.release)
	select {
	case <-settled:
	case <-time.After(5 * time.Second):
		t.Fatal("Settle never returned after the dump finished")
	}
}

func TestTheTreeLogShowsTheTrainingProcessBeforeTheDumpAndNotAfter(t *testing.T) {
	proc := fakeProc(t, map[int]int{supervisorPID: 1, trainingPID: supervisorPID})
	snap := &fakeSnapshotter{}
	snap.onDump = func() { os.RemoveAll(filepath.Join(proc, fmt.Sprint(trainingPID))) } // CRIU ended it
	cfg, logPath := snapshotConfig(t, snap, proc)
	sup := New(cfg, nil)
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	reply, answer := pauseForSnapshot(t, path, 2)
	sendAndHold(t, path, protocol.Request{Op: "app_checkpoint", Files: []string{writeApp(t, reply.Dir)}}).Close()
	receive(t, answer)

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	log := string(data)
	before := strings.Index(log, "before-checkpoint step=2")
	after := strings.Index(log, "after-dump")
	if before < 0 || after < before {
		t.Fatalf("log lacks before-checkpoint then after-dump:\n%s", log)
	}
	if !strings.Contains(log[before:after], "pid=200") {
		t.Fatalf("the before section lacks the training process:\n%s", log[before:after])
	}
	if strings.Contains(log[after:], "pid=200") {
		t.Fatalf("the after-dump section still shows the training process:\n%s", log[after:])
	}
}

func TestResumedMovesARestoringWorkloadToRunningAndLogsTheTree(t *testing.T) {
	proc := fakeProc(t, map[int]int{supervisorPID: 1, 150: supervisorPID, trainingPID: 150}) // hib, criu, python
	snap := &fakeSnapshotter{}
	cfg, logPath := snapshotConfig(t, snap, proc)
	sup := New(cfg, &protocol.Resume{Step: 5})
	sup.SetChild(trainingPID)
	path := serve(t, sup)
	if got := status(t, path).State; got != "Restoring" {
		t.Fatalf("state before the hook reconnects = %s, want Restoring", got)
	}

	var reply protocol.ResumedReply
	if err := protocol.Send(path, protocol.Request{Op: "resumed"}, &reply); err != nil {
		t.Fatal(err)
	}

	if !reply.OK {
		t.Fatalf("resumed reply = %+v, want ok", reply)
	}
	if got := status(t, path).State; got != "Running" {
		t.Fatalf("state after resumed = %s, want Running", got)
	}
	select {
	case <-sup.Resumed():
	default:
		t.Fatal("Resumed() is not closed after the hook reconnected")
	}
	data, _ := os.ReadFile(logPath)
	if !strings.Contains(string(data), "after-restore") || !strings.Contains(string(data), "pid=200") {
		t.Fatalf("log lacks the after-restore tree with the training process:\n%s", data)
	}
}

func TestResumedOnAWorkloadThatIsNotRestoringChangesNothing(t *testing.T) {
	path := serve(t, New(Config{Workload: "w", Store: t.TempDir()}, nil))

	var reply protocol.ResumedReply
	if err := protocol.Send(path, protocol.Request{Op: "resumed"}, &reply); err != nil {
		t.Fatal(err)
	}

	if got := status(t, path).State; got != "Running" {
		t.Fatalf("state = %s, want Running", got)
	}
}
