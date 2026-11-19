//go:build linux

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/snapshot"
	"github.com/krishmula/hibernus/internal/supervisor"
)

// fakeCriu stands in for CRIU. RestoreCommand runs a shell script, and Wake plays the restored
// hook: it sends "resumed" on a new connection, as the hook does after SIGUSR1.
type fakeCriu struct {
	script      string // what "criu restore" runs; $1 is the pid file
	hookReplies bool   // whether Wake makes the hook reconnect
	socket      string
	woken       chan int
}

func (f *fakeCriu) Dump(pid int, dir string) error { return nil }

func (f *fakeCriu) RestoreCommand(dir string) (*exec.Cmd, string) {
	pidFile := filepath.Join(filepath.Dir(dir), "restored.pid")
	return exec.Command("sh", "-c", f.script, "criu", pidFile), pidFile
}

func (f *fakeCriu) Wake(pid int) error {
	f.woken <- pid
	if f.hookReplies {
		go func() {
			var reply protocol.ResumedReply
			protocol.Send(f.socket, protocol.Request{Op: "resumed"}, &reply)
		}()
	}
	return nil
}

var _ snapshot.Snapshotter = (*fakeCriu)(nil)

// restorePoint makes a restore point whose process snapshot holds one image, and returns its manifest.
func restorePoint(t *testing.T) (string, manifest.Manifest) {
	t.Helper()
	root := t.TempDir()
	image := filepath.Join(root, "process", "core-1.img")
	if err := os.MkdirAll(filepath.Dir(image), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(image, []byte("image bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	files, err := manifest.Describe(root, []string{filepath.Join("process", "core-1.img")})
	if err != nil {
		t.Fatal(err)
	}
	m := manifest.Manifest{
		SchemaVersion: manifest.SchemaVersion, Workload: "w", Step: 5,
		Layers: manifest.Layers{ProcessSnapshot: manifest.Layer{Status: manifest.StatusOK, Files: files}},
	}
	return root, m
}

func restoreConfig(t *testing.T, criu *fakeCriu) (supervisor.Config, string) {
	t.Helper()
	runDir, err := os.MkdirTemp("", "hr") // a short path: Unix socket paths are limited to ~100 bytes
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(runDir) })
	t.Setenv("HIB_RUN_DIR", runDir)
	criu.socket = protocol.SocketPath("w")
	criu.woken = make(chan int, 4)
	logPath := filepath.Join(t.TempDir(), "tree.log")
	return supervisor.Config{Workload: "w", Store: t.TempDir(), Snapshotter: criu, TreeLog: logPath}, logPath
}

func shortRestoreTimeout(t *testing.T, d time.Duration) {
	t.Helper()
	old := restoreTimeout
	restoreTimeout = d
	t.Cleanup(func() { restoreTimeout = old })
}

func TestRestoreFromSnapshotSucceedsWhenTheRestoredHookReconnects(t *testing.T) {
	shortRestoreTimeout(t, 10*time.Second)
	// criu writes the restored root's pid (here its own, so nothing real is ever signalled), stays
	// for a moment as the root's parent, then exits 0.
	criu := &fakeCriu{script: `echo $$ > "$1"; sleep 0.5; exit 0`, hookReplies: true}
	cfg, logPath := restoreConfig(t, criu)
	root, m := restorePoint(t)

	code, ok, reason := restoreFromSnapshot(cfg, &protocol.Resume{Step: 5}, root, m)

	if !ok || code != 0 {
		t.Fatalf("restoreFromSnapshot = code %d, ok %v, reason %v; want a clean restore", code, ok, reason)
	}
	select {
	case pid := <-criu.woken:
		written, _ := os.ReadFile(filepath.Join(root, "restored.pid"))
		if strings.TrimSpace(string(written)) != fmt.Sprint(pid) {
			t.Fatalf("woke pid %d, but criu wrote %q as the restored root", pid, written)
		}
	default:
		t.Fatal("the restored process was never woken")
	}
	data, _ := os.ReadFile(logPath)
	log := string(data)
	before, after := strings.Index(log, "before-restore"), strings.Index(log, "after-restore")
	if before < 0 || after < before {
		t.Fatalf("tree log lacks before-restore then after-restore:\n%s", log)
	}
}

func TestRestoreFromSnapshotFailsWhenCriuExitsBeforeTheHookReconnects(t *testing.T) {
	shortRestoreTimeout(t, 10*time.Second)
	criu := &fakeCriu{script: `echo "Error: Restoring FAILED." >&2; exit 1`}
	cfg, _ := restoreConfig(t, criu)
	root, m := restorePoint(t)

	_, ok, reason := restoreFromSnapshot(cfg, &protocol.Resume{Step: 5}, root, m)

	if ok || reason == nil || !strings.Contains(reason.Error(), "criu") {
		t.Fatalf("ok = %v, reason = %v; want a failure that names criu", ok, reason)
	}
}

func TestRestoreFromSnapshotFailsAndLeavesNothingWhenTheHookNeverReconnects(t *testing.T) {
	shortRestoreTimeout(t, 400*time.Millisecond)
	criu := &fakeCriu{script: `echo $$ > "$1"; exec sleep 30`, hookReplies: false}
	cfg, _ := restoreConfig(t, criu)
	root, m := restorePoint(t)
	start := time.Now()

	_, ok, reason := restoreFromSnapshot(cfg, &protocol.Resume{Step: 5}, root, m)

	if ok || reason == nil || !strings.Contains(reason.Error(), "reconnect") {
		t.Fatalf("ok = %v, reason = %v; want a failure that says the hook did not reconnect", ok, reason)
	}
	if took := time.Since(start); took > 5*time.Second {
		t.Fatalf("took %s to give up, want about the 400ms timeout: criu was left running", took)
	}
	// The socket is free again, so the fallback can listen on it.
	listener, err := supervisor.Listen(protocol.SocketPath("w"))
	if err != nil {
		t.Fatalf("the socket is still taken after the failure: %v", err)
	}
	listener.Close()
}

func TestRestoreFromSnapshotRefusesCorruptedImagesBeforeStartingCriu(t *testing.T) {
	criu := &fakeCriu{script: `echo started > "$1.started"; exit 0`}
	cfg, _ := restoreConfig(t, criu)
	root, m := restorePoint(t)
	if err := os.WriteFile(filepath.Join(root, "process", "core-1.img"), []byte("tampered!!"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, ok, reason := restoreFromSnapshot(cfg, &protocol.Resume{Step: 5}, root, m)

	if ok || reason == nil || !strings.Contains(reason.Error(), "core-1.img") {
		t.Fatalf("ok = %v, reason = %v; want a refusal that names the corrupted image", ok, reason)
	}
	if _, err := os.Stat(filepath.Join(root, "restored.pid.started")); err == nil {
		t.Fatal("criu ran against corrupted images")
	}
}

func TestRestoreFromSnapshotSkipsAManifestWithoutAProcessSnapshot(t *testing.T) {
	criu := &fakeCriu{script: `exit 0`}
	cfg, _ := restoreConfig(t, criu)
	root, m := restorePoint(t)
	reason := "not built yet"
	m.Layers.ProcessSnapshot = manifest.Layer{Status: manifest.StatusFailed, Error: &reason}

	_, ok, why := restoreFromSnapshot(cfg, &protocol.Resume{Step: 5}, root, m)

	if ok || why == nil || !strings.Contains(why.Error(), "not built yet") {
		t.Fatalf("ok = %v, reason = %v; want a skip that repeats why there is no snapshot", ok, why)
	}
}
