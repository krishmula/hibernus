package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
)

var checkpointHost = manifest.Fingerprint{
	GPU: "NVIDIA A10G", Driver: "550.54.15", CUDA: "12.4", CuDNN: "90100", PyTorch: "2.5.1", Kernel: "6.8.0-1015-aws",
}

// fakePython stands in for the interpreter the manifest records. Run with -m hibernus.fingerprint, it
// prints here, as the fingerprint module would on this host.
func fakePython(t *testing.T, here manifest.Fingerprint) string {
	t.Helper()
	data, err := json.Marshal(here)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "python")
	script := fmt.Sprintf("#!/bin/sh\n[ \"$*\" = \"-m hibernus.fingerprint\" ] || exit 9\necho '%s'\n", data)
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// localRestorePoint writes a restore point at step 3, recorded on checkpointHost, whose training
// process only creates the returned marker file.
func localRestorePoint(t *testing.T, python string) (root, marker string) {
	t.Helper()
	root = t.TempDir()
	marker = filepath.Join(t.TempDir(), "started")
	if err := os.MkdirAll(filepath.Join(root, "app"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "app", "step.json"), []byte(`{"step": 3}`), 0o644); err != nil {
		t.Fatal(err)
	}
	files, err := manifest.Describe(root, []string{"app/step.json"})
	if err != nil {
		t.Fatal(err)
	}
	failed := "not built yet"
	m := manifest.Manifest{
		SchemaVersion: manifest.SchemaVersion,
		Workload:      "w",
		Step:          3,
		CreatedAt:     time.Unix(1700000000, 0).UTC(),
		Fingerprint:   checkpointHost,
		Python:        python,
		Command:       manifest.Command{Argv: []string{"touch", marker}, Cwd: root, Env: map[string]string{}},
		Layers: manifest.Layers{
			ApplicationCheckpoint: manifest.Layer{Status: manifest.StatusOK, Files: files},
			ProcessSnapshot:       manifest.Layer{Status: manifest.StatusFailed, Error: &failed, Files: []manifest.File{}},
		},
	}
	if err := manifest.Write(root, m); err != nil {
		t.Fatal(err)
	}
	return root, marker
}

func restoreOn(t *testing.T, here manifest.Fingerprint) (code int, started bool) {
	t.Helper()
	run, err := os.MkdirTemp("", "hib") // short, because a socket path must stay under about 100 bytes
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(run) })
	t.Setenv("HIB_RUN_DIR", run)

	root, marker := localRestorePoint(t, fakePython(t, here))
	code = restoreCommand([]string{"--store", t.TempDir(), root})
	_, err = os.Stat(marker)
	return code, err == nil
}

func TestRestoreRefusesAHostWithAnotherDriverBeforeStartingAnything(t *testing.T) {
	here := checkpointHost
	here.Driver = "535.183.01"
	if code, started := restoreOn(t, here); code != exitFailed || started {
		t.Fatalf("restore on a host with another driver: exit %d, training process started %v; want exit %d and no start", code, started, exitFailed)
	}
}

func TestRestoreStartsTheTrainingProcessOnAMatchingHost(t *testing.T) {
	if code, started := restoreOn(t, checkpointHost); code != exitOK || !started {
		t.Fatalf("restore on a matching host: exit %d, training process started %v; want exit 0 and a start", code, started)
	}
}

func TestTheRefusalNamesEachValueThatDiffers(t *testing.T) {
	here := checkpointHost
	here.Driver = "535.183.01"
	here.PyTorch = "2.4.0"
	m := manifest.Manifest{Fingerprint: checkpointHost, Python: fakePython(t, here), Command: manifest.Command{Cwd: t.TempDir()}}
	err := checkFingerprint(context.Background(), m)
	if err == nil {
		t.Fatal("checkFingerprint accepted a host with another driver and PyTorch")
	}
	for _, want := range []string{"driver: checkpoint host 550.54.15, this host 535.183.01", "pytorch: checkpoint host 2.5.1, this host 2.4.0"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal %q does not say %q", err, want)
		}
	}
	if strings.Contains(err.Error(), "gpu:") {
		t.Errorf("refusal %q names the GPU, which matches", err)
	}
}

func TestAnInterpreterThatCannotCollectTheFingerprintIsRefused(t *testing.T) {
	m := manifest.Manifest{Fingerprint: checkpointHost, Python: filepath.Join(t.TempDir(), "missing-python"), Command: manifest.Command{Cwd: t.TempDir()}}
	if err := checkFingerprint(context.Background(), m); err == nil || !strings.Contains(err.Error(), "collect this host's fingerprint") {
		t.Fatalf("checkFingerprint with a missing interpreter = %v, want a collection error", err)
	}
}
