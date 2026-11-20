package manifest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeRestorePoint(t *testing.T) (string, Layer) {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "app"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "app", "step.json"), []byte(`{"step": 7}`), 0o644); err != nil {
		t.Fatal(err)
	}
	files, err := Describe(root, []string{"app/step.json"})
	if err != nil {
		t.Fatal(err)
	}
	app := Layer{Status: StatusOK, Files: files}
	m := Manifest{SchemaVersion: SchemaVersion, Workload: "demo", Step: 7, Layers: Layers{ApplicationCheckpoint: app}}
	if err := Write(root, m); err != nil {
		t.Fatal(err)
	}
	return root, app
}

func TestWriteThenLoad(t *testing.T) {
	root, _ := writeRestorePoint(t)
	m, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if m.Workload != "demo" || m.Step != 7 {
		t.Fatalf("loaded %s at step %d, want demo at step 7", m.Workload, m.Step)
	}
	if err := m.Layers.ApplicationCheckpoint.Verify(root); err != nil {
		t.Fatalf("Verify of untouched files: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, FileName+".tmp")); !os.IsNotExist(err) {
		t.Fatalf("the temporary manifest is still there: %v", err)
	}
}

func TestVerifyCatchesAChangedFile(t *testing.T) {
	root, app := writeRestorePoint(t)
	if err := os.WriteFile(filepath.Join(root, "app", "step.json"), []byte(`{"step": 8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := app.Verify(root); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("Verify of a changed file = %v, want a checksum error", err)
	}
}

func TestVerifyRefusesAPathOutsideTheRestorePoint(t *testing.T) {
	root, _ := writeRestorePoint(t)
	outside := Layer{Status: StatusOK, Files: []File{{Path: "../elsewhere"}}}
	if err := outside.Verify(root); err == nil || !strings.Contains(err.Error(), "outside the restore point") {
		t.Fatalf("Verify of ../elsewhere = %v, want an outside-the-restore-point error", err)
	}
}

func TestLoadRefusesAFolderWithoutAManifest(t *testing.T) {
	if _, err := Load(t.TempDir()); err == nil || !strings.Contains(err.Error(), "no manifest.json") {
		t.Fatalf("Load of an empty folder = %v, want a missing-manifest error", err)
	}
}

func TestLoadRefusesAnotherSchemaVersion(t *testing.T) {
	root := t.TempDir()
	if err := Write(root, Manifest{SchemaVersion: 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), "schema_version 1") {
		t.Fatalf("Load of schema_version 1 = %v, want a schema version error", err)
	}
}

func TestDifferencesNamesEachValueThatDiffers(t *testing.T) {
	recorded := Fingerprint{GPU: "NVIDIA A10G", Driver: "550.54.15", CUDA: "12.4", CuDNN: "90100", PyTorch: "2.5.1", Kernel: "6.8.0"}
	if diffs := Differences(recorded, recorded); len(diffs) != 0 {
		t.Fatalf("Differences of equal fingerprints = %v, want none", diffs)
	}
	here := recorded
	here.Driver = "535.183.01"
	here.Kernel = "6.5.0"
	want := []string{
		"driver: checkpoint host 550.54.15, this host 535.183.01",
		"kernel: checkpoint host 6.8.0, this host 6.5.0",
	}
	if diffs := Differences(recorded, here); strings.Join(diffs, "|") != strings.Join(want, "|") {
		t.Fatalf("Differences = %q, want %q", diffs, want)
	}
}

func TestCompleteNamesEachMissingValue(t *testing.T) {
	err := Fingerprint{GPU: "none", Driver: "none", CUDA: "none", PyTorch: "none"}.Complete()
	if err == nil || err.Error() != "the fingerprint has no cudnn, kernel" {
		t.Fatalf("Complete = %v, want it to name cudnn and kernel", err)
	}
}
