package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/store"
)

var bg = context.Background()

// useMemoryBucket makes hib open an in-memory store, in place of S3, and records the bucket it was asked for.
func useMemoryBucket(t *testing.T, mem *store.Memory) *string {
	t.Helper()
	var asked string
	old := openBucket
	openBucket = func(ctx context.Context, bucket string) (store.Store, error) {
		asked = bucket
		return mem, nil
	}
	t.Cleanup(func() { openBucket = old })
	return &asked
}

// publishedRestorePoint publishes a small restore point to mem, as hib checkpoint would.
func publishedRestorePoint(t *testing.T, mem *store.Memory, workload string, step int) {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "app"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "app", "app.pt"), []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	files, err := manifest.Describe(root, []string{"app/app.pt"})
	if err != nil {
		t.Fatal(err)
	}
	failed := "not built yet"
	m := manifest.Manifest{
		SchemaVersion: manifest.SchemaVersion,
		Workload:      workload,
		Step:          step,
		CreatedAt:     time.Unix(1700000000, 0).UTC(),
		Fingerprint:   map[string]string{},
		Command:       manifest.Command{Argv: []string{"python", "train.py"}, Cwd: "/work", Env: map[string]string{}},
		Layers: manifest.Layers{
			ApplicationCheckpoint: manifest.Layer{Status: manifest.StatusOK, Files: files},
			ProcessSnapshot:       manifest.Layer{Status: manifest.StatusFailed, Error: &failed, Files: []manifest.File{}},
		},
	}
	if err := manifest.Write(root, m); err != nil {
		t.Fatal(err)
	}
	if err := store.Publish(bg, mem, root); err != nil {
		t.Fatal(err)
	}
}

func TestParseRemoteFlag(t *testing.T) {
	for value, want := range map[string]string{
		"s3://my-bucket":                 "my-bucket",
		"s3://my-bucket/":                "my-bucket",
		"s3://cloud-checkpointing-48213": "cloud-checkpointing-48213",
	} {
		got, err := parseRemoteFlag(value)
		if err != nil || got != want {
			t.Errorf("parseRemoteFlag(%q) = %q, %v; want %q", value, got, err, want)
		}
	}
	for _, value := range []string{
		"", "my-bucket", "s3://", "s3://My_Bucket", "s3://CloudCheckpointing",
		"s3://bucket/a-prefix", "http://my-bucket", "s3://ab",
	} {
		if got, err := parseRemoteFlag(value); err == nil {
			t.Errorf("parseRemoteFlag(%q) = %q, want an error", value, got)
		}
	}
}

func TestRemoteForAddressesRestorePointsInTheBucket(t *testing.T) {
	asked := useMemoryBucket(t, store.NewMemory())

	remote, err := remoteFor(bg, "my-bucket")
	if err != nil {
		t.Fatal(err)
	}

	if *asked != "my-bucket" {
		t.Fatalf("opened bucket %q, want my-bucket", *asked)
	}
	if got, want := remote.Location("demo", 1200), layout.S3URL("my-bucket", "demo", 1200); got != want {
		t.Fatalf("Location = %q, want %q", got, want)
	}
}

func TestALocalFolderIsUsedAsItIs(t *testing.T) {
	folder := t.TempDir()

	root, remote, err := locateRestorePoint(bg, folder, t.TempDir())

	if err != nil || root != folder || remote != nil {
		t.Fatalf("locateRestorePoint = %q, %v, %v; want %q, no remote", root, remote, err, folder)
	}
}

func TestAnS3LocationIsDownloadedIntoTheStore(t *testing.T) {
	mem := store.NewMemory()
	publishedRestorePoint(t, mem, "demo", 5)
	asked := useMemoryBucket(t, mem)
	storeDir := t.TempDir()

	root, remote, err := locateRestorePoint(bg, "s3://test-bucket/demo/step-5/", storeDir)
	if err != nil {
		t.Fatal(err)
	}

	if want := layout.LocalDir(storeDir, "demo", 5); root != want {
		t.Fatalf("root = %q, want %q", root, want)
	}
	if *asked != "test-bucket" {
		t.Fatalf("opened bucket %q, want test-bucket", *asked)
	}
	m, err := manifest.Load(root)
	if err != nil || m.Step != 5 {
		t.Fatalf("the downloaded manifest: %+v, %v", m, err)
	}
	if err := m.Layers.ApplicationCheckpoint.Verify(root); err != nil {
		t.Fatalf("the downloaded files do not verify: %v", err)
	}
	// The restored workload can be checkpointed again, into the same bucket.
	if remote == nil || remote.Location("demo", 9) != layout.S3URL("test-bucket", "demo", 9) {
		t.Fatalf("remote = %+v, want one that publishes to test-bucket", remote)
	}
}

func TestAnS3LocationWithNoRestorePointIsRefused(t *testing.T) {
	useMemoryBucket(t, store.NewMemory())
	storeDir := t.TempDir()

	_, _, err := locateRestorePoint(bg, "s3://test-bucket/demo/step-5/", storeDir)

	if err == nil || !strings.Contains(err.Error(), "not a complete restore point") {
		t.Fatalf("error = %v, want one that says it is not a complete restore point", err)
	}
	if _, statErr := os.Stat(filepath.Join(layout.LocalDir(storeDir, "demo", 5), manifest.FileName)); !os.IsNotExist(statErr) {
		t.Fatal("a manifest was left in the store after a failed download")
	}
}

func TestAWorkloadLocationRestoresTheNewestRestorePoint(t *testing.T) {
	mem := store.NewMemory()
	publishedRestorePoint(t, mem, "demo", 3)
	publishedRestorePoint(t, mem, "demo", 12)
	publishedRestorePoint(t, mem, "demo", 9)
	asked := useMemoryBucket(t, mem)
	storeDir := t.TempDir()

	root, remote, err := locateRestorePoint(bg, "s3://test-bucket/demo/", storeDir)
	if err != nil {
		t.Fatal(err)
	}

	if want := layout.LocalDir(storeDir, "demo", 12); root != want {
		t.Fatalf("root = %q, want %q: step 12 is the newest, though step 9 sorts after it as text", root, want)
	}
	if *asked != "test-bucket" {
		t.Fatalf("opened bucket %q, want test-bucket", *asked)
	}
	if m, err := manifest.Load(root); err != nil || m.Step != 12 {
		t.Fatalf("the downloaded manifest: %+v, %v", m, err)
	}
	if remote == nil {
		t.Fatal("no remote: a restored workload could not be checkpointed again into the same bucket")
	}
}

func TestAWorkloadLocationSkipsANewestRestorePointThatIsIncomplete(t *testing.T) {
	mem := store.NewMemory()
	publishedRestorePoint(t, mem, "demo", 3)
	// An upload that was cut off at step 8: a file, but no manifest.
	mem.Set("demo/step-8/app/app.pt", []byte("half"))
	useMemoryBucket(t, mem)
	storeDir := t.TempDir()

	root, _, err := locateRestorePoint(bg, "s3://test-bucket/demo/", storeDir)
	if err != nil {
		t.Fatal(err)
	}

	if want := layout.LocalDir(storeDir, "demo", 3); root != want {
		t.Fatalf("root = %q, want %q: step 8 has no manifest", root, want)
	}
}

func TestAWorkloadLocationWithNoRestorePointIsRefused(t *testing.T) {
	useMemoryBucket(t, store.NewMemory())

	_, _, err := locateRestorePoint(bg, "s3://test-bucket/demo/", t.TempDir())

	if err == nil || !strings.Contains(err.Error(), "no usable restore point") {
		t.Fatalf("error = %v, want one that says there is no usable restore point", err)
	}
}

func TestAMalformedS3LocationIsRefusedBeforeAnythingIsOpened(t *testing.T) {
	asked := useMemoryBucket(t, store.NewMemory())

	for _, location := range []string{"s3://test-bucket", "s3://test-bucket/demo/step-007/", "s3://Bad_Bucket/demo/step-5/", "s3://Bad_Bucket/demo/", "s3://test-bucket/demo/step-5/app/app.pt"} {
		if _, _, err := locateRestorePoint(bg, location, t.TempDir()); err == nil {
			t.Errorf("locateRestorePoint(%q) succeeded, want an error", location)
		}
	}
	if *asked != "" {
		t.Fatalf("opened bucket %q for a malformed location", *asked)
	}
}

func TestRunRefusesABadRemoteBeforeStartingAnything(t *testing.T) {
	if code := runCommand([]string{"--remote", "not-an-s3-address", "echo", "hi"}); code != exitBadCommand {
		t.Fatalf("hib run with a bad --remote exited %d, want %d", code, exitBadCommand)
	}
}
