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

// testAttempt is the checkpoint attempt that publishedRestorePoint publishes under.
const testAttempt = "20261119T000000Z-0000abcd"

// s3URL is where publishedRestorePoint put workload's restore point at step in test-bucket.
func s3URL(workload string, step int) string {
	return layout.RestorePoint{Bucket: "test-bucket", Workload: workload, Step: step, Attempt: testAttempt}.S3URL()
}

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
	if err := store.Publish(bg, mem, root, testAttempt); err != nil {
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
	if remote.Bucket != "my-bucket" {
		t.Fatalf("Bucket = %q, want my-bucket", remote.Bucket)
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

	root, remote, err := locateRestorePoint(bg, s3URL("demo", 5), storeDir)
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
	if remote == nil || remote.Bucket != "test-bucket" {
		t.Fatalf("remote = %+v, want one that publishes to test-bucket", remote)
	}
}

func TestAnS3LocationWithNoRestorePointIsRefused(t *testing.T) {
	useMemoryBucket(t, store.NewMemory())
	storeDir := t.TempDir()

	_, _, err := locateRestorePoint(bg, s3URL("demo", 5), storeDir)

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
	mem.Set("demo/step-8-20261119T090000Z-0000abcd/app/app.pt", []byte("half"))
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

	for _, location := range []string{"s3://test-bucket", "s3://test-bucket/demo/step-5/", "s3://test-bucket/demo/step-007-" + testAttempt + "/", "s3://Bad_Bucket/demo/step-5-" + testAttempt + "/", "s3://Bad_Bucket/demo/", s3URL("demo", 5) + "app/app.pt"} {
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

// runWithRemote runs hib run --remote s3://test-bucket for workload, with a command that leaves a
// file behind, and reports its exit code and whether the training command ran.
func runWithRemote(t *testing.T, workload string) (int, bool) {
	t.Helper()
	runDir, err := os.MkdirTemp("", "hib")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(runDir) })
	t.Setenv("HIB_RUN_DIR", runDir)
	marker := filepath.Join(t.TempDir(), "ran")
	code := runCommand([]string{"--remote", "s3://test-bucket", "--workload", workload, "--store", t.TempDir(), "sh", "-c", "touch " + marker})
	_, statErr := os.Stat(marker)
	return code, statErr == nil
}

func TestRunRefusesAWorkloadThatAlreadyHasARestorePoint(t *testing.T) {
	mem := store.NewMemory()
	publishedRestorePoint(t, mem, "demo", 5)
	useMemoryBucket(t, mem)

	code, ran := runWithRemote(t, "demo")

	if code != exitFailed || ran {
		t.Fatalf("hib run exited %d and ran the command: %v; want %d before anything starts", code, ran, exitFailed)
	}
}

func TestRunStartsAWorkloadWhoseOnlyUploadWasCutOff(t *testing.T) {
	mem := store.NewMemory()
	publishedRestorePoint(t, mem, "other", 5)
	mem.Set("demo/step-8-20261119T090000Z-0000abcd/app/app.pt", []byte("half"))
	useMemoryBucket(t, mem)

	code, ran := runWithRemote(t, "demo")

	if code != exitOK || !ran {
		t.Fatalf("hib run exited %d and ran the command: %v; want %d and the command run", code, ran, exitOK)
	}
}
