package store

// These tests run against a real S3 bucket, or any S3-compatible server such as MinIO. They are
// skipped unless HIB_TEST_S3_BUCKET is set. Credentials and the region come from the standard AWS
// chain (AWS_PROFILE, AWS_REGION, or the environment). Set HIB_TEST_S3_ENDPOINT to use MinIO.
//
// Every test writes under hibernus-test/<run id>/ and never deletes: stored files never change, and
// the IAM policy for Hibernus has no delete permission. Let a lifecycle rule expire that prefix.

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/krishmula/hibernus/internal/manifest"
)

var (
	runID     = fmt.Sprintf("%d", time.Now().UnixNano())
	runMu     sync.Mutex
	runNumber int
)

func requireIntegration(t *testing.T) {
	t.Helper()
	if os.Getenv("HIB_TEST_S3_BUCKET") == "" {
		t.Skip("set HIB_TEST_S3_BUCKET, and AWS credentials, to test against a real bucket")
	}
}

// integrationStore returns a store under a prefix of its own, so no two tests share a key.
func integrationStore(t *testing.T) *S3 {
	t.Helper()
	requireIntegration(t)
	runMu.Lock()
	runNumber++
	n := runNumber
	runMu.Unlock()

	var options []func(*s3.Options)
	if endpoint := os.Getenv("HIB_TEST_S3_ENDPOINT"); endpoint != "" {
		options = append(options, func(o *s3.Options) {
			o.BaseEndpoint = aws.String(endpoint)
			o.UsePathStyle = true
		})
	}
	st, err := NewS3(bg, os.Getenv("HIB_TEST_S3_BUCKET"), fmt.Sprintf("hibernus-test/%s/%d", runID, n), options...)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

// writeRandomFile writes size bytes of reproducible pseudo-random data and describes the file.
func writeRandomFile(t *testing.T, dir, name string, size, seed int64) (string, Object) {
	t.Helper()
	file := filepath.Join(dir, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(file)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	hash := sha256.New()
	random := rand.New(rand.NewSource(seed))
	buf := make([]byte, 1<<20)
	for written := int64(0); written < size; {
		n := int64(len(buf))
		if size-written < n {
			n = size - written
		}
		random.Read(buf[:n])
		if _, err := f.Write(buf[:n]); err != nil {
			t.Fatal(err)
		}
		hash.Write(buf[:n])
		written += n
	}
	return file, Object{Size: size, Sha256: hex.EncodeToString(hash.Sum(nil))}
}

// makeBigRestorePoint is a restore point whose process snapshot file has the given size.
func makeBigRestorePoint(t *testing.T, workload string, step int, size int64) string {
	t.Helper()
	root := t.TempDir()
	writeFile(t, root, "app/app.pt", "application checkpoint bytes")
	writeRandomFile(t, root, "process/pages-1.img", size, 3)
	app, err := manifest.Describe(root, []string{"app/app.pt"})
	if err != nil {
		t.Fatal(err)
	}
	process, err := manifest.Describe(root, []string{"process/pages-1.img"})
	if err != nil {
		t.Fatal(err)
	}
	m := manifest.Manifest{
		SchemaVersion: manifest.SchemaVersion,
		Workload:      workload,
		Step:          step,
		CreatedAt:     time.Unix(1700000000, 0).UTC(),
		Command:       manifest.Command{Argv: []string{"python", "train.py"}, Cwd: "/work", Env: map[string]string{}},
		Layers: manifest.Layers{
			ApplicationCheckpoint: manifest.Layer{Status: manifest.StatusOK, Files: app},
			ProcessSnapshot:       manifest.Layer{Status: manifest.StatusOK, Files: process},
		},
	}
	if err := manifest.Write(root, m); err != nil {
		t.Fatal(err)
	}
	return root
}

// The same contract that Memory, Local, and the fake S3 pass, against a real server. This is where
// a taken key must really come back as ErrExists, and a missing key as ErrNotFound.
func TestIntegrationS3Contract(t *testing.T) {
	requireIntegration(t)
	contract(t, func(t *testing.T) Store { return integrationStore(t) })
}

// A file bigger than the part size goes up in parts, and the no-overwrite rule has to hold on
// CompleteMultipartUpload, which the fake S3 never reaches.
func TestIntegrationS3MultipartUploadRoundTrip(t *testing.T) {
	st := integrationStore(t)
	st.PartSizeBytes = 5 << 20 // S3's smallest part, so 12 MiB is three parts
	const key = "demo/step-1/process/pages-1.img"
	src, want := writeRandomFile(t, t.TempDir(), "big", 12<<20, 1)

	if err := st.Put(bg, key, src, want); err != nil {
		t.Fatalf("multipart Put: %v", err)
	}

	got, err := st.Stat(bg, key)
	if err != nil || got != want {
		t.Fatalf("Stat = %+v, %v; want %+v", got, err, want)
	}
	dst := filepath.Join(t.TempDir(), "copy")
	if err := st.Get(bg, key, dst); err != nil {
		t.Fatal(err)
	}
	if size, sum, err := measure(dst); err != nil || size != want.Size || sum != want.Sha256 {
		t.Fatalf("the downloaded file has %d bytes and checksum %s (%v), want %d bytes and %s", size, sum, err, want.Size, want.Sha256)
	}

	other, otherWant := writeRandomFile(t, t.TempDir(), "other", 12<<20, 2)
	if err := st.Put(bg, key, other, otherWant); !errors.Is(err, ErrExists) {
		t.Fatalf("a second multipart Put of the same key: error = %v, want ErrExists", err)
	}
	if got, err := st.Stat(bg, key); err != nil || got != want {
		t.Fatalf("after the refused Put, Stat = %+v, %v; the stored file changed", got, err)
	}
}

// Publish and Fetch with a 100 MiB file at the default part size, with the time each way.
func TestIntegrationS3LargeRestorePointRoundTrip(t *testing.T) {
	if testing.Short() {
		t.Skip("uploads 100 MiB")
	}
	st := integrationStore(t)
	const size = 100 << 20
	root := makeBigRestorePoint(t, "demo", 7, size)

	start := time.Now()
	if err := Publish(bg, st, root, testAttempt); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	up := time.Since(start)
	t.Logf("published %d MiB in %s (%.1f MiB/s)", size>>20, up.Round(time.Millisecond), float64(size>>20)/up.Seconds())

	dest := filepath.Join(t.TempDir(), "restored")
	start = time.Now()
	if _, err := Fetch(bg, st, at("demo", 7), dest); err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	down := time.Since(start)
	t.Logf("fetched and verified %d MiB in %s (%.1f MiB/s)", size>>20, down.Round(time.Millisecond), float64(size>>20)/down.Seconds())
}

// Eight writers race for one key. S3 has to let exactly one in.
func TestIntegrationS3ConcurrentWritersOnlyOneWins(t *testing.T) {
	st := integrationStore(t)
	const key = "demo/step-1/app/app.pt"
	const writers = 8
	dir := t.TempDir()
	results := make([]error, writers)
	contents := make([]string, writers)
	var wg sync.WaitGroup
	for i := range writers {
		contents[i] = fmt.Sprintf("writer %d", i)
		src := writeFile(t, dir, fmt.Sprintf("src-%d", i), contents[i])
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i] = st.Put(bg, key, src, obj(contents[i]))
		}()
	}
	wg.Wait()

	winner := -1
	for i, err := range results {
		switch {
		case err == nil:
			if winner != -1 {
				t.Fatalf("writers %d and %d both succeeded: a stored file was overwritten", winner, i)
			}
			winner = i
		case errors.Is(err, ErrExists):
		default:
			t.Errorf("writer %d: unexpected error: %v", i, err)
		}
	}
	if winner == -1 {
		t.Fatal("no writer succeeded")
	}
	dst := filepath.Join(t.TempDir(), "copy")
	if err := st.Get(bg, key, dst); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(dst); string(data) != contents[winner] {
		t.Fatalf("the stored file holds %q, want the winner's %q", data, contents[winner])
	}
}

// A missing key and a missing bucket are different mistakes, and must look different.
func TestIntegrationS3MissingKeyAndMissingBucketAreDifferentErrors(t *testing.T) {
	st := integrationStore(t)
	if _, err := st.Stat(bg, "demo/step-1/nothing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Stat of a missing key: error = %v, want ErrNotFound", err)
	}
	if err := st.Get(bg, "demo/step-1/nothing", filepath.Join(t.TempDir(), "copy")); !errors.Is(err, ErrNotFound) {
		t.Errorf("Get of a missing key: error = %v, want ErrNotFound", err)
	}

	wrong := &S3{API: st.API, Bucket: "hibernus-no-such-bucket-" + runID}
	src := writeFile(t, t.TempDir(), "src", "hello")
	err := wrong.Put(bg, "demo/step-1/app/app.pt", src, obj("hello"))
	t.Logf("Put to a missing bucket: %v", err)
	if err == nil || errors.Is(err, ErrExists) {
		t.Errorf("Put to a missing bucket: error = %v, want a failure that is not ErrExists", err)
	}
	err = wrong.Get(bg, "demo/step-1/app/app.pt", filepath.Join(t.TempDir(), "copy"))
	t.Logf("Get from a missing bucket: %v", err)
	if err == nil || errors.Is(err, ErrNotFound) {
		t.Errorf("Get from a missing bucket: error = %v, want a failure that is not ErrNotFound, so a wrong bucket name is not mistaken for a missing file", err)
	}
}
