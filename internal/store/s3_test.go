package store

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/transfermanager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	s3types "github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
)

// fakeS3 is a small in-memory S3. It embeds the interface so that the methods these tests never
// reach, the multipart ones, exist and panic if something calls them. It answers like S3 does:
// If-None-Match: * on a taken key gives PreconditionFailed, and a missing key gives NotFound.
type fakeS3 struct {
	transfermanager.S3APIClient

	mu      sync.Mutex
	objects map[string]fakeObject
	puts    []*s3.PutObjectInput

	conflicts int   // how many PutObject calls answer 409 before one is accepted
	headErr   error // when set, HeadObject fails with it
	pageSize  int   // when set, ListObjectsV2 returns at most this many folders per page
}

type fakeObject struct {
	data     []byte
	metadata map[string]string
}

func newFakeS3() *fakeS3 { return &fakeS3{objects: map[string]fakeObject{}} }

func (f *fakeS3) PutObject(ctx context.Context, in *s3.PutObjectInput, _ ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	data, err := io.ReadAll(in.Body)
	if err != nil {
		return nil, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.puts = append(f.puts, in)
	if f.conflicts > 0 {
		f.conflicts--
		return nil, &smithy.GenericAPIError{Code: "ConditionalRequestConflict", Message: "a conflicting write is in progress"}
	}
	key := aws.ToString(in.Bucket) + "/" + aws.ToString(in.Key)
	if aws.ToString(in.IfNoneMatch) == "*" {
		if _, taken := f.objects[key]; taken {
			return nil, &smithy.GenericAPIError{Code: "PreconditionFailed", Message: "the pre-condition did not hold"}
		}
	}
	f.objects[key] = fakeObject{data: data, metadata: in.Metadata}
	return &s3.PutObjectOutput{}, nil
}

func (f *fakeS3) HeadObject(ctx context.Context, in *s3.HeadObjectInput, _ ...func(*s3.Options)) (*s3.HeadObjectOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.headErr != nil {
		return nil, f.headErr
	}
	object, ok := f.objects[aws.ToString(in.Bucket)+"/"+aws.ToString(in.Key)]
	if !ok {
		return nil, &s3types.NotFound{}
	}
	return &s3.HeadObjectOutput{ContentLength: aws.Int64(int64(len(object.data))), Metadata: object.metadata}, nil
}

func (f *fakeS3) GetObject(ctx context.Context, in *s3.GetObjectInput, _ ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	object, ok := f.objects[aws.ToString(in.Bucket)+"/"+aws.ToString(in.Key)]
	if !ok {
		return nil, &s3types.NoSuchKey{}
	}
	return &s3.GetObjectOutput{
		Body:          io.NopCloser(bytes.NewReader(object.data)),
		ContentLength: aws.Int64(int64(len(object.data))),
	}, nil
}

// ListObjectsV2 answers a listing with a "/" delimiter: the folders directly under the prefix, in
// order, in pages when pageSize is set. A page that is cut short says so and names where to go on.
func (f *fakeS3) ListObjectsV2(ctx context.Context, in *s3.ListObjectsV2Input, _ ...func(*s3.Options)) (*s3.ListObjectsV2Output, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	bucketPrefix := aws.ToString(in.Bucket) + "/"
	prefix := aws.ToString(in.Prefix)
	seen := map[string]bool{}
	var folders []string
	for full := range f.objects {
		key, ok := strings.CutPrefix(full, bucketPrefix)
		if !ok {
			continue
		}
		rest, ok := strings.CutPrefix(key, prefix)
		if !ok {
			continue
		}
		if i := strings.Index(rest, "/"); i >= 0 {
			folder := prefix + rest[:i+1]
			if !seen[folder] {
				seen[folder] = true
				folders = append(folders, folder)
			}
		}
	}
	sort.Strings(folders)

	if token := aws.ToString(in.ContinuationToken); token != "" {
		start := 0
		for start < len(folders) && folders[start] <= token {
			start++
		}
		folders = folders[start:]
	}
	out := &s3.ListObjectsV2Output{}
	if f.pageSize > 0 && len(folders) > f.pageSize {
		folders = folders[:f.pageSize]
		out.IsTruncated = aws.Bool(true)
		out.NextContinuationToken = aws.String(folders[len(folders)-1])
	}
	for _, folder := range folders {
		out.CommonPrefixes = append(out.CommonPrefixes, s3types.CommonPrefix{Prefix: aws.String(folder)})
	}
	return out, nil
}

func TestS3FoldersReadsEveryPage(t *testing.T) {
	fake := newFakeS3()
	fake.pageSize = 2
	st := &S3{API: fake, Bucket: "bucket", Prefix: "hib"}
	src := writeFile(t, t.TempDir(), "src", "x")
	for step := 1; step <= 5; step++ {
		if err := st.Put(bg, fmt.Sprintf("demo/step-%d/manifest.json", step), src, obj("x")); err != nil {
			t.Fatal(err)
		}
	}

	got, err := st.Folders(bg, "demo")
	if err != nil {
		t.Fatal(err)
	}

	sort.Strings(got)
	if want := []string{"step-1", "step-2", "step-3", "step-4", "step-5"}; !slices.Equal(got, want) {
		t.Fatalf("Folders = %v, want %v across three pages", got, want)
	}
}

func TestS3StoreContract(t *testing.T) {
	contract(t, func(t *testing.T) Store { return &S3{API: newFakeS3(), Bucket: "bucket"} })
}

func TestS3PutAsksForANoOverwriteWriteAndRecordsTheChecksum(t *testing.T) {
	fake := newFakeS3()
	st := &S3{API: fake, Bucket: "bucket"}
	src := writeFile(t, t.TempDir(), "src", "hello")

	if err := st.Put(bg, "demo/step-1/app/app.pt", src, obj("hello")); err != nil {
		t.Fatal(err)
	}

	in := fake.puts[0]
	if aws.ToString(in.IfNoneMatch) != "*" {
		t.Errorf("IfNoneMatch = %q, want *: without it S3 would overwrite a stored file", aws.ToString(in.IfNoneMatch))
	}
	if got := in.Metadata["sha256"]; got != sum("hello") {
		t.Errorf("recorded checksum = %q, want %q", got, sum("hello"))
	}
	if in.ChecksumAlgorithm != s3types.ChecksumAlgorithmCrc32 {
		t.Errorf("ChecksumAlgorithm = %q, want CRC32, so S3 verifies the upload itself", in.ChecksumAlgorithm)
	}
	if aws.ToString(in.Bucket) != "bucket" || aws.ToString(in.Key) != "demo/step-1/app/app.pt" {
		t.Errorf("uploaded to %s/%s", aws.ToString(in.Bucket), aws.ToString(in.Key))
	}
}

func TestS3PutDoesNotHashTheFile(t *testing.T) {
	// A file of the right size with other bytes is accepted: S3Store checks the size, and trusts
	// the caller's checksum, because hashing a huge file again would cost minutes. Fetch checks it.
	st := &S3{API: newFakeS3(), Bucket: "bucket"}
	src := writeFile(t, t.TempDir(), "src", "hello")

	if err := st.Put(bg, "demo/step-1/app/app.pt", src, obj("jello")); err != nil {
		t.Fatalf("Put refused a same-size file: %v", err)
	}
}

func TestS3PrefixGoesInFrontOfEveryKey(t *testing.T) {
	fake := newFakeS3()
	st := &S3{API: fake, Bucket: "bucket", Prefix: "hib"}
	src := writeFile(t, t.TempDir(), "src", "hello")

	if err := st.Put(bg, "demo/step-1/app/app.pt", src, obj("hello")); err != nil {
		t.Fatal(err)
	}

	if _, ok := fake.objects["bucket/hib/demo/step-1/app/app.pt"]; !ok {
		t.Fatalf("objects = %v, want bucket/hib/demo/step-1/app/app.pt", keysOf(fake.objects))
	}
	if _, err := st.Stat(bg, "demo/step-1/app/app.pt"); err != nil {
		t.Fatalf("Stat with the same prefix: %v", err)
	}
}

func keysOf(objects map[string]fakeObject) []string {
	var keys []string
	for key := range objects {
		keys = append(keys, key)
	}
	return keys
}

func TestS3PutRetriesWhenS3ReportsAConflict(t *testing.T) {
	fake := newFakeS3()
	fake.conflicts = 2
	st := &S3{API: fake, Bucket: "bucket"}
	src := writeFile(t, t.TempDir(), "src", "hello")

	if err := st.Put(bg, "demo/step-1/app/app.pt", src, obj("hello")); err != nil {
		t.Fatalf("Put gave up after a conflict: %v", err)
	}

	if len(fake.puts) != 3 {
		t.Fatalf("PutObject was called %d times, want 3 (two conflicts, then success)", len(fake.puts))
	}
}

func TestS3PutGivesUpWhenConflictsKeepComing(t *testing.T) {
	fake := newFakeS3()
	fake.conflicts = 100
	st := &S3{API: fake, Bucket: "bucket"}
	src := writeFile(t, t.TempDir(), "src", "hello")

	err := st.Put(bg, "demo/step-1/app/app.pt", src, obj("hello"))

	if err == nil || errors.Is(err, ErrExists) {
		t.Fatalf("Put error = %v, want a failure that is not ErrExists", err)
	}
	if want := conflictRetries + 1; len(fake.puts) != want {
		t.Fatalf("PutObject was called %d times, want %d", len(fake.puts), want)
	}
}

func TestS3PutStopsRetryingWhenTheContextEnds(t *testing.T) {
	fake := newFakeS3()
	fake.conflicts = 100
	st := &S3{API: fake, Bucket: "bucket"}
	src := writeFile(t, t.TempDir(), "src", "hello")
	ctx, cancel := context.WithCancel(bg)
	cancel()

	if err := st.Put(ctx, "demo/step-1/app/app.pt", src, obj("hello")); !errors.Is(err, context.Canceled) {
		t.Fatalf("Put error = %v, want context.Canceled", err)
	}
}

func TestS3StatDoesNotCallAnOtherFailureNotFound(t *testing.T) {
	fake := newFakeS3()
	fake.headErr = errors.New("connection reset")
	st := &S3{API: fake, Bucket: "bucket"}

	_, err := st.Stat(bg, "demo/step-1/app/app.pt")

	if err == nil || errors.Is(err, ErrNotFound) {
		t.Fatalf("Stat error = %v, want a failure that is not ErrNotFound: a network error must not look like a missing file", err)
	}
}

func TestS3StatOfAnObjectWithoutARecordedChecksumHasNone(t *testing.T) {
	fake := newFakeS3()
	fake.objects["bucket/demo/step-1/app/app.pt"] = fakeObject{data: []byte("hello")} // not written by Put
	st := &S3{API: fake, Bucket: "bucket"}

	got, err := st.Stat(bg, "demo/step-1/app/app.pt")
	if err != nil {
		t.Fatal(err)
	}

	if got.Size != 5 || got.Sha256 != "" {
		t.Fatalf("Stat = %+v, want size 5 and no checksum", got)
	}
}

func TestPublishAndFetchRoundTripThroughS3(t *testing.T) {
	st := &S3{API: newFakeS3(), Bucket: "bucket", Prefix: "hib"}
	root := makeRestorePoint(t, "demo", 5)

	if err := Publish(bg, st, root, testAttempt); err != nil {
		t.Fatal(err)
	}
	dest := t.TempDir() + "/restored"
	if _, err := Fetch(bg, st, at("demo", 5), dest); err != nil {
		t.Fatal(err)
	}

	original, _ := os.ReadFile(root + "/process/pages-1.img")
	fetched, _ := os.ReadFile(dest + "/process/pages-1.img")
	if !bytes.Equal(original, fetched) {
		t.Fatal("the fetched file differs from the published one")
	}
}

func TestPublishRefusesAnS3ObjectThatNoPutWrote(t *testing.T) {
	// Same size as the manifest says, but nobody recorded a checksum for it: not one of ours.
	fake := newFakeS3()
	fake.objects["bucket/demo/step-5-20261119T000000Z-0000abcd/app/app.pt"] = fakeObject{data: []byte(strings.Repeat("x", 28))}
	st := &S3{API: fake, Bucket: "bucket"}

	err := Publish(bg, st, makeRestorePoint(t, "demo", 5), testAttempt)

	if err == nil || !strings.Contains(err.Error(), "different content") {
		t.Fatalf("Publish error = %v, want one that says the stored file has different content", err)
	}
	if _, ok := fake.objects["bucket/demo/step-5-20261119T000000Z-0000abcd/manifest.json"]; ok {
		t.Fatal("a manifest was stored over a file that was not ours")
	}
}
