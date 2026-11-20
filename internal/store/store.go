// Package store moves a restore point between the host and durable storage, such as S3.
//
// A Store holds single files by key. The rules that make a restore point safe sit above it, in
// Publish and Fetch, so they are written once and hold for every Store:
//
//   - Publish uploads every file, checks each one in the store, and uploads the manifest last.
//     A key prefix without a manifest is not a restore point.
//   - Stored files never change. Publishing the same restore point again is fine, and publishing
//     different content under a key that is taken is an error. So each checkpoint attempt
//     publishes under its own folder, which no other attempt can have filled.
//   - Fetch downloads the manifest first, checks every file against it, and installs the manifest
//     last. A folder with a manifest is a verified restore point. A process snapshot that fails
//     its check is marked failed there, because the application checkpoint can still restore.
package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/manifest"
)

var (
	// ErrNotFound means the store has nothing under that key.
	ErrNotFound = errors.New("not found in the store")
	// ErrExists means a Put found something under that key already.
	ErrExists = errors.New("already in the store")
	// ErrMismatch means a Put found that the source file is not the file the caller described.
	ErrMismatch = errors.New("the file is not the one expected")
	// ErrConflict means the store holds a different file under a key than the one being published.
	// A stored file never changes, so retrying cannot fix it.
	ErrConflict = errors.New("the store holds a different file under that key")
	// ErrBadRestorePoint means a restore point cannot be used as it is stored: it has no manifest,
	// a file is missing or corrupt, or the manifest does not match its folder. A problem reaching
	// the store, such as a network or permission error, is not this, and says nothing about the
	// restore point.
	ErrBadRestorePoint = errors.New("bad restore point")
	// ErrNoRestorePoint means a workload has no restore point that downloads and verifies.
	ErrNoRestorePoint = errors.New("no usable restore point")
)

// permanentCodes are the S3 error codes that a retry cannot fix: the credentials or the bucket are
// wrong, or the request is not allowed.
var permanentCodes = map[string]bool{
	"AccessDenied":          true,
	"InvalidAccessKeyId":    true,
	"SignatureDoesNotMatch": true,
	"ExpiredToken":          true,
	"InvalidToken":          true,
	"NoSuchBucket":          true,
	"InvalidBucketName":     true,
}

// Permanent reports whether trying again cannot help. The Supervisor uses it to give up at once
// instead of retrying until the deadline: waiting does not fix a wrong credential, a missing
// bucket, a source file that changed, or a stored file that differs from ours.
func Permanent(err error) bool {
	return errors.Is(err, ErrMismatch) ||
		errors.Is(err, ErrConflict) ||
		errors.Is(err, os.ErrNotExist) ||
		permanentCodes[apiCode(err)]
}

// Object is what a store knows about one stored file.
type Object struct {
	Size   int64
	Sha256 string // lowercase hex
}

// Store keeps files by key. A key uses forward slashes, such as demo/step-1200/app/app.pt.
type Store interface {
	// Put stores the file at src under key. The caller says what the file is, in want, because it
	// already has the size and checksum from the manifest. Put returns ErrMismatch, and stores
	// nothing, when src is not that file. A store that must read the whole file anyway checks the
	// checksum too. A store that would have to read a very large file just to check it, such as S3,
	// checks the size, records want.Sha256 with the file, and leaves the full checksum check to Fetch.
	// A stored file never changes, so Put returns ErrExists when the key is taken, and leaves what
	// is there alone.
	Put(ctx context.Context, key, src string, want Object) error
	// Stat reports the size and checksum of the file under key, or ErrNotFound. A store that
	// does not hash the file itself reports the checksum that Put recorded.
	Stat(ctx context.Context, key string) (Object, error)
	// Get downloads the file under key to dst, or returns ErrNotFound. A failed Get leaves nothing
	// at dst: a partial download never appears there.
	Get(ctx context.Context, key, dst string) error
	// Folders lists the names of the folders directly under prefix, such as step-1200 under a
	// workload. A prefix with nothing under it has none, and that is not an error.
	Folders(ctx context.Context, prefix string) ([]string, error)
}

// Publish stores the restore point in root, which must hold a manifest, under the manifest's own
// workload and step and the given attempt. It checks every file in the store against the
// manifest, and stores the manifest last. It is safe to call again with the same attempt after a
// failure or a success.
func Publish(ctx context.Context, st Store, root, attempt string) error {
	m, err := manifest.Load(root)
	if err != nil {
		return err
	}
	if !layout.ValidWorkload(m.Workload) {
		return fmt.Errorf("the manifest names workload %q, which is not a valid name", m.Workload)
	}
	prefix := layout.RestorePoint{Workload: m.Workload, Step: m.Step, Attempt: attempt}.Key()

	for _, f := range layerFiles(m) {
		if !filepath.IsLocal(f.Path) {
			return fmt.Errorf("%s points outside the restore point", f.Path)
		}
		key := path.Join(prefix, filepath.ToSlash(f.Path))
		if err := putVerified(ctx, st, key, filepath.Join(root, f.Path), Object{Size: f.Size, Sha256: f.Sha256}); err != nil {
			return err
		}
	}

	manifestPath := filepath.Join(root, manifest.FileName)
	size, sum, err := measure(manifestPath)
	if err != nil {
		return err
	}
	return putVerified(ctx, st, path.Join(prefix, manifest.FileName), manifestPath, Object{Size: size, Sha256: sum})
}

// putVerified stores one file and checks that the store holds exactly what the manifest says. A
// key that is taken is fine when it already holds the same file, which is what a retry finds.
func putVerified(ctx context.Context, st Store, key, src string, want Object) error {
	err := st.Put(ctx, key, src, want)
	existed := errors.Is(err, ErrExists)
	if err != nil && !existed {
		return fmt.Errorf("upload %s: %w", key, err)
	}
	got, err := st.Stat(ctx, key)
	if err != nil {
		return fmt.Errorf("check %s in the store: %w", key, err)
	}
	if got == want {
		return nil
	}
	if existed {
		return fmt.Errorf("%w: %s is already in the store with different content, and stored files never change", ErrConflict, key)
	}
	return fmt.Errorf("%w: %s in the store has %d bytes and checksum %s, want %d bytes and checksum %s",
		ErrConflict, key, got.Size, got.Sha256, want.Size, want.Sha256)
}

// Fetch downloads the restore point rp into root and returns its manifest. It downloads the
// manifest first, refuses a location that has none, checks every file against it, and moves the
// manifest into root last, so root holds a manifest only when its files are verified. A process
// snapshot that does not verify is recorded as failed in that local manifest, and the restore
// point is still good. A manifest or application checkpoint that does not verify makes it bad.
func Fetch(ctx context.Context, st Store, rp layout.RestorePoint, root string) (manifest.Manifest, error) {
	if !layout.ValidWorkload(rp.Workload) {
		return manifest.Manifest{}, fmt.Errorf("%q is not a valid workload name", rp.Workload)
	}
	workload, step, prefix := rp.Workload, rp.Step, rp.Key()

	// A manifest left in root by an earlier attempt must not make this folder look complete.
	if err := os.Remove(filepath.Join(root, manifest.FileName)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return manifest.Manifest{}, err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return manifest.Manifest{}, err
	}
	staging, err := os.MkdirTemp(root, ".fetch-")
	if err != nil {
		return manifest.Manifest{}, err
	}
	defer os.RemoveAll(staging)

	if err := st.Get(ctx, path.Join(prefix, manifest.FileName), filepath.Join(staging, manifest.FileName)); err != nil {
		if errors.Is(err, ErrNotFound) {
			return manifest.Manifest{}, fmt.Errorf("%w: %s has no %s in the store, so it is not a complete restore point", ErrBadRestorePoint, prefix, manifest.FileName)
		}
		return manifest.Manifest{}, fmt.Errorf("download %s: %w", manifest.FileName, err)
	}
	m, err := manifest.Load(staging)
	if err != nil {
		return manifest.Manifest{}, fmt.Errorf("%w: %v", ErrBadRestorePoint, err)
	}
	if m.Workload != workload || m.Step != step {
		return manifest.Manifest{}, fmt.Errorf("%w: %s holds the manifest of workload %q step %d", ErrBadRestorePoint, prefix, m.Workload, m.Step)
	}

	problem, err := fetchLayer(ctx, st, prefix, root, m.Layers.ApplicationCheckpoint)
	if err != nil {
		return manifest.Manifest{}, err
	}
	if problem != "" {
		return manifest.Manifest{}, fmt.Errorf("%w: %s", ErrBadRestorePoint, problem)
	}
	// A bad process snapshot leaves the application checkpoint to restore from (N3), so it does not
	// make the restore point bad. The local manifest records it as failed.
	problem, err = fetchLayer(ctx, st, prefix, root, m.Layers.ProcessSnapshot)
	if err != nil {
		return manifest.Manifest{}, err
	}
	if problem != "" {
		m.Layers.ProcessSnapshot = manifest.Layer{Status: manifest.StatusFailed, Error: &problem, Files: []manifest.File{}}
		if err := manifest.Write(staging, m); err != nil {
			return manifest.Manifest{}, err
		}
	}
	if err := os.Rename(filepath.Join(staging, manifest.FileName), filepath.Join(root, manifest.FileName)); err != nil {
		return manifest.Manifest{}, err
	}
	return m, nil
}

// fetchLayer downloads the files of one layer into root and checks them against the manifest. A
// problem with the layer as stored, such as a missing or corrupt file, comes back as problem. A
// problem reaching the store comes back as err, and says nothing about the layer.
func fetchLayer(ctx context.Context, st Store, prefix, root string, layer manifest.Layer) (problem string, err error) {
	for _, f := range layer.Files {
		if !filepath.IsLocal(f.Path) {
			return f.Path + " points outside the restore point", nil
		}
		if err := st.Get(ctx, path.Join(prefix, filepath.ToSlash(f.Path)), filepath.Join(root, f.Path)); err != nil {
			if errors.Is(err, ErrNotFound) {
				return f.Path + " is listed in the manifest but is not in the store", nil
			}
			return "", fmt.Errorf("download %s: %w", f.Path, err)
		}
	}
	if err := layer.Verify(root); err != nil {
		return err.Error(), nil
	}
	return "", nil
}

// sameBytes reports whether data is the file that want describes.
func sameBytes(data []byte, want Object) bool {
	sum := sha256.Sum256(data)
	return int64(len(data)) == want.Size && hex.EncodeToString(sum[:]) == want.Sha256
}

// layerFiles lists the files of both layers. A failed layer has none.
func layerFiles(m manifest.Manifest) []manifest.File {
	var files []manifest.File
	files = append(files, m.Layers.ApplicationCheckpoint.Files...)
	files = append(files, m.Layers.ProcessSnapshot.Files...)
	return files
}

// copyAtomic writes r to dst through a temporary file in the same folder, so dst appears whole or
// not at all.
func copyAtomic(dst string, r io.Reader) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".part-")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // after a successful rename there is nothing to remove
	if _, err := io.Copy(tmp, r); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), dst)
}

// measure returns the size and sha256 of a file. internal/manifest has the same function, unexported.
func measure(file string) (int64, string, error) {
	f, err := os.Open(file)
	if err != nil {
		return 0, "", err
	}
	defer f.Close()
	hash := sha256.New()
	size, err := io.Copy(hash, f)
	if err != nil {
		return 0, "", err
	}
	return size, hex.EncodeToString(hash.Sum(nil)), nil
}
