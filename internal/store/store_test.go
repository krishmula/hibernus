package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/manifest"
)

var bg = context.Background()

// testAttempt is the checkpoint attempt that the tests publish under.
const testAttempt = "20261119T000000Z-0000abcd"

// at names the restore point of workload at step, published under testAttempt.
func at(workload string, step int) layout.RestorePoint {
	return layout.RestorePoint{Workload: workload, Step: step, Attempt: testAttempt}
}

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	file := filepath.Join(dir, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return file
}

func sum(content string) string {
	s := sha256.Sum256([]byte(content))
	return hex.EncodeToString(s[:])
}

// obj describes a file with the given content, as the manifest would.
func obj(content string) Object {
	return Object{Size: int64(len(content)), Sha256: sum(content)}
}

// makeRestorePoint writes a small restore point folder with both layers and a manifest.
func makeRestorePoint(t *testing.T, workload string, step int) string {
	t.Helper()
	root := t.TempDir()
	writeFile(t, root, "app/app.pt", "application checkpoint bytes")
	writeFile(t, root, "process/pages-1.img", strings.Repeat("p", 5000))
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
		Fingerprint:   map[string]string{},
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

func notExist(t *testing.T, file string) {
	t.Helper()
	if _, err := os.Stat(file); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("%s exists, or cannot be read (%v), want it absent", file, err)
	}
}

// --- what every Store must do ---

func TestMemoryStoreContract(t *testing.T) {
	contract(t, func(t *testing.T) Store { return NewMemory() })
}

func TestLocalStoreContract(t *testing.T) {
	contract(t, func(t *testing.T) Store { return Local{Root: t.TempDir()} })
}

func contract(t *testing.T, newStore func(t *testing.T) Store) {
	const key = "demo/step-1/app/app.pt"

	t.Run("a stored file reads back with its size and checksum", func(t *testing.T) {
		st := newStore(t)
		src := writeFile(t, t.TempDir(), "src", "hello")
		if err := st.Put(bg, key, src, obj("hello")); err != nil {
			t.Fatal(err)
		}

		got, err := st.Stat(bg, key)
		if err != nil {
			t.Fatal(err)
		}
		if want := (Object{Size: 5, Sha256: sum("hello")}); got != want {
			t.Fatalf("Stat = %+v, want %+v", got, want)
		}

		dst := filepath.Join(t.TempDir(), "out", "copy") // Get makes the folders
		if err := st.Get(bg, key, dst); err != nil {
			t.Fatal(err)
		}
		if data, _ := os.ReadFile(dst); string(data) != "hello" {
			t.Fatalf("Get wrote %q, want %q", data, "hello")
		}
	})

	t.Run("a missing key is not found", func(t *testing.T) {
		st := newStore(t)
		if _, err := st.Stat(bg, key); !errors.Is(err, ErrNotFound) {
			t.Fatalf("Stat error = %v, want ErrNotFound", err)
		}
		if err := st.Get(bg, key, filepath.Join(t.TempDir(), "copy")); !errors.Is(err, ErrNotFound) {
			t.Fatalf("Get error = %v, want ErrNotFound", err)
		}
	})

	t.Run("a failed get leaves nothing at the destination", func(t *testing.T) {
		st := newStore(t)
		dst := filepath.Join(t.TempDir(), "copy")
		_ = st.Get(bg, key, dst)
		notExist(t, dst)
	})

	t.Run("a stored file is never replaced", func(t *testing.T) {
		st := newStore(t)
		first := writeFile(t, t.TempDir(), "src", "first")
		second := writeFile(t, t.TempDir(), "src", "second")
		if err := st.Put(bg, key, first, obj("first")); err != nil {
			t.Fatal(err)
		}

		if err := st.Put(bg, key, second, obj("second")); !errors.Is(err, ErrExists) {
			t.Fatalf("second Put error = %v, want ErrExists", err)
		}

		dst := filepath.Join(t.TempDir(), "copy")
		if err := st.Get(bg, key, dst); err != nil {
			t.Fatal(err)
		}
		if data, _ := os.ReadFile(dst); string(data) != "first" {
			t.Fatalf("the stored file now holds %q, want %q", data, "first")
		}
	})

	t.Run("a missing source file is an error and stores nothing", func(t *testing.T) {
		st := newStore(t)
		if err := st.Put(bg, key, filepath.Join(t.TempDir(), "nope"), obj("x")); err == nil {
			t.Fatal("Put of a missing file succeeded")
		}
		if _, err := st.Stat(bg, key); !errors.Is(err, ErrNotFound) {
			t.Fatalf("Stat error = %v, want ErrNotFound", err)
		}
	})

	t.Run("a file that is not what the caller expects is refused and stores nothing", func(t *testing.T) {
		st := newStore(t)
		src := writeFile(t, t.TempDir(), "src", "hello")

		err := st.Put(bg, key, src, obj("hello!")) // one byte longer than the file

		if !errors.Is(err, ErrMismatch) {
			t.Fatalf("Put error = %v, want ErrMismatch", err)
		}
		if _, err := st.Stat(bg, key); !errors.Is(err, ErrNotFound) {
			t.Fatalf("Stat error = %v, want ErrNotFound", err)
		}
	})

	t.Run("folders lists the folders directly under a prefix", func(t *testing.T) {
		st := newStore(t)
		src := writeFile(t, t.TempDir(), "src", "x")
		for _, k := range []string{
			"demo/step-1/manifest.json", "demo/step-1/app/app.pt",
			"demo/step-10/manifest.json", "demo/loose.txt",
			"demo2/step-7/manifest.json", "other/step-3/manifest.json",
		} {
			if err := st.Put(bg, k, src, obj("x")); err != nil {
				t.Fatal(err)
			}
		}

		got, err := st.Folders(bg, "demo")
		if err != nil {
			t.Fatal(err)
		}

		slices.Sort(got)
		if want := []string{"step-1", "step-10"}; !slices.Equal(got, want) {
			t.Fatalf("Folders(demo) = %v, want %v: only folders, only under demo/, and not demo2", got, want)
		}
	})

	t.Run("a prefix with nothing under it has no folders", func(t *testing.T) {
		st := newStore(t)

		got, err := st.Folders(bg, "nothing")

		if err != nil || len(got) != 0 {
			t.Fatalf("Folders(nothing) = %v, %v; want no folders and no error", got, err)
		}
	})

	t.Run("a cancelled context stops the call", func(t *testing.T) {
		st := newStore(t)
		cancelled, cancel := context.WithCancel(bg)
		cancel()
		src := writeFile(t, t.TempDir(), "src", "hello")
		if err := st.Put(cancelled, key, src, obj("hello")); !errors.Is(err, context.Canceled) {
			t.Fatalf("Put error = %v, want context.Canceled", err)
		}
	})
}

func TestLocalStoreRefusesAKeyThatLeavesItsFolder(t *testing.T) {
	st := Local{Root: t.TempDir()}
	src := writeFile(t, t.TempDir(), "src", "hello")

	if err := st.Put(bg, "../escape", src, obj("hello")); err == nil {
		t.Fatal("Put of ../escape succeeded")
	}
	notExist(t, filepath.Join(filepath.Dir(st.Root), "escape"))
}

// A store that reads the whole file anyway checks its checksum, not only its size. S3 will not:
// it checks the size, and Fetch checks the checksum when the file comes back.
func TestStoresThatReadTheFileRefuseOneWithAnotherChecksum(t *testing.T) {
	stores := map[string]Store{"memory": NewMemory(), "local": Local{Root: t.TempDir()}}
	for name, st := range stores {
		t.Run(name, func(t *testing.T) {
			src := writeFile(t, t.TempDir(), "src", "hello")

			err := st.Put(bg, "demo/step-1/app/app.pt", src, obj("jello")) // same size, other bytes

			if !errors.Is(err, ErrMismatch) {
				t.Fatalf("Put error = %v, want ErrMismatch", err)
			}
		})
	}
}

// --- Publish ---

func TestPublishPutsTheManifestLast(t *testing.T) {
	mem := NewMemory()
	root := makeRestorePoint(t, "demo", 5)

	if err := Publish(bg, mem, root, testAttempt); err != nil {
		t.Fatal(err)
	}

	puts := mem.Puts()
	if len(puts) != 3 {
		t.Fatalf("stored %v, want 3 files", puts)
	}
	if last := puts[len(puts)-1]; last != "demo/step-5-20261119T000000Z-0000abcd/manifest.json" {
		t.Fatalf("the last file stored was %q, want the manifest", last)
	}
	rest := slices.Clone(puts[:len(puts)-1])
	slices.Sort(rest)
	if want := []string{"demo/step-5-20261119T000000Z-0000abcd/app/app.pt", "demo/step-5-20261119T000000Z-0000abcd/process/pages-1.img"}; !slices.Equal(rest, want) {
		t.Fatalf("stored %v before the manifest, want %v", rest, want)
	}
}

func TestPublishNamesTheFilesFromTheManifestsWorkloadAndStep(t *testing.T) {
	mem := NewMemory()
	if err := Publish(bg, mem, makeRestorePoint(t, "run-2", 1200), testAttempt); err != nil {
		t.Fatal(err)
	}

	for _, key := range mem.Puts() {
		if !strings.HasPrefix(key, "run-2/step-1200-20261119T000000Z-0000abcd/") {
			t.Errorf("key %q is not under run-2/step-1200/", key)
		}
	}
}

func TestPublishRefusesAFolderWithoutAManifest(t *testing.T) {
	mem := NewMemory()
	root := t.TempDir()
	writeFile(t, root, "app/app.pt", "bytes")

	if err := Publish(bg, mem, root, testAttempt); err == nil {
		t.Fatal("Publish of a folder without a manifest succeeded")
	}
	if puts := mem.Puts(); len(puts) != 0 {
		t.Fatalf("stored %v, want nothing", puts)
	}
}

func TestAFailedUploadLeavesNoManifestInTheStore(t *testing.T) {
	mem := NewMemory()
	mem.PutHook = func(key string) error {
		if strings.Contains(key, "/process/") {
			return errors.New("connection reset")
		}
		return nil
	}

	if err := Publish(bg, mem, makeRestorePoint(t, "demo", 5), testAttempt); err == nil {
		t.Fatal("Publish succeeded although an upload failed")
	}

	if _, err := mem.Stat(bg, "demo/step-5-20261119T000000Z-0000abcd/manifest.json"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("the store holds a manifest after a failed upload (%v): a half-uploaded restore point would look complete", err)
	}
}

func TestPublishCanBeRetriedAfterAFailure(t *testing.T) {
	mem := NewMemory()
	failed := false
	mem.PutHook = func(key string) error {
		if strings.Contains(key, "/process/") && !failed {
			failed = true
			return errors.New("connection reset")
		}
		return nil
	}
	root := makeRestorePoint(t, "demo", 5)
	if err := Publish(bg, mem, root, testAttempt); err == nil {
		t.Fatal("the first Publish should have failed")
	}

	if err := Publish(bg, mem, root, testAttempt); err != nil { // app.pt is already stored, and that is fine
		t.Fatalf("the retry failed: %v", err)
	}

	if _, err := mem.Stat(bg, "demo/step-5-20261119T000000Z-0000abcd/manifest.json"); err != nil {
		t.Fatalf("no manifest after the retry: %v", err)
	}
}

func TestPublishingTheSameRestorePointTwiceIsFine(t *testing.T) {
	mem := NewMemory()
	root := makeRestorePoint(t, "demo", 5)

	if err := Publish(bg, mem, root, testAttempt); err != nil {
		t.Fatal(err)
	}
	if err := Publish(bg, mem, root, testAttempt); err != nil {
		t.Fatalf("the second Publish failed: %v", err)
	}
}

func TestPublishRefusesToChangeAFileThatIsAlreadyStored(t *testing.T) {
	mem := NewMemory()
	mem.Set("demo/step-5-20261119T000000Z-0000abcd/app/app.pt", []byte("someone else's bytes"))

	err := Publish(bg, mem, makeRestorePoint(t, "demo", 5), testAttempt)

	if err == nil || !strings.Contains(err.Error(), "different content") {
		t.Fatalf("Publish error = %v, want one that says the stored file has different content", err)
	}
	if _, statErr := mem.Stat(bg, "demo/step-5-20261119T000000Z-0000abcd/manifest.json"); !errors.Is(statErr, ErrNotFound) {
		t.Fatalf("a manifest was stored over a conflicting file (%v)", statErr)
	}
}

func TestPublishCatchesAnUploadThatArrivedCorrupted(t *testing.T) {
	mem := NewMemory()
	mem.Mangle = func(key string, data []byte) []byte {
		if strings.HasSuffix(key, "pages-1.img") {
			data[0] ^= 0xff
		}
		return data
	}

	if err := Publish(bg, mem, makeRestorePoint(t, "demo", 5), testAttempt); err == nil {
		t.Fatal("Publish succeeded although a stored file differs from the manifest")
	}

	if _, err := mem.Stat(bg, "demo/step-5-20261119T000000Z-0000abcd/manifest.json"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a manifest was stored after a corrupted upload (%v)", err)
	}
}

func TestPublishRefusesAFileThatChangedAfterTheManifestWasWritten(t *testing.T) {
	mem := NewMemory()
	root := makeRestorePoint(t, "demo", 5)
	writeFile(t, root, "app/app.pt", "application checkpoint BYTES") // same length as before, other bytes

	err := Publish(bg, mem, root, testAttempt)

	if !errors.Is(err, ErrMismatch) {
		t.Fatalf("Publish error = %v, want ErrMismatch", err)
	}
	if _, statErr := mem.Stat(bg, "demo/step-5-20261119T000000Z-0000abcd/manifest.json"); !errors.Is(statErr, ErrNotFound) {
		t.Fatalf("a manifest was stored for a restore point whose file changed (%v)", statErr)
	}
}

// --- Fetch ---

func publish(t *testing.T, mem *Memory, workload string, step int) {
	t.Helper()
	if err := Publish(bg, mem, makeRestorePoint(t, workload, step), testAttempt); err != nil {
		t.Fatal(err)
	}
}

func TestFetchBringsBackWhatPublishStored(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 5)
	dest := filepath.Join(t.TempDir(), "restored")

	m, err := Fetch(bg, mem, at("demo", 5), dest)
	if err != nil {
		t.Fatal(err)
	}

	if m.Workload != "demo" || m.Step != 5 {
		t.Fatalf("Fetch returned the manifest of %q step %d", m.Workload, m.Step)
	}
	if data, _ := os.ReadFile(filepath.Join(dest, "app", "app.pt")); string(data) != "application checkpoint bytes" {
		t.Fatalf("app.pt holds %q", data)
	}
	loaded, err := manifest.Load(dest)
	if err != nil || loaded.Step != 5 {
		t.Fatalf("the manifest in the folder: %+v, %v", loaded, err)
	}
	for _, layer := range []manifest.Layer{loaded.Layers.ApplicationCheckpoint, loaded.Layers.ProcessSnapshot} {
		if err := layer.Verify(dest); err != nil {
			t.Fatalf("a fetched layer does not verify: %v", err)
		}
	}
	entries, _ := os.ReadDir(dest)
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if want := []string{"app", "manifest.json", "process"}; !slices.Equal(names, want) {
		t.Fatalf("the folder holds %v, want only %v (no leftover temporary files)", names, want)
	}
}

func TestFetchRefusesALocationWithoutAManifest(t *testing.T) {
	mem := NewMemory()
	mem.Set("demo/step-5-20261119T000000Z-0000abcd/app/app.pt", []byte("bytes")) // files, but the upload never finished
	dest := filepath.Join(t.TempDir(), "restored")

	_, err := Fetch(bg, mem, at("demo", 5), dest)

	if err == nil || !strings.Contains(err.Error(), "not a complete restore point") {
		t.Fatalf("Fetch error = %v, want one that says it is not a complete restore point", err)
	}
	notExist(t, filepath.Join(dest, manifest.FileName))
}

func TestFetchRefusesACorruptFileAndLeavesNoManifest(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 5)
	data, _ := mem.Data("demo/step-5-20261119T000000Z-0000abcd/app/app.pt")
	data[0] ^= 0xff
	mem.Set("demo/step-5-20261119T000000Z-0000abcd/app/app.pt", data)
	dest := filepath.Join(t.TempDir(), "restored")

	if _, err := Fetch(bg, mem, at("demo", 5), dest); err == nil {
		t.Fatal("Fetch succeeded with a corrupted file")
	}

	notExist(t, filepath.Join(dest, manifest.FileName))
}

func TestFetchRemovesAManifestLeftByAnEarlierAttempt(t *testing.T) {
	dest := t.TempDir()
	writeFile(t, dest, manifest.FileName, "an old manifest")

	if _, err := Fetch(bg, NewMemory(), at("demo", 5), dest); err == nil {
		t.Fatal("Fetch from an empty store succeeded")
	}

	notExist(t, filepath.Join(dest, manifest.FileName))
}

func TestFetchRefusesAManifestThatBelongsToAnotherStep(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 5)
	data, _ := mem.Data("demo/step-5-20261119T000000Z-0000abcd/manifest.json")
	mem.Set("demo/step-6-20261119T000000Z-0000abcd/manifest.json", data) // step 5's manifest, filed under step 6

	_, err := Fetch(bg, mem, at("demo", 6), filepath.Join(t.TempDir(), "restored"))

	if err == nil || !strings.Contains(err.Error(), "step 5") {
		t.Fatalf("Fetch error = %v, want one that names step 5", err)
	}
}

func TestFetchRefusesAManifestThatListsAFileOutsideTheFolder(t *testing.T) {
	mem := NewMemory()
	evil := manifest.Manifest{
		SchemaVersion: manifest.SchemaVersion, Workload: "demo", Step: 5,
		Layers: manifest.Layers{ApplicationCheckpoint: manifest.Layer{
			Status: manifest.StatusOK,
			Files:  []manifest.File{{Path: "../escape", Size: 1, Sha256: "x"}},
		}},
	}
	data, err := json.Marshal(evil)
	if err != nil {
		t.Fatal(err)
	}
	mem.Set("demo/step-5-20261119T000000Z-0000abcd/manifest.json", data)
	mem.Set("demo/step-5-20261119T000000Z-0000abcd/../escape", []byte("x"))
	parent := t.TempDir()
	dest := filepath.Join(parent, "restored")

	if _, err := Fetch(bg, mem, at("demo", 5), dest); err == nil {
		t.Fatal("Fetch accepted a file outside the restore point")
	}

	notExist(t, filepath.Join(parent, "escape"))
}

func TestABadProcessSnapshotFallsBackToTheApplicationCheckpoint(t *testing.T) {
	const pages = "demo/step-5-20261119T000000Z-0000abcd/process/pages-1.img"
	corrupt := NewMemory()
	publish(t, corrupt, "demo", 5)
	data, _ := corrupt.Data(pages)
	data[0] ^= 0xff
	corrupt.Set(pages, data)

	missing := NewMemory()
	publish(t, missing, "demo", 5)
	missing.Delete(pages)

	for name, mem := range map[string]*Memory{"corrupt": corrupt, "missing": missing} {
		dest := filepath.Join(t.TempDir(), "restored")

		m, err := Fetch(bg, mem, at("demo", 5), dest)
		if err != nil {
			t.Fatalf("%s process snapshot: Fetch = %v, want the restore point with the application checkpoint", name, err)
		}

		local, err := manifest.Load(dest)
		if err != nil {
			t.Fatalf("%s: no local manifest: %v", name, err)
		}
		for _, got := range []manifest.Manifest{m, local} {
			process := got.Layers.ProcessSnapshot
			if process.Status != manifest.StatusFailed || process.Error == nil || !strings.Contains(*process.Error, "process/pages-1.img") {
				t.Fatalf("%s: process snapshot = %+v, want failed, naming the bad file", name, process)
			}
			if len(process.Files) != 0 {
				t.Fatalf("%s: a failed process snapshot still lists files %+v", name, process.Files)
			}
			if got.Layers.ApplicationCheckpoint.Status != manifest.StatusOK {
				t.Fatalf("%s: application checkpoint = %+v, want ok", name, got.Layers.ApplicationCheckpoint)
			}
		}
		if err := local.Layers.ApplicationCheckpoint.Verify(dest); err != nil {
			t.Fatalf("%s: the application checkpoint does not verify: %v", name, err)
		}
	}
}
