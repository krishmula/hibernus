package store

import (
	"errors"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/aws/smithy-go"

	"github.com/krishmula/hibernus/internal/layout"
)

func rootsUnder(base string) func(step int) string {
	return func(step int) string { return layout.LocalDir(base, "demo", step) }
}

func putAll(t *testing.T, mem *Memory, keys ...string) {
	t.Helper()
	src := writeFile(t, t.TempDir(), "src", "x")
	for _, key := range keys {
		if err := mem.Put(bg, key, src, obj("x")); err != nil {
			t.Fatal(err)
		}
	}
}

func TestListListsRestorePointFoldersNewestFirst(t *testing.T) {
	mem := NewMemory()
	putAll(t, mem,
		"demo/step-9-20261119T000000Z-0000abcd/manifest.json",
		"demo/step-10-20261119T000000Z-0000abcd/manifest.json",
		"demo/step-100-20261119T000000Z-0000abcd/manifest.json",
		"demo/step-100-20261119T090000Z-0000abcd/manifest.json",  // a later attempt at step 100
		"demo/notes/readme.txt",                                  // not a restore point folder
		"demo/step-007-20261119T000000Z-0000abcd/manifest.json",  // not the spelling that is written
		"demo/step-5/manifest.json",                              // no attempt
		"demo2/step-500-20261119T000000Z-0000abcd/manifest.json", // another workload
	)

	got, err := List(bg, mem, "demo")
	if err != nil {
		t.Fatal(err)
	}

	var keys []string
	for _, rp := range got {
		keys = append(keys, rp.Key())
	}
	want := []string{
		"demo/step-100-20261119T090000Z-0000abcd",
		"demo/step-100-20261119T000000Z-0000abcd",
		"demo/step-10-20261119T000000Z-0000abcd",
		"demo/step-9-20261119T000000Z-0000abcd",
	}
	if !slices.Equal(keys, want) {
		t.Fatalf("List = %v, want %v: numeric order, the latest attempt first, and only real restore point folders", keys, want)
	}
}

func TestNewestCompleteFindsTheNewestFolderWithAManifest(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	putAll(t, mem, "demo/step-8-20261119T000000Z-0000abcd/app/app.pt") // cut off before the manifest

	got, found, err := NewestComplete(bg, mem, "demo")

	if err != nil || !found || got != at("demo", 3) {
		t.Fatalf("NewestComplete = %+v, %v, %v; want step 3", got, found, err)
	}
	if _, found, err := NewestComplete(bg, mem, "demo2"); err != nil || found {
		t.Fatalf("NewestComplete for a workload with nothing stored = %v, %v; want not found", found, err)
	}
}

func TestFetchLatestPicksTheNewestRestorePoint(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	publish(t, mem, "demo", 5)
	base := t.TempDir()

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(base))
	if err != nil {
		t.Fatal(err)
	}

	if got.Point != at("demo", 5) || got.Manifest.Step != 5 || got.Root != layout.LocalDir(base, "demo", 5) {
		t.Fatalf("FetchLatest = %+v, manifest step %d, root %q; want step 5 in %q", got.Point, got.Manifest.Step, got.Root, layout.LocalDir(base, "demo", 5))
	}
	if len(got.Skipped) != 0 {
		t.Fatalf("skipped %v, want nothing", got.Skipped)
	}
}

func TestFetchLatestSkipsANewestFolderThatHasNoManifest(t *testing.T) {
	// An upload that was cut off: files, but no manifest.
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	putAll(t, mem, "demo/step-8-20261119T000000Z-0000abcd/app/app.pt")

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}

	if got.Point.Step != 3 {
		t.Fatalf("restored step %d, want 3: step 8 has no manifest", got.Point.Step)
	}
	if len(got.Skipped) != 1 || got.Skipped[0].Point.Step != 8 || !errors.Is(got.Skipped[0].Reason, ErrBadRestorePoint) {
		t.Fatalf("skipped = %+v, want step 8 as a bad restore point", got.Skipped)
	}
}

func TestFetchLatestSkipsACorruptNewestRestorePoint(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	publish(t, mem, "demo", 5)
	data, _ := mem.Data("demo/step-5-20261119T000000Z-0000abcd/app/app.pt")
	data[0] ^= 0xff
	mem.Set("demo/step-5-20261119T000000Z-0000abcd/app/app.pt", data)

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}

	if got.Point.Step != 3 || len(got.Skipped) != 1 || got.Skipped[0].Point.Step != 5 {
		t.Fatalf("FetchLatest = step %d, skipped %+v; want step 3 after skipping the corrupt step 5", got.Point.Step, got.Skipped)
	}
}

func TestFetchLatestSkipsARestorePointWhoseListedFileIsMissing(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	publish(t, mem, "demo", 5)
	mem.Delete("demo/step-5-20261119T000000Z-0000abcd/app/app.pt")

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}

	if got.Point.Step != 3 || !errors.Is(got.Skipped[0].Reason, ErrBadRestorePoint) {
		t.Fatalf("FetchLatest = step %d, skipped %+v; want step 3, with step 5 a bad restore point", got.Point.Step, got.Skipped)
	}
}

func TestFetchLatestWithNothingToRestore(t *testing.T) {
	_, err := FetchLatest(bg, NewMemory(), "demo", rootsUnder(t.TempDir()))

	if !errors.Is(err, ErrNoRestorePoint) {
		t.Fatalf("error = %v, want ErrNoRestorePoint", err)
	}
}

func TestFetchLatestWhenEveryRestorePointIsBad(t *testing.T) {
	mem := NewMemory()
	putAll(t, mem, "demo/step-8-20261119T000000Z-0000abcd/app/app.pt")

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))

	if !errors.Is(err, ErrNoRestorePoint) || len(got.Skipped) != 1 {
		t.Fatalf("error = %v, skipped %+v; want ErrNoRestorePoint after skipping step 8", err, got.Skipped)
	}
}

func TestFetchLatestStopsOnAnErrorThatIsNotAboutTheRestorePoint(t *testing.T) {
	// A permission error says nothing about step 5. Trying step 3 would hide the real problem.
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	publish(t, mem, "demo", 5)
	var calls atomic.Int32
	mem.GetHook = func(key string) error {
		calls.Add(1)
		return &smithy.GenericAPIError{Code: "AccessDenied", Message: "not allowed"}
	}

	_, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))

	if err == nil || errors.Is(err, ErrNoRestorePoint) || !Permanent(err) {
		t.Fatalf("error = %v, want the AccessDenied error itself", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("Get was called %d times, want 1: it must not go on to an older restore point", calls.Load())
	}
}

func TestFetchLatestIgnoresOtherWorkloads(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	publish(t, mem, "demo2", 99)

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}

	if got.Point.Step != 3 {
		t.Fatalf("restored step %d, want 3: demo2 is another workload", got.Point.Step)
	}
}

func TestFetchErrorsAboutTheRestorePointSaySo(t *testing.T) {
	// A problem with the restore point itself is ErrBadRestorePoint. A problem reaching the store is not.
	missingManifest := NewMemory()
	putAll(t, missingManifest, "demo/step-5-20261119T000000Z-0000abcd/app/app.pt")

	corrupt := NewMemory()
	publish(t, corrupt, "demo", 5)
	data, _ := corrupt.Data("demo/step-5-20261119T000000Z-0000abcd/app/app.pt")
	data[0] ^= 0xff
	corrupt.Set("demo/step-5-20261119T000000Z-0000abcd/app/app.pt", data)

	wrongStep := NewMemory()
	publish(t, wrongStep, "demo", 5)
	manifestBytes, _ := wrongStep.Data("demo/step-5-20261119T000000Z-0000abcd/manifest.json")
	wrongStep.Set("demo/step-6-20261119T000000Z-0000abcd/manifest.json", manifestBytes)

	for name, c := range map[string]struct {
		mem  *Memory
		step int
	}{
		"no manifest":             {missingManifest, 5},
		"a corrupt file":          {corrupt, 5},
		"another step's manifest": {wrongStep, 6},
	} {
		if _, err := Fetch(bg, c.mem, at("demo", c.step), t.TempDir()); !errors.Is(err, ErrBadRestorePoint) {
			t.Errorf("%s: error = %v, want ErrBadRestorePoint", name, err)
		}
	}

	unreachable := NewMemory()
	publish(t, unreachable, "demo", 5)
	unreachable.GetHook = func(key string) error { return errors.New("connection reset") }
	if _, err := Fetch(bg, unreachable, at("demo", 5), t.TempDir()); err == nil || errors.Is(err, ErrBadRestorePoint) {
		t.Errorf("a network error: error = %v, want a failure that is not ErrBadRestorePoint", err)
	}
}
