package store

import (
	"errors"
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

func TestStepsListsRestorePointFoldersNewestFirst(t *testing.T) {
	mem := NewMemory()
	putAll(t, mem,
		"demo/step-9/manifest.json", "demo/step-10/manifest.json", "demo/step-100/manifest.json",
		"demo/notes/readme.txt",        // not a step folder
		"demo/step-007/manifest.json",  // not the spelling that is written
		"demo2/step-500/manifest.json", // another workload
	)

	got, err := Steps(bg, mem, "demo")
	if err != nil {
		t.Fatal(err)
	}

	if want := []int{100, 10, 9}; !equalInts(got, want) {
		t.Fatalf("Steps = %v, want %v: numeric order, newest first, and only real step folders", got, want)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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

	if got.Step != 5 || got.Manifest.Step != 5 || got.Root != layout.LocalDir(base, "demo", 5) {
		t.Fatalf("FetchLatest = step %d, manifest step %d, root %q; want step 5 in %q", got.Step, got.Manifest.Step, got.Root, layout.LocalDir(base, "demo", 5))
	}
	if len(got.Skipped) != 0 {
		t.Fatalf("skipped %v, want nothing", got.Skipped)
	}
}

func TestFetchLatestSkipsANewestFolderThatHasNoManifest(t *testing.T) {
	// An upload that was cut off: files, but no manifest.
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	putAll(t, mem, "demo/step-8/app/app.pt")

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}

	if got.Step != 3 {
		t.Fatalf("restored step %d, want 3: step 8 has no manifest", got.Step)
	}
	if len(got.Skipped) != 1 || got.Skipped[0].Step != 8 || !errors.Is(got.Skipped[0].Reason, ErrBadRestorePoint) {
		t.Fatalf("skipped = %+v, want step 8 as a bad restore point", got.Skipped)
	}
}

func TestFetchLatestSkipsACorruptNewestRestorePoint(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	publish(t, mem, "demo", 5)
	data, _ := mem.Data("demo/step-5/process/pages-1.img")
	data[0] ^= 0xff
	mem.Set("demo/step-5/process/pages-1.img", data)

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}

	if got.Step != 3 || len(got.Skipped) != 1 || got.Skipped[0].Step != 5 {
		t.Fatalf("FetchLatest = step %d, skipped %+v; want step 3 after skipping the corrupt step 5", got.Step, got.Skipped)
	}
}

func TestFetchLatestSkipsARestorePointWhoseListedFileIsMissing(t *testing.T) {
	mem := NewMemory()
	publish(t, mem, "demo", 3)
	publish(t, mem, "demo", 5)
	mem.Delete("demo/step-5/app/app.pt")

	got, err := FetchLatest(bg, mem, "demo", rootsUnder(t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}

	if got.Step != 3 || !errors.Is(got.Skipped[0].Reason, ErrBadRestorePoint) {
		t.Fatalf("FetchLatest = step %d, skipped %+v; want step 3, with step 5 a bad restore point", got.Step, got.Skipped)
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
	putAll(t, mem, "demo/step-8/app/app.pt")

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

	if got.Step != 3 {
		t.Fatalf("restored step %d, want 3: demo2 is another workload", got.Step)
	}
}

func TestFetchErrorsAboutTheRestorePointSaySo(t *testing.T) {
	// A problem with the restore point itself is ErrBadRestorePoint. A problem reaching the store is not.
	missingManifest := NewMemory()
	putAll(t, missingManifest, "demo/step-5/app/app.pt")

	corrupt := NewMemory()
	publish(t, corrupt, "demo", 5)
	data, _ := corrupt.Data("demo/step-5/process/pages-1.img")
	data[0] ^= 0xff
	corrupt.Set("demo/step-5/process/pages-1.img", data)

	wrongStep := NewMemory()
	publish(t, wrongStep, "demo", 5)
	manifestBytes, _ := wrongStep.Data("demo/step-5/manifest.json")
	wrongStep.Set("demo/step-6/manifest.json", manifestBytes)

	for name, c := range map[string]struct {
		mem  *Memory
		step int
	}{
		"no manifest":             {missingManifest, 5},
		"a corrupt file":          {corrupt, 5},
		"another step's manifest": {wrongStep, 6},
	} {
		if _, err := Fetch(bg, c.mem, "demo", c.step, t.TempDir()); !errors.Is(err, ErrBadRestorePoint) {
			t.Errorf("%s: error = %v, want ErrBadRestorePoint", name, err)
		}
	}

	unreachable := NewMemory()
	publish(t, unreachable, "demo", 5)
	unreachable.GetHook = func(key string) error { return errors.New("connection reset") }
	if _, err := Fetch(bg, unreachable, "demo", 5, t.TempDir()); err == nil || errors.Is(err, ErrBadRestorePoint) {
		t.Errorf("a network error: error = %v, want a failure that is not ErrBadRestorePoint", err)
	}
}
