package snapshot

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestDumpArgsDumpTheTreeIntoTheImageFolderAndKillIt(t *testing.T) {
	dir := filepath.Join("store", "w", "step-5", "process")

	args := DumpArgs(1234, dir)

	for _, want := range [][]string{{"dump"}, {"-t", "1234"}, {"-D", dir}} {
		if !containsRun(args, want) {
			t.Errorf("dump args %v lack %v", args, want)
		}
	}
	// The checkpoint host is going away, so the process must not keep running after the dump.
	if slices.Contains(args, "--leave-running") {
		t.Errorf("dump args %v keep the process running", args)
	}
	// The training process starts in its own session, so it has no terminal for CRIU to record.
	if slices.Contains(args, "--shell-job") {
		t.Errorf("dump args %v ask for shell-job handling", args)
	}
	if !slices.Contains(args, "-o") {
		t.Errorf("dump args %v write no CRIU log", args)
	}
}

func TestDumpLogLivesBesideTheImagesNotInsideThem(t *testing.T) {
	dir := filepath.Join("store", "w", "step-5", "process")

	log := DumpLog(dir)

	// Images are checksummed into the manifest. A log written during the dump must not be one of them.
	if filepath.Dir(log) != filepath.Dir(dir) {
		t.Fatalf("dump log %s is not beside the image folder %s", log, dir)
	}
}

func TestRestoreArgsRestoreInTheForegroundAndReportThePID(t *testing.T) {
	dir := filepath.Join("store", "w", "step-5", "process")
	pidFile := filepath.Join("store", "w", "step-5", "restored.pid")

	args := RestoreArgs(dir, pidFile)

	for _, want := range [][]string{{"restore"}, {"-D", dir}, {"--pidfile", pidFile}} {
		if !containsRun(args, want) {
			t.Errorf("restore args %v lack %v", args, want)
		}
	}
	// In the foreground, CRIU stays as the restored process's parent and passes its exit code on.
	if slices.Contains(args, "--restore-detached") {
		t.Errorf("restore args %v detach the restored process from its parent", args)
	}
}

func TestImageFilesListsEveryImageRelativeToTheRestorePoint(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "process", "core-1.img"))
	mustWrite(t, filepath.Join(root, "process", "pages-1.img"))
	mustWrite(t, filepath.Join(root, "criu-dump.log")) // beside the images, not one of them
	mustWrite(t, filepath.Join(root, "app", "app.pt"))

	files, err := ImageFiles(root)
	if err != nil {
		t.Fatal(err)
	}

	want := []string{filepath.Join("process", "core-1.img"), filepath.Join("process", "pages-1.img")}
	if !slices.Equal(files, want) {
		t.Fatalf("image files = %v, want %v", files, want)
	}
}

func TestImageFilesOfAFolderWithNoImagesFails(t *testing.T) {
	if _, err := ImageFiles(t.TempDir()); err == nil {
		t.Fatal("a restore point with no process folder listed images")
	}
}

func mustWrite(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// containsRun reports whether want appears in args as consecutive items.
func containsRun(args, want []string) bool {
	for i := 0; i+len(want) <= len(args); i++ {
		if strings.Join(args[i:i+len(want)], "\x00") == strings.Join(want, "\x00") {
			return true
		}
	}
	return false
}
