//go:build linux

package snapshot

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeCriu writes a shell script that stands in for the criu binary.
func fakeCriu(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "criu")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestDumpRunsCriuAndSucceedsWhenCriuDoes(t *testing.T) {
	record := filepath.Join(t.TempDir(), "args")
	criu := &Criu{Bin: fakeCriu(t, `echo "$@" > `+record)}
	dir := filepath.Join(t.TempDir(), "process")

	if err := criu.Dump(4321, dir); err != nil {
		t.Fatal(err)
	}

	args, _ := os.ReadFile(record)
	if !strings.Contains(string(args), "dump") || !strings.Contains(string(args), "4321") {
		t.Fatalf("criu ran with %q, want a dump of pid 4321", args)
	}
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		t.Fatalf("Dump left no image folder at %s: %v", dir, err)
	}
}

func TestDumpFailureCarriesTheLastLinesOfCriusLog(t *testing.T) {
	// CRIU writes its reason to the log file named by -o, which is the argument after "-o".
	criu := &Criu{Bin: fakeCriu(t, `
while [ $# -gt 0 ]; do [ "$1" = "-o" ] && log="$2"; shift; done
printf 'line one\nError (criu/tty.c:1453): tty: No task found with sid 285\n' > "$log"
exit 1`)}

	err := criu.Dump(4321, filepath.Join(t.TempDir(), "process"))

	if err == nil {
		t.Fatal("a failing criu gave no error")
	}
	if !strings.Contains(err.Error(), "No task found with sid 285") {
		t.Fatalf("error %q lacks the reason from CRIU's log", err)
	}
}

func TestDumpOfAMissingBinaryTellsHowToInstallIt(t *testing.T) {
	criu := &Criu{Bin: filepath.Join(t.TempDir(), "no-such-criu")}

	err := criu.Dump(1, filepath.Join(t.TempDir(), "process"))

	if err == nil || !strings.Contains(err.Error(), "criu") {
		t.Fatalf("error = %v, want one that names criu", err)
	}
}

func TestRestoreCommandIsNotStartedAndNamesItsPIDFile(t *testing.T) {
	criu := &Criu{Bin: fakeCriu(t, `true`)}
	root := t.TempDir()

	cmd, pidFile := criu.RestoreCommand(filepath.Join(root, "process"))

	if cmd.Process != nil {
		t.Fatal("RestoreCommand already started criu")
	}
	if filepath.Dir(pidFile) != root {
		t.Fatalf("pid file %s is not in the restore point %s", pidFile, root)
	}
	if !strings.Contains(strings.Join(cmd.Args, " "), pidFile) {
		t.Fatalf("criu args %v do not name the pid file %s", cmd.Args, pidFile)
	}
}
