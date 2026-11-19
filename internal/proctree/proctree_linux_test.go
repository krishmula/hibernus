//go:build linux

package proctree

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// The fake /proc tests prove the parser against proc(5). This one proves it against a real kernel.
func TestSnapshotReadsTheRealProcFolder(t *testing.T) {
	child := exec.Command("sleep", "30")
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { child.Process.Kill(); child.Wait() })

	root, err := Snapshot("/proc", os.Getpid())
	if err != nil {
		t.Fatal(err)
	}

	if root.PPID != os.Getppid() {
		t.Errorf("ppid = %d, want %d", root.PPID, os.Getppid())
	}
	if root.Threads < 1 || root.FDs < 3 || root.RSSKB <= 0 || root.State == "" {
		t.Errorf("root = %+v, want threads >= 1, fds >= 3, rss > 0, and a state", root)
	}
	if len(root.Cmdline) == 0 {
		t.Error("root has no command line")
	}
	var sleep *Node
	for _, e := range root.Flatten() {
		if e.PID == child.Process.Pid {
			sleep = e.Node
		}
	}
	if sleep == nil {
		t.Fatalf("the child %d is not in the tree:\n%s", child.Process.Pid, root.Render())
	}
	if sleep.Comm != "sleep" || sleep.PPID != os.Getpid() || !strings.Contains(strings.Join(sleep.Cmdline, " "), "30") {
		t.Errorf("child = %+v, want comm sleep, ppid %d, cmdline with 30", sleep, os.Getpid())
	}
}
