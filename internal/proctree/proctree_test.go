package proctree

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeProcess is one entry of a fake /proc folder.
type fakeProcess struct {
	pid, ppid, pgid, sid int
	comm                 string
	state                string
	threads              int
	rssPages             int
	cmdline              []string
	fds                  int
}

// writeProc builds a fake /proc folder, laid out as Linux does.
func writeProc(t *testing.T, procs ...fakeProcess) string {
	t.Helper()
	root := t.TempDir()
	for _, p := range procs {
		dir := filepath.Join(root, fmt.Sprint(p.pid))
		if err := os.MkdirAll(filepath.Join(dir, "fd"), 0o755); err != nil {
			t.Fatal(err)
		}
		// Fields after the command name, as in proc(5): state ppid pgrp session tty tpgid flags
		// minflt cminflt majflt cmajflt utime stime cutime cstime priority nice num_threads
		// itrealvalue starttime vsize rss
		stat := fmt.Sprintf("%d (%s) %s %d %d %d 0 0 0 0 0 0 0 0 0 0 0 0 0 %d 0 0 0 %d\n",
			p.pid, p.comm, p.state, p.ppid, p.pgid, p.sid, p.threads, p.rssPages)
		write(t, filepath.Join(dir, "stat"), stat)
		write(t, filepath.Join(dir, "cmdline"), strings.Join(p.cmdline, "\x00")+"\x00")
		for i := range p.fds {
			write(t, filepath.Join(dir, "fd", fmt.Sprint(i)), "")
		}
	}
	return root
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// the process tree of a training run: hib, then python under it, then a worker under python.
func trainingRun() []fakeProcess {
	return []fakeProcess{
		{pid: 100, ppid: 1, pgid: 100, sid: 50, comm: "hib", state: "S", threads: 6, rssPages: 1000, cmdline: []string{"hib", "run", "train.py"}, fds: 5},
		{pid: 200, ppid: 100, pgid: 200, sid: 200, comm: "python3", state: "S", threads: 4, rssPages: 25000, cmdline: []string{"python3", "train.py", "--updates", "20"}, fds: 9},
		{pid: 201, ppid: 200, pgid: 200, sid: 200, comm: "pt_data_worker", state: "R", threads: 1, rssPages: 4000, cmdline: []string{"python3", "train.py"}, fds: 3},
		{pid: 300, ppid: 1, pgid: 300, sid: 300, comm: "bash", state: "S", threads: 1, rssPages: 500, cmdline: []string{"bash"}, fds: 3},
	}
}

func TestSnapshotFollowsChildrenAndLeavesOutOtherProcesses(t *testing.T) {
	proc := writeProc(t, trainingRun()...)

	root, err := Snapshot(proc, 100)
	if err != nil {
		t.Fatal(err)
	}

	if root.PID != 100 || len(root.Children) != 1 {
		t.Fatalf("root = pid %d with %d children, want pid 100 with 1", root.PID, len(root.Children))
	}
	python := root.Children[0]
	if python.PID != 200 || python.PPID != 100 || python.SID != 200 {
		t.Fatalf("python = %+v, want pid 200 under 100, session 200", python)
	}
	if len(python.Children) != 1 || python.Children[0].PID != 201 {
		t.Fatalf("python's children = %+v, want the worker 201", python.Children)
	}
	for _, entry := range root.Flatten() {
		if entry.PID == 300 {
			t.Fatal("the tree holds pid 300, an unrelated process")
		}
	}
}

func TestSnapshotReadsEveryField(t *testing.T) {
	proc := writeProc(t, trainingRun()...)

	root, err := Snapshot(proc, 200)
	if err != nil {
		t.Fatal(err)
	}

	if root.State != "S" || root.Comm != "python3" || root.Threads != 4 || root.FDs != 9 {
		t.Fatalf("node = %+v, want state S, comm python3, 4 threads, 9 fds", root)
	}
	if got := strings.Join(root.Cmdline, " "); got != "python3 train.py --updates 20" {
		t.Fatalf("cmdline = %q", got)
	}
	if root.RSSKB != 25000*4 {
		t.Fatalf("rss = %d KB, want %d (25000 pages of 4 KB)", root.RSSKB, 25000*4)
	}
}

func TestSnapshotHandlesACommandNameWithSpacesAndParentheses(t *testing.T) {
	proc := writeProc(t, fakeProcess{pid: 7, ppid: 1, pgid: 7, sid: 7, comm: "my (odd) name", state: "S", threads: 2, rssPages: 10, cmdline: []string{"x"}})

	root, err := Snapshot(proc, 7)
	if err != nil {
		t.Fatal(err)
	}
	if root.Comm != "my (odd) name" || root.PPID != 1 || root.Threads != 2 {
		t.Fatalf("node = %+v, want comm %q and the fields after it read correctly", root, "my (odd) name")
	}
}

func TestSnapshotOfAMissingProcessFails(t *testing.T) {
	proc := writeProc(t, trainingRun()...)

	if _, err := Snapshot(proc, 999); err == nil {
		t.Fatal("a snapshot of a process that does not exist succeeded")
	}
}

func TestSnapshotSkipsAProcessThatVanishesMidWalk(t *testing.T) {
	proc := writeProc(t, trainingRun()...)
	// 250 has a folder but no stat file, as when a process exits while we read /proc.
	if err := os.MkdirAll(filepath.Join(proc, "250"), 0o755); err != nil {
		t.Fatal(err)
	}

	root, err := Snapshot(proc, 100)
	if err != nil {
		t.Fatalf("a vanished process broke the walk: %v", err)
	}
	if len(root.Flatten()) != 3 {
		t.Fatalf("tree has %d processes, want 3", len(root.Flatten()))
	}
}

func TestRenderIndentsChildrenAndShowsTheFieldsThatIdentifyAProcess(t *testing.T) {
	proc := writeProc(t, trainingRun()...)
	root, _ := Snapshot(proc, 100)

	text := root.Render()

	lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("render has %d lines, want 3:\n%s", len(lines), text)
	}
	if strings.HasPrefix(lines[0], " ") || !strings.HasPrefix(lines[1], "  ") || !strings.HasPrefix(lines[2], "    ") {
		t.Fatalf("lines are not indented by depth:\n%s", text)
	}
	for _, want := range []string{"pid=200", "ppid=100", "sid=200", "state=S", "threads=4", "fds=9", "python3 train.py --updates 20"} {
		if !strings.Contains(lines[1], want) {
			t.Errorf("python's line %q lacks %q", lines[1], want)
		}
	}
}

func TestShapeIgnoresWhatChangesAcrossARestoreAndKeepsWhatShouldNot(t *testing.T) {
	// Before: python 200 under hib 100. After a restore: the same python, now under criu 150,
	// itself under a new hib 900. PIDs of the training process and its worker must match.
	before := writeProc(t, trainingRun()...)
	after := writeProc(t,
		fakeProcess{pid: 900, ppid: 1, pgid: 900, sid: 50, comm: "hib", state: "S", threads: 6, rssPages: 1000, cmdline: []string{"hib", "restore", "x"}, fds: 5},
		fakeProcess{pid: 150, ppid: 900, pgid: 150, sid: 150, comm: "criu", state: "S", threads: 1, rssPages: 500, cmdline: []string{"criu", "restore"}, fds: 4},
		fakeProcess{pid: 200, ppid: 150, pgid: 200, sid: 200, comm: "python3", state: "S", threads: 4, rssPages: 25000, cmdline: []string{"python3", "train.py", "--updates", "20"}, fds: 9},
		fakeProcess{pid: 201, ppid: 200, pgid: 200, sid: 200, comm: "pt_data_worker", state: "R", threads: 1, rssPages: 4000, cmdline: []string{"python3", "train.py"}, fds: 3},
	)
	b, _ := Snapshot(before, 200)
	a, _ := Snapshot(after, 200)

	if b.Shape() != a.Shape() {
		t.Fatalf("the training tree changed shape across the restore:\nbefore:\n%s\nafter:\n%s", b.Shape(), a.Shape())
	}
	if !strings.Contains(b.Shape(), "200 python3") || !strings.Contains(b.Shape(), "201 pt_data_worker") {
		t.Fatalf("shape lacks pids and command names:\n%s", b.Shape())
	}
}

func TestAppendAddsLabelledSectionsToTheLog(t *testing.T) {
	proc := writeProc(t, trainingRun()...)
	root, _ := Snapshot(proc, 100)
	log := filepath.Join(t.TempDir(), "logs", "w.log") // the folder does not exist yet

	if err := Append(log, "before-checkpoint step=5", root, nil); err != nil {
		t.Fatal(err)
	}
	if err := Append(log, "after-dump", nil, fmt.Errorf("process 200 is gone")); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	first := strings.Index(text, "before-checkpoint step=5")
	second := strings.Index(text, "after-dump")
	if first < 0 || second < first {
		t.Fatalf("log lacks the two sections in order:\n%s", text)
	}
	if !strings.Contains(text, "pid=200") || !strings.Contains(text, "process 200 is gone") {
		t.Fatalf("log lacks the tree or the error:\n%s", text)
	}
}

func TestAppendWithAnEmptyPathDoesNothing(t *testing.T) {
	if err := Append("", "label", nil, nil); err != nil {
		t.Fatalf("logging to no file failed: %v", err)
	}
}
