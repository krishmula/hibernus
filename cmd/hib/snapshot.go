package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/snapshot"
	"github.com/krishmula/hibernus/internal/supervisor"
)

// restoreTimeout is how long a restore from the process snapshot gets before the application
// checkpoint takes over. A variable so tests can shorten it.
var restoreTimeout = 60 * time.Second

// criuGrace is how long to wait for criu's output after we kill it.
const criuGrace = time.Second

// logDir holds the process tree log and the training output of every workload.
func logDir() string {
	if dir := os.Getenv("HIB_LOG_DIR"); dir != "" {
		return dir
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".hib", "logs")
}

func treeLogPath(workload string) string { return filepath.Join(logDir(), workload+".tree.log") }

// outputPath is where a snapshotted workload's training process writes its output. CRIU can't
// restore a pipe to a process outside the dump, but it restores a regular file by its path.
func outputPath(workload string) string { return filepath.Join(logDir(), workload+".out") }

// restoreFromSnapshot brings the workload back from the process snapshot. When it returns ok, the
// workload ran to its end, or was checkpointed again, and code is what `hib restore` exits with.
// When it returns not ok, reason says why, and nothing of the attempt is left running, so the
// caller can fall back to the application checkpoint.
func restoreFromSnapshot(cfg supervisor.Config, resume *protocol.Resume, root string, m manifest.Manifest) (code int, ok bool, reason error) {
	layer := m.Layers.ProcessSnapshot
	if layer.Status != manifest.StatusOK {
		why := "no reason recorded"
		if layer.Error != nil {
			why = *layer.Error
		}
		return 0, false, fmt.Errorf("the process snapshot is %s: %s", layer.Status, why)
	}
	if err := layer.Verify(root); err != nil {
		return 0, false, fmt.Errorf("process snapshot: %w", err)
	}

	listener, err := supervisor.Listen(protocol.SocketPath(cfg.Workload))
	if err != nil {
		return 0, false, err
	}
	sup := supervisor.New(cfg, resume)
	served := make(chan struct{})
	go func() {
		sup.Serve(listener)
		close(served)
	}()
	shutdown := func() {
		listener.Close()
		select {
		case <-served:
		case <-time.After(2 * time.Second):
		}
	}

	sup.LogTree(fmt.Sprintf("before-restore step=%d", m.Step))

	cmd, pidFile := cfg.Snapshotter.RestoreCommand(filepath.Join(root, snapshot.ImageDir))
	os.Remove(pidFile) // a pid file left by an earlier restore would point at the wrong process
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	cmd.WaitDelay = criuGrace
	if err := cmd.Start(); err != nil {
		shutdown()
		return 0, false, fmt.Errorf("start criu: %w", err)
	}
	exited := make(chan struct{})
	var exitCode int
	go func() {
		exitCode = childExitCode(cmd.Wait())
		close(exited)
	}()

	// abort ends whatever the attempt started, so the fallback begins from a clean host.
	abort := func(why error) (int, bool, error) {
		cmd.Process.Kill()
		if pid := readPID(pidFile); pid > 0 {
			if process, err := os.FindProcess(pid); err == nil {
				process.Kill()
			}
		}
		<-exited
		shutdown()
		return 0, false, why
	}
	criuFailed := func(when string) error {
		return fmt.Errorf("criu exited with code %d %s%s", exitCode, when, tail(&output))
	}

	deadline := time.After(restoreTimeout)
	pid := 0
	for pid == 0 {
		select {
		case <-exited:
			shutdown()
			return 0, false, criuFailed("before it restored the process")
		case <-deadline:
			return abort(fmt.Errorf("criu did not restore the process within %s%s", restoreTimeout, tail(&output)))
		case <-time.After(20 * time.Millisecond):
			pid = readPID(pidFile)
		}
	}
	sup.SetChild(pid)
	if err := cfg.Snapshotter.Wake(pid); err != nil {
		return abort(err)
	}

	select {
	case <-sup.Resumed():
	case <-exited:
		shutdown()
		return 0, false, criuFailed("before the restored training hook reconnected")
	case <-deadline:
		return abort(fmt.Errorf("the restored training hook did not reconnect within %s", restoreTimeout))
	}

	// The restore worked. criu now waits for the restored process, and exits with its exit code.
	<-exited
	sup.ChildExited(exitCode)
	sup.Settle()
	shutdown()
	if sup.Checkpointed() {
		return exitOK, true, nil
	}
	return exitCode, true, nil
}

// readPID reads the pid that criu wrote. It returns 0 while the file is missing or incomplete.
func readPID(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 0 {
		return 0
	}
	return pid
}

// tail quotes the end of criu's output for an error message.
func tail(output *bytes.Buffer) string {
	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) > 6 {
		lines = lines[len(lines)-6:]
	}
	if len(lines) == 1 && lines[0] == "" {
		return ""
	}
	return "\n" + strings.Join(lines, "\n")
}
