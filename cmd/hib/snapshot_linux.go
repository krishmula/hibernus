//go:build linux

package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/krishmula/hibernus/internal/supervisor"
)

// prSetChildSubreaper is PR_SET_CHILD_SUBREAPER from linux/prctl.h.
const prSetChildSubreaper = 36

// processSnapshotter returns CRIU when this host can make a process snapshot. Otherwise it says
// why not, and hib warns that restore points hold only the application checkpoint. On a host with
// an NVIDIA GPU, it also warns when a training process that uses the GPU can't go into the snapshot.
func processSnapshotter(output string) (supervisor.Snapshotter, string) {
	version, err := criuVersion()
	if err == nil {
		err = version.usable()
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib: no process snapshot, so restore points hold only the application checkpoint: %v\n", err)
		return nil, err.Error()
	}
	_, lookErr := exec.LookPath("cuda-checkpoint")
	gpu := version.gpuProblem(hostHasGPU(), lookErr == nil)
	if gpu != "" {
		fmt.Fprintf(os.Stderr, "hib: a training process that uses the GPU gets no process snapshot: %s\n", gpu)
	}
	return criu{output: output, gpuProblem: gpu}, ""
}

// criuVersion requires root, then asks criu for its version.
func criuVersion() (criuRelease, error) {
	if os.Geteuid() != 0 {
		return criuRelease{}, errors.New("CRIU needs root, and hib runs as a regular user")
	}
	out, err := exec.Command("criu", "--version").Output()
	if err != nil {
		return criuRelease{}, fmt.Errorf("run criu --version: %v", err)
	}
	return parseCRIUVersion(string(out))
}

// hostHasGPU reports whether the NVIDIA driver is loaded.
func hostHasGPU() bool {
	_, err := os.Stat("/proc/driver/nvidia/version")
	return err == nil
}

// criu makes the process snapshot with the criu command.
type criu struct {
	output     string // the training process's output file
	gpuProblem string // why a training process with the GPU open can't be dumped; empty when it can
}

// Dump dumps the training process. CRIU freezes it for the dump and ends it afterwards. hib sends
// no SIGSTOP first: CRIU's CUDA plugin has to talk to the running process to pause CUDA, and the
// training hook wakes on SIGCONT whether or not the process was stopped.
func (c criu) Dump(pid int, root string) ([]string, error) {
	if c.gpuProblem != "" && gpuOpen(pid) {
		// criu dump would fail or hang on the GPU, so don't try. The training process keeps
		// waiting in the training hook, and the restore point holds the application checkpoint.
		return nil, fmt.Errorf("the training process uses the GPU, and %s", c.gpuProblem)
	}
	images := filepath.Join(root, imagesDir)
	if err := os.RemoveAll(images); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(images, 0o700); err != nil {
		return nil, err
	}
	log := filepath.Join(root, "criu-dump.log") // outside images, so the manifest doesn't checksum it
	out, err := exec.Command("criu", "dump", "-t", strconv.Itoa(pid), "-D", images, "-o", log, "-v4").CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("criu dump: %v: %s", err, criuReason(log, out))
	}
	// The training process has ended, so its output file no longer changes.
	if err := copyFile(c.output, filepath.Join(root, outputCopy)); err != nil {
		return nil, fmt.Errorf("copy the output file: %w", err)
	}
	var files []string
	err = filepath.WalkDir(images, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		relative, err := filepath.Rel(root, path)
		files = append(files, relative)
		return err
	})
	return append(files, outputCopy), err
}

// gpuOpen reports whether the process has an NVIDIA device open.
func gpuOpen(pid int) bool {
	fds := fmt.Sprintf("/proc/%d/fd", pid)
	entries, err := os.ReadDir(fds)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if target, err := os.Readlink(filepath.Join(fds, entry.Name())); err == nil && strings.HasPrefix(target, "/dev/nvidia") {
			return true
		}
	}
	return false
}

// restoreProcess brings the training process back from the images in root, waiting for SIGCONT. hib
// becomes a subreaper first, so that when criu exits, the kernel hands the restored process to hib.
func restoreProcess(root string) (*os.Process, error) {
	if _, _, errno := syscall.RawSyscall(syscall.SYS_PRCTL, prSetChildSubreaper, 1, 0); errno != 0 {
		return nil, fmt.Errorf("become a subreaper: %v", errno)
	}
	pidFile := filepath.Join(root, "restored.pid")
	os.Remove(pidFile) // a pid file from an earlier restore would name the wrong process
	log := filepath.Join(root, "criu-restore.log")
	out, err := exec.Command("criu", "restore", "-D", filepath.Join(root, imagesDir), "--restore-detached",
		"--pidfile", pidFile, "-o", log, "-v4").CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("criu restore: %v: %s", err, criuReason(log, out))
	}
	data, err := os.ReadFile(pidFile)
	if err != nil {
		return nil, fmt.Errorf("read the restored process's PID: %w", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return nil, fmt.Errorf("read the restored process's PID from %q", data)
	}
	return os.FindProcess(pid)
}

// wake sends SIGCONT, which ends the training hook's wait.
func wake(process *os.Process) error {
	if err := process.Signal(syscall.SIGCONT); err != nil {
		return fmt.Errorf("wake the restored training process: %w", err)
	}
	return nil
}

// criuReason quotes CRIU's error lines from its log, or its output when the log has none.
func criuReason(log string, out []byte) string {
	var errs []string
	if data, err := os.ReadFile(log); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.Contains(line, "Error") {
				errs = append(errs, strings.TrimSpace(line))
			}
		}
	}
	if len(errs) == 0 {
		return strings.TrimSpace(string(out))
	}
	if len(errs) > 3 {
		errs = errs[:3] // the first errors name the cause; later ones follow from it
	}
	return strings.Join(errs, "; ")
}
