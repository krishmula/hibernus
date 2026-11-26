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
// why not, and hib warns that restore points hold only the application checkpoint.
func processSnapshotter(output string) (supervisor.Snapshotter, string) {
	if err := checkCRIU(); err != nil {
		fmt.Fprintf(os.Stderr, "hib: no process snapshot, so restore points hold only the application checkpoint: %v\n", err)
		return nil, err.Error()
	}
	return criu{output: output}, ""
}

// checkCRIU requires root and CRIU 3.17 or later.
func checkCRIU() error {
	if os.Geteuid() != 0 {
		return errors.New("CRIU needs root, and hib runs as a regular user")
	}
	out, err := exec.Command("criu", "--version").Output()
	if err != nil {
		return fmt.Errorf("run criu --version: %v", err)
	}
	var major, minor int
	if _, err := fmt.Sscanf(string(out), "Version: %d.%d", &major, &minor); err != nil {
		return fmt.Errorf("read the CRIU version from %q", strings.TrimSpace(string(out)))
	}
	if major < 3 || major == 3 && minor < 17 {
		return fmt.Errorf("CRIU %d.%d is older than 3.17, which can't restore glibc's rseq", major, minor)
	}
	return nil
}

// criu makes the process snapshot with the criu command.
type criu struct {
	output string // the training process's output file
}

// Dump dumps the training process. CRIU freezes it for the dump and ends it afterwards. hib sends
// no SIGSTOP first: CRIU's CUDA plugin has to talk to the running process to pause CUDA, and the
// training hook wakes on SIGCONT whether or not the process was stopped.
func (c criu) Dump(pid int, root string) ([]string, error) {
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
