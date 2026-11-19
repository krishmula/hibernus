// Package snapshot takes and restores the process snapshot layer of a restore point with CRIU.
//
// The Supervisor works against the Snapshotter interface, so its tests use a fake and need no CRIU.
package snapshot

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	// ImageDir is the folder inside a restore point that holds CRIU's image files.
	ImageDir = "process"
	// logLines is how many lines of CRIU's log an error message quotes.
	logLines = 6
)

// Snapshotter captures a process tree and brings it back.
type Snapshotter interface {
	// Dump freezes the tree under pid, writes its images into dir, and ends the tree.
	Dump(pid int, dir string) error
	// RestoreCommand returns the unstarted command that restores the images in dir, and the file
	// where it writes the restored root process's PID. The caller starts it and waits on it.
	RestoreCommand(dir string) (cmd *exec.Cmd, pidFile string)
	// Wake tells a restored process to carry on from the point where it was paused.
	Wake(pid int) error
}

// Criu runs the criu binary. Dump and restore need root, or CAP_CHECKPOINT_RESTORE and CAP_SYS_PTRACE.
type Criu struct {
	Bin string // the criu binary; empty means "criu" on PATH
}

func (c *Criu) bin() string {
	if c.Bin != "" {
		return c.Bin
	}
	return "criu"
}

// DumpArgs is the criu command line that dumps the tree under pid into dir. CRIU ends the tree
// after the dump, which is what we want, because the checkpoint host is going away. The training
// process starts in its own session, so there is no terminal to record and no --shell-job.
func DumpArgs(pid int, dir string) []string {
	return []string{"dump", "-t", fmt.Sprint(pid), "-D", dir, "-o", DumpLog(dir), "-v4"}
}

// RestoreArgs is the criu command line that restores the images in dir. Without --restore-detached
// CRIU stays as the restored process's parent and exits with its exit code.
func RestoreArgs(dir, pidFile string) []string {
	return []string{"restore", "-D", dir, "-o", RestoreLog(dir), "-v4", "--pidfile", pidFile}
}

// DumpLog and RestoreLog sit beside the image folder, not inside it, because the manifest
// checksums every file in the image folder, and CRIU is still writing its log during the dump.
func DumpLog(dir string) string    { return filepath.Join(filepath.Dir(dir), "criu-dump.log") }
func RestoreLog(dir string) string { return filepath.Join(filepath.Dir(dir), "criu-restore.log") }

func (c *Criu) Dump(pid int, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	out, err := exec.Command(c.bin(), DumpArgs(pid, dir)...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("criu dump of process %d failed (is criu installed, and does hib run as root?): %w%s",
			pid, err, reason(DumpLog(dir), out))
	}
	return nil
}

func (c *Criu) RestoreCommand(dir string) (*exec.Cmd, string) {
	pidFile := filepath.Join(filepath.Dir(dir), "restored.pid")
	return exec.Command(c.bin(), RestoreArgs(dir, pidFile)...), pidFile
}

// reason quotes the end of CRIU's log, where CRIU says why it failed, and falls back to its output.
func reason(logPath string, output []byte) string {
	text := ""
	if data, err := os.ReadFile(logPath); err == nil {
		text = string(data)
	}
	if strings.TrimSpace(text) == "" {
		text = string(output)
	}
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) > logLines {
		lines = lines[len(lines)-logLines:]
	}
	if len(lines) == 1 && lines[0] == "" {
		return ""
	}
	return "\n" + strings.Join(lines, "\n")
}

// ImageFiles lists the files in the restore point's image folder, as paths relative to root.
func ImageFiles(root string) ([]string, error) {
	var files []string
	imageRoot := filepath.Join(root, ImageDir)
	err := filepath.WalkDir(imageRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		files = append(files, relative)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, errors.New("the process folder holds no image files")
	}
	return files, nil
}
