package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const (
	imagesDir  = "process"    // CRIU's images, inside the restore point folder
	outputCopy = "output.log" // the training process's output file, as it was at the dump
)

// restoredTimeout is how long a restored training process gets to send restored. After that, hib
// kills it and uses the application checkpoint. That is safe, because the training hook doesn't
// train until the Supervisor answers restored.
const restoredTimeout = 30 * time.Second

// superviseSnapshot restores the workload from the process snapshot and supervises it until it
// exits. It returns an error only when training never continued, and then nothing of the attempt
// is left running, so hib restore can use the application checkpoint instead.
func superviseSnapshot(cfg supervisor.Config, resume *protocol.Resume, root string, m manifest.Manifest) (int, error) {
	layer := m.Layers.ProcessSnapshot
	if layer.Status != manifest.StatusOK {
		reason := "no reason recorded"
		if layer.Error != nil {
			reason = *layer.Error
		}
		return 0, fmt.Errorf("the restore point has no process snapshot: %s", reason)
	}
	if cfg.Snapshot == nil {
		return 0, fmt.Errorf("this host can't restore a process snapshot: %s", cfg.NoSnapshot)
	}
	if err := layer.Verify(root); err != nil {
		return 0, fmt.Errorf("the process snapshot: %w", err)
	}
	if m.Output == "" {
		return 0, errors.New("the manifest records no output file for the process snapshot")
	}

	// CRIU reopens the output file by its path and offset, so it must be back at that path first.
	if err := copyFile(filepath.Join(root, outputCopy), m.Output); err != nil {
		return 0, fmt.Errorf("put the output file back: %w", err)
	}
	output, err := os.Open(m.Output)
	if err != nil {
		return 0, err
	}
	defer output.Close()
	if _, err := output.Seek(0, io.SeekEnd); err != nil { // copy only what the training process writes from now on
		return 0, err
	}

	se, err := openSession(cfg, resume)
	if err != nil {
		return 0, err
	}
	process, err := restoreProcess(root)
	if err != nil {
		se.close()
		return 0, err
	}
	exited := waitFor(process)
	if err := wake(process); err != nil {
		process.Kill()
		<-exited
		se.close()
		return 0, err
	}
	select {
	case <-se.sup.Restored():
	case code := <-exited:
		se.close()
		return 0, fmt.Errorf("the restored training process exited with code %d before it sent restored", code)
	case <-time.After(restoredTimeout):
		process.Kill()
		<-exited
		se.close()
		return 0, fmt.Errorf("the restored training process sent no restored within %s", restoredTimeout)
	}
	fmt.Fprintf(os.Stderr, "hib restore: restored the process snapshot at step %d\n", m.Step)
	return se.run(process, exited, output), nil
}

// copyFile copies src to dst, replacing dst, and makes dst's folder if it is missing.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o700); err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
