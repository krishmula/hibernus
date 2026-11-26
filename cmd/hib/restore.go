package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const restoreUsage = "usage: hib restore [--store <folder>] <folder | s3://<bucket>/<workload>/step-<K>-<attempt>/ | s3://<bucket>/<workload>/>\n"

func restoreCommand(args []string) int {
	flags := flag.NewFlagSet("restore", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, restoreUsage) }
	store := storeFlag(flags)
	if code, ok := parseFlags(flags, args); !ok {
		return code
	}
	if flags.NArg() != 1 {
		fmt.Fprint(os.Stderr, "hib restore: give exactly one restore point location\n"+restoreUsage)
		return exitBadCommand
	}

	storeDir, err := filepath.Abs(*store)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib restore: %v\n", err)
		return exitFailed
	}
	// An s3:// location is downloaded and verified into the store folder first. After that, a
	// restore from S3 is a restore from a local folder.
	root, remote, err := locateRestorePoint(context.Background(), flags.Arg(0), storeDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib restore: %v\n", err)
		return exitFailed
	}
	m, err := manifest.Load(root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib restore: %v\n", err)
		return exitFailed
	}
	resume, err := resumeFromApplicationCheckpoint(root, m)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib restore: %v\n", err)
		return exitFailed
	}
	if err := checkFingerprint(context.Background(), m); err != nil {
		fmt.Fprintf(os.Stderr, "hib restore: %v\n", err)
		return exitFailed
	}

	cfg := supervisor.Config{Workload: m.Workload, Store: storeDir, Command: m.Command, Remote: remote, Output: protocol.OutputPath(m.Workload)}
	cfg.Snapshot, cfg.NoSnapshot = processSnapshotter(cfg.Output)
	// The process snapshot first. It brings back the training process exactly as it was.
	code, err := superviseSnapshot(cfg, resume, root, m)
	if err == nil || errors.Is(err, errCanceled) {
		return code
	}
	fmt.Fprintf(os.Stderr, "hib restore: using the application checkpoint, because %v\n", err)
	return supervise(cfg, resume)
}

// resumeFromApplicationCheckpoint checks that the application checkpoint can bring the workload back
// in a fresh training process, and returns what the training hook loads. hib restore needs it even
// when the process snapshot works, as the fallback.
func resumeFromApplicationCheckpoint(root string, m manifest.Manifest) (*protocol.Resume, error) {
	if !validWorkloadName.MatchString(m.Workload) {
		return nil, fmt.Errorf("the manifest names workload %q, which is not a valid name", m.Workload)
	}
	if len(m.Command.Argv) == 0 {
		return nil, errors.New("the manifest has no command to start the training process")
	}
	app := m.Layers.ApplicationCheckpoint
	if app.Status != manifest.StatusOK {
		reason := "no reason recorded"
		if app.Error != nil {
			reason = *app.Error
		}
		return nil, fmt.Errorf("the application checkpoint is %s: %s", app.Status, reason)
	}
	if err := app.Verify(root); err != nil {
		return nil, fmt.Errorf("application checkpoint: %w", err)
	}

	files := make([]string, len(app.Files))
	for i, file := range app.Files {
		files[i] = filepath.Join(root, file.Path)
	}
	return &protocol.Resume{Step: m.Step, Files: files}, nil
}

// probeTimeout bounds the fingerprint module, which loads PyTorch and can take a while on a cold disk.
const probeTimeout = 2 * time.Minute

// checkFingerprint runs the fingerprint module with the interpreter the manifest records, the same
// module the training hook ran on the checkpoint host, and refuses this host if any value differs.
// It runs before the training process starts, so a refusal leaves nothing to clean up.
func checkFingerprint(ctx context.Context, m manifest.Manifest) error {
	if err := m.Fingerprint.Complete(); err != nil {
		return fmt.Errorf("the manifest: %w", err)
	}
	if m.Python == "" {
		return errors.New("the manifest names no python to collect this host's fingerprint with")
	}
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	probe := exec.CommandContext(ctx, m.Python, "-m", "hibernus.fingerprint")
	probe.Dir = m.Command.Cwd
	var stderr strings.Builder
	probe.Stderr = &stderr
	out, err := probe.Output()
	if err != nil {
		return fmt.Errorf("collect this host's fingerprint with %s: %v: %s", m.Python, err, strings.TrimSpace(stderr.String()))
	}
	var here manifest.Fingerprint
	if err := json.Unmarshal(out, &here); err != nil {
		return fmt.Errorf("read this host's fingerprint from %s: %v", m.Python, err)
	}
	if err := here.Complete(); err != nil {
		return fmt.Errorf("this host: %w", err)
	}
	if diffs := manifest.Differences(m.Fingerprint, here); len(diffs) > 0 {
		return fmt.Errorf("this host's fingerprint differs from the checkpoint host's, so the restore would not match:\n  %s", strings.Join(diffs, "\n  "))
	}
	return nil
}
