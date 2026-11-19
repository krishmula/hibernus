package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/snapshot"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const restoreUsage = "usage: hib restore [--store <folder>] [--no-snapshot] <folder | s3://<bucket>/<workload>/step-<K>/ | s3://<bucket>/<workload>/>\n"

func restoreCommand(args []string) int {
	flags := flag.NewFlagSet("restore", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, restoreUsage) }
	store := storeFlag(flags)
	noSnapshot := flags.Bool("no-snapshot", false, "restore from the application checkpoint only")
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return exitOK
		}
		return exitBadCommand
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
	resume, err := fallback(root, m)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib restore: %v\n", err)
		return exitFailed
	}

	cfg := supervisor.Config{Workload: m.Workload, Store: storeDir, Command: m.Command, TreeLog: treeLogPath(m.Workload), Remote: remote}
	if !*noSnapshot && m.Layers.ProcessSnapshot.Status == manifest.StatusOK {
		// The process snapshot comes first. The restored workload can be checkpointed again the
		// same way, so the Supervisor keeps the snapshotter.
		cfg.Snapshotter = &snapshot.Criu{}
		code, ok, reason := restoreFromSnapshot(cfg, resume, root, m)
		if ok {
			return code
		}
		fmt.Fprintf(os.Stderr, "hib restore: the process snapshot did not restore, so restoring from the application checkpoint instead: %v\n", reason)
	}
	return supervise(cfg, resume)
}

// fallback checks that the application checkpoint can bring the workload back in a fresh training
// process, and returns what the training hook loads. The first build never writes a process
// snapshot, so every restore takes this path.
func fallback(root string, m manifest.Manifest) (*protocol.Resume, error) {
	if !validWorkloadName.MatchString(m.Workload) {
		return nil, fmt.Errorf("the manifest names workload %q, which is not a valid name", m.Workload)
	}
	if len(m.Fingerprint) > 0 {
		return nil, errors.New("the manifest has a fingerprint, and checking one is not built yet")
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
