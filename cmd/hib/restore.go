package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const restoreUsage = "usage: hib restore [--store <folder>] <location>\n"

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

	root, err := filepath.Abs(flags.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib restore: %v\n", err)
		return exitFailed
	}
	storeDir, err := filepath.Abs(*store)
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

	return supervise(supervisor.Config{Workload: m.Workload, Store: storeDir, Command: m.Command}, resume)
}

// resumeFromApplicationCheckpoint checks that the application checkpoint can bring the workload back
// in a fresh training process, and returns what the training hook loads. The first build never writes a process
// snapshot, so every restore takes this path.
func resumeFromApplicationCheckpoint(root string, m manifest.Manifest) (*protocol.Resume, error) {
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
