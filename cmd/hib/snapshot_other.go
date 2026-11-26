//go:build !linux

package main

import (
	"errors"
	"os"

	"github.com/krishmula/hibernus/internal/supervisor"
)

var errNeedsLinux = errors.New("the process snapshot needs Linux and CRIU")

// processSnapshotter returns no Snapshotter, because only Linux has CRIU. It warns nothing, because
// a host without Linux is a development host.
func processSnapshotter(output string) (supervisor.Snapshotter, string) {
	return nil, errNeedsLinux.Error()
}

func restoreProcess(root string) (*os.Process, error) { return nil, errNeedsLinux }

func wake(process *os.Process) error { return errNeedsLinux }
