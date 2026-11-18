package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
)

var validWorkloadName = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// parseFlags parses args. When ok is false, the command stops and exits with code: exitOK after -h,
// or exitBadCommand after a bad flag, which the flag package has already reported.
func parseFlags(flags *flag.FlagSet, args []string) (code int, ok bool) {
	err := flags.Parse(args)
	switch {
	case err == nil:
		return exitOK, true
	case errors.Is(err, flag.ErrHelp):
		return exitOK, false
	default:
		return exitBadCommand, false
	}
}

// workloadFlag adds --workload, the name of the workload.
func workloadFlag(flags *flag.FlagSet) *string {
	return flags.String("workload", "", "name of the workload")
}

// storeFlag adds --store, the folder that holds restore points.
func storeFlag(flags *flag.FlagSet) *string {
	home, _ := os.UserHomeDir()
	return flags.String("store", filepath.Join(home, ".hib", "restore-points"), "folder that holds restore points")
}

func checkWorkloadName(name string) error {
	if !validWorkloadName.MatchString(name) {
		return fmt.Errorf("workload name %q may only use letters, digits, '.', '_', and '-'", name)
	}
	return nil
}
