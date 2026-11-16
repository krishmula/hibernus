package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const runUsage = "usage: hib run [--workload <name>] <command> [args...]\n"

var validWorkloadName = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

func runCommand(args []string) int {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, runUsage) }
	workload := flags.String("workload", "", "name of the workload")
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return exitOK
		}
		return exitBadCommand
	}

	command := flags.Args()
	if len(command) == 0 {
		fmt.Fprint(os.Stderr, "hib run: missing the command to run\n"+runUsage)
		return exitBadCommand
	}
	if *workload == "" {
		*workload = defaultWorkloadName(command)
	}
	if !validWorkloadName.MatchString(*workload) {
		fmt.Fprintf(os.Stderr, "hib run: workload name %q may only use letters, digits, '.', '_', and '-'\n", *workload)
		return exitBadCommand
	}

	fmt.Printf("workload: %s\ncommand: %s\n", *workload, strings.Join(command, " "))
	return exitOK
}

func defaultWorkloadName(command []string) string {
	for _, arg := range command {
		if strings.HasSuffix(arg, ".py") {
			return strings.TrimSuffix(filepath.Base(arg), ".py")
		}
	}
	return filepath.Base(command[0])
}
