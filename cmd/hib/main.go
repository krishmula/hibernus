package main

import (
	"fmt"
	"os"
)

const (
	exitOK           = 0
	exitFailed       = 1
	exitBadCommand   = 2
	exitNoSupervisor = 3
	exitFinished     = 4
)

const usage = `usage: hib <command> [flags]

commands:
  run [--workload <name>] [--store <folder>] <command> [args...]
                      start a workload under a Supervisor
  checkpoint [--workload <name>] [--deadline <90s | 2026-11-16T10:02:00Z>]
                      checkpoint a running workload and print the restore point location
  status [--workload <name>]
                      show a workload's state
  restore [--store <folder>] <location>
                      restore a workload from a restore point
`

func main() {
	os.Exit(dispatch(os.Args[1:]))
}

func dispatch(args []string) int {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		return exitBadCommand
	}

	switch command := args[0]; command {
	case "run":
		return runCommand(args[1:])
	case "status":
		return statusCommand(args[1:])
	case "checkpoint":
		return checkpointCommand(args[1:])
	case "restore":
		return restoreCommand(args[1:])
	case "help", "-h", "--help":
		fmt.Print(usage)
		return exitOK
	default:
		fmt.Fprintf(os.Stderr, "hib: unknown command %q\n\n%s", command, usage)
		return exitBadCommand
	}
}
