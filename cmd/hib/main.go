package main

import (
	"fmt"
	"os"
)

const (
	exitOK         = 0
	exitFailed     = 1
	exitBadCommand = 2
)

const usage = `usage: hib <command> [flags]

commands:
  run [--workload <name>] <command> [args...]   start a workload under a Supervisor
  checkpoint                                    checkpoint a running workload
  status                                        show a workload's state
  restore <location>                            restore a workload from a restore point
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
	case "checkpoint", "status", "restore":
		fmt.Fprintf(os.Stderr, "hib %s: not built yet\n", command)
		return exitFailed
	case "help", "-h", "--help":
		fmt.Print(usage)
		return exitOK
	default:
		fmt.Fprintf(os.Stderr, "hib: unknown command %q\n\n%s", command, usage)
		return exitBadCommand
	}
}
