package main

import (
	"fmt"
	"os"
)

const exitBadCommand = 2

const usage = `usage: hib <command> [flags]

commands:
  run <script> [args...]   start a workload under a Supervisor
  checkpoint               checkpoint a running workload
  status                   show a workload's state
  restore <location>       restore a workload from a restore point
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(exitBadCommand)
	}

	switch command := os.Args[1]; command {
	case "run", "checkpoint", "status", "restore":
		fmt.Fprintf(os.Stderr, "hib %s: not built yet\n", command)
		os.Exit(1)
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "hib: unknown command %q\n\n%s", command, usage)
		os.Exit(exitBadCommand)
	}
}
