package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/krishmula/hibernus/internal/protocol"
)

const statusUsage = "usage: hib status [--workload <name>]\n"

func statusCommand(args []string) int {
	flags := flag.NewFlagSet("status", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, statusUsage) }
	workload := workloadFlag(flags)
	if code, ok := parseFlags(flags, args); !ok {
		return code
	}
	if flags.NArg() > 0 {
		fmt.Fprint(os.Stderr, "hib status: unexpected arguments\n"+statusUsage)
		return exitBadCommand
	}

	name, code := findWorkload("status", *workload)
	if code != exitOK {
		return code
	}

	var reply protocol.StatusReply
	if err := protocol.Send(protocol.SocketPath(name), protocol.Request{Op: protocol.OpStatus}, &reply); err != nil {
		fmt.Fprintf(os.Stderr, "hib status: no Supervisor answers for workload %q: %v\n", name, err)
		return exitNoSupervisor
	}
	fmt.Printf("workload: %s\nstate: %s\nstep: %d\n", name, reply.State, reply.Step)
	return exitOK
}

// findWorkload picks the workload a command talks to. Without a name, it picks the only workload
// running on this host.
func findWorkload(command, name string) (string, int) {
	if name != "" {
		if err := checkWorkloadName(name); err != nil {
			fmt.Fprintf(os.Stderr, "hib %s: %v\n", command, err)
			return "", exitBadCommand
		}
		return name, exitOK
	}

	sockets, _ := filepath.Glob(filepath.Join(protocol.RunDir(), "*.sock"))
	switch len(sockets) {
	case 0:
		fmt.Fprintf(os.Stderr, "hib %s: no Supervisor runs on this host\n", command)
		return "", exitNoSupervisor
	case 1:
		return strings.TrimSuffix(filepath.Base(sockets[0]), ".sock"), exitOK
	default:
		names := make([]string, len(sockets))
		for i, socket := range sockets {
			names[i] = strings.TrimSuffix(filepath.Base(socket), ".sock")
		}
		fmt.Fprintf(os.Stderr, "hib %s: several workloads run (%s); pick one with --workload\n", command, strings.Join(names, ", "))
		return "", exitBadCommand
	}
}
