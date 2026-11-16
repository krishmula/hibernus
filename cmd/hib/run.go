package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
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

	return supervise(*workload, command)
}

func supervise(workload string, command []string) int {
	socketPath := protocol.SocketPath(workload)
	listener, err := supervisor.Listen(socketPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib run: %v\n", err)
		return exitFailed
	}
	defer listener.Close()

	sup := supervisor.New()
	go sup.Serve(listener)

	child := exec.Command(command[0], command[1:]...)
	child.Env = append(os.Environ(), protocol.SocketEnv+"="+socketPath)
	child.Stdin = os.Stdin
	child.Stdout = os.Stdout
	child.Stderr = os.Stderr

	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	defer signal.Stop(interrupts)

	if err := child.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "hib run: %v\n", err)
		return exitFailed
	}

	code := childExitCode(child.Wait())
	sup.ChildExited(code)
	return code
}

// childExitCode turns the result of Wait into the exit code hib run passes on.
func childExitCode(err error) int {
	if err == nil {
		return exitOK
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) && exitErr.ExitCode() >= 0 {
		return exitErr.ExitCode()
	}
	fmt.Fprintf(os.Stderr, "hib run: training process %v\n", err)
	return exitFailed
}

func defaultWorkloadName(command []string) string {
	for _, arg := range command {
		if strings.HasSuffix(arg, ".py") {
			return strings.TrimSuffix(filepath.Base(arg), ".py")
		}
	}
	return filepath.Base(command[0])
}
