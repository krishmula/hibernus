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
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const runUsage = "usage: hib run [--workload <name>] [--store <folder>] <command> [args...]\n"

var validWorkloadName = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

func runCommand(args []string) int {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, runUsage) }
	workload := flags.String("workload", "", "name of the workload")
	store := storeFlag(flags)
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

	storeDir, err := filepath.Abs(*store)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib run: %v\n", err)
		return exitFailed
	}
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib run: %v\n", err)
		return exitFailed
	}

	return supervise(supervisor.Config{
		Workload: *workload,
		Store:    storeDir,
		// The manifest records no environment variables yet, because they can hold secrets.
		Command: manifest.Command{Argv: command, Cwd: cwd, Env: map[string]string{}},
	}, nil)
}

// storeFlag adds --store, the folder that holds restore points.
func storeFlag(flags *flag.FlagSet) *string {
	home, _ := os.UserHomeDir()
	return flags.String("store", filepath.Join(home, ".hib", "restore-points"), "folder that holds restore points")
}

// supervise runs the Supervisor for one training process, from start to exit. hib run and hib restore
// both use it. With resume nil, training starts fresh. With resume set, it continues from a restore point.
func supervise(cfg supervisor.Config, resume *protocol.Resume) int {
	socketPath := protocol.SocketPath(cfg.Workload)
	listener, err := supervisor.Listen(socketPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}
	defer listener.Close()

	sup := supervisor.New(cfg, resume)
	served := make(chan struct{})
	go func() {
		sup.Serve(listener)
		close(served)
	}()

	command := cfg.Command.Argv
	child := exec.Command(command[0], command[1:]...)
	child.Dir = cfg.Command.Cwd
	child.Env = append(os.Environ(), protocol.SocketEnv+"="+socketPath)
	child.Stdin = os.Stdin
	child.Stdout = os.Stdout
	child.Stderr = os.Stderr

	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	defer signal.Stop(interrupts)

	if err := child.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}

	code := childExitCode(child.Wait())
	sup.ChildExited(code)

	// Close the socket, then give each waiting hib checkpoint a moment to read its answer.
	listener.Close()
	select {
	case <-served:
	case <-time.After(2 * time.Second):
	}
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
	fmt.Fprintf(os.Stderr, "hib: training process %v\n", err)
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
