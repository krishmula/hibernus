package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const runUsage = "usage: hib run [--workload <name>] [--store <folder>] [--remote s3://<bucket>] <command> [args...]\n"

// closeGrace is how long hib run waits, after the training process exits, for each waiting
// hib checkpoint to read its answer.
const closeGrace = 2 * time.Second

func runCommand(args []string) int {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, runUsage) }
	workload := workloadFlag(flags)
	store := storeFlag(flags)
	remoteFlag := flags.String("remote", "", "publish each restore point to s3://<bucket> before a checkpoint counts as done")
	if code, ok := parseFlags(flags, args); !ok {
		return code
	}

	command := flags.Args()
	if len(command) == 0 {
		fmt.Fprint(os.Stderr, "hib run: missing the command to run\n"+runUsage)
		return exitBadCommand
	}
	if *workload == "" {
		*workload = defaultWorkloadName(command)
	}
	if err := checkWorkloadName(*workload); err != nil {
		fmt.Fprintf(os.Stderr, "hib run: %v\n", err)
		return exitBadCommand
	}

	bucket := ""
	if *remoteFlag != "" {
		var err error
		if bucket, err = parseRemoteFlag(*remoteFlag); err != nil {
			fmt.Fprintf(os.Stderr, "hib run: %v\n", err)
			return exitBadCommand
		}
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

	cfg := supervisor.Config{
		Workload: *workload,
		Store:    storeDir,
		// The manifest records no environment variables yet, because they can hold secrets.
		Command: manifest.Command{Argv: command, Cwd: cwd, Env: map[string]string{}},
	}
	if bucket != "" {
		remote, err := remoteFor(context.Background(), bucket)
		if err != nil {
			fmt.Fprintf(os.Stderr, "hib run: open the bucket: %v\n", err)
			return exitFailed
		}
		cfg.Remote = remote
	}
	return supervise(cfg, nil)
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

	child := trainingProcess(cfg.Command, socketPath)

	// Ctrl-C reaches the training process too, because it shares the terminal. hib catches the
	// signal and never reads it, so hib keeps running until the training process exits, and then
	// exits with that process's code.
	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	defer signal.Stop(interrupts)

	if err := child.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}

	code := childExitCode(child.Wait())
	sup.ChildExited(code)
	sup.Settle() // an upload in flight must finish before hib exits

	listener.Close()
	select {
	case <-served:
	case <-time.After(closeGrace):
	}
	return code
}

// trainingProcess prepares the training process. It shares hib's terminal, and its environment
// tells the training hook where the socket is.
func trainingProcess(command manifest.Command, socketPath string) *exec.Cmd {
	child := exec.Command(command.Argv[0], command.Argv[1:]...)
	child.Dir = command.Cwd
	child.Env = append(os.Environ(), protocol.SocketEnv+"="+socketPath)
	child.Stdin = os.Stdin
	child.Stdout = os.Stdout
	child.Stderr = os.Stderr
	return child
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
