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
	"regexp"
	"strings"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/snapshot"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const runUsage = "usage: hib run [--workload <name>] [--store <folder>] [--remote s3://<bucket>] [--snapshot] <command> [args...]\n"

var validWorkloadName = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

func runCommand(args []string) int {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, runUsage) }
	workload := flags.String("workload", "", "name of the workload")
	store := storeFlag(flags)
	withSnapshot := flags.Bool("snapshot", false, "also take a CRIU process snapshot at each checkpoint (Linux, needs root)")
	remoteFlag := flags.String("remote", "", "publish each restore point to s3://<bucket> before a checkpoint counts as done")
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
		TreeLog: treeLogPath(*workload),
	}
	if *withSnapshot {
		cfg.Snapshotter = &snapshot.Criu{}
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
	if cfg.Snapshotter != nil {
		// CRIU dumps the training process, so it needs no terminal and no pipe to this process.
		output, err := openOutput(cfg.Workload)
		if err != nil {
			fmt.Fprintf(os.Stderr, "hib: %v\n", err)
			return exitFailed
		}
		defer output.Close()
		child.Stdin, child.Stdout, child.Stderr = nil, output, output
		isolateChild(child)
		fmt.Fprintf(os.Stderr, "hib: training output goes to %s, and the process tree log to %s\n",
			output.Name(), cfg.TreeLog)
	}

	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	defer signal.Stop(interrupts)

	if err := child.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}
	sup.SetChild(child.Process.Pid)

	code := childExitCode(child.Wait())
	sup.ChildExited(code)
	sup.Settle() // a dump in flight ends the child on purpose, and must finish before we exit

	// Close the socket, then give each waiting hib checkpoint a moment to read its answer.
	listener.Close()
	select {
	case <-served:
	case <-time.After(2 * time.Second):
	}
	if sup.Checkpointed() {
		return exitOK // the checkpoint ended the training process on purpose
	}
	return code
}

// openOutput opens the file that a snapshotted workload's training process writes to.
func openOutput(workload string) (*os.File, error) {
	path := outputPath(workload)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
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
