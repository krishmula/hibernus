package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const runUsage = "usage: hib run [--workload <name>] [--store <folder>] [--remote s3://<bucket>] <command> [args...]\n"

// closeGrace is how long hib run waits, after the training process exits, for each waiting
// hib checkpoint to read its answer.
const closeGrace = 2 * time.Second

// killGrace is how long the training process has to exit after hib passes it SIGINT or SIGTERM.
// After that, hib sends SIGKILL.
const killGrace = 10 * time.Second

// outputPoll is how often hib looks for new output from the training process.
const outputPoll = 50 * time.Millisecond

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
		if err := refuseTakenWorkload(context.Background(), remote, *workload); err != nil {
			fmt.Fprintf(os.Stderr, "hib run: %v\n", err)
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

	// The training process writes to a file, not to hib's terminal, because CRIU can't restore a
	// terminal that belongs to a process outside the snapshot. hib copies the file to its own output.
	outputPath := protocol.OutputPath(cfg.Workload)
	output, err := os.OpenFile(outputPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC|os.O_APPEND, 0o600)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}
	reader, err := os.Open(outputPath)
	if err != nil {
		output.Close()
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}
	defer reader.Close()

	child := trainingProcess(cfg.Command, socketPath, output)

	// The training process runs in its own session, so Ctrl+C at the terminal reaches only hib.
	// hib passes it on, along with SIGTERM.
	signals := make(chan os.Signal, 2)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signals)

	err = child.Start()
	output.Close() // the training process has its own copy
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}
	stopCopy := copyOutput(reader, os.Stdout)
	exited := make(chan struct{})
	go relaySignals(signals, child.Process, killGrace, exited)

	code := childExitCode(child.Wait())
	close(exited)
	stopCopy()
	sup.ChildExited(code)
	sup.Settle() // an upload in flight must finish before hib exits

	listener.Close()
	select {
	case <-served:
	case <-time.After(closeGrace):
	}
	return code
}

// trainingProcess prepares the training process. It reads from /dev/null and writes to output, and
// its environment tells the training hook where the socket is.
func trainingProcess(command manifest.Command, socketPath string, output *os.File) *exec.Cmd {
	child := exec.Command(command.Argv[0], command.Argv[1:]...)
	child.Dir = command.Cwd
	// Python buffers output to a file in large blocks, so without PYTHONUNBUFFERED a print would
	// reach hib's output long after it ran.
	child.Env = append(os.Environ(), protocol.SocketEnv+"="+socketPath, "PYTHONUNBUFFERED=1")
	child.Stdout = output
	child.Stderr = output
	isolate(child)
	return child
}

// copyOutput copies what the training process writes to w as it arrives, until stop is called.
// stop copies what is left, then returns.
func copyOutput(r io.Reader, w io.Writer) (stop func()) {
	quit := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			io.Copy(w, r) // stops at the end of what the training process has written so far
			select {
			case <-quit:
				io.Copy(w, r)
				return
			case <-time.After(outputPoll):
			}
		}
	}()
	return func() {
		close(quit)
		<-done
	}
}

// relaySignals passes each signal to the training process until exited closes. If the training
// process still runs grace after the first signal, relaySignals kills it.
func relaySignals(signals <-chan os.Signal, process *os.Process, grace time.Duration, exited <-chan struct{}) {
	var kill <-chan time.Time
	for {
		select {
		case sig := <-signals:
			process.Signal(sig)
			if kill == nil {
				kill = time.After(grace)
			}
		case <-kill:
			process.Kill()
		case <-exited:
			return
		}
	}
}

// childExitCode turns the result of Wait into the exit code hib run passes on.
func childExitCode(err error) int {
	if err == nil {
		return exitOK
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		if status, ok := exitErr.Sys().(syscall.WaitStatus); ok && status.Signaled() {
			// The shell's convention, so a training process killed by SIGKILL makes hib exit 137.
			fmt.Fprintf(os.Stderr, "hib: the training process ended on %v\n", status.Signal())
			return 128 + int(status.Signal())
		}
		if exitErr.ExitCode() >= 0 {
			return exitErr.ExitCode()
		}
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
