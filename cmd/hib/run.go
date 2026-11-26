package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net"
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
		Output:  protocol.OutputPath(*workload),
	}
	cfg.Snapshot, cfg.NoSnapshot = processSnapshotter(cfg.Output)
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

// supervise runs the Supervisor for a fresh training process, from start to exit. hib run and the
// application checkpoint path of hib restore use it. With resume nil, training starts fresh. With
// resume set, the training hook loads the application checkpoint.
func supervise(cfg supervisor.Config, resume *protocol.Resume) int {
	se, err := openSession(cfg, resume)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}
	process, output, err := startTraining(cfg)
	if err != nil {
		se.close()
		fmt.Fprintf(os.Stderr, "hib: %v\n", err)
		return exitFailed
	}
	defer output.Close()
	return se.run(process, waitFor(process), output)
}

// session is the Supervisor and its socket for one training process.
type session struct {
	sup      *supervisor.Supervisor
	listener net.Listener
	served   chan struct{}
	signals  chan os.Signal
}

func openSession(cfg supervisor.Config, resume *protocol.Resume) (*session, error) {
	listener, err := supervisor.Listen(protocol.SocketPath(cfg.Workload))
	if err != nil {
		return nil, err
	}
	se := &session{
		sup:      supervisor.New(cfg, resume),
		listener: listener,
		served:   make(chan struct{}),
		signals:  make(chan os.Signal, 2),
	}
	go func() {
		se.sup.Serve(listener)
		close(se.served)
	}()
	// The training process runs in its own session, so Ctrl+C at the terminal reaches only hib.
	// hib passes it on, along with SIGTERM.
	signal.Notify(se.signals, os.Interrupt, syscall.SIGTERM)
	return se, nil
}

// close stops answering on the socket. Each waiting hib checkpoint gets closeGrace to read its answer.
func (se *session) close() {
	signal.Stop(se.signals)
	se.listener.Close()
	select {
	case <-se.served:
	case <-time.After(closeGrace):
	}
}

// run supervises the training process until it exits, and returns what hib exits with.
func (se *session) run(process *os.Process, exited <-chan int, output io.Reader) int {
	se.sup.TrainingProcess(process.Pid)
	stopCopy := copyOutput(output, os.Stdout)
	stopRelay := make(chan struct{})
	go relaySignals(se.signals, process, killGrace, stopRelay)

	code := <-exited
	close(stopRelay)
	stopCopy()
	se.sup.ChildExited(code)
	se.sup.Settle() // a process snapshot or an upload in flight must finish before hib exits
	se.close()
	if se.sup.Checkpointed() {
		// CRIU ended the training process, or the host is going away. Either way the workload is safe.
		fmt.Fprintln(os.Stderr, "hib: the workload is checkpointed; continue it with hib restore")
		return exitOK
	}
	if code > 128 {
		fmt.Fprintf(os.Stderr, "hib: the training process ended on %v\n", syscall.Signal(code-128))
	}
	return code
}

// waitFor returns where the training process's exit code arrives.
func waitFor(process *os.Process) <-chan int {
	exited := make(chan int, 1)
	go func() { exited <- exitCode(process.Wait()) }()
	return exited
}

// startTraining starts a fresh training process, and returns it with a reader of its output file.
// The training process writes to a file, not to hib's terminal, because CRIU can't restore a
// terminal that belongs to a process outside the snapshot.
func startTraining(cfg supervisor.Config) (*os.Process, *os.File, error) {
	output, err := os.OpenFile(cfg.Output, os.O_WRONLY|os.O_CREATE|os.O_TRUNC|os.O_APPEND, 0o600)
	if err != nil {
		return nil, nil, err
	}
	defer output.Close() // the training process has its own copy
	reader, err := os.Open(cfg.Output)
	if err != nil {
		return nil, nil, err
	}
	child := trainingProcess(cfg.Command, protocol.SocketPath(cfg.Workload), output)
	if err := child.Start(); err != nil {
		reader.Close()
		return nil, nil, err
	}
	return child.Process, reader, nil
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
	// Its own session, so it has no terminal for CRIU to record. It gets no PID namespace: CUDA
	// fails in one that shares the host's /proc, and CRIU can't restore one with its own /proc.
	child.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
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

// exitCode turns the result of Wait into the exit code hib passes on.
func exitCode(state *os.ProcessState, err error) int {
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib: wait for the training process: %v\n", err)
		return exitFailed
	}
	if status, ok := state.Sys().(syscall.WaitStatus); ok && status.Signaled() {
		// The shell's convention, so a training process killed by SIGKILL makes hib exit 137.
		return 128 + int(status.Signal())
	}
	return state.ExitCode()
}

func defaultWorkloadName(command []string) string {
	for _, arg := range command {
		if strings.HasSuffix(arg, ".py") {
			return strings.TrimSuffix(filepath.Base(arg), ".py")
		}
	}
	return filepath.Base(command[0])
}
