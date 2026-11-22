//go:build linux

// Command criu-test is the stand-in Supervisor for the CRIU test. It checks the phase 5 design on
// one Linux host, as root, before any of it goes into hib.
//
//	criu-test dump -dir D -python P [-torch]    start the training process, pause it, and dump it
//	criu-test restore -dir D [-sibling]         restore it, wake it, and let it finish
//
// Each check prints one PASS or FAIL line. The command exits 1 if any check fails.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const prSetChildSubreaper = 36

// dumped is what the dump run leaves for the restore run.
type dumped struct {
	Step    int `json:"step"`
	HostPID int `json:"host_pid"`
}

// message holds every field the stand-in training process sends.
type message struct {
	Op            string `json:"op"`
	Step          int    `json:"step"`
	Threads       int    `json:"threads"`
	GetPID        int    `json:"getpid"`
	ProcSelfPID   int    `json:"proc_self_pid"`
	ProcGetPIDCom string `json:"proc_getpid_comm"`
}

var failed bool

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: criu-test dump|restore [flags]")
		os.Exit(2)
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	dir := fs.String("dir", "", "the folder for this case's images, logs, and notes")
	socketPath := fs.String("socket", "/run/hib-criu-test.sock", "the socket path, the same for dump and restore")
	python := fs.String("python", "python3", "dump: the interpreter for trainer.py")
	torch := fs.Bool("torch", false, "dump: have the training process load torch")
	pauseAt := fs.Int("pause-at", 5, "dump: the step at which to pause")
	sibling := fs.Bool("sibling", false, "restore: use criu --restore-sibling instead of a subreaper")
	fs.Parse(os.Args[2:])
	if *dir == "" {
		fmt.Fprintln(os.Stderr, "criu-test: -dir is required")
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "dump":
		err = dump(*dir, *socketPath, *python, *torch, *pauseAt)
	case "restore":
		err = restore(*dir, *socketPath, *sibling)
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		fmt.Printf("FAIL %s: %v\n", os.Args[1], err)
		failed = true
	}
	if failed {
		os.Exit(1)
	}
}

// dump runs the checkpoint host's side: start the training process as PID 1 of its own PID
// namespace, pause it at an update boundary, stop it from outside, and run criu dump.
func dump(dir, socketPath, python string, torch bool, pauseAt int) error {
	images := filepath.Join(dir, "images")
	if err := os.MkdirAll(images, 0o755); err != nil {
		return err
	}
	ln, err := listen(socketPath)
	if err != nil {
		return err
	}
	defer ln.Close()

	trainer, err := filepath.Abs(filepath.Join(filepath.Dir(os.Args[0]), "trainer.py"))
	if err != nil {
		return err
	}
	args := []string{trainer}
	if torch {
		args = append(args, "--torch")
	}
	// The log is a regular file, so CRIU can reopen it by path. Restore on another host would need
	// the same file there, or another plan for the training process's output.
	logFile, err := os.Create(filepath.Join(dir, "trainer.log"))
	if err != nil {
		return err
	}
	defer logFile.Close()
	cmd := exec.Command(python, args...)
	cmd.Env = append(os.Environ(), "HIB_SOCKET="+socketPath)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	// Its own session and process group too: CRIU refuses a tree whose session leader is outside it.
	cmd.SysProcAttr = &syscall.SysProcAttr{Cloneflags: syscall.CLONE_NEWPID, Setsid: true}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start the training process: %w", err)
	}
	pid := cmd.Process.Pid
	fmt.Printf("training process: host PID %d\n", pid)

	conn, err := accept(ln, 60*time.Second)
	if err != nil {
		return err
	}
	in := bufio.NewReader(conn)
	step := 0
	for {
		m, err := read(in)
		if err != nil {
			return fmt.Errorf("before the pause: %w", err)
		}
		if m.Op != "boundary" {
			return fmt.Errorf("got %q, want boundary", m.Op)
		}
		step = m.Step
		pause := step >= pauseAt
		if err := write(conn, map[string]any{"pause": pause}); err != nil {
			return err
		}
		if pause {
			fmt.Printf("paused at step %d with %d threads\n", step, m.Threads)
			break
		}
	}
	if m, err := read(in); err != nil || m.Op != "app_checkpoint" {
		return fmt.Errorf("want app_checkpoint, got %+v, %v", m, err)
	}
	// The training hook closes its connection once its SIGCONT handler is in place.
	if _, err := in.ReadByte(); err == nil {
		return errors.New("want the connection to close after app_checkpoint")
	}
	conn.Close()

	// Stop it from outside. PID 1 of a namespace would ignore a SIGSTOP from inside it.
	if err := syscall.Kill(pid, syscall.SIGSTOP); err != nil {
		return fmt.Errorf("SIGSTOP: %w", err)
	}
	var ws syscall.WaitStatus
	if _, err := syscall.Wait4(pid, &ws, syscall.WUNTRACED, nil); err != nil {
		return fmt.Errorf("wait for the stop: %w", err)
	}
	check("stopped before the dump", ws.Stopped() && ws.StopSignal() == syscall.SIGSTOP, "wait status %v", ws)
	status, err := procStatus(pid)
	if err != nil {
		return err
	}
	check("PID 1 of its own namespace", lastField(status["NSpid"]) == "1", "NSpid %q, State %q", status["NSpid"], status["State"])

	start := time.Now()
	err = runCRIU("dump", "-t", strconv.Itoa(pid), "-D", images, "-v4", "-o", "dump.log")
	check("criu dump", err == nil, "%v in %.1f s (log %s)", err, time.Since(start).Seconds(), filepath.Join(images, "dump.log"))
	if err != nil {
		return nil
	}
	if _, err := syscall.Wait4(pid, &ws, 0, nil); err != nil {
		return fmt.Errorf("reap the training process: %w", err)
	}
	check("criu ended the training process", ws.Signaled(), "wait status %v", ws)
	fmt.Printf("images: %s\n", du(images))
	return writeJSON(filepath.Join(dir, "dumped.json"), dumped{Step: step, HostPID: pid})
}

// restore runs the restore host's side: adopt the restored training process, wake it with
// SIGCONT, and accept its reconnect.
func restore(dir, socketPath string, sibling bool) error {
	var d dumped
	if err := readJSON(filepath.Join(dir, "dumped.json"), &d); err != nil {
		return err
	}
	images := filepath.Join(dir, "images")
	if !sibling {
		if _, _, errno := syscall.RawSyscall(syscall.SYS_PRCTL, prSetChildSubreaper, 1, 0); errno != 0 {
			return fmt.Errorf("become a subreaper: %w", errno)
		}
	}
	// Listen before the restore, so the socket is ready when the training process wakes.
	ln, err := listen(socketPath)
	if err != nil {
		return err
	}
	defer ln.Close()

	pidfile := filepath.Join(dir, "restored.pid")
	os.Remove(pidfile)
	args := []string{"-D", images, "-v4", "-o", "restore.log", "--restore-detached", "--pidfile", pidfile}
	if sibling {
		args = append(args, "--restore-sibling")
	}
	start := time.Now()
	err = runCRIU("restore", args...)
	check("criu restore", err == nil, "%v in %.1f s (log %s)", err, time.Since(start).Seconds(), filepath.Join(images, "restore.log"))
	if err != nil {
		return nil
	}
	raw, err := os.ReadFile(pidfile)
	if err != nil {
		return err
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil {
		return fmt.Errorf("pidfile: %w", err)
	}
	fmt.Printf("restored training process: host PID %d (it was %d)\n", pid, d.HostPID)

	status, err := procStatus(pid)
	if err != nil {
		return err
	}
	check("restored still stopped", strings.HasPrefix(status["State"], "T"), "State %q", status["State"])
	check("restored as PID 1 of a new namespace", lastField(status["NSpid"]) == "1", "NSpid %q", status["NSpid"])
	check("the Supervisor adopted it", status["PPid"] == strconv.Itoa(os.Getpid()), "PPid %s, Supervisor %d", status["PPid"], os.Getpid())

	if err := syscall.Kill(pid, syscall.SIGCONT); err != nil {
		return fmt.Errorf("SIGCONT: %w", err)
	}
	conn, err := accept(ln, 10*time.Second)
	check("SIGCONT woke it and it reconnected", err == nil, "%v", err)
	if err != nil {
		syscall.Kill(pid, syscall.SIGKILL)
		syscall.Wait4(pid, nil, 0, nil)
		return nil
	}
	defer conn.Close()
	in := bufio.NewReader(conn)
	m, err := read(in)
	if err != nil {
		return err
	}
	check("it reports the dumped step", m.Op == "restored" && m.Step == d.Step, "got %s step %d, want restored step %d", m.Op, m.Step, d.Step)
	fmt.Printf("after the restore: %d threads, getpid %d, /proc/self is PID %d, /proc/%d/comm is %q\n",
		m.Threads, m.GetPID, m.ProcSelfPID, m.GetPID, m.ProcGetPIDCom)
	if err := write(conn, map[string]any{}); err != nil {
		return err
	}

	next := d.Step + 1
	for {
		m, err := read(in)
		if err != nil {
			break // the training process closes the connection when it finishes
		}
		check(fmt.Sprintf("continued at step %d", next), m.Op == "boundary" && m.Step == next, "got %s step %d", m.Op, m.Step)
		next = m.Step + 1
		if err := write(conn, map[string]any{"pause": false}); err != nil {
			return err
		}
	}
	var ws syscall.WaitStatus
	_, err = syscall.Wait4(pid, &ws, 0, nil)
	check("the Supervisor reaped it, exit 0", err == nil && ws.Exited() && ws.ExitStatus() == 0, "%v, wait status %v", err, ws)
	return nil
}

// runCRIU runs criu with this command's own stdout and stderr, not pipes, so a restored process
// that kept a pipe open could not make the wait hang.
func runCRIU(command string, args ...string) error {
	cmd := exec.Command("criu", append([]string{command}, args...)...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

func check(name string, ok bool, format string, args ...any) {
	result := "PASS"
	if !ok {
		result, failed = "FAIL", true
	}
	fmt.Printf("%s %s: %s\n", result, name, fmt.Sprintf(format, args...))
}

func listen(path string) (*net.UnixListener, error) {
	os.Remove(path)
	return net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
}

func accept(ln *net.UnixListener, limit time.Duration) (net.Conn, error) {
	ln.SetDeadline(time.Now().Add(limit))
	conn, err := ln.Accept()
	if err != nil {
		return nil, fmt.Errorf("no connection within %s: %w", limit, err)
	}
	return conn, nil
}

func read(in *bufio.Reader) (message, error) {
	var m message
	line, err := in.ReadBytes('\n')
	if err != nil {
		return m, err
	}
	return m, json.Unmarshal(line, &m)
}

func write(conn net.Conn, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = conn.Write(append(b, '\n'))
	return err
}

func procStatus(pid int) (map[string]string, error) {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return nil, err
	}
	fields := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		if k, v, ok := strings.Cut(line, ":"); ok {
			fields[k] = strings.TrimSpace(v)
		}
	}
	return fields, nil
}

func lastField(s string) string {
	f := strings.Fields(s)
	if len(f) == 0 {
		return ""
	}
	return f[len(f)-1]
}

func du(path string) string {
	out, _ := exec.Command("du", "-sh", path).Output()
	return strings.TrimSpace(string(out))
}

func writeJSON(path string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
