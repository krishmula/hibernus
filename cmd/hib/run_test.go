package main

import (
	"bufio"
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestDefaultWorkloadName(t *testing.T) {
	tests := []struct {
		command []string
		want    string
	}{
		{[]string{"python", "examples/toy/train.py", "--steps", "3"}, "train"},
		{[]string{"python", "-m", "trainer"}, "python"},
		{[]string{"./my-trainer", "--lr", "3e-4"}, "my-trainer"},
	}
	for _, tt := range tests {
		if got := defaultWorkloadName(tt.command); got != tt.want {
			t.Errorf("defaultWorkloadName(%q) = %q, want %q", tt.command, got, tt.want)
		}
	}
}

func TestCopyOutputCopiesWhatArrivesUntilStop(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.log")
	writer, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	var got syncBuffer
	stop := copyOutput(reader, &got)
	writer.WriteString("step 1\n")
	waitUntil(t, func() bool { return got.String() == "step 1\n" })
	writer.WriteString("step 2\n") // written just before stop, so only stop's last copy can catch it
	stop()

	if got.String() != "step 1\nstep 2\n" {
		t.Errorf("copied %q, want %q", got.String(), "step 1\nstep 2\n")
	}
}

func TestRelaySignalsPassesTheSignalOn(t *testing.T) {
	// The training process exits 7 on SIGTERM, well before the grace period ends.
	child := startShell(t, `trap "exit 7" TERM; echo ready; while :; do sleep 0.05; done`)
	signals := make(chan os.Signal, 1)
	exited := make(chan struct{})
	defer close(exited)
	go relaySignals(signals, child.Process, time.Minute, exited)

	signals <- syscall.SIGTERM
	if code := exitCode(child.Process.Wait()); code != 7 {
		t.Errorf("the training process exited %d, want 7", code)
	}
}

func TestRelaySignalsKillsATrainingProcessThatIgnoresTheSignal(t *testing.T) {
	child := startShell(t, `trap "" TERM INT; echo ready; while :; do sleep 0.05; done`)
	signals := make(chan os.Signal, 1)
	exited := make(chan struct{})
	defer close(exited)
	go relaySignals(signals, child.Process, 200*time.Millisecond, exited)

	start := time.Now()
	signals <- syscall.SIGTERM
	code := exitCode(child.Process.Wait())
	if code != 128+int(syscall.SIGKILL) {
		t.Errorf("hib would exit %d, want %d", code, 128+int(syscall.SIGKILL))
	}
	if waited := time.Since(start); waited < 200*time.Millisecond {
		t.Errorf("killed after %v, before the grace period ended", waited)
	}
}

// startShell starts sh with script, and waits until the shell has installed its traps.
func startShell(t *testing.T, script string) *exec.Cmd {
	t.Helper()
	child := exec.Command("sh", "-c", script)
	stdout, err := child.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { child.Process.Kill() })
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil || line != "ready\n" {
		t.Fatalf("the shell printed %q, %v", line, err)
	}
	return child
}

func waitUntil(t *testing.T, done func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if done() {
			return
		}
	}
	t.Fatal("timed out")
}

// syncBuffer is a bytes.Buffer that copyOutput's goroutine and the test can both use.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
