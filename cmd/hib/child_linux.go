//go:build linux

package main

import (
	"os/exec"
	"syscall"
)

// isolateChild starts the training process in its own session. It then has no controlling
// terminal for CRIU to record, which is why a dump needs no --shell-job.
func isolateChild(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
