//go:build !linux

package main

import (
	"os/exec"
	"syscall"
)

// isolate starts the training process in its own session. Only Linux has PID namespaces and CRIU,
// so elsewhere the session is all.
func isolate(child *exec.Cmd) {
	child.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
