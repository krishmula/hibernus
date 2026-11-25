//go:build linux

package main

import (
	"os"
	"os/exec"
	"syscall"
)

// isolate starts the training process in its own session. As root, it also makes the training
// process PID 1 of a new PID namespace, so that criu restore can give it the same PID on the
// restore host. Without root, CRIU can't run, so hib makes no process snapshot and needs no namespace.
func isolate(child *exec.Cmd) {
	child.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if os.Geteuid() == 0 {
		child.SysProcAttr.Cloneflags = syscall.CLONE_NEWPID
	}
}
