//go:build !linux

package main

import "os/exec"

// isolateChild does nothing off Linux, where there are no process snapshots.
func isolateChild(cmd *exec.Cmd) {}
