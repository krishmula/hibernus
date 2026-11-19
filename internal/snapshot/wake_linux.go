//go:build linux

package snapshot

import (
	"fmt"
	"syscall"
)

// Wake sends SIGUSR1, which the paused training hook waits for.
func (c *Criu) Wake(pid int) error {
	if err := syscall.Kill(pid, syscall.SIGUSR1); err != nil {
		return fmt.Errorf("wake process %d: %w", pid, err)
	}
	return nil
}
