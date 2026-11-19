//go:build !linux

package snapshot

import "errors"

// CRIU runs only on Linux, so there is nothing to wake on other systems.
func (c *Criu) Wake(pid int) error {
	return errors.New("process snapshots need Linux")
}
