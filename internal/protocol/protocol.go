// Package protocol defines the messages on the Supervisor's socket and where that socket lives.
package protocol

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
)

// SocketEnv names the environment variable that tells the training hook where the socket is.
const SocketEnv = "HIB_SOCKET"

// ErrNoSupervisor means that nothing answers on the socket.
var ErrNoSupervisor = errors.New("no Supervisor answers")

type Request struct {
	Op       string    `json:"op"`
	Step     int       `json:"step,omitempty"`
	Deadline time.Time `json:"deadline,omitzero"`
	Files    []string  `json:"files,omitempty"`
	Error    string    `json:"error,omitempty"`
}

type StatusReply struct {
	State    string  `json:"state"`
	Step     int     `json:"step"`
	Location *string `json:"location"`
	Error    *string `json:"error"`
}

type StartReply struct {
	Resume *Resume `json:"resume"`
}

type Resume struct {
	Step  int      `json:"step"`
	Files []string `json:"files"`
}

// BoundaryReply tells the training hook whether to pause. With Pause set, Dir is the folder where
// the hook writes the application checkpoint.
type BoundaryReply struct {
	Pause bool   `json:"pause"`
	Dir   string `json:"dir,omitempty"`
}

type CheckpointReply struct {
	OK       bool               `json:"ok"`
	Location string             `json:"location,omitempty"`
	Manifest *manifest.Manifest `json:"manifest,omitempty"`
	State    string             `json:"state,omitempty"`
	Error    string             `json:"error,omitempty"`
}

// Refusal is the reply to a message the Supervisor can't answer, such as an unknown op. Its key
// differs from the "error" field of other replies, so the two never mix.
type Refusal struct {
	Reason string `json:"refused"`
}

// RunDir is the folder that holds one socket per running workload.
func RunDir() string {
	if dir := os.Getenv("HIB_RUN_DIR"); dir != "" {
		return dir
	}
	return filepath.Join(os.TempDir(), fmt.Sprintf("hib-%d", os.Getuid()))
}

func SocketPath(workload string) string {
	return filepath.Join(RunDir(), workload+".sock")
}

// Send sends one request and reads one reply into reply.
func Send(socketPath string, req Request, reply any) error {
	return SendBefore(socketPath, req, reply, time.Time{})
}

// SendBefore is Send with a time limit. A zero deadline means no limit.
func SendBefore(socketPath string, req Request, reply any, deadline time.Time) error {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return fmt.Errorf("%w at %s: %v", ErrNoSupervisor, socketPath, err)
	}
	defer conn.Close()
	if err := conn.SetDeadline(deadline); err != nil {
		return err
	}

	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return err
	}
	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return err
	}
	var refusal Refusal
	if json.Unmarshal(line, &refusal) == nil && refusal.Reason != "" {
		return errors.New(refusal.Reason)
	}
	return json.Unmarshal(line, reply)
}
