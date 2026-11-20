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

// Op names what a message to the Supervisor asks for.
type Op string

const (
	OpStatus        Op = "status"
	OpCheckpoint    Op = "checkpoint"
	OpInitialize    Op = "initialize"
	OpBoundary      Op = "boundary"
	OpAppCheckpoint Op = "app_checkpoint"
)

// Request holds the fields for every message sent to the Supervisor. Op selects
// the operation; each operation uses only the fields shown in these JSON examples:
//
//	{"op":"status"}                                  // hib status
//	{"op":"checkpoint","deadline":"2026-11-17T18:00:00Z"} // hib checkpoint
//	{"op":"initialize","fingerprint":{...},"python":"/venv/bin/python"} // training hook asks where to begin
//	{"op":"boundary","step":12}                      // training hook reports an update boundary
//	{"op":"app_checkpoint","files":["/path/step.json"]} // training hook reports the files it wrote
//	{"op":"app_checkpoint","error":"disk full"}      // training hook reports a write failure
//
// The socket path selects the workload, so requests need no workload name or PID.
type Request struct {
	Op       Op        `json:"op"`
	Step     int       `json:"step,omitempty"`
	Deadline time.Time `json:"deadline,omitzero"`
	Files    []string  `json:"files,omitempty"`
	Error    string    `json:"error,omitempty"`

	// Initialize carries the fingerprint of the checkpoint host and the interpreter that collected it.
	Fingerprint *manifest.Fingerprint `json:"fingerprint,omitempty"`
	Python      string                `json:"python,omitempty"`
}

type StatusReply struct {
	State    string  `json:"state"`
	Step     int     `json:"step"`
	Location *string `json:"location"`
	Error    *string `json:"error"`
}

type InitializeReply struct {
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

// RunDir returns the folder path used for workload sockets. It uses HIB_RUN_DIR
// when nonempty; otherwise, it uses hib-<user ID> inside the OS temporary folder.
// It does not create the folder or check whether the folder or sockets exist.
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
