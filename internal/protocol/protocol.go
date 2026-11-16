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
)

// SocketEnv names the environment variable that tells the training hook where the socket is.
const SocketEnv = "HIB_SOCKET"

type Request struct {
	Op   string `json:"op"`
	Step int    `json:"step,omitempty"`
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

type BoundaryReply struct {
	Pause bool `json:"pause"`
}

type ErrorReply struct {
	Error string `json:"error"`
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
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return err
	}
	defer conn.Close()

	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return err
	}
	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return err
	}
	var failure ErrorReply
	if json.Unmarshal(line, &failure) == nil && failure.Error != "" {
		return errors.New(failure.Error)
	}
	return json.Unmarshal(line, reply)
}
