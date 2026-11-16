// Package supervisor holds a workload's state and answers messages on its socket.
package supervisor

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"

	"github.com/krishmula/hibernus/internal/protocol"
)

// maxSocketPath stays under the smallest limit Unix systems put on a socket's path (104 bytes on macOS).
const maxSocketPath = 100

type Supervisor struct {
	mu    sync.Mutex
	state State
	step  int
}

func New() *Supervisor {
	return &Supervisor{state: Running}
}

func (s *Supervisor) ChildExited(code int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = afterChildExit(s.state, code)
}

// Listen opens the workload's socket. It refuses if another Supervisor already answers there,
// and it removes a socket file that a crashed Supervisor left behind.
func Listen(path string) (net.Listener, error) {
	if len(path) > maxSocketPath {
		return nil, fmt.Errorf("socket path %s is too long; set HIB_RUN_DIR to a shorter folder", path)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	if conn, err := net.Dial("unix", path); err == nil {
		conn.Close()
		return nil, fmt.Errorf("a Supervisor already runs at %s", path)
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	return net.Listen("unix", path)
}

// Serve answers connections until the listener closes.
func (s *Supervisor) Serve(listener net.Listener) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		go s.serveConn(conn)
	}
}

func (s *Supervisor) serveConn(conn net.Conn) {
	defer conn.Close()
	scanner := bufio.NewScanner(conn)
	encoder := json.NewEncoder(conn)
	for scanner.Scan() {
		var req protocol.Request
		var reply any
		if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
			reply = protocol.ErrorReply{Error: "bad message: " + err.Error()}
		} else {
			reply = s.handle(req)
		}
		if err := encoder.Encode(reply); err != nil {
			return
		}
	}
}

func (s *Supervisor) handle(req protocol.Request) any {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch req.Op {
	case "status":
		return protocol.StatusReply{State: string(s.state), Step: s.step}
	case "start":
		return protocol.StartReply{Resume: nil}
	case "boundary":
		s.step = req.Step
		return protocol.BoundaryReply{Pause: false}
	default:
		return protocol.ErrorReply{Error: fmt.Sprintf("unknown op %q", req.Op)}
	}
}
