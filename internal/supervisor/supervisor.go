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

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
)

// maxSocketPath stays under the smallest limit Unix systems put on a socket's path (104 bytes on macOS).
const maxSocketPath = 100

// Config is what a Supervisor knows about its workload from the start.
type Config struct {
	Workload string
	Store    string           // the folder that holds restore points, one subfolder per workload
	Command  manifest.Command // how the training process started, recorded in each manifest
}

type Supervisor struct {
	cfg Config

	// mu guards every field below. handle, join, ChildExited, and giveUp take it, and every method
	// they call runs with it already held.
	mu      sync.Mutex
	state   State
	step    int
	resume  *protocol.Resume // what initialize answers; nil for a fresh run
	attempt *attempt         // the latest checkpoint, in progress or finished
	err     string           // why the workload Failed
}

// New makes a Supervisor. With resume nil, it starts in Running. With resume set, it starts in
// Restoring and hands resume to the training hook when the hook asks initialize.
func New(cfg Config, resume *protocol.Resume) *Supervisor {
	s := &Supervisor{cfg: cfg, state: Running}
	if resume != nil {
		s.state = Restoring
		s.step = resume.Step
		s.resume = resume
	}
	return s
}

func (s *Supervisor) ChildExited(code int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	next := afterChildExit(s.state, code)
	if next == Failed && s.state != Failed {
		s.fail(fmt.Sprintf("the training process exited with code %d", code))
		return
	}
	s.state = next
}

// fail moves to Failed and answers every hib checkpoint that waits.
func (s *Supervisor) fail(reason string) {
	s.state = Failed
	s.err = reason
	if s.attempt != nil {
		s.attempt.resolve(protocol.CheckpointReply{State: string(Failed), Error: reason})
	}
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

// Serve answers connections until the listener closes. Then it waits for open connections to end,
// so that each waiting hib checkpoint gets its answer.
func (s *Supervisor) Serve(listener net.Listener) {
	var conns sync.WaitGroup
	defer conns.Wait()
	for {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		conns.Add(1)
		go func() {
			defer conns.Done()
			s.serveConn(conn)
		}()
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
			reply = protocol.Refusal{Reason: "bad message: " + err.Error()}
		} else {
			reply = s.handle(req)
		}
		if reply == nil {
			continue
		}
		if err := encoder.Encode(reply); err != nil {
			return
		}
	}
}

// handle answers one message. A nil answer means no reply.
func (s *Supervisor) handle(req protocol.Request) any {
	if req.Op == protocol.OpCheckpoint {
		// It waits until the checkpoint ends, so it must not hold the lock while it waits.
		return s.checkpoint(req.Deadline)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	switch req.Op {
	case protocol.OpStatus:
		return s.status()
	case protocol.OpInitialize:
		return s.initialize()
	case protocol.OpBoundary:
		return s.boundary(req.Step)
	case protocol.OpAppCheckpoint:
		s.appCheckpointWritten(req.Files, req.Error)
		return nil
	default:
		return protocol.Refusal{Reason: fmt.Sprintf("unknown op %q", req.Op)}
	}
}

// status copies the fields it reports, because the reply is sent after s.mu is released.
func (s *Supervisor) status() protocol.StatusReply {
	reply := protocol.StatusReply{State: string(s.state), Step: s.step}
	if s.state == Checkpointed {
		location := s.attempt.root
		reply.Location = &location
	}
	if s.err != "" {
		reason := s.err
		reply.Error = &reason
	}
	return reply
}

// initialize answers the training hook's first message: where training begins.
func (s *Supervisor) initialize() protocol.InitializeReply {
	if s.state == Restoring {
		s.state = Running
		if s.attempt != nil {
			// A hib checkpoint arrived during Restoring and waits. Start it now.
			s.state = Checkpointing
		}
	}
	return protocol.InitializeReply{Resume: s.resume}
}

func (s *Supervisor) boundary(step int) any {
	s.step = step
	switch s.state {
	case Running:
		return protocol.BoundaryReply{Pause: false}
	case Checkpointing:
		return s.pause()
	default:
		return protocol.Refusal{Reason: fmt.Sprintf("the workload is %s, so training stops here", s.state)}
	}
}
