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
	"github.com/krishmula/hibernus/internal/store"
)

// maxSocketPath stays under the smallest limit Unix systems put on a socket's path (104 bytes on macOS).
const maxSocketPath = 100

// Config is what a Supervisor knows about its workload from the start.
type Config struct {
	Workload string
	Store    string           // the folder that holds restore points, one subfolder per workload
	Command  manifest.Command // how the training process started, recorded in each manifest

	// Remote is where a checkpoint is published before it counts as done. Nil means the restore
	// point stays in Store, and a checkpoint is done as soon as its manifest is written.
	Remote *Remote
}

// Remote is durable storage for restore points: an S3 bucket.
type Remote struct {
	Store  store.Store
	Bucket string // names the published restore point in what hib checkpoint prints
}

type Supervisor struct {
	cfg Config

	// mu guards the fields in this block. handle, join, ChildExited, giveUp, and publish take it,
	// and every method they call runs with it already held.
	mu        sync.Mutex
	state     State
	step      int
	resume    *protocol.Resume // what initialize answers; nil for a fresh run
	host      *host            // what initialize reported, recorded in each manifest; nil before initialize
	attempt   *attempt         // the latest checkpoint, in progress or finished
	err       string           // why the workload Failed
	uploading bool             // the restore point is being published; the training process may already be gone

	work sync.WaitGroup // uploads in flight, so hib run doesn't exit under one
}

// host is the checkpoint host's fingerprint and the interpreter that collected it.
type host struct {
	fingerprint manifest.Fingerprint
	python      string
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

// Settle waits for an upload in flight. hib run calls it after the training process exits, so
// that it doesn't leave while the upload still runs.
func (s *Supervisor) Settle() { s.work.Wait() }

func (s *Supervisor) ChildExited(code int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.uploading {
		// The restore point is already complete on this host, so the process ending does not
		// change it. publish decides the state when the upload ends.
		return
	}
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
		return s.initialize(req.Fingerprint, req.Python)
	case protocol.OpRestored:
		return s.restored(req.Step, req.Fingerprint, req.Python)
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
		location := s.attempt.result.Location
		reply.Location = &location
	}
	if s.err != "" {
		reason := s.err
		reply.Error = &reason
	}
	return reply
}

// initialize answers the training hook's first message: where training begins. It keeps the
// fingerprint the hook sends, for the manifests of later checkpoints. hib restore has already
// compared this host's fingerprint with the restore point's before it started the training process.
func (s *Supervisor) initialize(fingerprint *manifest.Fingerprint, python string) any {
	if refusal := s.keepHost("initialize", fingerprint, python); refusal != nil {
		return refusal
	}
	// Only a restore from the application checkpoint reaches this point in Restoring: the training
	// process starts over, and s.resume tells it which step to load. A restored process snapshot
	// never sends initialize. It wakes up already past it.
	if s.state == Restoring {
		s.endRestoring()
	}
	return protocol.InitializeReply{Resume: s.resume}
}

// restored answers a training process that a restore from the process snapshot brought back. It
// woke up at the step the restore point holds, so it never asks initialize. The training hook
// waits for this answer before it trains, so a refusal here leaves nothing half done.
func (s *Supervisor) restored(step int, fingerprint *manifest.Fingerprint, python string) any {
	if s.state != Restoring {
		return protocol.Refusal{Reason: fmt.Sprintf("restored arrives only during a restore, and the workload is %s", s.state)}
	}
	if step != s.step {
		return protocol.Refusal{Reason: fmt.Sprintf("the training process is at step %d, but the restore point holds step %d", step, s.step)}
	}
	if refusal := s.keepHost("restored", fingerprint, python); refusal != nil {
		return refusal
	}
	s.endRestoring()
	return protocol.RestoredReply{Continue: true}
}

// keepHost checks and keeps the fingerprint the training hook sends, for the manifests of later
// checkpoints.
func (s *Supervisor) keepHost(op string, fingerprint *manifest.Fingerprint, python string) *protocol.Refusal {
	if fingerprint == nil || python == "" {
		return &protocol.Refusal{Reason: op + " carries no fingerprint and python; update the training hook"}
	}
	if err := fingerprint.Complete(); err != nil {
		return &protocol.Refusal{Reason: err.Error()}
	}
	s.host = &host{fingerprint: *fingerprint, python: python}
	return nil
}

// endRestoring moves to Running, or begins a hib checkpoint that arrived during Restoring and waits.
func (s *Supervisor) endRestoring() {
	s.state = Running
	if s.attempt != nil {
		s.state = Checkpointing
	}
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
