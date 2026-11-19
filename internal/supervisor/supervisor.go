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
	"github.com/krishmula/hibernus/internal/snapshot"
	"github.com/krishmula/hibernus/internal/store"
)

// maxSocketPath stays under the smallest limit Unix systems put on a socket's path (104 bytes on macOS).
const maxSocketPath = 100

// Config is what a Supervisor knows about its workload from the start.
type Config struct {
	Workload string
	Store    string           // the folder that holds restore points, one subfolder per workload
	Command  manifest.Command // how the training process started, recorded in each manifest

	// Snapshotter takes the process snapshot layer. Nil means the application checkpoint only.
	Snapshotter snapshot.Snapshotter
	// Remote is where a checkpoint is published before it counts as done. Nil means the restore
	// point stays in Store, and a checkpoint is done as soon as its manifest is written.
	Remote *Remote
	// TreeLog is the file that gets the process tree before a checkpoint, after the dump, and
	// after a restore. Empty means no tree logging.
	TreeLog string
	// ProcRoot and SelfPID default to /proc and this process. Tests point them at a fake /proc.
	ProcRoot string
	SelfPID  int
}

// Remote is durable storage for restore points, such as an S3 bucket.
type Remote struct {
	Store store.Store
	// Location is the address of a published restore point, such as s3://bucket/workload/step-1200/.
	// hib checkpoint prints it, and hib restore takes it.
	Location func(workload string, step int) string
}

type Supervisor struct {
	cfg Config

	mu       sync.Mutex
	state    State
	step     int
	resume   *protocol.Resume // what start answers; nil for a fresh run
	attempt  *attempt         // the latest checkpoint, in progress or finished
	location string           // the restore point folder, once Checkpointed
	err      string           // why the workload Failed

	childPID    int            // the training process, the root of what a dump captures
	dumping     bool           // CRIU is ending the training process on purpose
	uploading   bool           // the restore point is being published; the training process may already be gone
	work        sync.WaitGroup // dumps and uploads in flight, so `hib run` doesn't exit under one
	resumed     chan struct{}  // closed when a restored hook reconnects
	resumedOnce sync.Once
}

// New makes a Supervisor. With resume nil, it starts in Running. With resume set, it starts in
// Restoring and hands resume to the training hook when the hook asks start.
func New(cfg Config, resume *protocol.Resume) *Supervisor {
	if cfg.ProcRoot == "" {
		cfg.ProcRoot = "/proc"
	}
	if cfg.SelfPID == 0 {
		cfg.SelfPID = os.Getpid()
	}
	s := &Supervisor{cfg: cfg, state: Running, resumed: make(chan struct{})}
	if resume != nil {
		s.state = Restoring
		s.step = resume.Step
		s.resume = resume
	}
	return s
}

// SetChild records the training process, the root of the tree a dump captures. A restore calls it
// again with the restored process.
func (s *Supervisor) SetChild(pid int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.childPID = pid
}

// Resumed is closed when a restored training hook reconnects and sends "resumed".
func (s *Supervisor) Resumed() <-chan struct{} { return s.resumed }

// Checkpointed reports whether a restore point is complete, so `hib run` can exit cleanly when
// the dump ended the training process.
func (s *Supervisor) Checkpointed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == Checkpointed
}

// Settle waits for a dump or an upload in flight. `hib run` calls it after the training process
// exits, so that it doesn't leave while CRIU still writes the images or the upload still runs.
func (s *Supervisor) Settle() { s.work.Wait() }

func (s *Supervisor) ChildExited(code int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dumping || s.uploading {
		// CRIU ends the tree after the dump. finishCheckpoint decides what that means. While the
		// upload runs, the restore point is already complete, so the process ending does not
		// change it.
		return
	}
	next := afterChildExit(s.state, code)
	if next == Failed && s.state != Failed {
		s.fail(fmt.Sprintf("the training process exited with code %d", code))
		return
	}
	s.state = next
}

// fail moves to Failed and answers every hib checkpoint that waits. The caller holds s.mu.
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
	sentAppCheckpoint := false
	for scanner.Scan() {
		var req protocol.Request
		var reply any
		if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
			reply = protocol.Refusal{Reason: "bad message: " + err.Error()}
		} else {
			reply = s.handle(req)
			sentAppCheckpoint = sentAppCheckpoint || req.Op == "app_checkpoint"
		}
		if reply == nil {
			continue
		}
		if err := encoder.Encode(reply); err != nil {
			break
		}
	}
	// The hook that wrote the application checkpoint closes this connection before CRIU may dump
	// it. Only this connection's end starts the dump, because another client's end proves nothing
	// about the hook's socket.
	if sentAppCheckpoint {
		conn.Close()
		s.finishCheckpoint()
	}
}

// handle answers one message. A nil answer means no reply.
func (s *Supervisor) handle(req protocol.Request) any {
	if req.Op == "checkpoint" {
		// It waits until the checkpoint ends, so it must not hold the lock while it waits.
		return s.checkpoint(req.Deadline)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	switch req.Op {
	case "status":
		reply := protocol.StatusReply{State: string(s.state), Step: s.step}
		if s.location != "" {
			reply.Location = &s.location
		}
		if s.err != "" {
			reply.Error = &s.err
		}
		return reply
	case "start":
		return s.start()
	case "boundary":
		return s.boundary(req.Step)
	case "app_checkpoint":
		s.appCheckpointWritten(req.Files, req.Error)
		return nil
	case "resumed":
		return s.resumedFromSnapshot()
	default:
		return protocol.Refusal{Reason: fmt.Sprintf("unknown op %q", req.Op)}
	}
}

// start answers the training hook's first message: where training begins.
func (s *Supervisor) start() protocol.StartReply {
	if s.state == Restoring {
		s.state = Running
		if s.attempt != nil {
			// A hib checkpoint arrived during Restoring and waits. Start it now.
			s.state = Checkpointing
		}
	}
	return protocol.StartReply{Resume: s.resume}
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
