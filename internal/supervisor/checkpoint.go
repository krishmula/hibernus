package supervisor

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
)

// defaultDeadline applies when a checkpoint request carries no deadline.
const defaultDeadline = 90 * time.Second

// maxGiveUpMargin caps how long before the deadline the Supervisor gives up on a checkpoint.
const maxGiveUpMargin = 5 * time.Second

// attempt is one checkpoint. Every hib checkpoint that arrives while it runs waits for the same answer.
type attempt struct {
	root   string        // the restore point folder, chosen at the update boundary
	done   chan struct{} // closed once result is set
	result protocol.CheckpointReply
}

func newAttempt() *attempt {
	return &attempt{done: make(chan struct{})}
}

// resolve sets the answer once. Later calls do nothing.
func (a *attempt) resolve(result protocol.CheckpointReply) {
	select {
	case <-a.done:
		return
	default:
	}
	a.result = result
	close(a.done)
}

// checkpoint handles one hib checkpoint request and waits for its answer.
func (s *Supervisor) checkpoint(deadline time.Time) protocol.CheckpointReply {
	a := s.join(deadline)
	<-a.done
	return a.result
}

// join returns the attempt this request waits for. It starts one when none runs.
func (s *Supervisor) join(deadline time.Time) *attempt {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch s.state {
	case Finished:
		return answered(protocol.CheckpointReply{State: string(Finished), Error: "training already finished"})
	case Failed:
		return answered(protocol.CheckpointReply{State: string(Failed), Error: s.err})
	case Checkpointing, Checkpointed:
		// Wait for the checkpoint in progress, or get the answer of the one that finished.
	case Running:
		// Switch state before any checkpoint work, so a second request joins this one.
		s.state = Checkpointing
		s.startAttempt(deadline)
	case Restoring:
		// Queue the checkpoint. start begins it when Restoring ends.
		if s.attempt == nil {
			s.startAttempt(deadline)
		}
	}
	return s.attempt
}

// answered returns an attempt that already holds its answer, for a request the Supervisor answers at once.
func answered(reply protocol.CheckpointReply) *attempt {
	a := newAttempt()
	a.resolve(reply)
	return a
}

// startAttempt makes a new attempt and sets a timer that gives up on it near the deadline.
func (s *Supervisor) startAttempt(deadline time.Time) {
	now := time.Now()
	if deadline.IsZero() {
		deadline = now.Add(defaultDeadline)
	}
	a := newAttempt()
	s.attempt = a
	time.AfterFunc(giveUpAt(now, deadline).Sub(now), func() { s.giveUp(a) })
}

// giveUpAt keeps a tenth of the time left, at most maxGiveUpMargin, so the answer reaches the
// Disruption handler before its deadline.
func giveUpAt(now, deadline time.Time) time.Time {
	margin := min(deadline.Sub(now)/10, maxGiveUpMargin)
	return deadline.Add(-max(margin, 0))
}

func (s *Supervisor) giveUp(a *attempt) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if a != s.attempt {
		return
	}
	select {
	case <-a.done:
		return
	default:
	}
	if s.state == Restoring {
		// The restore itself goes on. Only this checkpoint request fails.
		s.attempt = nil
		a.resolve(protocol.CheckpointReply{State: string(Restoring), Error: "the deadline arrived before the restore finished"})
		return
	}
	s.fail("the deadline arrived before the restore point was written")
}

// pause runs at the first update boundary in Checkpointing. It makes the restore point folder and
// tells the training hook to write the application checkpoint into it.
func (s *Supervisor) pause() any {
	if s.attempt.root != "" {
		return protocol.Refusal{Reason: "the training hook is already paused"}
	}
	root := filepath.Join(s.cfg.Store, s.cfg.Workload, fmt.Sprintf("step-%d", s.step))
	app := filepath.Join(root, "app")
	if err := prepareFolder(root, app); err != nil {
		s.fail("make the restore point folder: " + err.Error())
		return protocol.Refusal{Reason: s.err}
	}
	s.attempt.root = root
	return protocol.BoundaryReply{Pause: true, Dir: app}
}

// prepareFolder makes the folders. It removes a manifest left there by an earlier run, so the
// folder doesn't look complete before this checkpoint finishes.
func prepareFolder(root, app string) error {
	if err := os.MkdirAll(app, 0o755); err != nil {
		return err
	}
	if err := os.Remove(filepath.Join(root, manifest.FileName)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// appCheckpointWritten runs when the training hook reports the application checkpoint. It writes
// the manifest last and moves to Checkpointed.
func (s *Supervisor) appCheckpointWritten(files []string, hookError string) {
	if s.state != Checkpointing || s.attempt.root == "" {
		return
	}
	if hookError != "" {
		s.fail("the training hook could not write the application checkpoint: " + hookError)
		return
	}
	m, err := s.writeManifest(s.attempt.root, files)
	if err != nil {
		s.fail("write the manifest: " + err.Error())
		return
	}
	s.state = Checkpointed
	s.attempt.resolve(protocol.CheckpointReply{OK: true, Location: s.attempt.root, Manifest: &m})
}

func (s *Supervisor) writeManifest(root string, written []string) (manifest.Manifest, error) {
	paths := make([]string, len(written))
	for i, path := range written {
		relative, err := filepath.Rel(root, path)
		if err != nil || !filepath.IsLocal(relative) {
			return manifest.Manifest{}, fmt.Errorf("the training hook wrote %s, outside the restore point", path)
		}
		paths[i] = relative
	}
	files, err := manifest.Describe(root, paths)
	if err != nil {
		return manifest.Manifest{}, err
	}

	notBuilt := "not built yet"
	m := manifest.Manifest{
		SchemaVersion: manifest.SchemaVersion,
		Workload:      s.cfg.Workload,
		Step:          s.step,
		CreatedAt:     time.Now().UTC().Truncate(time.Second),
		Fingerprint:   map[string]string{},
		Command:       s.cfg.Command,
		Layers: manifest.Layers{
			ProcessSnapshot:       manifest.Layer{Status: manifest.StatusFailed, Error: &notBuilt, Files: []manifest.File{}},
			ApplicationCheckpoint: manifest.Layer{Status: manifest.StatusOK, Files: files},
		},
	}
	return m, manifest.Write(root, m)
}
