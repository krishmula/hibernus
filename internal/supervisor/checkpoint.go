package supervisor

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/manifest"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/store"
)

// defaultDeadline applies when a checkpoint request carries no deadline.
const defaultDeadline = 90 * time.Second

// maxGiveUpMargin caps how long before the deadline the Supervisor gives up on a checkpoint.
const maxGiveUpMargin = 5 * time.Second

// retryWait is how long the first retry of a failed upload waits. It doubles up to maxRetryWait.
// A variable so tests can shorten it.
var retryWait = 500 * time.Millisecond

const maxRetryWait = 5 * time.Second

// attempt is one checkpoint. Every hib checkpoint that arrives while it runs waits for the same answer.
type attempt struct {
	root     string        // the restore point folder, chosen at the update boundary
	deadline time.Time     // when the attempt gives up; an upload's context ends then
	done     chan struct{} // closed once result is set
	result   protocol.CheckpointReply
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
	// Start a checkpoint attempt or join the one already underway.
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
		// Queue the checkpoint. initialize begins it when Restoring ends.
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
	a.deadline = giveUpAt(now, deadline)
	s.attempt = a
	time.AfterFunc(a.deadline.Sub(now), func() { s.giveUp(a) })
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
	root := layout.LocalDir(s.cfg.Store, s.cfg.Workload, s.step)
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
// the manifest last. With no remote, the restore point is done, and the state moves to
// Checkpointed. With a remote, the restore point still has to be published, so the state stays
// Checkpointing while publish runs in the background.
func (s *Supervisor) appCheckpointWritten(files []string, hookError string) {
	a := s.attempt
	if s.state != Checkpointing || a.root == "" {
		return
	}
	if hookError != "" {
		s.fail("the training hook could not write the application checkpoint: " + hookError)
		return
	}
	m, err := s.writeManifest(a.root, files)
	if err != nil {
		s.fail("write the manifest: " + err.Error())
		return
	}
	if s.cfg.Remote == nil {
		s.markCheckpointed(a, a.root, m)
		return
	}
	rp := layout.RestorePoint{Bucket: s.cfg.Remote.Bucket, Workload: m.Workload, Step: m.Step, Attempt: layout.NewAttempt(time.Now())}
	s.uploading = true
	s.work.Add(1)
	go s.publish(a, rp, m)
}

// markCheckpointed records a finished restore point and answers every hib checkpoint that waits.
func (s *Supervisor) markCheckpointed(a *attempt, location string, m manifest.Manifest) {
	s.state = Checkpointed
	a.resolve(protocol.CheckpointReply{OK: true, Location: location, Manifest: &m})
}

// publish uploads the attempt's restore point to the remote as rp, and then finishes or fails the
// checkpoint. It runs without the lock: an upload can take minutes, and meanwhile hib status and
// the deadline timer need the lock.
func (s *Supervisor) publish(a *attempt, rp layout.RestorePoint, m manifest.Manifest) {
	defer s.work.Done()
	ctx, cancel := context.WithDeadline(context.Background(), a.deadline)
	defer cancel()
	err := publishWithRetry(ctx, s.cfg.Remote.Store, a.root, rp.Attempt)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.uploading = false
	if s.state != Checkpointing || s.attempt != a {
		return // the deadline passed during the upload, and the attempt already failed
	}
	switch {
	case err == nil:
		s.markCheckpointed(a, rp.S3URL(), m)
	case ctx.Err() != nil:
		s.fail("the deadline arrived before the restore point was uploaded: " + err.Error())
	default:
		s.fail("publish the restore point: " + err.Error())
	}
}

// publishWithRetry publishes until it works, the context ends, or an error comes that waiting
// cannot fix, such as a wrong credential. Publish is safe to repeat: a file that is already stored
// with the same content is accepted.
func publishWithRetry(ctx context.Context, st store.Store, root, attempt string) error {
	wait := retryWait
	for {
		err := store.Publish(ctx, st, root, attempt)
		if err == nil || store.Permanent(err) || ctx.Err() != nil {
			return err
		}
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return fmt.Errorf("%w (the last upload error: %v)", ctx.Err(), err)
		}
		wait = min(wait*2, maxRetryWait)
	}
}

func (s *Supervisor) writeManifest(root string, written []string) (manifest.Manifest, error) {
	if s.host == nil {
		return manifest.Manifest{}, errors.New("the training hook never sent initialize, so the fingerprint is unknown")
	}
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
		Fingerprint:   s.host.fingerprint,
		Python:        s.host.python,
		Command:       s.cfg.Command,
		Layers: manifest.Layers{
			ProcessSnapshot:       manifest.Layer{Status: manifest.StatusFailed, Error: &notBuilt, Files: []manifest.File{}},
			ApplicationCheckpoint: manifest.Layer{Status: manifest.StatusOK, Files: files},
		},
	}
	return m, manifest.Write(root, m)
}
