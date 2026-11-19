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
	"github.com/krishmula/hibernus/internal/proctree"
	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/snapshot"
	"github.com/krishmula/hibernus/internal/store"
)

// defaultDeadline applies when a checkpoint request carries no deadline.
const defaultDeadline = 90 * time.Second

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

	// With a process snapshot, the hook's report waits here until the hook closes its socket.
	snapshot  bool     // the hook was told to close its socket, and a dump follows
	staged    bool     // the hook reported, and the dump has not started
	files     []string // the application checkpoint files the hook reported
	hookError string
}

func newAttempt() *attempt {
	return &attempt{done: make(chan struct{})}
}

// resolve sets the answer once. Later calls do nothing. The caller holds the Supervisor's lock.
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
		a := newAttempt()
		a.resolve(protocol.CheckpointReply{State: string(Finished), Error: "training already finished"})
		return a
	case Failed:
		a := newAttempt()
		a.resolve(protocol.CheckpointReply{State: string(Failed), Error: s.err})
		return a
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

// startAttempt makes a new attempt and sets a timer that gives up on it near the deadline.
// The caller holds s.mu.
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

// giveUpAt keeps a tenth of the time left, at most 5 seconds, so the answer reaches the
// Disruption handler before its deadline.
func giveUpAt(now, deadline time.Time) time.Time {
	margin := min(deadline.Sub(now)/10, 5*time.Second)
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
// tells the training hook to write the application checkpoint into it. The caller holds s.mu.
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
	s.attempt.snapshot = s.snapshotting()
	return protocol.BoundaryReply{Pause: true, Dir: app, Snapshot: s.attempt.snapshot}
}

// snapshotting reports whether this checkpoint takes a process snapshot. The caller holds s.mu.
func (s *Supervisor) snapshotting() bool {
	return s.cfg.Snapshotter != nil && s.childPID != 0
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
// the manifest last and moves to Checkpointed. The caller holds s.mu.
func (s *Supervisor) appCheckpointWritten(files []string, hookError string) {
	if s.state != Checkpointing || s.attempt.root == "" {
		return
	}
	if s.attempt.snapshot {
		// Hold the report. CRIU can't dump while the hook's socket is open, so the dump starts
		// when the hook closes it. See finishCheckpoint.
		s.attempt.staged = true
		s.attempt.files = files
		s.attempt.hookError = hookError
		return
	}
	if hookError != "" {
		s.fail("the training hook could not write the application checkpoint: " + hookError)
		return
	}
	notBuilt := "not built yet"
	s.completeCheckpoint(files, failedLayer(notBuilt))
}

// finishCheckpoint runs when the connection that carried app_checkpoint closes. It dumps the
// training process, and then completes the checkpoint with both layers. A dump that fails still
// completes the checkpoint, from the application checkpoint alone.
func (s *Supervisor) finishCheckpoint() {
	s.mu.Lock()
	a := s.attempt
	if a == nil || !a.staged || s.state != Checkpointing {
		s.mu.Unlock()
		return
	}
	a.staged = false
	if a.hookError != "" {
		s.fail("the training hook could not write the application checkpoint: " + a.hookError)
		s.mu.Unlock()
		return
	}
	s.dumping = true
	s.work.Add(1)
	root, pid, step, files := a.root, s.childPID, s.step, a.files
	s.mu.Unlock()
	defer s.work.Done()

	s.logTree(fmt.Sprintf("before-checkpoint step=%d training-pid=%d", step, pid))
	dumpErr := s.cfg.Snapshotter.Dump(pid, filepath.Join(root, snapshot.ImageDir))
	s.logTree("after-dump")

	s.mu.Lock()
	defer s.mu.Unlock()
	s.dumping = false
	if s.state != Checkpointing || s.attempt != a {
		return // the deadline passed while CRIU worked, and the attempt already failed
	}
	s.completeCheckpoint(files, processLayer(root, dumpErr))
}

// processLayer describes the process snapshot after a dump: its image files, or why it failed.
func processLayer(root string, dumpErr error) manifest.Layer {
	if dumpErr != nil {
		return failedLayer(dumpErr.Error())
	}
	paths, err := snapshot.ImageFiles(root)
	if err != nil {
		return failedLayer(err.Error())
	}
	files, err := manifest.Describe(root, paths)
	if err != nil {
		return failedLayer(err.Error())
	}
	return manifest.Layer{Status: manifest.StatusOK, Files: files}
}

func failedLayer(reason string) manifest.Layer {
	return manifest.Layer{Status: manifest.StatusFailed, Error: &reason, Files: []manifest.File{}}
}

// completeCheckpoint writes the manifest last. With no remote, the restore point is done, and the
// state moves to Checkpointed. With a remote, the restore point still has to be published, so the
// state stays Checkpointing while publish runs in the background. The caller holds s.mu.
func (s *Supervisor) completeCheckpoint(appFiles []string, process manifest.Layer) {
	a := s.attempt
	m, err := s.writeManifest(a.root, appFiles, process)
	if err != nil {
		s.fail("write the manifest: " + err.Error())
		return
	}
	if s.cfg.Remote == nil {
		s.markCheckpointed(a, a.root, m)
		return
	}
	s.uploading = true
	s.work.Add(1)
	go s.publish(a, a.root, m)
}

// markCheckpointed records a finished restore point and answers every hib checkpoint that waits.
// The caller holds s.mu.
func (s *Supervisor) markCheckpointed(a *attempt, location string, m manifest.Manifest) {
	s.state = Checkpointed
	s.location = location
	a.resolve(protocol.CheckpointReply{OK: true, Location: location, Manifest: &m})
}

// publish uploads the restore point in root to the remote, and then finishes or fails the
// checkpoint. It runs without the lock: an upload can take minutes, and meanwhile hib status and
// the deadline timer need the lock.
func (s *Supervisor) publish(a *attempt, root string, m manifest.Manifest) {
	defer s.work.Done()
	ctx, cancel := context.WithDeadline(context.Background(), a.deadline)
	defer cancel()
	err := publishWithRetry(ctx, s.cfg.Remote.Store, root)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.uploading = false
	if s.state != Checkpointing || s.attempt != a {
		return // the deadline passed during the upload, and the attempt already failed
	}
	switch {
	case err == nil:
		s.markCheckpointed(a, s.cfg.Remote.Location(m.Workload, m.Step), m)
	case ctx.Err() != nil:
		s.fail("the deadline arrived before the restore point was uploaded: " + err.Error())
	default:
		s.fail("publish the restore point: " + err.Error())
	}
}

// publishWithRetry publishes until it works, the context ends, or an error comes that waiting
// cannot fix, such as a wrong credential. Publish is safe to repeat: a file that is already stored
// with the same content is accepted.
func publishWithRetry(ctx context.Context, st store.Store, root string) error {
	wait := retryWait
	for {
		err := store.Publish(ctx, st, root)
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

// resumedFromSnapshot runs when a restored hook has woken, reconnected, and says so. The caller
// holds s.mu.
func (s *Supervisor) resumedFromSnapshot() any {
	if s.state == Restoring {
		s.start() // the same move as the first message of a fresh process
		s.logTree(fmt.Sprintf("after-restore step=%d training-pid=%d", s.step, s.childPID))
		s.resumedOnce.Do(func() { close(s.resumed) })
	}
	return protocol.ResumedReply{OK: true}
}

// LogTree writes the process tree under the Supervisor to the tree log. `hib restore` calls it
// before it starts CRIU.
func (s *Supervisor) LogTree(label string) { s.logTree(label) }

// logTree writes the process tree under the Supervisor to the tree log.
func (s *Supervisor) logTree(label string) {
	tree, err := proctree.Snapshot(s.cfg.ProcRoot, s.cfg.SelfPID)
	if logErr := proctree.Append(s.cfg.TreeLog, label, tree, err); logErr != nil {
		fmt.Fprintf(os.Stderr, "hib: write the process tree log: %v\n", logErr)
	}
}

func (s *Supervisor) writeManifest(root string, written []string, process manifest.Layer) (manifest.Manifest, error) {
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

	m := manifest.Manifest{
		SchemaVersion: manifest.SchemaVersion,
		Workload:      s.cfg.Workload,
		Step:          s.step,
		CreatedAt:     time.Now().UTC().Truncate(time.Second),
		Fingerprint:   map[string]string{},
		Command:       s.cfg.Command,
		Layers: manifest.Layers{
			ProcessSnapshot:       process,
			ApplicationCheckpoint: manifest.Layer{Status: manifest.StatusOK, Files: files},
		},
	}
	return m, manifest.Write(root, m)
}
