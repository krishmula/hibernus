package store

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"path"
	"slices"
	"strings"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/manifest"
)

// Skip is a restore point that FetchLatest passed over, and why.
type Skip struct {
	Point  layout.RestorePoint
	Reason error
}

// Latest is the restore point that FetchLatest brought back.
type Latest struct {
	Point    layout.RestorePoint
	Root     string // the local folder it was downloaded into
	Manifest manifest.Manifest
	Skipped  []Skip // newer restore points that could not be used, newest first
}

// List lists a workload's restore point folders, newest first: the highest step first, and within
// one step the latest attempt first. A folder counts whether or not it holds a manifest: only a
// download shows whether it is usable. Folders that are not named step-<K>-<attempt> are ignored.
// The restore points have no bucket set.
func List(ctx context.Context, st Store, workload string) ([]layout.RestorePoint, error) {
	if !layout.ValidWorkload(workload) {
		return nil, fmt.Errorf("%q is not a valid workload name", workload)
	}
	names, err := st.Folders(ctx, workload)
	if err != nil {
		return nil, err
	}
	var points []layout.RestorePoint
	for _, name := range names {
		if step, attempt, ok := layout.ParseFolder(name); ok {
			points = append(points, layout.RestorePoint{Workload: workload, Step: step, Attempt: attempt})
		}
	}
	slices.SortFunc(points, func(a, b layout.RestorePoint) int {
		return cmp.Or(cmp.Compare(b.Step, a.Step), strings.Compare(b.Attempt, a.Attempt))
	})
	return points, nil
}

// NewestComplete returns the newest restore point of workload that has a manifest in the store,
// without downloading it. found is false when the workload has none.
func NewestComplete(ctx context.Context, st Store, workload string) (rp layout.RestorePoint, found bool, err error) {
	points, err := List(ctx, st, workload)
	if err != nil {
		return layout.RestorePoint{}, false, err
	}
	for _, rp := range points {
		_, err := st.Stat(ctx, path.Join(rp.Key(), manifest.FileName))
		if err == nil {
			return rp, true, nil
		}
		if !errors.Is(err, ErrNotFound) {
			return layout.RestorePoint{}, false, err
		}
	}
	return layout.RestorePoint{}, false, nil
}

// FetchLatest downloads the newest restore point of workload that downloads and verifies. It goes
// down from the newest, and passes over a restore point that is bad as stored, such as one whose
// upload was cut off. It stops at once on any other error, such as a permission or network error:
// that says nothing about the restore point, and trying an older one would hide the real problem.
// rootFor names the local folder for a step. The result lists what was skipped, even on failure.
func FetchLatest(ctx context.Context, st Store, workload string, rootFor func(step int) string) (Latest, error) {
	points, err := List(ctx, st, workload)
	if err != nil {
		return Latest{}, err
	}
	var latest Latest
	for _, rp := range points {
		root := rootFor(rp.Step)
		m, err := Fetch(ctx, st, rp, root)
		if err == nil {
			latest.Point, latest.Root, latest.Manifest = rp, root, m
			return latest, nil
		}
		if !errors.Is(err, ErrBadRestorePoint) {
			return latest, err
		}
		latest.Skipped = append(latest.Skipped, Skip{Point: rp, Reason: err})
	}
	return latest, fmt.Errorf("%w: %s has %d restore point folders, and none downloads and verifies", ErrNoRestorePoint, workload, len(points))
}
