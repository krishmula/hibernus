package store

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/manifest"
)

// Skip is a restore point that FetchLatest passed over, and why.
type Skip struct {
	Step   int
	Reason error
}

// Latest is the restore point that FetchLatest brought back.
type Latest struct {
	Step     int
	Root     string // the local folder it was downloaded into
	Manifest manifest.Manifest
	Skipped  []Skip // newer restore points that could not be used, newest first
}

// Steps lists the steps of a workload's restore point folders, newest first. A folder counts
// whether or not it holds a manifest: only a download shows whether it is usable. Folders that
// are not named step-<K> are ignored.
func Steps(ctx context.Context, st Store, workload string) ([]int, error) {
	if !layout.ValidWorkload(workload) {
		return nil, fmt.Errorf("%q is not a valid workload name", workload)
	}
	names, err := st.Folders(ctx, workload)
	if err != nil {
		return nil, err
	}
	var steps []int
	for _, name := range names {
		if step, ok := layout.ParseStepDir(name); ok {
			steps = append(steps, step)
		}
	}
	slices.Sort(steps)
	slices.Reverse(steps)
	return steps, nil
}

// FetchLatest downloads the newest restore point of workload that downloads and verifies. It goes
// down from the newest, and passes over a restore point that is bad as stored, such as one whose
// upload was cut off. It stops at once on any other error, such as a permission or network error:
// that says nothing about the restore point, and trying an older one would hide the real problem.
// rootFor names the local folder for a step. The result lists what was skipped, even on failure.
func FetchLatest(ctx context.Context, st Store, workload string, rootFor func(step int) string) (Latest, error) {
	steps, err := Steps(ctx, st, workload)
	if err != nil {
		return Latest{}, err
	}
	var latest Latest
	for _, step := range steps {
		root := rootFor(step)
		m, err := Fetch(ctx, st, workload, step, root)
		if err == nil {
			latest.Step, latest.Root, latest.Manifest = step, root, m
			return latest, nil
		}
		if !errors.Is(err, ErrBadRestorePoint) {
			return latest, err
		}
		latest.Skipped = append(latest.Skipped, Skip{Step: step, Reason: err})
	}
	return latest, fmt.Errorf("%w: %s has %d restore point folders, and none downloads and verifies", ErrNoRestorePoint, workload, len(steps))
}
