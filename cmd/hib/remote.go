package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/krishmula/hibernus/internal/layout"
	"github.com/krishmula/hibernus/internal/store"
	"github.com/krishmula/hibernus/internal/supervisor"
)

// openBucket opens the store for an S3 bucket. Credentials and the region come from the standard
// AWS chain. A test replaces it with an in-memory store.
var openBucket = func(ctx context.Context, bucket string) (store.Store, error) {
	return store.NewS3(ctx, bucket, "")
}

// parseRemoteFlag reads --remote, which names a bucket as s3://<bucket>. A folder inside the bucket
// is not supported yet, because an s3:// restore point location has no room for one.
func parseRemoteFlag(value string) (string, error) {
	rest, ok := strings.CutPrefix(value, "s3://")
	bucket := strings.TrimSuffix(rest, "/")
	if !ok || !layout.ValidBucket(bucket) {
		return "", fmt.Errorf("--remote %q should look like s3://<bucket>", value)
	}
	return bucket, nil
}

// remoteFor makes the Supervisor's remote for a bucket, where a checkpoint is published.
func remoteFor(ctx context.Context, bucket string) (*supervisor.Remote, error) {
	st, err := openBucket(ctx, bucket)
	if err != nil {
		return nil, err
	}
	return &supervisor.Remote{Store: st, Bucket: bucket}, nil
}

// refuseTakenWorkload returns an error when the bucket already holds a complete restore point of
// workload. A fresh hib run under that name would mix its restore points with the old run's, and
// s3://<bucket>/<workload>/ could then restore the old run.
func refuseTakenWorkload(ctx context.Context, remote *supervisor.Remote, workload string) error {
	rp, found, err := store.NewestComplete(ctx, remote.Store, workload)
	if err != nil {
		return fmt.Errorf("check s3://%s/%s/ for earlier restore points: %w", remote.Bucket, workload, err)
	}
	if found {
		rp.Bucket = remote.Bucket
		return fmt.Errorf("s3://%s/%s/ already holds restore point %s from an earlier run; choose another --workload, or continue that run with hib restore s3://%s/%s/",
			remote.Bucket, workload, rp.S3URL(), remote.Bucket, workload)
	}
	return nil
}

// locateRestorePoint returns the local folder that holds the restore point named by arg. A folder
// is used as it is. An s3://<bucket>/<workload>/step-<K>-<attempt>/ location is downloaded and verified into
// the store folder. An s3://<bucket>/<workload>/ location means the newest restore point of that
// workload: the newest one that downloads and verifies, passing over any that is incomplete or
// corrupt. For both, the remote it came from is returned too, so a restored workload that is
// checkpointed again publishes to the same bucket.
func locateRestorePoint(ctx context.Context, arg, storeDir string) (string, *supervisor.Remote, error) {
	if !strings.HasPrefix(arg, "s3://") {
		root, err := filepath.Abs(arg)
		return root, nil, err
	}
	if rp, err := layout.ParseS3URL(arg); err == nil {
		remote, err := remoteFor(ctx, rp.Bucket)
		if err != nil {
			return "", nil, err
		}
		root := layout.LocalDir(storeDir, rp.Workload, rp.Step)
		if _, err := store.Fetch(ctx, remote.Store, rp, root); err != nil {
			return "", nil, fmt.Errorf("download %s: %w", arg, err)
		}
		return root, remote, nil
	}
	bucket, workload, err := layout.ParseS3Workload(arg)
	if err != nil {
		_, exactErr := layout.ParseS3URL(arg) // its message names both accepted forms
		return "", nil, exactErr
	}
	remote, err := remoteFor(ctx, bucket)
	if err != nil {
		return "", nil, err
	}
	latest, err := store.FetchLatest(ctx, remote.Store, workload, func(step int) string {
		return layout.LocalDir(storeDir, workload, step)
	})
	for _, skip := range latest.Skipped {
		fmt.Fprintf(os.Stderr, "hib restore: passed over %s: %v\n", skip.Point.Key(), skip.Reason)
	}
	if err != nil {
		return "", nil, fmt.Errorf("find the newest restore point of %s in s3://%s: %w", workload, bucket, err)
	}
	latest.Point.Bucket = bucket
	fmt.Fprintf(os.Stderr, "hib restore: the newest usable restore point is %s\n", latest.Point.S3URL())
	return latest.Root, remote, nil
}
