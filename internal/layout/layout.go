// Package layout says where a restore point lives, so every other package asks here and never
// builds the path itself. The local folder is <workload>/step-<K>. In S3 the folder is
// <workload>/step-<K>-<attempt>: stored files never change, so each checkpoint attempt needs keys
// that an earlier attempt at the same step, even one cut off halfway, cannot hold.
package layout

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const stepPrefix = "step-"

// attemptTime is the first part of an attempt. It sorts as text in time order.
const attemptTime = "20060102T150405Z"

var (
	workloadName = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
	attemptName  = regexp.MustCompile(`^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$`)
	// bucketName is S3's rule in short: 3 to 63 lowercase letters, digits, dots, and hyphens.
	bucketName = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$`)
)

// RestorePoint names one restore point in S3: one checkpoint attempt of a workload at a step.
type RestorePoint struct {
	Bucket   string
	Workload string
	Step     int
	Attempt  string // from NewAttempt
}

// NewAttempt names a checkpoint attempt that starts at now: the UTC time, then random hex, such
// as 20261119T163000Z-3fa94c1e. Attempts sort as text in time order.
func NewAttempt(now time.Time) string {
	random := make([]byte, 4)
	rand.Read(random) // never returns an error
	return now.UTC().Format(attemptTime) + "-" + hex.EncodeToString(random)
}

// ValidWorkload reports whether name can be a workload name and a folder name. A dot or two dots
// would match the character rule but point at the folder itself or its parent, so they are refused.
func ValidWorkload(name string) bool {
	return name != "." && name != ".." && workloadName.MatchString(name)
}

// ValidBucket reports whether name is a valid S3 bucket name, in the short form of S3's rule.
func ValidBucket(name string) bool { return bucketName.MatchString(name) }

func stepDir(step int) string { return stepPrefix + strconv.Itoa(step) } // never padded

// Key is the restore point's folder relative to the bucket, with forward slashes, such as
// demo/step-1200-20261119T163000Z-3fa94c1e.
func (rp RestorePoint) Key() string {
	return rp.Workload + "/" + stepDir(rp.Step) + "-" + rp.Attempt
}

// LocalDir is the restore point's folder under a local store.
func LocalDir(store, workload string, step int) string {
	return filepath.Join(store, workload, stepDir(step))
}

// S3URL is the restore point's location in S3. It ends with a slash, because it is a folder.
func (rp RestorePoint) S3URL() string {
	return "s3://" + rp.Bucket + "/" + rp.Key() + "/"
}

// ParseS3URL reads a location such as s3://my-bucket/demo/step-1200-20261119T163000Z-3fa94c1e/.
// It accepts only the folder of one restore point, in the one spelling that S3URL writes, so two
// names never mean one folder.
func ParseS3URL(url string) (RestorePoint, error) {
	rest, ok := strings.CutPrefix(url, "s3://")
	if !ok {
		return RestorePoint{}, fmt.Errorf("%q is not an s3:// location", url)
	}
	parts := strings.Split(strings.TrimSuffix(rest, "/"), "/")
	if len(parts) != 3 {
		return RestorePoint{}, fmt.Errorf("%q should look like s3://<bucket>/<workload>/step-<K>-<attempt>/, or s3://<bucket>/<workload>/ for the newest", url)
	}
	bucket, workload, dir := parts[0], parts[1], parts[2]
	if !bucketName.MatchString(bucket) {
		return RestorePoint{}, fmt.Errorf("%q: %q is not a valid bucket name", url, bucket)
	}
	if !ValidWorkload(workload) {
		return RestorePoint{}, fmt.Errorf("%q: %q is not a valid workload name", url, workload)
	}
	step, attempt, ok := ParseFolder(dir)
	if !ok {
		return RestorePoint{}, fmt.Errorf("%q: %q is not a restore point folder such as step-1200-20261119T163000Z-3fa94c1e", url, dir)
	}
	return RestorePoint{Bucket: bucket, Workload: workload, Step: step, Attempt: attempt}, nil
}

// ParseFolder reads a restore point folder name such as step-1200-20261119T163000Z-3fa94c1e. It
// accepts only the spelling that Key writes, so two names never mean one folder.
func ParseFolder(name string) (step int, attempt string, ok bool) {
	rest, ok := strings.CutPrefix(name, stepPrefix)
	digits, attempt, ok2 := strings.Cut(rest, "-")
	step, err := strconv.Atoi(digits)
	if !ok || !ok2 || err != nil || step < 0 || strconv.Itoa(step) != digits || !attemptName.MatchString(attempt) {
		return 0, "", false
	}
	return step, attempt, true
}

// ParseS3Workload reads s3://<bucket>/<workload>/, which names every restore point of a workload.
// hib restore takes it to mean the newest one.
func ParseS3Workload(url string) (bucket, workload string, err error) {
	rest, ok := strings.CutPrefix(url, "s3://")
	if !ok {
		return "", "", fmt.Errorf("%q is not an s3:// location", url)
	}
	parts := strings.Split(strings.TrimSuffix(rest, "/"), "/")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("%q should look like s3://<bucket>/<workload>/", url)
	}
	if !bucketName.MatchString(parts[0]) {
		return "", "", fmt.Errorf("%q: %q is not a valid bucket name", url, parts[0])
	}
	if !ValidWorkload(parts[1]) {
		return "", "", fmt.Errorf("%q: %q is not a valid workload name", url, parts[1])
	}
	return parts[0], parts[1], nil
}
