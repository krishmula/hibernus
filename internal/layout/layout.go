// Package layout says where a restore point lives. The local folder and the S3 location share one
// relative path, <workload>/step-<K>, so every other package asks here and never builds the path itself.
package layout

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const stepPrefix = "step-"

var (
	workloadName = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)
	// bucketName is S3's rule in short: 3 to 63 lowercase letters, digits, dots, and hyphens.
	bucketName = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$`)
)

// RestorePoint names one restore point in S3.
type RestorePoint struct {
	Bucket   string
	Workload string
	Step     int
}

// ValidWorkload reports whether name can be a workload name and a folder name. A dot or two dots
// would match the character rule but point at the folder itself or its parent, so they are refused.
func ValidWorkload(name string) bool {
	return name != "." && name != ".." && workloadName.MatchString(name)
}

// ValidBucket reports whether name is a valid S3 bucket name, in the short form of S3's rule.
func ValidBucket(name string) bool { return bucketName.MatchString(name) }

func stepDir(step int) string { return stepPrefix + strconv.Itoa(step) } // never padded

// Key is the restore point's path relative to the store or the bucket, with forward slashes.
func Key(workload string, step int) string {
	return workload + "/" + stepDir(step)
}

// LocalDir is the restore point's folder under a local store.
func LocalDir(store, workload string, step int) string {
	return filepath.Join(store, workload, stepDir(step))
}

// S3URL is the restore point's location in S3. It ends with a slash, because it is a folder.
func S3URL(bucket, workload string, step int) string {
	return "s3://" + bucket + "/" + Key(workload, step) + "/"
}

// ParseS3URL reads a location such as s3://my-bucket/demo/step-1200/. It accepts only the folder of
// one restore point, in the one spelling that S3URL writes, so two names never mean one folder.
func ParseS3URL(url string) (RestorePoint, error) {
	rest, ok := strings.CutPrefix(url, "s3://")
	if !ok {
		return RestorePoint{}, fmt.Errorf("%q is not an s3:// location", url)
	}
	parts := strings.Split(strings.TrimSuffix(rest, "/"), "/")
	if len(parts) != 3 {
		return RestorePoint{}, fmt.Errorf("%q should look like s3://<bucket>/<workload>/step-<K>/, or s3://<bucket>/<workload>/ for the newest", url)
	}
	bucket, workload, dir := parts[0], parts[1], parts[2]
	if !bucketName.MatchString(bucket) {
		return RestorePoint{}, fmt.Errorf("%q: %q is not a valid bucket name", url, bucket)
	}
	if !ValidWorkload(workload) {
		return RestorePoint{}, fmt.Errorf("%q: %q is not a valid workload name", url, workload)
	}
	step, ok := ParseStepDir(dir)
	if !ok {
		return RestorePoint{}, fmt.Errorf("%q: %q is not a step folder such as step-1200", url, dir)
	}
	return RestorePoint{Bucket: bucket, Workload: workload, Step: step}, nil
}

// ParseStepDir reads a folder name such as step-1200. It accepts only the spelling that Key
// writes, so two names never mean one folder.
func ParseStepDir(name string) (int, bool) {
	digits, ok := strings.CutPrefix(name, stepPrefix)
	step, err := strconv.Atoi(digits)
	if !ok || err != nil || step < 0 || strconv.Itoa(step) != digits {
		return 0, false
	}
	return step, true
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
