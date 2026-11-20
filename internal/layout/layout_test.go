package layout

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const attempt = "20261119T163000Z-3fa94c1e"

func TestKeyIsWorkloadSlashStepNumberAndAttempt(t *testing.T) {
	rp := RestorePoint{Workload: "demo", Step: 1200, Attempt: attempt}
	if got, want := rp.Key(), "demo/step-1200-20261119T163000Z-3fa94c1e"; got != want {
		t.Fatalf("Key = %q, want %q", got, want)
	}
}

func TestTheStepNumberIsNotPadded(t *testing.T) {
	for step, want := range map[int]string{9: "demo/step-9-" + attempt, 10: "demo/step-10-" + attempt} {
		if got := (RestorePoint{Workload: "demo", Step: step, Attempt: attempt}).Key(); got != want {
			t.Fatalf("Key = %q, want %q", got, want)
		}
	}
}

func TestS3URLEndsWithASlash(t *testing.T) {
	rp := RestorePoint{Bucket: "my-bucket", Workload: "demo", Step: 1200, Attempt: attempt}
	if got, want := rp.S3URL(), "s3://my-bucket/demo/step-1200-20261119T163000Z-3fa94c1e/"; got != want {
		t.Fatalf("S3URL = %q, want %q", got, want)
	}
}

func TestLocalDirIsWorkloadSlashStepNumber(t *testing.T) {
	got := LocalDir("store", "demo", 1200)
	want := filepath.Join("store", "demo", "step-1200")
	if got != want {
		t.Fatalf("LocalDir = %q, want %q", got, want)
	}
}

func TestNewAttemptsAreValidAndSortInTimeOrder(t *testing.T) {
	early := NewAttempt(time.Date(2026, 11, 19, 9, 0, 0, 0, time.UTC))
	late := NewAttempt(time.Date(2026, 11, 19, 10, 0, 0, 0, time.FixedZone("x", 3600))) // 09:00 UTC
	later := NewAttempt(time.Date(2026, 11, 19, 9, 0, 1, 0, time.UTC))
	if !strings.HasPrefix(early, "20261119T090000Z-") || !strings.HasPrefix(late, "20261119T090000Z-") {
		t.Fatalf("attempts %q and %q do not start with the UTC time", early, late)
	}
	if early == late {
		t.Fatalf("two attempts in the same second are both %q", early)
	}
	if !(early < later && late < later) {
		t.Fatalf("%q and %q do not sort before %q", early, late, later)
	}
	for _, a := range []string{early, late, later} {
		if _, got, ok := ParseFolder("step-1-" + a); !ok || got != a {
			t.Fatalf("ParseFolder does not read back attempt %q", a)
		}
	}
}

func TestParseS3URLReadsBackWhatS3URLWrote(t *testing.T) {
	for _, step := range []int{0, 9, 1200} {
		want := RestorePoint{Bucket: "my-bucket", Workload: "demo", Step: step, Attempt: attempt}
		got, err := ParseS3URL(want.S3URL())
		if err != nil {
			t.Fatalf("ParseS3URL(%q): %v", want.S3URL(), err)
		}
		if got != want {
			t.Fatalf("ParseS3URL(%q) = %+v, want %+v", want.S3URL(), got, want)
		}
	}
}

func TestParseS3URLAcceptsAMissingTrailingSlash(t *testing.T) {
	got, err := ParseS3URL("s3://my-bucket/demo/step-1200-" + attempt)
	if err != nil || got.Step != 1200 || got.Attempt != attempt {
		t.Fatalf("got %+v, %v", got, err)
	}
}

func TestParseS3URLRefusesWhatIsNotARestorePointFolder(t *testing.T) {
	for _, url := range []string{
		"my-bucket/demo/step-1-" + attempt + "/",                // no scheme
		"http://my-bucket/demo/step-1-" + attempt + "/",         // wrong scheme
		"s3:///demo/step-1-" + attempt + "/",                    // no bucket
		"s3://my-bucket/step-1-" + attempt + "/",                // no workload
		"s3://my-bucket/demo/",                                  // no step folder
		"s3://my-bucket/demo/step-1/",                           // no attempt
		"s3://my-bucket/demo/step--" + attempt + "/",            // no number
		"s3://my-bucket/demo/step-abc-" + attempt + "/",         // not a number
		"s3://my-bucket/demo/step-007-" + attempt + "/",         // padded: two names for one folder
		"s3://my-bucket/demo/step-1-" + attempt + "/app/app.pt", // a file inside, not the folder
		"s3://my-bucket/demo/step-1-" + attempt + "//",          // empty path part
		"s3://my-bucket/a b/step-1-" + attempt + "/",            // bad workload name
		"s3://my-bucket/../step-1-" + attempt + "/",             // workload that escapes the folder
		"s3://Bad_Bucket/demo/step-1-" + attempt + "/",          // not a valid bucket name
	} {
		if got, err := ParseS3URL(url); err == nil {
			t.Errorf("ParseS3URL(%q) = %+v, want an error", url, got)
		}
	}
}

func TestParseFolder(t *testing.T) {
	for name, want := range map[string]int{"step-0-" + attempt: 0, "step-9-" + attempt: 9, "step-1200-" + attempt: 1200} {
		if got, gotAttempt, ok := ParseFolder(name); !ok || got != want || gotAttempt != attempt {
			t.Errorf("ParseFolder(%q) = %d, %q, %v; want %d, %q, true", name, got, gotAttempt, ok, want, attempt)
		}
	}
	for _, name := range []string{
		"", "step-", "step-1", "step-1-", "step-007-" + attempt, "step--1-" + attempt, "step-abc-" + attempt,
		"step-1-" + attempt + "/", "steps-1-" + attempt, "Step-1-" + attempt, "other",
		"step-1-20261119T163000Z-3FA94C1E", "step-1-20261119T163000Z", "step-1-20261119T163000Z-3fa94c1e-x",
	} {
		if got, gotAttempt, ok := ParseFolder(name); ok {
			t.Errorf("ParseFolder(%q) = %d, %q, true; want false", name, got, gotAttempt)
		}
	}
}

func TestParseS3WorkloadReadsAWorkloadLocation(t *testing.T) {
	for _, url := range []string{"s3://my-bucket/demo", "s3://my-bucket/demo/"} {
		bucket, workload, err := ParseS3Workload(url)
		if err != nil || bucket != "my-bucket" || workload != "demo" {
			t.Errorf("ParseS3Workload(%q) = %q, %q, %v; want my-bucket, demo", url, bucket, workload, err)
		}
	}
	for _, url := range []string{
		"my-bucket/demo/", "http://my-bucket/demo/", "s3://my-bucket", "s3://my-bucket/",
		"s3://my-bucket/demo/step-1-" + attempt + "/", "s3://Bad_Bucket/demo/", "s3://my-bucket/../", "s3://my-bucket/a b/",
	} {
		if bucket, workload, err := ParseS3Workload(url); err == nil {
			t.Errorf("ParseS3Workload(%q) = %q, %q; want an error", url, bucket, workload)
		}
	}
}

func TestValidBucket(t *testing.T) {
	for _, name := range []string{"my-bucket", "cloud-checkpointing-48213", "a.b-c", "abc"} {
		if !ValidBucket(name) {
			t.Errorf("ValidBucket(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"", "ab", "CloudCheckpointing", "has_underscore", "has/slash", "-leading", "trailing-", "has space"} {
		if ValidBucket(name) {
			t.Errorf("ValidBucket(%q) = true, want false", name)
		}
	}
}

func TestValidWorkload(t *testing.T) {
	for _, name := range []string{"demo", "run-1", "a.b_c"} {
		if !ValidWorkload(name) {
			t.Errorf("ValidWorkload(%q) = false, want true", name)
		}
	}
	for _, name := range []string{"", ".", "..", "a/b", "a b", "a\\b"} {
		if ValidWorkload(name) {
			t.Errorf("ValidWorkload(%q) = true, want false", name)
		}
	}
}
