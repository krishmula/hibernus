package layout

import (
	"path/filepath"
	"testing"
)

func TestKeyIsWorkloadSlashStepNumber(t *testing.T) {
	if got, want := Key("demo", 1200), "demo/step-1200"; got != want {
		t.Fatalf("Key = %q, want %q", got, want)
	}
}

func TestTheStepNumberIsNotPadded(t *testing.T) {
	if got, want := Key("demo", 9), "demo/step-9"; got != want {
		t.Fatalf("Key = %q, want %q", got, want)
	}
	if got, want := Key("demo", 10), "demo/step-10"; got != want {
		t.Fatalf("Key = %q, want %q", got, want)
	}
}

func TestS3URLEndsWithASlash(t *testing.T) {
	if got, want := S3URL("my-bucket", "demo", 1200), "s3://my-bucket/demo/step-1200/"; got != want {
		t.Fatalf("S3URL = %q, want %q", got, want)
	}
}

func TestLocalDirHasTheSameRelativePathAsTheKey(t *testing.T) {
	got := LocalDir("store", "demo", 1200)
	want := filepath.Join("store", "demo", "step-1200")
	if got != want {
		t.Fatalf("LocalDir = %q, want %q", got, want)
	}
}

func TestParseS3URLReadsBackWhatS3URLWrote(t *testing.T) {
	for _, step := range []int{0, 9, 1200} {
		url := S3URL("my-bucket", "demo", step)
		got, err := ParseS3URL(url)
		if err != nil {
			t.Fatalf("ParseS3URL(%q): %v", url, err)
		}
		want := RestorePoint{Bucket: "my-bucket", Workload: "demo", Step: step}
		if got != want {
			t.Fatalf("ParseS3URL(%q) = %+v, want %+v", url, got, want)
		}
	}
}

func TestParseS3URLAcceptsAMissingTrailingSlash(t *testing.T) {
	got, err := ParseS3URL("s3://my-bucket/demo/step-1200")
	if err != nil || got.Step != 1200 {
		t.Fatalf("got %+v, %v", got, err)
	}
}

func TestParseS3URLRefusesWhatIsNotARestorePointFolder(t *testing.T) {
	for _, url := range []string{
		"my-bucket/demo/step-1/",                // no scheme
		"http://my-bucket/demo/step-1/",         // wrong scheme
		"s3:///demo/step-1/",                    // no bucket
		"s3://my-bucket/step-1/",                // no workload
		"s3://my-bucket/demo/",                  // no step folder
		"s3://my-bucket/demo/step-/",            // no number
		"s3://my-bucket/demo/step-abc/",         // not a number
		"s3://my-bucket/demo/step-007/",         // padded: two names for one folder
		"s3://my-bucket/demo/step--1/",          // negative
		"s3://my-bucket/demo/step-1/app/app.pt", // a file inside, not the folder
		"s3://my-bucket/demo/step-1//",          // empty path part
		"s3://my-bucket/a b/step-1/",            // bad workload name
		"s3://my-bucket/../step-1/",             // workload that escapes the folder
		"s3://Bad_Bucket/demo/step-1/",          // not a valid bucket name
	} {
		if got, err := ParseS3URL(url); err == nil {
			t.Errorf("ParseS3URL(%q) = %+v, want an error", url, got)
		}
	}
}

func TestParseStepDir(t *testing.T) {
	for name, want := range map[string]int{"step-0": 0, "step-9": 9, "step-1200": 1200} {
		if got, ok := ParseStepDir(name); !ok || got != want {
			t.Errorf("ParseStepDir(%q) = %d, %v; want %d, true", name, got, ok, want)
		}
	}
	for _, name := range []string{"", "step-", "step-007", "step--1", "step-abc", "step-1/", "steps-1", "other", "Step-1"} {
		if got, ok := ParseStepDir(name); ok {
			t.Errorf("ParseStepDir(%q) = %d, true; want false", name, got)
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
		"s3://my-bucket/demo/step-1/", "s3://Bad_Bucket/demo/", "s3://my-bucket/../", "s3://my-bucket/a b/",
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
