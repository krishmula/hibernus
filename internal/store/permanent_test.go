package store

import (
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/aws/smithy-go"
)

func TestPermanent(t *testing.T) {
	apiError := func(code string) error {
		return fmt.Errorf("upload demo/step-1/app/app.pt: %w", &smithy.GenericAPIError{Code: code, Message: "from S3"})
	}
	for name, err := range map[string]error{
		"access denied":           apiError("AccessDenied"),
		"unknown access key":      apiError("InvalidAccessKeyId"),
		"expired token":           apiError("ExpiredToken"),
		"no such bucket":          apiError("NoSuchBucket"),
		"the file changed":        fmt.Errorf("upload x: %w", ErrMismatch),
		"a different stored file": fmt.Errorf("%w: x", ErrConflict),
		"the source is missing":   fmt.Errorf("open: %w", os.ErrNotExist),
	} {
		if !Permanent(err) {
			t.Errorf("Permanent(%s) = false, want true: waiting cannot fix it", name)
		}
	}

	for name, err := range map[string]error{
		"a network error":      errors.New("connection reset"),
		"S3 asks to slow down": apiError("SlowDown"),
		"S3 is unavailable":    apiError("ServiceUnavailable"),
		"an internal error":    apiError("InternalError"),
	} {
		if Permanent(err) {
			t.Errorf("Permanent(%s) = true, want false: a retry may work", name)
		}
	}
}

func TestPublishConflictsAreErrConflict(t *testing.T) {
	mem := NewMemory()
	mem.Set("demo/step-5-20261119T000000Z-0000abcd/app/app.pt", []byte("someone else's bytes"))

	err := Publish(bg, mem, makeRestorePoint(t, "demo", 5), testAttempt)

	if !errors.Is(err, ErrConflict) || !Permanent(err) {
		t.Fatalf("Publish error = %v, want an ErrConflict that is permanent", err)
	}
}

func TestACorruptedUploadIsErrConflict(t *testing.T) {
	mem := NewMemory()
	mem.Mangle = func(key string, data []byte) []byte {
		data[0] ^= 0xff
		return data
	}

	err := Publish(bg, mem, makeRestorePoint(t, "demo", 5), testAttempt)

	if !errors.Is(err, ErrConflict) {
		t.Fatalf("Publish error = %v, want ErrConflict: the stored file cannot be replaced, so a retry cannot fix it", err)
	}
}
