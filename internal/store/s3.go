package store

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/feature/s3/transfermanager"
	tmtypes "github.com/aws/aws-sdk-go-v2/feature/s3/transfermanager/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	s3types "github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
)

const (
	// checksumMetadata is the object metadata key that holds the SHA-256 that Put was given. S3
	// stores it as x-amz-meta-sha256. S3's own checksum of a multipart upload is a checksum of the
	// parts, not of the whole file, so it can't be compared with the manifest's SHA-256.
	checksumMetadata = "sha256"

	defaultPartSize    = 64 << 20 // S3 allows at most 10,000 parts, so this reaches 640 GB
	defaultConcurrency = 8

	// conflictRetries is how many times Put tries again after S3 answers 409. S3 gives 409 when
	// two writes to one key overlap, and says to retry.
	conflictRetries = 3
)

// S3 is a Store that keeps files as objects in one S3 bucket. Credentials and the region come
// from the standard AWS chain (environment, profile, or the instance role), never from Hibernus.
type S3 struct {
	API    transfermanager.S3APIClient
	Bucket string
	Prefix string // an optional folder inside the bucket, without a slash at either end

	PartSizeBytes int64 // 0 means 64 MiB. S3 refuses parts under 5 MiB, except the last one.
	Concurrency   int   // parts uploaded at once; 0 means 8
}

var _ Store = (*S3)(nil)

// NewS3 makes an S3 store for bucket. optFns adjust the client, which a test against MinIO uses to
// set its endpoint.
func NewS3(ctx context.Context, bucket, prefix string, optFns ...func(*s3.Options)) (*S3, error) {
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("load the AWS configuration: %w", err)
	}
	return &S3{API: s3.NewFromConfig(cfg, optFns...), Bucket: bucket, Prefix: strings.Trim(prefix, "/")}, nil
}

func (s *S3) objectKey(key string) string {
	if s.Prefix == "" {
		return key
	}
	return s.Prefix + "/" + key
}

func (s *S3) partSize() int64 {
	if s.PartSizeBytes > 0 {
		return s.PartSizeBytes
	}
	return defaultPartSize
}

func (s *S3) concurrency() int {
	if s.Concurrency > 0 {
		return s.Concurrency
	}
	return defaultConcurrency
}

// Put uploads src, in parts when it is large. It writes only if the key is free (If-None-Match: *),
// so a stored object never changes, and S3 itself decides a race between two writers. It checks
// the size of src, records want.Sha256 as object metadata, and asks S3 to verify each upload with
// its own checksum. It does not hash src: that would read a very large file a second time.
func (s *S3) Put(ctx context.Context, key, src string, want Object) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	file, err := os.Open(src)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if info.Size() != want.Size {
		return fmt.Errorf("%w: %s has %d bytes, want %d", ErrMismatch, src, info.Size(), want.Size)
	}

	uploader := transfermanager.New(s.API, func(o *transfermanager.Options) {
		o.PartSizeBytes = s.partSize()
		o.MultipartUploadThreshold = s.partSize()
		o.Concurrency = s.concurrency()
	})
	for attempt := 0; ; attempt++ {
		if _, err := file.Seek(0, io.SeekStart); err != nil {
			return err
		}
		_, err := uploader.UploadObject(ctx, &transfermanager.UploadObjectInput{
			Bucket:            aws.String(s.Bucket),
			Key:               aws.String(s.objectKey(key)),
			Body:              file,
			IfNoneMatch:       aws.String("*"),
			Metadata:          map[string]string{checksumMetadata: want.Sha256},
			ChecksumAlgorithm: tmtypes.ChecksumAlgorithmCrc32,
		})
		switch {
		case err == nil:
			return nil
		case httpStatus(err) == 412 || apiCode(err) == "PreconditionFailed":
			return fmt.Errorf("%w: %s", ErrExists, key)
		case (httpStatus(err) == 409 || apiCode(err) == "ConditionalRequestConflict") && attempt < conflictRetries:
			select {
			case <-time.After(time.Duration(attempt+1) * 200 * time.Millisecond):
			case <-ctx.Done():
				return ctx.Err()
			}
		default:
			return fmt.Errorf("upload %s to bucket %s: %w", key, s.Bucket, err)
		}
	}
}

// Stat reports the size S3 holds and the SHA-256 that Put recorded as metadata. An object that no
// Put wrote has no recorded checksum, and reports an empty one.
func (s *S3) Stat(ctx context.Context, key string) (Object, error) {
	if err := ctx.Err(); err != nil {
		return Object{}, err
	}
	out, err := s.API.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(s.Bucket),
		Key:    aws.String(s.objectKey(key)),
	})
	if err != nil {
		if isNotFound(err) {
			return Object{}, ErrNotFound
		}
		return Object{}, fmt.Errorf("stat %s in bucket %s: %w", key, s.Bucket, err)
	}
	return Object{Size: aws.ToInt64(out.ContentLength), Sha256: out.Metadata[checksumMetadata]}, nil
}

// Get downloads the object into a temporary file and renames it to dst, so a failed download
// leaves nothing at dst. It reads the object as one stream. Fetch checks its checksum afterwards.
func (s *S3) Get(ctx context.Context, key, dst string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	out, err := s.API.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.Bucket),
		Key:    aws.String(s.objectKey(key)),
	})
	if err != nil {
		if isNotFound(err) {
			return ErrNotFound
		}
		return fmt.Errorf("download %s from bucket %s: %w", key, s.Bucket, err)
	}
	defer out.Body.Close()
	return copyAtomic(dst, out.Body)
}

// Folders lists the folders directly under prefix. S3 has no folders, so it asks for the common
// prefixes up to the next "/", page by page, because one answer holds at most 1000 entries.
func (s *S3) Folders(ctx context.Context, prefix string) ([]string, error) {
	full := s.objectKey(prefix) + "/"
	var names []string
	var token *string
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		out, err := s.API.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
			Bucket:            aws.String(s.Bucket),
			Prefix:            aws.String(full),
			Delimiter:         aws.String("/"),
			ContinuationToken: token,
		})
		if err != nil {
			return nil, fmt.Errorf("list %s in bucket %s: %w", prefix, s.Bucket, err)
		}
		for _, common := range out.CommonPrefixes {
			names = append(names, strings.TrimSuffix(strings.TrimPrefix(aws.ToString(common.Prefix), full), "/"))
		}
		if !aws.ToBool(out.IsTruncated) {
			return names, nil
		}
		token = out.NextContinuationToken
	}
}

// httpStatus returns the HTTP status that S3 answered with, or 0 when err carries none.
func httpStatus(err error) int {
	var withStatus interface{ HTTPStatusCode() int }
	if errors.As(err, &withStatus) {
		return withStatus.HTTPStatusCode()
	}
	return 0
}

// apiCode returns the error code S3 answered with, such as PreconditionFailed, or "".
func apiCode(err error) string {
	var api smithy.APIError
	if errors.As(err, &api) {
		return api.ErrorCode()
	}
	return ""
}

// isNotFound reports whether err says the object is missing. It does not look at the HTTP status
// alone, because S3 also answers 404 for a missing bucket, and a wrong bucket name must not look
// like a missing file.
func isNotFound(err error) bool {
	var notFound *s3types.NotFound
	var noSuchKey *s3types.NoSuchKey
	return errors.As(err, &notFound) || errors.As(err, &noSuchKey)
}
