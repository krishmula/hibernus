package store

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Local is a Store that keeps files in a folder, such as a shared or mounted disk. Each key is a
// path under Root.
type Local struct {
	Root string
}

var _ Store = Local{}

// path turns a key into a file under Root, and refuses a key that would leave Root.
func (l Local) path(key string) (string, error) {
	relative := filepath.FromSlash(key)
	if !filepath.IsLocal(relative) {
		return "", fmt.Errorf("key %q points outside the store", key)
	}
	return filepath.Join(l.Root, relative), nil
}

// Put copies src to the key. It checks that the key is free and then renames the copy into place,
// so two Supervisors racing for one key is not guarded against. One Supervisor runs per workload.
func (l Local) Put(ctx context.Context, key, src string, want Object) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	dst, err := l.path(key)
	if err != nil {
		return err
	}
	if _, err := os.Stat(dst); err == nil {
		return ErrExists
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	size, sum, err := measure(src)
	if err != nil {
		return err
	}
	if size != want.Size || sum != want.Sha256 {
		return fmt.Errorf("%w: %s has %d bytes and checksum %s, want %d bytes and checksum %s",
			ErrMismatch, src, size, sum, want.Size, want.Sha256)
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	return copyAtomic(dst, in)
}

func (l Local) Stat(ctx context.Context, key string) (Object, error) {
	if err := ctx.Err(); err != nil {
		return Object{}, err
	}
	file, err := l.path(key)
	if err != nil {
		return Object{}, err
	}
	size, sum, err := measure(file)
	if errors.Is(err, os.ErrNotExist) {
		return Object{}, ErrNotFound
	}
	if err != nil {
		return Object{}, err
	}
	return Object{Size: size, Sha256: sum}, nil
}

func (l Local) Folders(ctx context.Context, prefix string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	dir, err := l.path(prefix)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var names []string
	for _, entry := range entries {
		if entry.IsDir() {
			names = append(names, entry.Name())
		}
	}
	return names, nil
}

func (l Local) Get(ctx context.Context, key, dst string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	file, err := l.path(key)
	if err != nil {
		return err
	}
	in, err := os.Open(file)
	if errors.Is(err, os.ErrNotExist) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	defer in.Close()
	return copyAtomic(dst, in)
}
