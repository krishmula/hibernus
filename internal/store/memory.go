package store

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
)

// Memory is a Store that keeps files in memory. Tests use it in place of S3, and its two hooks
// play the failures that S3 can have.
type Memory struct {
	// PutHook, if set, runs at the start of each Put. An error from it fails that Put and stores nothing.
	PutHook func(key string) error
	// Mangle, if set, changes the bytes of each Put before they are stored, to play a corrupted upload.
	Mangle func(key string, data []byte) []byte
	// GetHook, if set, runs at the start of each Get. An error from it fails that Get.
	GetHook func(key string) error

	mu      sync.Mutex
	objects map[string][]byte
	puts    []string
}

var _ Store = (*Memory)(nil)

func NewMemory() *Memory {
	return &Memory{objects: map[string][]byte{}}
}

func (m *Memory) Put(ctx context.Context, key, src string, want Object) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.PutHook != nil {
		if err := m.PutHook(key); err != nil {
			return err
		}
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if !sameBytes(data, want) {
		return fmt.Errorf("%w: %s is not the %d bytes with checksum %s", ErrMismatch, src, want.Size, want.Sha256)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, taken := m.objects[key]; taken {
		return ErrExists
	}
	if m.Mangle != nil {
		data = m.Mangle(key, data)
	}
	m.objects[key] = data
	m.puts = append(m.puts, key)
	return nil
}

func (m *Memory) Stat(ctx context.Context, key string) (Object, error) {
	if err := ctx.Err(); err != nil {
		return Object{}, err
	}
	data, ok := m.Data(key)
	if !ok {
		return Object{}, ErrNotFound
	}
	sum := sha256.Sum256(data)
	return Object{Size: int64(len(data)), Sha256: hex.EncodeToString(sum[:])}, nil
}

func (m *Memory) Get(ctx context.Context, key, dst string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.GetHook != nil {
		if err := m.GetHook(key); err != nil {
			return err
		}
	}
	data, ok := m.Data(key)
	if !ok {
		return ErrNotFound
	}
	return copyAtomic(dst, bytes.NewReader(data))
}

func (m *Memory) Folders(ctx context.Context, prefix string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	seen := map[string]bool{}
	var names []string
	for key := range m.objects {
		rest, ok := strings.CutPrefix(key, prefix+"/")
		if !ok {
			continue
		}
		if name, _, nested := strings.Cut(rest, "/"); nested && !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names, nil
}

// Delete removes the file under key, to play a file that was lost. Put never replaces or deletes.
func (m *Memory) Delete(key string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.objects, key)
}

// Data returns the bytes stored under key.
func (m *Memory) Data(key string) ([]byte, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	data, ok := m.objects[key]
	return append([]byte(nil), data...), ok
}

// Set stores bytes under key whether or not it is taken, to set up a test. Put never replaces.
func (m *Memory) Set(key string, data []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.objects[key] = append([]byte(nil), data...)
}

// Puts lists the keys that Put stored, in order.
func (m *Memory) Puts() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.puts...)
}
