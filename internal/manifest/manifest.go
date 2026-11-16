// Package manifest reads and writes manifest.json, the file that describes a restore point.
package manifest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

const (
	FileName      = "manifest.json"
	SchemaVersion = 1
	StatusOK      = "ok"
	StatusFailed  = "failed"
)

type Manifest struct {
	SchemaVersion int               `json:"schema_version"`
	Workload      string            `json:"workload"`
	Step          int               `json:"step"`
	CreatedAt     time.Time         `json:"created_at"`
	Fingerprint   map[string]string `json:"fingerprint"`
	Command       Command           `json:"command"`
	Layers        Layers            `json:"layers"`
}

// Command is how to start a fresh training process when a restore falls back to the application checkpoint.
type Command struct {
	Argv []string          `json:"argv"`
	Cwd  string            `json:"cwd"`
	Env  map[string]string `json:"env"`
}

type Layers struct {
	ProcessSnapshot       Layer `json:"process_snapshot"`
	ApplicationCheckpoint Layer `json:"application_checkpoint"`
}

type Layer struct {
	Status string  `json:"status"`
	Error  *string `json:"error"`
	Files  []File  `json:"files"`
}

// File is one file of a layer. Path is relative to the restore point folder.
type File struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	Sha256 string `json:"sha256"`
}

// Describe measures the size and checksum of each file. Paths are relative to root, the restore point folder.
func Describe(root string, paths []string) ([]File, error) {
	files := make([]File, 0, len(paths))
	for _, path := range paths {
		size, sum, err := measure(filepath.Join(root, path))
		if err != nil {
			return nil, err
		}
		files = append(files, File{Path: path, Size: size, Sha256: sum})
	}
	return files, nil
}

// Write writes the manifest under a temporary name and then renames it, so root never holds a
// half-written manifest. A folder with a manifest is a complete restore point.
func Write(root string, m Manifest) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	temporary := filepath.Join(root, FileName+".tmp")
	if err := os.WriteFile(temporary, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(temporary, filepath.Join(root, FileName))
}

// Load reads the manifest in root. It refuses a folder without one, and a manifest from another schema version.
func Load(root string) (Manifest, error) {
	data, err := os.ReadFile(filepath.Join(root, FileName))
	if errors.Is(err, os.ErrNotExist) {
		return Manifest{}, fmt.Errorf("%s has no %s, so it is not a complete restore point", root, FileName)
	}
	if err != nil {
		return Manifest{}, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return Manifest{}, fmt.Errorf("read %s: %w", filepath.Join(root, FileName), err)
	}
	if m.SchemaVersion != SchemaVersion {
		return Manifest{}, fmt.Errorf("%s has schema_version %d, and this hib reads only %d", FileName, m.SchemaVersion, SchemaVersion)
	}
	return m, nil
}

// Verify checks that each file of the layer is inside root and has the size and checksum the manifest records.
func (l Layer) Verify(root string) error {
	for _, f := range l.Files {
		if !filepath.IsLocal(f.Path) {
			return fmt.Errorf("%s points outside the restore point", f.Path)
		}
		size, sum, err := measure(filepath.Join(root, f.Path))
		if err != nil {
			return err
		}
		if size != f.Size {
			return fmt.Errorf("%s has %d bytes, want %d", f.Path, size, f.Size)
		}
		if sum != f.Sha256 {
			return fmt.Errorf("%s has checksum %s, want %s", f.Path, sum, f.Sha256)
		}
	}
	return nil
}

func measure(path string) (int64, string, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, "", err
	}
	defer file.Close()
	hash := sha256.New()
	size, err := io.Copy(hash, file)
	if err != nil {
		return 0, "", err
	}
	return size, hex.EncodeToString(hash.Sum(nil)), nil
}
