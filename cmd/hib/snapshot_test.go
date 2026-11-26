package main

import "testing"

func TestParseCRIUVersion(t *testing.T) {
	v, err := parseCRIUVersion("Version: 4.3\nGitID: v4.3\n")
	if err != nil || v != (criuRelease{4, 3}) {
		t.Errorf("parseCRIUVersion = %v, %v; want 4.3", v, err)
	}
	if _, err := parseCRIUVersion("criu: command not found"); err == nil {
		t.Error("parseCRIUVersion accepted output with no version")
	}
}

func TestCRIUOlderThan317IsRefused(t *testing.T) {
	if err := (criuRelease{3, 16}).usable(); err == nil {
		t.Error("CRIU 3.16 was accepted, but it can't restore rseq")
	}
	if err := (criuRelease{3, 17}).usable(); err != nil {
		t.Errorf("CRIU 3.17 was refused: %v", err)
	}
}

func TestAGPUProcessNeedsCRIU4AndCudaCheckpoint(t *testing.T) {
	tests := []struct {
		name          string
		version       criuRelease
		gpu, cudaCkpt bool
		wantNoProblem bool
	}{
		{"no GPU on the host", criuRelease{3, 17}, false, false, true},
		{"CRIU 3.19 has no CUDA plugin", criuRelease{3, 19}, true, true, false},
		{"no cuda-checkpoint", criuRelease{4, 3}, true, false, false},
		{"CRIU 4.3 and cuda-checkpoint", criuRelease{4, 3}, true, true, true},
	}
	for _, tt := range tests {
		problem := tt.version.gpuProblem(tt.gpu, tt.cudaCkpt)
		if (problem == "") != tt.wantNoProblem {
			t.Errorf("%s: gpuProblem = %q", tt.name, problem)
		}
	}
}
