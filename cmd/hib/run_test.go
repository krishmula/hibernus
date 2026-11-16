package main

import "testing"

func TestDefaultWorkloadName(t *testing.T) {
	tests := []struct {
		command []string
		want    string
	}{
		{[]string{"python", "examples/toy/train.py", "--steps", "3"}, "train"},
		{[]string{"python", "-m", "trainer"}, "python"},
		{[]string{"./my-trainer", "--lr", "3e-4"}, "my-trainer"},
	}
	for _, tt := range tests {
		if got := defaultWorkloadName(tt.command); got != tt.want {
			t.Errorf("defaultWorkloadName(%q) = %q, want %q", tt.command, got, tt.want)
		}
	}
}
