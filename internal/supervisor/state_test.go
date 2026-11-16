package supervisor

import "testing"

func TestAfterChildExit(t *testing.T) {
	tests := []struct {
		current State
		code    int
		want    State
	}{
		{Running, 0, Finished},
		{Running, 1, Failed},
		{Running, 7, Failed},
		{Checkpointed, 1, Checkpointed},
		{Finished, 1, Finished},
	}
	for _, tt := range tests {
		if got := afterChildExit(tt.current, tt.code); got != tt.want {
			t.Errorf("afterChildExit(%s, %d) = %s, want %s", tt.current, tt.code, got, tt.want)
		}
	}
}
