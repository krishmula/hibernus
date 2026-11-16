package main

import (
	"testing"
	"time"
)

func TestParseDeadline(t *testing.T) {
	now := time.Date(2026, 11, 16, 10, 0, 0, 0, time.UTC)
	tests := []struct {
		value string
		want  time.Time
		ok    bool
	}{
		{"90s", now.Add(90 * time.Second), true},
		{"2m", now.Add(2 * time.Minute), true},
		{"2026-11-16T10:02:00Z", now.Add(2 * time.Minute), true},
		{"0s", time.Time{}, false},
		{"-5s", time.Time{}, false},
		{"2026-11-16T09:59:00Z", time.Time{}, false},
		{"soon", time.Time{}, false},
	}
	for _, tt := range tests {
		got, err := parseDeadline(tt.value, now)
		if (err == nil) != tt.ok || !got.Equal(tt.want) {
			t.Errorf("parseDeadline(%q) = %v, %v; want %v, ok %v", tt.value, got, err, tt.want, tt.ok)
		}
	}
}
