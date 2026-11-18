package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/krishmula/hibernus/internal/protocol"
	"github.com/krishmula/hibernus/internal/supervisor"
)

const checkpointUsage = "usage: hib checkpoint [--workload <name>] [--deadline <90s | 2026-11-16T10:02:00Z>]\n"

// answerGrace is how long past the deadline hib checkpoint waits for the Supervisor, which answers
// before the deadline unless something is badly wrong.
const answerGrace = 5 * time.Second

func checkpointCommand(args []string) int {
	flags := flag.NewFlagSet("checkpoint", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprint(os.Stderr, checkpointUsage) }
	workload := workloadFlag(flags)
	deadlineFlag := flags.String("deadline", "90s", "when the host goes away: a duration or an RFC 3339 time")
	if code, ok := parseFlags(flags, args); !ok {
		return code
	}
	if flags.NArg() > 0 {
		fmt.Fprint(os.Stderr, "hib checkpoint: unexpected arguments\n"+checkpointUsage)
		return exitBadCommand
	}
	deadline, err := parseDeadline(*deadlineFlag, time.Now())
	if err != nil {
		fmt.Fprintf(os.Stderr, "hib checkpoint: %v\n", err)
		return exitBadCommand
	}

	name, code := findWorkload("checkpoint", *workload)
	if code != exitOK {
		return code
	}

	var reply protocol.CheckpointReply
	request := protocol.Request{Op: protocol.OpCheckpoint, Deadline: deadline}
	err = protocol.SendBefore(protocol.SocketPath(name), request, &reply, deadline.Add(answerGrace))
	switch {
	case errors.Is(err, protocol.ErrNoSupervisor):
		fmt.Fprintf(os.Stderr, "hib checkpoint: workload %q: %v\n", name, err)
		return exitNoSupervisor
	case err != nil:
		fmt.Fprintf(os.Stderr, "hib checkpoint: no restore point: %v\n", err)
		return exitFailed
	case reply.OK:
		fmt.Println(reply.Location)
		return exitOK
	case reply.State == string(supervisor.Finished):
		fmt.Fprintln(os.Stderr, "hib checkpoint: training already finished, so there is nothing to checkpoint")
		return exitFinished
	default:
		fmt.Fprintf(os.Stderr, "hib checkpoint: no restore point: %s\n", reply.Error)
		return exitFailed
	}
}

// parseDeadline reads --deadline: a duration from now, such as 90s, or a time, such as 2026-11-16T10:02:00Z.
func parseDeadline(value string, now time.Time) (time.Time, error) {
	if duration, err := time.ParseDuration(value); err == nil {
		if duration <= 0 {
			return time.Time{}, fmt.Errorf("--deadline %s must be positive", value)
		}
		return now.Add(duration), nil
	}
	deadline, err := time.Parse(time.RFC3339, value)
	if err != nil {
		return time.Time{}, fmt.Errorf("--deadline %q is neither a duration, such as 90s, nor a time, such as 2026-11-16T10:02:00Z", value)
	}
	if !deadline.After(now) {
		return time.Time{}, fmt.Errorf("--deadline %s has already passed", value)
	}
	return deadline, nil
}
