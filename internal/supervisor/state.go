package supervisor

// State is where a workload is right now. docs/design/data-shapes.md lists the moves between states.
type State string

const (
	Restoring     State = "Restoring"
	Running       State = "Running"
	Checkpointing State = "Checkpointing"
	Checkpointed  State = "Checkpointed"
	Finished      State = "Finished"
	Failed        State = "Failed"
)

// afterChildExit is the state a workload moves to when its training process exits with code.
func afterChildExit(current State, code int) State {
	switch current {
	case Running:
		if code == 0 {
			return Finished
		}
		return Failed
	case Restoring, Checkpointing:
		return Failed
	default:
		return current
	}
}
