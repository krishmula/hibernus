# Hibernus

## Glossary

**Supervisor**:
The long-running `hib` process for one workload on one host. It starts the training process as its child, and it checkpoints and restores that process on request.
_Avoid_: agent, node agent

**Disruption handler**:
The script or person on the workload's host that notices a disruption signal and runs `hib checkpoint`. It is outside Hibernus.
_Avoid_: caller

**Launcher**:
Whatever brings up the restore host and runs `hib restore` there, such as an Auto Scaling group, SkyPilot, or a person. It is outside Hibernus.
_Avoid_: relauncher

**Update boundary**:
The point in a training step after the optimizer, scheduler, and gradient reset finish, and before the next batch is read. Hibernus checkpoints only here.
_Avoid_: safe point, optimizer-update boundary

**Host**:
A Linux computer with an NVIDIA GPU where Hibernus runs.
_Avoid_: node, box, machine

**Checkpoint host**:
The host where a checkpoint happens, because it is about to go away.
_Avoid_: old host, saving host

**Restore host**:
The host where `hib restore` brings the workload back.
_Avoid_: new host

**Disruption**:
An event that takes a host away from a workload, such as a spot reclaim or host maintenance.
_Avoid_: preemption, interruption

**Restore point**:
The process snapshot and application checkpoint from one checkpoint, plus a manifest. A restore uses one restore point.
_Avoid_: checkpoint (for the pair), save set

**Checkpoint**:
The act of capturing a workload at an update boundary and writing a restore point. A disruption handler starts one with `hib checkpoint`.
_Avoid_: save, save request

**Workload**:
One run of a program, from `hib run` until it finishes, that Hibernus keeps alive across disruptions. It is either a training workload or an inference workload.
_Avoid_: job

**Training process**:
The Linux process, with its DataLoader child processes, that runs a training workload on one host. The Supervisor starts it as its child.

**Process snapshot**:
The layer of a restore point that holds the training process as the operating system sees it, including GPU memory. CRIU and `cuda-checkpoint` make it, and a restore tries it first.

**Application checkpoint**:
The layer of a restore point that holds the state the training loop owns, such as the model, optimizer, RNG states, and data position. A restore falls back to it when the process snapshot fails.

**Training hook**:
A small Hibernus library inside the training process that stops the loop at the update boundary and writes and loads the application checkpoint.

**Fingerprint**:
The GPU model and the driver, CUDA, cuDNN, PyTorch, and kernel versions of a host. A restore runs only on a host whose fingerprint matches the checkpoint host's.

**Initialize**:
The training hook’s first request to the Supervisor, asking whether to begin fresh or load a restore point. Both fresh training and a restore from the application checkpoint use it.
_Avoid_: start (for this request)

**Restored**:
The training hook's first request to the Supervisor after a restore from the process snapshot. It tells the Supervisor the step where the training process already is.
_Avoid_: reconnect (for this request)
