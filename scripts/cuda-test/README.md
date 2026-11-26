# CRIU with CUDA

Checkpoints and restores a PyTorch process that holds GPU memory, using `cuda-checkpoint` and
CRIU's CUDA plugin, then compares its losses with an uninterrupted run.

Needs an NVIDIA GPU with driver 550 or newer, CRIU 4.x with `/usr/lib/criu/cuda_plugin.so`, and a
venv with CUDA torch. `setup.sh` installs CRIU from the CRIU PPA and downloads `cuda-checkpoint`
from the NVIDIA/cuda-checkpoint repository into `~/cc`. Put `gpu_train.py` and `run.sh` in `~/cc`
and make `~/hibernus/.venv` the venv, or edit the `PY` line in `run.sh`. Then run `bash run.sh`.

What `run.sh` does:

1. Trains 20 steps without interruption to get reference losses.
2. Trains again, and at step 8 runs `cuda-checkpoint --action lock` then `--action checkpoint`.
   The GPU memory moves to the host, and `nvidia-smi` shows 0 MiB in use.
3. `criu dump`, and the original process is gone.
4. `criu restore`. The CUDA plugin restores the GPU state itself, so the script's own
   `cuda-checkpoint --action restore` and `unlock` steps report "the operation cannot be performed in
   the present state". That is expected and does not fail the test.
5. Compares every loss with the reference run.

Result on one g5.xlarge (A10G, driver 595, CRIU 4.3): dump 2.6 s, restore 2.2 s, 1.6 GB of images,
512 MB of GPU memory carried over, and all 20 losses bit-identical to the uninterrupted run.
Not tested: moving the GPU snapshot to a second GPU host.
