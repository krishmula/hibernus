# Prior art for minimal-loss checkpointing

Survey date: 2026-11-07. Six parallel web surveys covered academic training papers, industry systems, transparent GPU checkpointing, analogues outside ML, inference, and mid-epoch data resumption. Numbers are the authors' or vendors' own unless stated. Nothing here was measured by us. Items marked **unverified** were seen only in search snippets or secondary sources.

## Verdict

Every piece of the Hibernus design already exists somewhere. No shipped product combines them for a single GPU on plain EC2 with S3.

- The two-stage save (GPU to pinned host memory, then a background write to durable storage) is the industry standard. Examples are PyTorch DCP, MegaScale, ByteRobust, DLRover, NVRx, and the Orbax Snapshotter.
- A separate process that holds the snapshot in shared memory and survives a trainer crash already exists. DLRover Flash Checkpoint, BitSnap, and REFT all do it.
- Per-step snapshots are proven in production. ByteRobust reports 0.01 to 0.04 s stalls and under 0.9% overhead on 9,600 GPUs.
- Nobody restores mid-step. Orbax, NeMo, and Ray all save at the step where the preemption notice lands.
- Big labs are moving to "checkpointless" recovery from peer replicas (torchft, HyperPod, Gemini). A single GPU has no peers, so host RAM plus S3 is the only redundancy Hibernus has.

So the research contribution is small. The product contribution is real. It is one installable tool that closes the loop on plain EC2, with lost work measured per failure.

## Design changes the research supports

| # | Change | Evidence |
|---|---|---|
| 1 | Start the GPU-to-host copy right after `optimizer.step`, on a side stream. Block only before the *next* `optimizer.step`. Weights do not change during forward and backward. | DataStates-LLM (HPDC'24), PHOENIX (arXiv 2026), Redis fork copy-on-write |
| 2 | Keep two pinned host buffers and alternate between them. A complete snapshot always exists, even if a copy is cut off. | PHOENIX, BitSnap |
| 3 | The sidecar keeps only the newest snapshot. If an upload is still running, skip or replace, never queue. | Orbax Snapshotter, PCcheck, AccelPact (656.7% latency from backpressure) |
| 4 | Two restore paths. If the host survived (trainer crash, CUDA Xid, OOM), reload from shared memory on the same node. Otherwise load from S3. | DLRover Flash Checkpoint, HyperPod tiered checkpointing |
| 5 | Fence old writers. Each run attempt gets a generation number. Write manifests with S3 conditional writes. Restore picks the highest valid generation and step. | ARIES, Kafka producer epochs, Flink attempt fencing |
| 6 | Use AWS warnings in two tiers. A rebalance recommendation starts a drain and shrinks K. The 2-minute notice triggers the final checkpoint, which must *finish*, including upload, inside the window. | HPC proactive checkpointing (recall beats precision), SpotCheck, Cho et al. (IEEE Access 2024: 2 minutes can be too short) |
| 7 | Choose K with Young/Daly, τ ≈ √(2δM). δ is the training stall only. M is the mean time between failures that come *without* a usable warning. K is never shorter than the drain time. | Young 1974, Daly 2006, CheckFreq |
| 8 | Treat restore time as the main metric. It now dominates lost work. Use parallel ranged S3 GETs, a pre-baked AMI, and fast loaders. | Google multi-tier (30 min to under 1 min), NVRx on EKS, SkyServe (183 s bring-up) |
| 9 | Make augmentation randomness a pure function of `(seed, epoch, sample_idx)`. StatefulDataLoader does not save worker RNG state. | Grain, torchdata source |
| 10 | Record an environment fingerprint in the manifest (GPU model, driver, CUDA, cuDNN, PyTorch, deterministic flags). | PyTorch and cuDNN reproducibility docs |
| 11 | Validate with three runs: an uninterrupted reference, a run that saves at step K, and a resume to step N. Assert bitwise on the same stack. | TorchTitan `checkpoint_compat_test.py`, Megatron Core |
| 12 | Write the controller (detect, relaunch, restore) as an idempotent state machine. Cap restarts per time window to stop crash loops. | Temporal, Erlang supervisors |
| 13 | Make side effects (metrics, eval files) idempotent by step, or hold them until the commit. | Remus output commit, Spark commit log, MillWheel |

## Training checkpointing: academic

| System | Org, venue | Mechanism | Reported result | Takeaway |
|---|---|---|---|---|
| [CheckFreq](https://www.usenix.org/conference/fast21/presentation/mohan) | UT Austin and MSR, FAST'21 | Iteration-level checkpoints in two phases (snapshot, persist), with a profiled frequency and a resumable data iterator | Under 3.5% overhead, recovery in seconds | Profile to choose K |
| [DataStates-LLM](https://www.arxiv.org/abs/2406.10707) | Argonne, HPDC'24 | Lazy copy during forward and backward, flushed in the background | Up to 48x faster checkpoints | Block only before `optimizer.step` |
| [FastPersist](https://arxiv.org/html/2406.13768) | Microsoft, 2024 | Double-buffered NVMe writes overlapped with the next iteration | Negligible overhead at every iteration | NVMe as a middle tier |
| [PCcheck](https://senbon-sakura.github.io/slide_html/pccheck-asplos-2025/) | ETH Zurich, ASPLOS'25 | Several checkpoints in flight at once | About 3% at every 10 iterations (unverified) | Memory grows with N in flight |
| [Gemini](https://cdn.amazon.science/29/31/6523473f48e4af52252bac56ef51/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints.pdf) | Rice and AWS, SOSP'23 | Checkpoints in local and peer CPU memory, remote storage as fallback | Every iteration, over 13x faster recovery | Without peers, RAM survives only software failures |
| [REFT](https://arxiv.org/html/2310.12670v4) | HKBU, arXiv | Snapshot-management processes that survive trainer crashes | 0 to 5% overhead | Prior art for the sidecar |
| [BitSnap](https://arxiv.org/html/2511.12376v1) | InfiniGence and Tsinghua, arXiv 2025 | Shared memory plus an async agent daemon, several iterations kept, incomplete snapshots dropped | 7 to 12x faster than `torch.save` | Closest match to Hibernus. Avoid its lossy compression |
| [PHOENIX](https://arxiv.org/html/2607.01646v2) | Rutgers and GMU, arXiv 2026 | Full snapshot every iteration into two alternating pinned buffers | Overhead in the noise, at most 1 step lost | Alternating buffers |
| [Just-In-Time Checkpointing](https://dl.acm.org/doi/abs/10.1145/3627703.3650085) | MSR India, EuroSys'24 | Saves only when a failure is detected, then replays one minibatch | Near-zero steady overhead | Needs a healthy replica for unwarned failures (unverified detail) |
| [Mnemosyne](https://zhangmenghao.github.io/papers/APNet2025-Mnemosyne.pdf) | Beihang, APNet'25 | JIT with a lightweight device proxy | Recovery 58.8% faster than JIT | Reactive path |
| [MegaScale](https://arxiv.org/pdf/2402.15627) | ByteDance, NSDI'24 | Pinned host memory in seconds, then a background HDFS write | Over 90% effective training time | Confirms the plan |
| [ByteCheckpoint](https://www.usenix.org/conference/nsdi25/presentation/wan-borui) | ByteDance, NSDI'25 | Checkpoint format independent of parallelism, resharded at load | 54x less stall | Store device-independent tensors plus a manifest |
| [Bamboo](https://www.usenix.org/conference/nsdi23/presentation/thorpe), [Oobleck](https://arxiv.org/pdf/2309.08125), [ReCycle](https://www.arxiv.org/abs/2405.14009), [FlashRecovery](https://arxiv.org/abs/2509.03047), [Checkmate](https://www.usenix.org/conference/nsdi26/presentation/bhardwaj) | Various, 2023 to 2026 | Redundant computation or peer replicas instead of checkpoints | Up to 1 step lost | All need peers. Not for v1 |
| [Check-N-Run](https://www.usenix.org/conference/nsdi22/presentation/eisenman), [ExCP](https://proceedings.mlr.press/v235/li24m.html), [LowDiff](https://arxiv.org/html/2509.04084v3) | Meta, Huawei, others | Incremental, quantized, or differential checkpoints | 6 to 70x smaller | With Adam every parameter changes every step. Big savings are lossy. Use lossless compression only |
| [Can't Be Late](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao) | Berkeley, NSDI'24 | Switch to on-demand when a spot job falls behind its deadline | Not recorded | Possible later `hib` feature |

## Training checkpointing: industry

| System | When it saves | Where the snapshot lives | Preemption handling | Takeaway |
|---|---|---|---|---|
| [PyTorch DCP async](https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/) | User interval (TorchTitan default 500 steps) | Pinned shared memory, then a separate writer process | None built in | Process writer beat threads because of the GIL. Stall fell from 18.5 s to 1.5 s. Copy its API shape |
| [torchft](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/) | Every step, as commit or rollback | Peer replicas | Quorum via a lighthouse server | The per-step transaction model |
| [Orbax multi-tier](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/multi_tier_checkpointing.html) | RAM every N steps, GCS every 30 min | Ramdisk, then peer, then GCS, via a replicator service | [Saves at the preemption step](https://orbax.readthedocs.io/en/latest/guides/checkpoint/preemption_checkpointing.html), on by default | Separate cadence per tier. Skip a snapshot if one is in flight |
| [SageMaker HyperPod](https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/managed-tier-checkpointing.html) | Per-tier frequency | CPU memory, peer copies, then S3 | Auto-resume | Restore from memory first, then S3 |
| [SageMaker managed spot](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) | User-defined | Local folder synced to S3 | No documented notice delivery | A plain directory sync, SageMaker only |
| [NVRx / NeMo](https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/training/resiliency.html) | Interval plus local checkpoints | Node-local, replicated | Saves on SIGTERM at the end of the step | Watchdog with separate timeouts per phase |
| [ByteRobust](https://arxiv.org/html/2509.16293v4) | Every iteration | CPU memory and SSD, peer backup | Not recorded | Per-step is realistic. CUDA errors are 36.1% of 38,236 failures |
| [DLRover Flash Checkpoint](https://github.com/intelligent-machine-learning/dlrover/blob/master/docs/blogs/megatron_flash_checkpoint.md) | User interval | Shared memory, persisted by an agent process | Agent persists after a trainer crash | Closest shipped match. Same-node reload from shared memory |
| [Composer autoresume](https://docs.mosaicml.com/projects/composer/en/latest/notes/resumption.html) | `save_interval` | Local, then object store | None found | One command both starts and resumes |
| [Ray Train](https://docs.ray.io/en/latest/train/api/doc/ray.train.get_preemption_info.html) | User interval | `storage_path` (S3) | `get_preemption_info()` for a just-in-time save | Needs a Ray cluster |
| [SkyPilot managed jobs](https://docs.skypilot.co/en/stable/examples/managed-jobs.html) | User's job | Mounted bucket | Relaunches across regions and clouds | Placement without state. Ship a SkyPilot recipe |
| [Modal GPU snapshots](https://modal.com/blog/gpu-mem-snapshots) | Once, after init | Host memory via cuda-checkpoint and gVisor | Not applicable | Cold start only, not training |
| [Llama 3 paper](https://arxiv.org/html/2407.21783) | Not applicable | Tectonic storage | Not applicable | 466 interruptions in 54 days. 78% were hardware |

## Transparent GPU checkpointing (the CRIU mode)

| System | Mechanism | Reported numbers | Limits |
|---|---|---|---|
| [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) | Driver API: lock, checkpoint to host memory, restore, unlock | About 3.0 GB/s checkpoint and 7.2 GB/s restore on A100 (measured by GCR) | Same chip type on restore. No UVM. Driver 550 or newer, 570 for CRIU 4 process trees |
| [CRIUgpu](https://arxiv.org/html/2502.16631v1) | CRIU plugins for CUDA and AMD | GPT-2 124M on A100: 12.8 s checkpoint, 8.3 s restore, 9.9 GB image | Image is the whole allocator pool, not just model state. Mixed GPU types fail. Use ptrace, not the freezer cgroup |
| [PhoenixOS](https://github.com/fslongjin/phoenixos) | Interception with soft copy-on-write | About 185 ms stall for Llama2-13B | Research code for CUDA 11.3, single GPU only |
| [GCR](https://www.usenix.org/conference/fast26/presentation/zeng) | Driver checkpoint for control state, intercepted data copies | 72% lower checkpoint latency than cuda-checkpoint | Incremental mode does not help training |
| [Singularity](https://arxiv.org/pdf/2202.07848) | Device proxy plus CRIU | Under 3% overhead, migration in 28 to 141 s | Closed source |
| [Gandiva](https://www.usenix.org/conference/osdi18/presentation/xiao) | Suspend at the mini-batch boundary | Under 100 ms suspend and resume | Needed modified frameworks |

**Verdict.**
- Always-on periodic transparent checkpointing is not viable with shipping tools. Each snapshot stops the process and pays for both checkpoint and restore.
- Transparent checkpointing on the warning only is viable as a best effort. It needs a hard deadline and a fallback to the latest application snapshot.
- Migration to a different GPU model is not viable. Only an application-level checkpoint can move from an A10G to an L4 or A100.

## Analogues outside ML

| Field | Idea | Source | What Hibernus takes |
|---|---|---|---|
| HPC | Optimal interval √(2δM) | Young (CACM 1974), [Daly (FGCS 2006)](https://blog.acolyer.org/2015/06/10/a-higher-order-estimate-of-the-optimun-checkpoint-interval-for-restart-dumps/) | Derive K. The formula is forgiving, so rough δ and M are enough |
| HPC | Multi-level checkpointing | [SCR (SC10)](https://www.osti.gov/biblio/984082), [FTI (SC11)](https://github.com/leobago/fti), [VeloC](https://arxiv.org/pdf/2103.02131) | Each tier covers one failure class. RAM covers process crashes. S3 covers instance loss |
| HPC | Proactive checkpointing on failure prediction | [Bouguerra et al. (IPDPS 2013)](https://arxiv.org/pdf/1302.3752) | Recall beats precision. Act on the rebalance recommendation |
| Databases | WAL, ARIES, commit record | [ARIES (TODS 1992)](https://cs.uwaterloo.ca/%7Edavid/cs448/aries-mohan.pdf) | The manifest is the commit record. Ignore shards no manifest points to |
| Databases | Fork copy-on-write snapshot | [Redis persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/) | Only `optimizer.step` writes state, so copy during forward and backward |
| Streaming | Asynchronous barrier snapshots | [Flink (arXiv 1506.08603)](https://arxiv.org/abs/1506.08603) | The step boundary is the barrier. Count consumed batches, not prefetched ones |
| Streaming | Changelog plus periodic materialization | [Flink FLIP-158](https://flink.apache.org/2022/05/30/improving-speed-and-stability-of-checkpointing-with-generic-log-based-incremental-checkpoints/) | A log of batch indices gives exactness, not saved compute. Fence old attempts |
| Streaming | Offsets committed with output | Kafka exactly-once, [Spark checkpoints](https://docs.databricks.com/aws/en/structured-streaming/checkpoints) | Put the data cursor and RNG in the same manifest as the weights |
| VMs | Pre-copy live migration | [Clark et al. (NSDI'05)](https://www.usenix.org/conference/nsdi-05/live-migration-virtual-machines) | Dense training dirties all state every step, so pre-copy never converges. Stop and copy |
| VMs | Continuous replication, output commit | [Remus (NSDI'08)](https://www.usenix.org/conference/nsdi-08/remus-high-availability-asynchronous-virtual-machine-replication) | Hold side effects until commit |
| VMs | Migrate within the revocation deadline | [SpotCheck (EuroSys'15)](https://cgi.luddy.indiana.edu/~prateeks/papers/SpotCheck.pdf) | Design the final checkpoint to finish, not just start, within the deadline |
| Replay | Lineage recompute | [RDDs (NSDI'12)](https://people.csail.mit.edu/matei/talks/2012/nsdi_rdds.pdf), [Ray ownership](https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html) | Recompute is too expensive to cut lost work, but replaying one step after restore verifies it |
| Durable execution | Event history and replay | [Temporal](https://docs.temporal.io/dev-guide/python/durable-execution) | Applies to the `hib` controller, not the training loop |
| Mobile | Save on background, not at kill time | [Android saved state](https://developer.android.com/topic/libraries/architecture/viewmodel-savedstate) | The death warning is best effort. Periodic commits must be correct without it |

## Inference

The hypothesis mostly holds, with three corrections.

1. **Long generations need saved tokens.** Record the generated token IDs (kilobytes) with the item ID. On resume, prefill the prompt plus those tokens, which costs about as much as generating a tenth as many tokens. Sources: [ServerlessLLM (OSDI'24)](https://arxiv.org/abs/2401.14351v2) and [NVIDIA Dynamo request migration](https://docs.nvidia.com/dynamo/v1.1.1/user-guides/fault-tolerance/request-migration).
2. **Startup time is the real loss.** [SkyServe (EuroSys'25)](https://arxiv.org/abs/2411.01438v2) measured 183 s to bring up a 7B vLLM endpoint, more than the 2-minute notice. Use fast weight loading ([Run:ai Model Streamer](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer), [fastsafetensors](https://arxiv.org/html/2505.23072v1)) and a pre-baked AMI.
3. **Replay has limits.** Guided decoding, `n>1`, and beam search cannot resume from tokens alone. Sampled output continues correctly but is not identical.

KV cache fault tolerance ([DéjàVu](https://arxiv.org/abs/2403.01876), [FailSafe](https://arxiv.org/abs/2511.14116), [GhostServe](https://arxiv.org/abs/2605.00831)) and live migration ([Llumnix](https://arxiv.org/abs/2406.03243), [SpotServe](https://arxiv.org/abs/2311.15566)) all assume surviving peers. They are out of scope for single-GPU v1.

**v1 inference scope.**
- A batch runner with an input manifest of item IDs, per-item outputs committed to S3 with no-overwrite writes, and a ledger of finished IDs.
- Partial token IDs saved for long items.
- On the warning: stop taking items, flush, save tokens.
- Fast restart, with time to first token after preemption as the headline number.

## Mid-epoch resume and determinism

**What to capture at a step boundary:**
- Model, optimizer, scheduler, and GradScaler.
- Step, epoch, and the sample offset within the epoch.
- All RNG states, including every explicit `torch.Generator`.
- The StatefulDataLoader `state_dict`, with `num_workers` kept fixed.
- An environment fingerprint.

**Notable findings.**
- [StatefulDataLoader](https://pytorch.org/data/beta/torchdata.stateful_dataloader.html) handles prefetch correctly. It does not save each worker's global RNG, so augmentation must use per-sample seeding, as [Grain](https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html) does.
- HF `skip_first_batches` loses shuffle-buffer and prefetch state ([datasets #7444](https://github.com/huggingface/datasets/issues/7444)). Avoid it for exact resume.
- Megatron's sampler resumes in O(1) from `consumed_samples`. A known bug applied the offset again every epoch, which is worth a regression test.
- Bitwise equality is realistic only on the same GPU model and software stack, with deterministic mode on. [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/misc.html) guarantees nothing across architectures. Across GPU models, assert the exact next batch and bitwise state at load, then use a tolerance for later steps.
- The cost of deterministic mode is workload-dependent. [Megatron Core](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/) reduced its cost from about 17% to 1.5 to 2% with custom kernels. Measure it for single-GPU stock PyTorch.
- Saving at micro-batch boundaries has no mainstream implementation. It would cost one extra gradient-sized buffer. Defer it.

## Open gaps Hibernus could own

- **Single-GPU spot training.** Almost every paper assumes data-parallel or pipeline peers.
- **Snapshot survival.** No paper reports how often a sidecar snapshot survives a crash, or a crash during a copy.
- **The final save under a 2-minute deadline.** It is not well characterized, including what happens when the notice never arrives.
- **Lost-work accounting per failure.** Vendors report only aggregate goodput. Report lost steps, notice-to-commit time, and restore time for every interruption.
