set -u
export CUDA_VISIBLE_DEVICES=0
PY=~/hibernus/.venv/bin/python
CC=~/cc/cuda-checkpoint
W=~/cc/work; sudo rm -rf $W; mkdir -p $W/img
echo "== reference run (no interruption)"
$PY ~/cc/gpu_train.py 20 0 > $W/ref.log 2>&1; tail -2 $W/ref.log
echo "== run to be checkpointed"
setsid $PY ~/cc/gpu_train.py 20 1 > $W/run.log 2>&1 < /dev/null &
sleep 2; PID=$(pgrep -f "gpu_train.py 20 1" | head -1); echo "pid $PID"
until grep -q "^step 8 " $W/run.log; do sleep 0.5; done
echo "state before: $($CC --get-state --pid $PID)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "-- GPU memory in use before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
T0=$(date +%s.%N)
sudo $CC --action lock --pid $PID --timeout 10000 && echo locked
sudo $CC --action checkpoint --pid $PID && echo "gpu state moved to host; state=$($CC --get-state --pid $PID)"
echo "-- GPU memory in use after cuda-checkpoint: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
sudo criu dump -t $PID -D $W/img --shell-job --tcp-close -v4 -o $W/img/dump.log; echo "criu dump rc=$?"
T1=$(date +%s.%N); echo "dump took $(echo "$T1 - $T0" | bc) s; images: $(du -sh $W/img | cut -f1)"
sleep 1; kill -0 $PID 2>/dev/null && echo "STILL ALIVE" || echo "original process is gone"
echo "== restore"
T2=$(date +%s.%N)
sudo criu restore -D $W/img --shell-job --tcp-close -d -v4 -o $W/img/restore.log; echo "criu restore rc=$?"
sleep 1; NEW=$(pgrep -f "gpu_train.py 20 1" | head -1); echo "restored pid $NEW"
sudo $CC --action restore --pid $NEW && echo "gpu state back on device"
sudo $CC --action unlock --pid $NEW && echo unlocked
T3=$(date +%s.%N); echo "restore took $(echo "$T3 - $T2" | bc) s"
echo "state after: $($CC --get-state --pid $NEW)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
sleep 14
echo "== log of the restored run (it reuses the old stdout file)"; tail -14 $W/run.log
echo "== compare losses with the uninterrupted run"
grep '^step' $W/ref.log | sed 's/ ballast.*//' > $W/ref.txt
grep '^step' $W/run.log | sed 's/ ballast.*//' | awk '!s[$2]++' > $W/run.txt
echo "ref steps: $(wc -l < $W/ref.txt), run steps: $(wc -l < $W/run.txt)"
diff $W/ref.txt $W/run.txt && echo "RESULT: every loss is bit-identical to the uninterrupted run" || echo "RESULT: DIFFERENT"
grep -c "ballast 3584" $W/run.log
