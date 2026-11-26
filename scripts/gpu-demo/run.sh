#!/usr/bin/env bash
# A timed disruption drill. It trains, runs `hib checkpoint` after a set time, kills the training
# host's process, waits, runs `hib restore` from S3, and lets the run finish. An uninterrupted
# reference run trains alongside, so the restored losses can be compared update by update.
#
# While it runs, live.json in the output folder is rewritten every 2 seconds from the files the
# drill leaves behind. With --serve, the dashboard in dashboard/ reads that file and updates itself.
#
# Run it from the repository root, after `make build venv` and `uv pip install --python .venv -e
# 'python[train]'`. It needs AWS credentials (for example an instance role) allowed to use the
# bucket, and a region: it sets AWS_REGION to us-east-1 when none is set.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cd "$root"

BUCKET=${HIB_BUCKET:-hibernus-restore-points-nitish-01}
UPDATES=2400
STEP_SLEEP=0.025
CKPT_AFTER=20
DOWN_FOR=10
OUT=/var/tmp/hib-demo
SERVE=""

usage() {
	cat <<EOF
usage: scripts/gpu-demo/run.sh [options]
  --bucket NAME          S3 bucket for restore points (default \$HIB_BUCKET, then $BUCKET)
  --updates N            training updates (default $UPDATES)
  --step-sleep SECONDS   pause after each update, so the run lasts long enough to watch (default $STEP_SLEEP)
  --checkpoint-after S   seconds of training before hib checkpoint runs (default $CKPT_AFTER)
  --down-for S           seconds between losing the host and hib restore (default $DOWN_FOR)
  --out DIR              work folder and live.json (default $OUT)
  --serve PORT           serve the dashboard and live.json on 127.0.0.1:PORT
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	--bucket) BUCKET=$2; shift 2 ;;
	--updates) UPDATES=$2; shift 2 ;;
	--step-sleep) STEP_SLEEP=$2; shift 2 ;;
	--checkpoint-after) CKPT_AFTER=$2; shift 2 ;;
	--down-for) DOWN_FOR=$2; shift 2 ;;
	--out) OUT=$2; shift 2 ;;
	--serve) SERVE=$2; shift 2 ;;
	-h | --help) usage; exit 0 ;;
	*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
done

export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$AWS_REGION}
PY=.venv/bin/python
SCRIPT=examples/plain/train.py
HIB=./bin/hib
WL=drill-$(date +%s)
W=$OUT
rm -rf "$W"
mkdir -p "$W"
trap 'pkill -P $$ >/dev/null 2>&1 || true' EXIT

for need in "$HIB" "$PY"; do
	[ -x "$need" ] || { echo "missing $need: run make build venv first" >&2; exit 1; }
done

ms() { date +%s%3N; }
ev() { printf '%s %s %s %s\n' "$(date +%T.%3N)" "$1" "$2" "$3" >>"$W/events.log"; echo "$1 $2 $3"; }
nz() { if [ -n "${1:-}" ]; then printf '%s' "$1"; else printf null; fi; }

GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU, unknown")
GPU_NAME=${GPU%%,*}
GPU_DRIVER=$(echo "${GPU##*,}" | tr -d ' ')
REGION=$AWS_REGION
INSTANCE=$(curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "local")
HOSTJSON="{\"gpu\":\"$GPU_NAME\",\"driver\":\"$GPU_DRIVER\",\"instance\":\"$INSTANCE\",\"region\":\"$REGION\"}"

put_state() {
	cat >"$W/state.tmp" <<EOF
{"workload":"$WL","upd":$UPDATES,"host":$HOSTJSON,"phase":$1,"status":"$2","mode":"$3","k":$(nz "${K:-}"),"disruptStep":$(nz "${DISRUPT:-}"),"restoredStep":$(nz "${RESTORED:-}"),"ckptMs":$(nz "${CK_MS:-}"),"restoreMs":$(nz "${RESTORE_MS:-}"),"ckptId":"${CKID:-}","ckptBytes":$(nz "${CKBYTES:-}"),"loc":"${LOC:-}","fin":${FIN:-false}}
EOF
	mv "$W/state.tmp" "$W/state.json"
}

# Refresh live.json every 2 seconds for as long as this script runs.
(
	while kill -0 $$ 2>/dev/null; do
		python3 "$here/bridge.py" "$W" >"$W/live.tmp" 2>/dev/null && mv "$W/live.tmp" "$W/live.json"
		sleep 2
	done
) &

if [ -n "$SERVE" ]; then
	mkdir -p "$W/web"
	cp "$root/dashboard/index.html" "$W/web/index.html"
	ln -sf "$W/live.json" "$W/web/live.json"
	python3 -m http.server "$SERVE" --directory "$W/web" --bind 127.0.0.1 >/dev/null 2>&1 &
	echo "dashboard: http://127.0.0.1:$SERVE/  (from your laptop: ssh -L $SERVE:127.0.0.1:$SERVE <host>)"
fi

ev INFO sched "workload $WL on $GPU_NAME $INSTANCE $REGION"
put_state 1 Running idle

"$PY" $SCRIPT --updates "$UPDATES" --step-sleep "$STEP_SLEEP" >"$W/baseline.log" 2>"$W/baseline.err" &
BASEPID=$!
ev INFO baseline "uninterrupted reference run started: seed 0, $UPDATES updates"

$HIB run --workload "$WL" --store "$W/store" --remote "s3://$BUCKET" "$PY" $SCRIPT --updates "$UPDATES" --step-sleep "$STEP_SLEEP" >"$W/run.log" 2>"$W/run.err" &
RUNPID=$!
ev INFO hibernus "supervisor attached: remote s3://$BUCKET"

# 1. Train for CKPT_AFTER seconds.
started=$(date +%s)
while [ $(($(date +%s) - started)) -lt "$CKPT_AFTER" ]; do
	kill -0 $RUNPID 2>/dev/null || { ev ERROR hibernus "supervisor exited early: $(head -c 300 "$W/run.err")"; exit 1; }
	sleep 0.5
done
ev INFO train "trained for ${CKPT_AFTER}s"

# 2. The interruption notice arrives: checkpoint to S3.
ev WARN agent "interruption notice (simulated trigger): running hib checkpoint"
put_state 2 Checkpointing idle
t0=$(ms)
LOC=$($HIB checkpoint --workload "$WL" --deadline 180s) || { ev ERROR hib "checkpoint failed"; exit 1; }
t1=$(ms)
CK_MS=$((t1 - t0))
K=$($HIB status --workload "$WL" | sed -n 's/^step: //p')
CKID=$(basename "${LOC%/}")
LDIR=$(find "$W/store/$WL" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
CKBYTES=$(du -sb "$LDIR" | cut -f1)
ev OK ckpt "$CKID saved at step $K, $((CKBYTES / 1024)) KiB"
ev OK s3 "published to $LOC, sha256 verified, ${CK_MS} ms (checkpoint plus upload)"
TPID=$(pgrep -P $RUNPID -f "$SCRIPT" | head -1)
"$PY" "$here/probe.py" "$LDIR" "${TPID:-0}" "$W/before.json" 2>>"$W/probe.err"
ev INFO probe "snapshot read: pid ${TPID:-?}, weights, optimizer and RNG hashed"
put_state 2 Checkpointing idle
sleep 3

# 3. The host is lost.
DISRUPT=$K
ev WARN kernel "host lost (simulated): killing supervisor and training process"
put_state 2 Disrupted idle
[ -n "${TPID:-}" ] && kill -9 "$TPID" 2>/dev/null
kill $RUNPID 2>/dev/null
sleep 1
kill -9 $RUNPID 2>/dev/null
sleep "$DOWN_FOR"

# 4. Restore from S3.
put_state 3 Checkpointing restoring
ev INFO restore "hib restore $LOC"
t0=$(ms)
$HIB restore --store "$W/store2" "$LOC" >"$W/restore.log" 2>"$W/restore.err" &
RPID=$!
for _ in $(seq 1 1200); do
	grep -q '^step ' "$W/restore.log" 2>/dev/null && break
	kill -0 $RPID 2>/dev/null || break
	sleep 0.1
done
t1=$(ms)
RESTORE_MS=$((t1 - t0))
RESTORED=$K
ev OK restore "downloaded and verified from S3, resumed at step $((K + 1)), ${RESTORE_MS} ms to first update"
NPID=$(pgrep -P $RPID -f "$SCRIPT" | head -1)
RDIR=$(find "$W/store2" -name manifest.json | head -1 | xargs -r dirname)
put_state 4 Recovered restoring
"$PY" "$here/probe.py" "$RDIR" "${NPID:-0}" "$W/after.json" 2>>"$W/probe.err"
ev OK probe "restored process pid ${NPID:-?}: state hashed and compared"
put_state 4 Recovered done

# 5. Let the restored run finish, then compare it with the reference.
wait $RPID
last=$(grep '^step ' "$W/restore.log" | tail -1 | awk '{print $2}')
wait $BASEPID 2>/dev/null
FIN=true
ev OK train "restored run finished at step ${last:-?} of $UPDATES"
put_state 4 Recovered done
sleep 3
verdict=$(python3 -c "import json;d=json.load(open('$W/live.json'))['verdict'];print('%d of %d losses identical, %d mismatches' % (d['compared'], d['of'], d['mismatches']))" 2>/dev/null)
ev OK verdict "${verdict:-no verdict}"
ev INFO done "drill complete; results in $W"
[ -n "$SERVE" ] && echo "dashboard still served on port $SERVE; press Ctrl-C to stop" && wait
exit 0
