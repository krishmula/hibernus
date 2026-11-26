#!/usr/bin/env bash
# The first build's whole story: hib run, hib checkpoint at step K, the checkpoint host goes away,
# and hib restore continues training at step K+1. Run it with `make e2e`.
set -euo pipefail

cd "$(dirname "$0")/.."
hib=$PWD/bin/hib
python=.venv/bin/python
steps=12

# A short folder, because a socket path must stay under about 100 bytes.
work=$(mktemp -d /tmp/hib-e2e.XXXXXX)
export HIB_RUN_DIR=$work/run
store=$work/store
trap 'pkill -P $$ >/dev/null 2>&1 || true; rm -rf "$work"' EXIT

fail() {
	echo "e2e failed: $*" >&2
	exit 1
}

expect_exit() {
	local want=$1
	shift
	local code=0
	"$@" >"$work/out" 2>&1 || code=$?
	[ "$code" -eq "$want" ] || fail "$* exited $code, want $want: $(cat "$work/out")"
}

# wait_for_step waits until workload $1 passes step $2. hib status exits 3 until the Supervisor
# listens, so a failed status only means another try.
wait_for_step() {
	local step
	for _ in $(seq 100); do
		step=$("$hib" status --workload "$1" 2>/dev/null | sed -n 's/^step: //p' || true)
		[ "${step:-0}" -ge "$2" ] && return
		sleep 0.1
	done
	fail "workload $1 never reached step $2"
}

# 1. Start the workload, and wait until it passes step 3.
"$hib" run --workload e2e --store "$store" "$python" examples/toy/train.py --steps $steps --step-seconds 0.1 >"$work/run.log" 2>"$work/run.err" &
run_pid=$!
wait_for_step e2e 3

# 2. Checkpoint. hib checkpoint prints the restore point location.
location=$("$hib" checkpoint --workload e2e --deadline 10s) || fail "hib checkpoint exited $?"
k=$(sed -n 's/.*"step": \([0-9]*\).*/\1/p' "$location/manifest.json" | head -1)
[ -n "$k" ] || fail "no step in $location/manifest.json"
[ "$location" = "$store/e2e/step-$k" ] || fail "location $location, want $store/e2e/step-$k"

# On Linux as root with CRIU, the restore point also holds the process snapshot.
snapshot=$("$python" -c "import json, sys; print(json.load(open(sys.argv[1]))['layers']['process_snapshot']['status'])" "$location/manifest.json")
echo "process snapshot: $snapshot"

if [ "$snapshot" = ok ]; then
	# 3. CRIU ended the training process at step K, and hib run exits 0: the workload is checkpointed.
	code=0
	wait "$run_pid" || code=$?
	[ "$code" -eq 0 ] || fail "hib run exited $code after the process snapshot, want 0: $(cat "$work/run.err")"
	[ "$(tail -1 "$work/run.log")" = "step $k" ] || fail "training went on after the checkpoint: $(tail -1 "$work/run.log")"
else
	# 3. Training stays paused at step K, and a second checkpoint returns the same restore point.
	"$hib" status --workload e2e | grep -q '^state: Checkpointed$' || fail "state is not Checkpointed"
	sleep 0.5
	[ "$(tail -1 "$work/run.log")" = "step $k" ] || fail "training went on after the checkpoint: $(tail -1 "$work/run.log")"
	[ "$("$hib" checkpoint --workload e2e)" = "$location" ] || fail "a second hib checkpoint gave another location"

	# 4. The checkpoint host goes away: stop the training process, and hib run exits with it.
	pkill -P "$run_pid"
	wait "$run_pid" || true
fi
expect_exit 3 "$hib" status --workload e2e

# 5. hib restore refuses a damaged restore point, and a folder without a manifest.
cp -R "$location" "$work/damaged"
# Change one character, so the size stays the same and only the checksum can catch it.
sed -i.orig 's/[0-9]}/x}/' "$work/damaged/app/step.json"
expect_exit 1 "$hib" restore --store "$store" "$work/damaged"
grep -q checksum "$work/out" || fail "the damaged restore point failed for another reason: $(cat "$work/out")"
mkdir "$work/empty"
expect_exit 1 "$hib" restore --store "$store" "$work/empty"

# 6. hib restore refuses a restore point from a host with another driver, before training starts.
cp -R "$location" "$work/other-driver"
sed -i.orig 's/"driver": "[^"]*"/"driver": "999.99"/' "$work/other-driver/manifest.json"
expect_exit 1 "$hib" restore --store "$store" "$work/other-driver"
grep -q 'driver: checkpoint host 999.99' "$work/out" || fail "the other-driver restore point failed for another reason: $(cat "$work/out")"
grep -q '^step ' "$work/out" && fail "training started on a host whose fingerprint differs"

# 7. Restore. Training continues at step K+1 and runs to the end, from the process snapshot when
# the restore point has one.
"$hib" restore --store "$store" "$location" >"$work/restore.log" 2>"$work/restore.err" || fail "hib restore exited $?: $(cat "$work/restore.err")"
[ "$(head -1 "$work/restore.log")" = "step $((k + 1))" ] || fail "restore began at '$(head -1 "$work/restore.log")', want 'step $((k + 1))'"
[ "$(tail -1 "$work/restore.log")" = "step $steps" ] || fail "restore ended at '$(tail -1 "$work/restore.log")', want 'step $steps'"
if [ "$snapshot" = ok ]; then
	grep -q "restored the process snapshot at step $k" "$work/restore.err" || fail "the restore didn't use the process snapshot: $(cat "$work/restore.err")"

	# 7b. A damaged process snapshot falls back to the application checkpoint.
	cp -R "$location" "$work/damaged-snapshot"
	image=$(find "$work/damaged-snapshot/process" -name 'pages-*.img' | head -1)
	printf x | dd of="$image" bs=1 seek=100 conv=notrunc 2>/dev/null
	"$hib" restore --store "$store" "$work/damaged-snapshot" >"$work/fallback.log" 2>"$work/fallback.err" || fail "the fallback restore exited $?: $(cat "$work/fallback.err")"
	grep -q "using the application checkpoint, because the process snapshot: .*checksum" "$work/fallback.err" || fail "the damaged snapshot didn't fall back: $(cat "$work/fallback.err")"
	[ "$(head -1 "$work/fallback.log")" = "step $((k + 1))" ] || fail "the fallback began at '$(head -1 "$work/fallback.log")', want 'step $((k + 1))'"
fi

# 8. The training process runs in its own session, so Ctrl+C at the terminal reaches only hib. hib
# passes SIGINT and SIGTERM on, and exits with the training process's code.
for sig in INT TERM; do
	"$hib" run --workload signals --store "$store" "$python" examples/toy/train.py --steps 1000 --step-seconds 0.1 >"$work/signals.log" 2>&1 &
	pid=$!
	wait_for_step signals 1
	kill -"$sig" "$pid"
	code=0
	wait "$pid" || code=$?
	want=130 # 128 + SIGINT, from KeyboardInterrupt
	[ "$sig" = TERM ] && want=143 # 128 + SIGTERM, from the training hook's handler
	[ "$code" -eq "$want" ] || fail "after SIG$sig, hib run exited $code, want $want: $(cat "$work/signals.log")"
done

echo "e2e passed: checkpointed at step $k, restored at step $((k + 1)), process snapshot: $snapshot"
