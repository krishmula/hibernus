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

# 1. Start the workload, and wait until it passes step 3.
"$hib" run --workload e2e --store "$store" "$python" examples/toy/train.py --steps $steps --step-seconds 0.1 >"$work/run.log" 2>"$work/run.err" &
run_pid=$!
for _ in $(seq 100); do
	step=$("$hib" status --workload e2e 2>/dev/null | sed -n 's/^step: //p')
	[ "${step:-0}" -ge 3 ] && break
	sleep 0.1
done
[ "${step:-0}" -ge 3 ] || fail "training never reached step 3"

# 2. Checkpoint. hib checkpoint prints the restore point location.
location=$("$hib" checkpoint --workload e2e --deadline 10s) || fail "hib checkpoint exited $?"
k=$(sed -n 's/.*"step": \([0-9]*\).*/\1/p' "$location/manifest.json" | head -1)
[ -n "$k" ] || fail "no step in $location/manifest.json"
[ "$location" = "$store/e2e/step-$k" ] || fail "location $location, want $store/e2e/step-$k"
"$hib" status --workload e2e | grep -q '^state: Checkpointed$' || fail "state is not Checkpointed"

# 3. Training stays paused at step K, and a second checkpoint returns the same restore point.
sleep 0.5
[ "$(tail -1 "$work/run.log")" = "step $k" ] || fail "training went on after the checkpoint: $(tail -1 "$work/run.log")"
[ "$("$hib" checkpoint --workload e2e)" = "$location" ] || fail "a second hib checkpoint gave another location"

# 4. The checkpoint host goes away: stop the training process, and hib run exits with it.
pkill -P "$run_pid"
wait "$run_pid" || true
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

# 7. Restore. Training continues at step K+1 and runs to the end.
"$hib" restore --store "$store" "$location" >"$work/restore.log" || fail "hib restore exited $?: $(cat "$work/restore.log")"
[ "$(head -1 "$work/restore.log")" = "step $((k + 1))" ] || fail "restore began at '$(head -1 "$work/restore.log")', want 'step $((k + 1))'"
[ "$(tail -1 "$work/restore.log")" = "step $steps" ] || fail "restore ended at '$(tail -1 "$work/restore.log")', want 'step $steps'"

echo "e2e passed: checkpointed at step $k, restored at step $((k + 1))"
