#!/usr/bin/env bash
# End-to-end test of the process snapshot: hib run --snapshot, hib checkpoint, hib restore.
#
# Needs Linux, root (CRIU), criu, go, and a python with torch and numpy. Run it as root:
#   PY=/opt/hibvenv/bin/python scripts/e2e-criu.sh
#
# It checks that a restore from the process snapshot resumes the same training process (same PID),
# that the training output matches a run that never stopped, that the process tree log shows the
# tree before the checkpoint and after the restore, and that a corrupted snapshot falls back to the
# application checkpoint.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
GO="${GO:-go}"
WORK="${WORK:-/tmp/hib-e2e}"
UPDATES="${UPDATES:-30}"
STEP_SLEEP="${STEP_SLEEP:-0.25}"
PAUSE_AT="${PAUSE_AT:-8}"          # checkpoint once this many steps have printed

pass=0
fail=0
check() {
  local desc="$1"; shift
  if "$@"; then echo "PASS  $desc"; pass=$((pass + 1)); else echo "FAIL  $desc"; fail=$((fail + 1)); fi
}
section() { echo; echo "== $*"; }
steps_of() { grep '^step' "$1"; }

rm -rf "$WORK"; mkdir -p "$WORK"
export HIB_RUN_DIR="$WORK/run" HIB_LOG_DIR="$WORK/logs"
export PYTHONPATH="$REPO/python"
STORE="$WORK/store"
TRAIN="$REPO/examples/classifier/train.py"
OUT="$HIB_LOG_DIR/e2e.out"
TREELOG="$HIB_LOG_DIR/e2e.tree.log"

section "build and reference run"
(cd "$REPO" && "$GO" build -o "$WORK/hib" ./cmd/hib) || { echo "build failed"; exit 1; }
HIB="$WORK/hib"
"$PY" "$TRAIN" --updates "$UPDATES" > "$WORK/reference.txt" || { echo "reference run failed"; exit 1; }
echo "reference: $(steps_of "$WORK/reference.txt" | wc -l) steps"

section "run under hib with a process snapshot"
"$HIB" run --snapshot --workload e2e --store "$STORE" \
  "$PY" "$TRAIN" --updates "$UPDATES" --step-sleep "$STEP_SLEEP" > "$WORK/run.hib.log" 2>&1 &
HIB_PID=$!
for _ in $(seq 1 600); do
  [ -f "$OUT" ] && [ "$(steps_of "$OUT" | wc -l)" -ge "$PAUSE_AT" ] && break
  sleep 0.2
done
TRAIN_PID="$(pgrep -P "$HIB_PID" | head -1)"
echo "hib run pid=$HIB_PID, training pid=$TRAIN_PID, $(steps_of "$OUT" | wc -l) steps printed"
check "the training process is a child of hib run" test -n "$TRAIN_PID"
pstree -ap "$HIB_PID" > "$WORK/tree-external-before.txt" 2>&1

section "disruption: hib checkpoint"
LOCATION="$("$HIB" checkpoint --workload e2e --deadline 60s 2> "$WORK/checkpoint.err")"
CP_RC=$?
check "hib checkpoint exits 0 and prints the restore point" test "$CP_RC" -eq 0 -a -n "$LOCATION"
[ "$CP_RC" -ne 0 ] && cat "$WORK/checkpoint.err"
wait "$HIB_PID"; RUN_RC=$?
check "hib run exits 0 after the checkpoint ended the training process" test "$RUN_RC" -eq 0
check "the training process is gone" bash -c "! kill -0 $TRAIN_PID 2>/dev/null"
pstree -ap "$HIB_PID" > "$WORK/tree-external-after.txt" 2>&1

STEP="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$LOCATION/manifest.json")"
LAST_PRINTED="$(steps_of "$OUT" | tail -1 | awk '{print $2}')"
check "the manifest step ($STEP) is the last step the process printed ($LAST_PRINTED)" test "$STEP" = "$LAST_PRINTED"
check "the manifest's process snapshot is ok, with image files" "$PY" -c "
import json,sys
m=json.load(open(sys.argv[1]))['layers']['process_snapshot']
sys.exit(0 if m['status']=='ok' and len(m['files'])>0 else 1)" "$LOCATION/manifest.json"
check "the manifest's application checkpoint is ok" "$PY" -c "
import json,sys
m=json.load(open(sys.argv[1]))['layers']['application_checkpoint']
sys.exit(0 if m['status']=='ok' else 1)" "$LOCATION/manifest.json"
echo "images: $(ls "$LOCATION/process" | wc -l) files, $(du -sh "$LOCATION/process" | cut -f1)"
rm -rf "$WORK/rp_copy"; cp -a "$LOCATION" "$WORK/rp_copy"   # kept for the fallback test

section "restore from the process snapshot"
timeout 180 "$HIB" restore --store "$STORE" "$LOCATION" > "$WORK/restore.hib.log" 2>&1
RESTORE_RC=$?
check "hib restore exits 0" test "$RESTORE_RC" -eq 0
check "the restore did not fall back to the application checkpoint" bash -c "! grep -q 'did not restore' '$WORK/restore.hib.log'"
[ "$RESTORE_RC" -ne 0 ] && { cat "$WORK/restore.hib.log"; tail -15 "$LOCATION/criu-restore.log" 2>/dev/null; }

STEP_NUMBERS="$(steps_of "$OUT" | awk '{printf "%s ", $2}')"
check "every step 1..$UPDATES appears exactly once, in order, across the checkpoint and restore" \
  test "$STEP_NUMBERS" = "$(seq -s ' ' 1 "$UPDATES") "
check "the training output equals a run that never stopped (same losses)" diff <(steps_of "$OUT") <(steps_of "$WORK/reference.txt")

section "process tree log"
python_pid() { # python_pid <label>: the training process's pid in that section of the tree log
  awk -v label="$1" '$0 ~ "^=== " label {f=1; next} /^===/ {f=0} f' "$TREELOG" | grep -E '^ +python' | head -1 | sed -E 's/.* pid=([0-9]+) .*/\1/'
}
order_ok() {
  local a b c d
  a=$(grep -n '^=== before-checkpoint' "$TREELOG" | head -1 | cut -d: -f1)
  b=$(grep -n '^=== after-dump' "$TREELOG" | head -1 | cut -d: -f1)
  c=$(grep -n '^=== before-restore' "$TREELOG" | head -1 | cut -d: -f1)
  d=$(grep -n '^=== after-restore' "$TREELOG" | head -1 | cut -d: -f1)
  [ -n "$a" ] && [ -n "$b" ] && [ -n "$c" ] && [ -n "$d" ] && [ "$a" -lt "$b" ] && [ "$b" -lt "$c" ] && [ "$c" -lt "$d" ]
}
PID_BEFORE="$(python_pid before-checkpoint)"
PID_AFTER="$(python_pid after-restore)"
check "the log has before-checkpoint, after-dump, before-restore, after-restore, in that order" order_ok
check "before the checkpoint the log shows the training process (pid ${PID_BEFORE:-?})" test -n "$PID_BEFORE"
check "after the dump the log no longer shows it" test -z "$(python_pid after-dump)"
check "before the restore the log shows no training process" test -z "$(python_pid before-restore)"
check "after the restore the log shows it again, with the same pid (${PID_AFTER:-?})" test -n "$PID_AFTER" -a "$PID_BEFORE" = "$PID_AFTER"
echo; echo "---- $TREELOG"; cat "$TREELOG"

section "fallback: a corrupted process snapshot"
PAGES="$(ls "$WORK/rp_copy"/process/pages-*.img | head -1)"
printf 'X' >> "$PAGES"
export HIB_LOG_DIR="$WORK/logs2"
FB_OUT="$HIB_LOG_DIR/e2e.out"
timeout 180 "$HIB" restore --store "$WORK/store2" "$WORK/rp_copy" > "$WORK/fallback.hib.log" 2>&1
FB_RC=$?
check "hib restore exits 0 from the application checkpoint" test "$FB_RC" -eq 0
check "it says the process snapshot did not restore, and names the corrupted image" \
  grep -q "did not restore.*$(basename "$PAGES")" "$WORK/fallback.hib.log"
FIRST_AFTER=$((STEP + 1))
check "the fallback continues at step $FIRST_AFTER and matches the reference losses" \
  diff <(steps_of "$FB_OUT") <(steps_of "$WORK/reference.txt" | awk -v s="$STEP" '$2 > s')
[ "$FB_RC" -ne 0 ] && cat "$WORK/fallback.hib.log"

echo
echo "passed: $pass   failed: $fail   (logs and images in $WORK)"
[ "$fail" -eq 0 ]
