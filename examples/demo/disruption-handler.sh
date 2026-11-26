#!/usr/bin/env bash
# An example disruption handler: the piece outside Hibernus that notices a disruption and runs
# hib checkpoint. It polls for the spot notice that AWS posts 2 minutes before it reclaims a spot
# host. For a demo on on-demand hosts, it also reads the same notice from NOTICE_FILE.
#
# Usage, as root on the checkpoint host: disruption-handler.sh <workload>
# HIB is the hib binary (default: hib on PATH). The restore point's location goes to LOCATION_FILE.
set -u
workload=$1
hib=${HIB:-hib}
notice_file=${NOTICE_FILE:-/run/hib-demo/spot-notice}
location_file=${LOCATION_FILE:-/run/hib-demo/location}

spot_notice() {
	local token
	token=$(curl -sf -m 2 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" http://169.254.169.254/latest/api/token) || return 1
	curl -sf -m 2 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/spot/instance-action
}

# The notice is JSON, such as {"action": "terminate", "time": "2026-11-26T03:12:00Z"}.
field() { python3 -c "import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$1" "$2"; }

while :; do
	notice=$(spot_notice 2>/dev/null || cat "$notice_file" 2>/dev/null || true)
	[ -n "$notice" ] && break
	sleep 2
done

action=$(field "$notice" action)
deadline=$(field "$notice" time)
echo "[disruption handler] spot notice: $action at $deadline. Running hib checkpoint." >&2
start=$(date +%s.%N)
if location=$("$hib" checkpoint --workload "$workload" --deadline "$deadline"); then
	took=$(python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$start" "$(date +%s.%N)")
	echo "[disruption handler] checkpointed in ${took} s, before the $deadline deadline: $location" >&2
	echo "$location" >"$location_file"
else
	echo "[disruption handler] hib checkpoint failed" >&2
	echo "failed" >"$location_file"
fi
