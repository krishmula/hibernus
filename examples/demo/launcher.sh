#!/usr/bin/env bash
# An example launcher's last step: the piece outside Hibernus that brings a workload back. On a
# restore host that is already up, it waits until the workload's newest restore point is complete
# in S3, then runs hib restore. A real launcher, such as an Auto Scaling group, would also bring up
# the host first.
#
# Usage, as root on the restore host: launcher.sh s3://<bucket>/<workload>/
# HIB is the hib binary (default: hib on PATH). It needs the AWS CLI.
set -euo pipefail
location=${1:?usage: launcher.sh s3://<bucket>/<workload>/}
hib=${HIB:-hib}
path=${location#s3://}
bucket=${path%%/*}
prefix=${path#*/}

complete() {
	aws s3api list-objects-v2 --bucket "$bucket" --prefix "$prefix" \
		--query "Contents[?ends_with(Key, 'manifest.json')].Key" --output text 2>/dev/null | grep -q manifest
}

echo "[launcher] waiting for a complete restore point at $location" >&2
until complete; do sleep 2; done
echo "[launcher] found one. Running: hib restore $location" >&2
exec "$hib" restore "$location"
