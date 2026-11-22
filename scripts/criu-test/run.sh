#!/usr/bin/env bash
# Runs the CRIU test cases on one host, as root, after setup.sh. Each case dumps a stand-in
# training process and restores it on the same host. Exits 1 if any check fails.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=${1:-/var/tmp/hib-criu-test}
rm -rf "$work"
echo "os: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r), $(criu --version | head -1)"
echo "== criu check"
criu check 2>&1 | tail -5

result=0
run_case() {
	local name=$1 dump_flags=$2 restore_flags=$3
	echo "== case $name"
	"$here/criu-test" dump -dir "$work/$name" $dump_flags || { result=1; return; }
	"$here/criu-test" restore -dir "$work/$name" $restore_flags || result=1
	echo "-- training process output"
	cat "$work/$name/trainer.log"
}

# The design: plain Python, restore under a subreaper.
run_case plain "-python python3" ""
# The same with torch loaded, so its thread pool exists when SIGCONT arrives.
run_case torch "-python /opt/criu-test-venv/bin/python -torch" ""
# An alternative to the subreaper: CRIU makes the restored process the Supervisor's own child.
run_case sibling "-python python3" "-sibling"

echo "== result: $([ $result -eq 0 ] && echo PASS || echo FAIL)"
exit $result
