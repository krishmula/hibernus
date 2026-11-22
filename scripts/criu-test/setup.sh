#!/usr/bin/env bash
# Prepares a fresh Ubuntu host for the CRIU test, as root: CRIU from the CRIU project's PPA, and a
# CPU-only torch in its own venv for the torch case. Ubuntu 22.04's own CRIU 3.16 can't restore
# rseq, which glibc 2.35 turns on, so the restored process segfaults. CRIU 3.17 added rseq.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

. /etc/os-release
echo "os: $PRETTY_NAME, kernel $(uname -r)"
apt-get update -q
add-apt-repository -y ppa:criu/ppa
apt-get install -y -q criu
version=$(criu --version | sed -n 's/^Version: //p')
echo "criu $version"
printf '3.17\n%s\n' "$version" | sort -V -C || { echo "criu $version is older than 3.17"; exit 1; }
apt-get install -y -q python3-venv
python3 -m venv /opt/criu-test-venv
/opt/criu-test-venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/opt/criu-test-venv/bin/python -c 'import torch; print("torch", torch.__version__)'
