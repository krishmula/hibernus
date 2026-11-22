#!/usr/bin/env bash
# Prepares a fresh Ubuntu host for the CRIU test, as root: CRIU from the CRIU project's PPA, and a
# CPU-only torch in its own venv for the torch case.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

. /etc/os-release
echo "os: $PRETTY_NAME, kernel $(uname -r)"
apt-get update -q
if ! apt-get install -y -q criu; then
	echo "criu is not in the $VERSION_CODENAME archive, so add the CRIU PPA"
	add-apt-repository -y ppa:criu/ppa
	apt-get install -y -q criu
fi
criu --version
apt-get install -y -q python3-venv
python3 -m venv /opt/criu-test-venv
/opt/criu-test-venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/opt/criu-test-venv/bin/python -c 'import torch; print("torch", torch.__version__)'
