#!/usr/bin/env bash
# Prepares a Linux GPU host for process snapshots. Run it as root on every checkpoint host and
# restore host of a workload, after Hibernus and its venv are installed.
#
# It installs CRIU 4.x and cuda-checkpoint, and freezes the host's packages. CRIU restores a
# training process only when every library it loaded is byte-identical on the restore host, so all
# hosts of a workload must hold the same packages. Launch them from one image, run this on each,
# and compare the "packages" line it prints.
set -u

# Apply today's security updates now, then stop automatic upgrades, so hosts can't drift apart.
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades >/dev/null 2>&1
while pgrep -f unattended-upgr >/dev/null || fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 2; done
apt-get update -qq >/dev/null 2>&1
unattended-upgrade >/dev/null 2>&1

# CRIU 4.x from its PPA: Ubuntu 22.04's CRIU 3.16 can't restore glibc's rseq, and CRIU 4.0 is the
# first with the CUDA plugin, which runs cuda-checkpoint during dump and restore.
if ! criu --version 2>/dev/null | grep -q 'Version: [4-9]'; then
	add-apt-repository -y ppa:criu/ppa >/dev/null 2>&1
	apt-get install -y -qq criu >/dev/null 2>&1
fi
if ! command -v cuda-checkpoint >/dev/null; then
	curl -fsSL -o /usr/local/bin/cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint
	chmod +x /usr/local/bin/cuda-checkpoint
fi

echo "criu:            $(criu --version | head -1)"
echo "cuda-checkpoint: $(cuda-checkpoint --help 2>&1 | sed -n 2p | cut -d. -f1-4)"
echo "gpu, driver:     $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo none)"
echo "packages:        $(dpkg-query -W -f='${Package} ${Version}\n' | md5sum | cut -c1-12)"
