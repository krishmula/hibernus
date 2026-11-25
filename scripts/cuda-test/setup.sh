set -e
sudo DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:criu/ppa >/dev/null 2>&1
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq criu >/dev/null
criu --version
ls /usr/lib/criu /usr/lib/x86_64-linux-gnu/criu 2>/dev/null || echo "no plugin dir"
find / -name "cuda_plugin.so" 2>/dev/null | head
mkdir -p ~/cc && cd ~/cc
curl -fsSL -o cuda-checkpoint https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint
chmod +x cuda-checkpoint; ls -l cuda-checkpoint; sha256sum cuda-checkpoint
./cuda-checkpoint --help 2>&1 | head -12
