import sys, time, torch
from torch import nn
total = int(sys.argv[1]); sleep = float(sys.argv[2])
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
dev = torch.device("cuda")
model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 8)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
x = torch.randn(1024, 256, device=dev); y = torch.randint(0, 8, (1024,), device=dev)
ballast = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev).fill_(7)  # 512 MB of VRAM that must survive
print("pid", __import__("os").getpid(), flush=True)
for step in range(1, total + 1):
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward(); opt.step(); opt.zero_grad()
    print(f"step {step} loss {loss.item():.9f} ballast {int(ballast[::1048576].sum())}", flush=True)
    time.sleep(sleep)
