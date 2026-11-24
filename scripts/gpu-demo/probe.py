"""Reads real values out of one restore point and one running process.

usage: probe.py <restore point folder> <training pid or 0> <out.json>
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch


def feed(h, x):
    if isinstance(x, dict):
        for k in sorted(x, key=lambda z: str(z)):
            h.update(str(k).encode())
            feed(h, x[k])
    elif isinstance(x, (list, tuple)):
        for v in x:
            feed(h, v)
    elif isinstance(x, torch.Tensor):
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(x, np.ndarray):
        h.update(x.tobytes())
    else:
        h.update(repr(x).encode())


def digest(x):
    h = hashlib.sha256()
    feed(h, x)
    return h.hexdigest()


def proc(pid):
    out = {"pid": pid}
    try:
        for line in open("/proc/%d/status" % pid):
            if line.startswith("VmRSS:"):
                out["rss_mib"] = round(int(line.split()[1]) / 1024.0, 1)
        out["fds"] = len(os.listdir("/proc/%d/fd" % pid))
    except Exception:
        pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        for line in q.splitlines():
            p, m = [t.strip() for t in line.split(",")]
            if int(p) == pid:
                out["vram_mib"] = int(m)
    except Exception:
        pass
    return out


def main():
    folder, pid, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    payload = torch.load(os.path.join(folder, "app", "app.pt"), map_location="cpu", weights_only=False)
    manifest = json.load(open(os.path.join(folder, "manifest.json")))
    cfg = payload["config"]
    step = payload["step"]
    bpe = max(1, cfg["samples"] // cfg["batch_size"])
    opt = payload["state"]["optimizer"]
    out = {
        "step": step,
        "epoch": step // bpe,
        "batch": step % bpe,
        "bpe": bpe,
        "offset": (step % bpe) * cfg["batch_size"],
        "lr": opt["param_groups"][0]["lr"],
        "model_hash": digest(payload["state"]["model"]),
        "opt_hash": digest(opt),
        "rng_hash": digest(payload["rng"]),
        "config": cfg,
        "fp": manifest.get("fingerprint", {}),
        "layers": {k: (v.get("error") or v.get("status")) for k, v in manifest.get("layers", {}).items()},
        "files": [{"path": f["path"], "size": f["size"]} for f in manifest.get("layers", {}).get("application_checkpoint", {}).get("files", [])],
        "created_at": manifest.get("created_at"),
    }
    if pid:
        out.update(proc(pid))
    json.dump(out, open(out_path, "w"))


main()
