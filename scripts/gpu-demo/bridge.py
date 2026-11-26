"""Turns the files one scenario run leaves in a work folder into the one JSON document the
dashboard reads. Prints ASCII JSON. usage: bridge.py <work folder>
"""
import json
import math
import os
import re
import sys
import time

W = sys.argv[1]
STEP = re.compile(r"^step (\d+) epoch (\d+) loss ([0-9.eE+-]+)")


def load(name, default=None):
    try:
        return json.load(open(os.path.join(W, name)))
    except Exception:
        return default


def steps(name):
    out = []
    try:
        for line in open(os.path.join(W, name), errors="replace"):
            m = STEP.match(line)
            if m:
                out.append((int(m.group(1)), m.group(3)))
    except Exception:
        pass
    return out


def thin(points, limit=520):
    if len(points) <= limit:
        return points
    stride = int(math.ceil(len(points) / float(limit)))
    kept = points[::stride]
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


def fmt(n):
    return "{:,}".format(int(n))


def fmt_lr(v):
    mant, exp = ("%.2e" % v).split("e")
    return "%se%d" % (mant, int(exp))


def short(h):
    return h[:10] + "..." + h[-6:]


def match(eq):
    return {"text": "Exact Match", "tone": "ok"} if eq else {"text": "Mismatch", "tone": "warn"}


def row(label, before, after, delta, bsub=None, asub=None, wrap=False):
    r = {"label": label, "before": before, "after": after, "delta": delta, "wrap": wrap}
    if bsub:
        r["beforeSub"] = bsub
    if asub and after is not None:
        r["afterSub"] = asub
    return r


def diff_delta(b, a, unit, tol):
    d = a - b
    sign = "+" if d >= 0 else "-"
    txt = "%s%s %s" % (sign, ("%g" % abs(round(d, 1))), unit)
    return {"text": txt, "tone": "ok" if abs(d) <= tol else "info"}


state = load("state.json", {})
before = load("before.json")
after = load("after.json")
upd = state.get("upd", 0)
k = state.get("k")

run = steps("run.log")
rest = steps("restore.log")
base = steps("baseline.log")
base_map = dict(base)
rest_map = dict(rest)

app_rows, sys_rows = [], []
if before:
    b, a = before, after
    total_epochs = int(math.ceil(upd / float(b["bpe"]))) if upd else 0
    nxt = None
    if k is not None:
        nxt_b = base_map.get(k + 1)
        nxt_a = rest_map.get(k + 1)
        nxt = (nxt_b, nxt_a)
    app_rows = [
        row("Epoch", "%d / %d" % (b["epoch"] + 1, total_epochs), a and "%d / %d" % (a["epoch"] + 1, total_epochs),
            a and match(a["epoch"] == b["epoch"])),
        row("Global step", fmt(b["step"]), a and fmt(a["step"]),
            a and {"text": "%+d steps" % (a["step"] - b["step"]), "tone": "ok" if a["step"] == b["step"] else "warn"}),
        row("Model weights hash", short(b["model_hash"]), a and short(a["model_hash"]), a and match(a["model_hash"] == b["model_hash"]), wrap=True),
        row("Optimizer state hash", short(b["opt_hash"]), a and short(a["opt_hash"]), a and match(a["opt_hash"] == b["opt_hash"]), wrap=True),
        row("Learning rate", fmt_lr(b["lr"]), a and fmt_lr(a["lr"]), a and match(a["lr"] == b["lr"])),
        row("Dataloader position", "batch %d / %d" % (b["batch"], b["bpe"]), a and "batch %d / %d" % (a["batch"], a["bpe"]),
            a and match(a["batch"] == b["batch"])),
        row("Sample offset", fmt(b["offset"]), a and fmt(a["offset"]),
            a and {"text": "%+d samples" % (a["offset"] - b["offset"]), "tone": "ok" if a["offset"] == b["offset"] else "warn"}),
        row("RNG state hash", short(b["rng_hash"]), a and short(a["rng_hash"]), a and match(a["rng_hash"] == b["rng_hash"]), wrap=True),
    ]
    if nxt:
        have = nxt[0] is not None and nxt[1] is not None
        app_rows.append(row("Loss at step %s" % fmt(k + 1), nxt[0] if nxt[0] else "-", nxt[1] if nxt[1] else None,
                            match(nxt[0] == nxt[1]) if have else None,
                            bsub="uninterrupted reference run", asub="restored run"))
    fp_b = b.get("fp", {})
    fp_a = a.get("fp", {}) if a else None
    fp_line = lambda f: "%s / driver %s" % (f.get("gpu", "?"), f.get("driver", "?"))
    fp_sub = lambda f: "CUDA %s / PyTorch %s" % (f.get("cuda", "?"), f.get("pytorch", "?"))
    sys_rows = [
        row("Process PID", str(b.get("pid", "-")), a and str(a.get("pid", "-")),
            a and {"text": "new process", "tone": "info"}, bsub="paused at checkpoint", asub="started by hib restore"),
        row("GPU memory (process)", "%d MiB" % b.get("vram_mib", 0), a and "%d MiB" % a.get("vram_mib", 0),
            a and diff_delta(b.get("vram_mib", 0), a.get("vram_mib", 0), "MiB", 64)),
        row("Host RAM (RSS)", "%.0f MiB" % b.get("rss_mib", 0), a and "%.0f MiB" % a.get("rss_mib", 0),
            a and diff_delta(b.get("rss_mib", 0), a.get("rss_mib", 0), "MiB", 64)),
        row("Open file descriptors", fmt(b.get("fds", 0)), a and fmt(a.get("fds", 0)),
            a and diff_delta(b.get("fds", 0), a.get("fds", 0), "fds", 8)),
        row("Environment fingerprint", fp_line(fp_b), a and fp_line(fp_a), a and match(fp_a == fp_b),
            bsub=fp_sub(fp_b), asub=fp_sub(fp_a) if fp_a else None),
        row("GPU process snapshot (CRIU)", str(b.get("layers", {}).get("process_snapshot", "-")),
            a and str(a.get("layers", {}).get("process_snapshot", "-")), a and {"text": "not in v1", "tone": "mute"}),
    ]

# Verdict: every loss the restored run printed against the uninterrupted reference.
compared = mism = 0
for s_, loss in rest:
    if s_ in base_map:
        compared += 1
        if base_map[s_] != loss:
            mism += 1
verdict = None
if rest:
    verdict = {"compared": compared, "mismatches": mism, "of": max(0, upd - (k or 0)), "final": bool(state.get("fin"))}

series = [[s_, float(l), 0] for s_, l in run] + [[s_, float(l), 1] for s_, l in rest]
base_series = [[s_, float(l)] for s_, l in base]
latest = rest[-1][0] if rest else (run[-1][0] if run else 0)

logs = []
try:
    for line in open(os.path.join(W, "events.log"), errors="replace"):
        parts = line.rstrip("\n").split(" ", 3)
        if len(parts) == 4:
            logs.append(parts)
except Exception:
    pass

doc = {
    "v": 1,
    "updatedAt": int(time.time() * 1000),
    "workload": state.get("workload", ""),
    "host": state.get("host", {}),
    "total": upd,
    "phase": state.get("phase", 1),
    "status": state.get("status", "Running"),
    "mode": state.get("mode", "idle"),
    "finished": bool(state.get("fin")),
    "step": latest,
    "disruptStep": state.get("disruptStep"),
    "restoredStep": state.get("restoredStep"),
    "ckpt": {"id": state.get("ckptId"), "step": k, "bytes": state.get("ckptBytes"), "ms": state.get("ckptMs"), "loc": state.get("loc")} if k is not None else None,
    "restoreMs": state.get("restoreMs"),
    "appRows": app_rows,
    "sysRows": sys_rows,
    "verdict": verdict,
    "series": thin(series),
    "baseline": thin(base_series),
    "logs": logs[-140:],
}
print(json.dumps(doc, ensure_ascii=True, separators=(",", ":")))
