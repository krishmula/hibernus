import sys
W, H = 1640, 1010
BLUE = ("#E8F0FE", "#1A73E8", "#174EA6")
GREEN = ("#E6F4EA", "#188038", "#0D652D")
AMBER = ("#FEF7E0", "#E37400", "#8A4B00")
GRAY = ("#F1F3F4", "#5F6368", "#3C4043")
SAVE, REST = "#1A73E8", "#8430CE"
F = "Helvetica, Arial, sans-serif"
o = []
def box(x, y, w, h, c, title, sub=None, dash=False, tsize=20):
    d = ' stroke-dasharray="7 5"' if dash else ""
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{c[0]}" stroke="{c[1]}" stroke-width="2"{d}/>')
    cy = y + h / 2 + (-9 if sub else 7)
    o.append(f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" font-family="{F}" font-size="{tsize}" font-weight="700" fill="{c[2]}">{title}</text>')
    if sub:
        o.append(f'<text x="{x+w/2}" y="{cy+24}" text-anchor="middle" font-family="{F}" font-size="15" fill="#3C4043">{sub}</text>')
def arrow(x1, y1, x2, y2, col, dash=False):
    d = ' stroke-dasharray="6 5"' if dash else ""
    m = "s" if col == SAVE else ("r" if col == REST else "g")
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.5"{d} marker-end="url(#{m})"/>')
def badge(x, y, t, col):
    o.append(f'<circle cx="{x}" cy="{y}" r="15" fill="{col}"/>')
    o.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-family="{F}" font-size="13" font-weight="700" fill="#fff">{t}</text>')
def label(x, y, lines, col="#3C4043", anchor="start", size=15, bold_first=False):
    for i, s in enumerate(lines):
        wgt = ' font-weight="700"' if (bold_first and i == 0) else ""
        o.append(f'<text x="{x}" y="{y+i*19}" text-anchor="{anchor}" font-family="{F}" font-size="{size}"{wgt} fill="{col}">{s}</text>')
def host(x, y, w, h, title):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="#FAFAFA" stroke="#9AA0A6" stroke-width="1.5"/>')
    o.append(f'<text x="{x+20}" y="{y+h-18}" font-family="{F}" font-size="17" font-weight="700" fill="#3C4043">{title}</text>')

o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
o.append('<defs>' + "".join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for i, c in (("s", SAVE), ("r", REST), ("g", "#5F6368"))) + '</defs>')
o.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# Saving host
AX, AW = 40, 520; ax, aw = AX + 30, AW - 60; acx = ax + aw / 2
box(ax, 40, aw, 70, GRAY, "Disruption signal", "spot notice, maintenance, or the demo's emulated signal", dash=True, tsize=18)
host(AX, 170, AW, 690, "Saving host (Linux, NVIDIA GPU)")
box(ax, 220, aw, 66, GRAY, "Disruption handler", "outside Hibernus", dash=True)
box(ax, 330, aw, 56, BLUE, "hib checkpoint", "short-lived client", tsize=18)
box(ax, 440, aw, 84, BLUE, "Supervisor", "hib run · parent of the training process")
box(ax, 600, aw, 100, GREEN, "Training process", "train.py + training hook")
box(ax, 740, aw, 50, GREEN, "GPU", None, tsize=18)
arrow(acx, 110, acx, 218, "#5F6368", dash=True)
arrow(acx, 286, acx, 328, SAVE); badge(acx + 34, 307, "1", SAVE); label(acx + 58, 312, ["request a checkpoint"])
arrow(ax + 120, 386, ax + 120, 438, SAVE); label(ax + 108, 418, ["local socket"], anchor="end", size=14)
arrow(ax + aw - 80, 440, ax + aw - 80, 388, SAVE); badge(ax + aw - 112, 414, "5", SAVE); label(ax + aw - 132, 419, ["confirm: step, checkpoint time"], anchor="end", size=14)
arrow(ax + 80, 524, ax + 80, 598, SAVE); badge(ax + 110, 548, "2", SAVE); label(ax + 134, 553, ["pause at the update boundary"], size=14)
arrow(ax + aw - 80, 524, ax + aw - 80, 598, SAVE); badge(ax + aw - 110, 578, "3", SAVE); label(ax + aw - 134, 583, ["snapshot GPU and process"], anchor="end", size=14)
o.append(f'<line x1="{acx}" y1="700" x2="{acx}" y2="740" stroke="#188038" stroke-width="2"/>')

# Store
SX, SW = 700, 240; scx = SX + SW / 2
o.append(f'<rect x="{SX}" y="400" width="{SW}" height="250" rx="12" fill="{AMBER[0]}" stroke="{AMBER[1]}" stroke-width="2"/>')
label(scx, 432, ["Checkpoint store"], AMBER[2], "middle", 19, True)
label(scx, 452, ["S3"], "#3C4043", "middle", 15)
for i, (t, s) in enumerate((("Process snapshot", "CRIU + cuda-checkpoint"), ("Application checkpoint", "model, optimizer, data position"), ("Manifest", "written last, marks it complete"))):
    y = 472 + i * 58
    o.append(f'<rect x="{SX+16}" y="{y}" width="{SW-32}" height="48" rx="6" fill="#FFFFFF" stroke="{AMBER[1]}" stroke-width="1"/>')
    label(scx, y + 21, [t], "#3C4043", "middle", 14, True)
    label(scx, y + 39, [s], "#5F6368", "middle", 12)
arrow(ax + aw, 482, SX - 2, 482, SAVE); badge(625, 455, "4", SAVE); label(630, 512, ["upload, verify"], anchor="middle", size=14)

# Restore host
BX, BW = 1080, 520; bx, bw = BX + 30, BW - 60; bcx = bx + bw / 2
box(bx, 40, bw, 70, GRAY, "Launcher", "Auto Scaling group, SkyPilot, or a person", dash=True, tsize=18)
host(BX, 170, BW, 690, "Restore host (same GPU model and software stack)")
arrow(bcx, 110, bcx, 438, REST); badge(bcx + 34, 260, "A", REST); label(bcx + 58, 255, ["bring up the host,"], size=14); label(bcx + 58, 273, ["run hib restore"], size=14)
box(bx, 440, bw, 84, BLUE, "Supervisor", "hib restore")
box(bx, 600, bw, 100, GREEN, "Training process", "restored, continues from the same step")
box(bx, 740, bw, 50, GREEN, "GPU", None, tsize=18)
arrow(SX + SW + 2, 482, bx - 2, 482, REST); badge(1015, 455, "B", REST); label(1010, 512, ["download,"], anchor="middle", size=14); label(1010, 530, ["check fingerprint"], anchor="middle", size=14)
arrow(bcx, 524, bcx, 598, REST); badge(bcx + 34, 561, "C", REST); label(bcx + 58, 548, ["restore the snapshot,", "or fall back to the", "application checkpoint"], size=14)
o.append(f'<line x1="{bcx}" y1="700" x2="{bcx}" y2="740" stroke="#188038" stroke-width="2"/>')

# Legend
ly = 920
items = [(BLUE, "Hibernus", False), (GREEN, "Workload", False), (AMBER, "Storage", False), (GRAY, "Outside Hibernus", True)]
x = 40
for c, t, d in items:
    dd = ' stroke-dasharray="5 4"' if d else ""
    o.append(f'<rect x="{x}" y="{ly-14}" width="26" height="18" rx="4" fill="{c[0]}" stroke="{c[1]}" stroke-width="2"{dd}/>')
    label(x + 36, ly, [t], size=15); x += 70 + len(t) * 8
x += 30
for col, t in ((SAVE, "Checkpoint path, steps 1 to 5"), (REST, "Restore path, steps A to C")):
    m = "s" if col == SAVE else "r"
    o.append(f'<line x1="{x}" y1="{ly-5}" x2="{x+40}" y2="{ly-5}" stroke="{col}" stroke-width="2.5" marker-end="url(#{m})"/>')
    label(x + 52, ly, [t], size=15); x += 90 + len(t) * 8
label(40, ly + 38, ["Step 2: the training hook stops the loop before the next batch and writes the application checkpoint.", "Step 3: cuda-checkpoint moves GPU state into host memory, then CRIU writes the process snapshot."], "#5F6368", size=14)
o.append('</svg>')
open(sys.argv[1], "w").write("\n".join(o))
