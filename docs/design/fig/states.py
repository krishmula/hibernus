"""Draw the workload states figure. Usage: python3 states.py states.svg"""
import sys

W, H = 1240, 560
F = "Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"
BW, BH = 220, 76
COL = {1: 280, 2: 640, 3: 1000}
ROW = {0: 40, 1: 210, 2: 400}

# name: (light fill, light stroke, light title, dark fill, dark stroke, dark title)
PALETTE = {
    "run":  ("#E8F0FE", "#1A73E8", "#174EA6", "#10243E", "#4C8DF6", "#A8C7FA"),
    "ckpt": ("#FEF7E0", "#E37400", "#8A4B00", "#33260A", "#E8A33D", "#F5CF8A"),
    "done": ("#E6F4EA", "#188038", "#0D652D", "#0F2A19", "#3FB950", "#A6E3B5"),
    "fin":  ("#F1F3F4", "#5F6368", "#3C4043", "#1F242B", "#8B949E", "#C9D1D9"),
    "bad":  ("#FCE8E6", "#C5221F", "#A50E0E", "#3A1414", "#F85149", "#FFB3AD"),
    "rest": ("#F3E8FD", "#8430CE", "#681DA8", "#2A1A3D", "#BC8CFF", "#DEC8FF"),
}
BASE = {"bg": ("#FFFFFF", "#0D1117"), "ink": ("#202124", "#E6EDF3"),
        "muted": ("#5F6368", "#9198A1"), "edge": ("#5F6368", "#9198A1"),
        "edgebad": ("#C5221F", "#F85149")}


def css(dark):
    i = 1 if dark else 0
    rules = [f".bg{{fill:{BASE['bg'][i]}}}", f".ink{{fill:{BASE['ink'][i]}}}",
             f".muted{{fill:{BASE['muted'][i]}}}",
             f".edge{{stroke:{BASE['edge'][i]}}}", f".edgebad{{stroke:{BASE['edgebad'][i]}}}",
             f".head-edge{{fill:{BASE['edge'][i]}}}", f".head-edgebad{{fill:{BASE['edgebad'][i]}}}",
             f".pill{{fill:{BASE['bg'][i]};stroke:{BASE['edge'][i]}}}"]
    for k, c in PALETTE.items():
        f, s, t = c[3:] if dark else c[:3]
        rules.append(f".f-{k}{{fill:{f};stroke:{s}}} .t-{k}{{fill:{t}}}")
    return "".join(rules)


o = []


def text(x, y, s, cls="ink", size=14, anchor="start", weight="400", font=F):
    o.append(f'<text x="{x}" y="{y}" class="{cls}" font-family="{font}" font-size="{size}" '
             f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def state(col, row, kind, name, sub, final=False):
    x, y = COL[col], ROW[row]
    o.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="12" class="f-{kind}" stroke-width="2"/>')
    if final:
        o.append(f'<rect x="{x+5}" y="{y+5}" width="{BW-10}" height="{BH-10}" rx="8" fill="none" '
                 f'class="f-{kind}" stroke-width="1.5" style="fill:none"/>')
    text(x + BW / 2, y + 33, name, f"t-{kind}", 19, "middle", "700")
    text(x + BW / 2, y + 55, sub, "muted", 13, "middle")


def pill(x, cy, label):
    w, h = 200, 44
    o.append(f'<rect x="{x}" y="{cy-h/2}" width="{w}" height="{h}" rx="22" class="pill" '
             f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    text(x + w / 2, cy + 5, label, "ink", 13, "middle", "600", MONO)


def arrow(d, bad=False):
    k = "edgebad" if bad else "edge"
    o.append(f'<path d="{d}" fill="none" class="{k}" stroke-width="2" marker-end="url(#h-{k})"/>')


def cx(col):
    return COL[col] + BW / 2


def cy(row):
    return ROW[row] + BH / 2


o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
o.append(f'<style>{css(False)}@media (prefers-color-scheme: dark){{{css(True)}}}</style>')
o.append("<defs>" + "".join(
    f'<marker id="h-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="head-{k}"/></marker>'
    for k in ("edge", "edgebad")) + "</defs>")
o.append(f'<rect width="{W}" height="{H}" class="bg"/>')

state(1, 0, "fin", "Finished", "training ended by itself", final=True)
state(1, 1, "run", "Running", "training runs steps")
state(2, 1, "ckpt", "Checkpointing", "paused, writing the restore point")
state(3, 1, "done", "Checkpointed", "paused until the host goes away", final=True)
state(1, 2, "rest", "Restoring", "bringing training back")
state(2, 2, "bad", "Failed", "no restore point", final=True)

pill(30, cy(1), "hib run")
pill(30, cy(2), "hib restore &lt;location&gt;")
arrow(f"M230,{cy(1)} L{COL[1]-2},{cy(1)}")
arrow(f"M230,{cy(2)} L{COL[1]-2},{cy(2)}")

# Running -> Checkpointing -> Checkpointed
arrow(f"M{COL[1]+BW},{cy(1)} L{COL[2]-2},{cy(1)}")
text((COL[1] + BW + COL[2]) / 2, cy(1) - 10, "hib checkpoint", "ink", 13, "middle", "600", MONO)
arrow(f"M{COL[2]+BW},{cy(1)} L{COL[3]-2},{cy(1)}")
text((COL[2] + BW + COL[3]) / 2, cy(1) - 28, "1 or 2 layers", "ink", 13, "middle")
text((COL[2] + BW + COL[3]) / 2, cy(1) - 10, "written", "ink", 13, "middle")

# Running -> Finished, Restoring -> Running
arrow(f"M{cx(1)},{ROW[1]} L{cx(1)},{ROW[0]+BH+2}")
text(cx(1) - 12, (ROW[0] + BH + ROW[1]) / 2 + 5, "training exits 0", "ink", 13, "end")
arrow(f"M{cx(1)},{ROW[2]} L{cx(1)},{ROW[1]+BH+2}")
text(cx(1) - 12, (ROW[1] + BH + ROW[2]) / 2 - 4, "training is back", "ink", 13, "end")
text(cx(1) - 12, (ROW[1] + BH + ROW[2]) / 2 + 14, "at step K", "ink", 13, "end")

# Into Failed
arrow(f"M{COL[1]+BW},{cy(2)} L{COL[2]-2},{cy(2)}", bad=True)
text((COL[1] + BW + COL[2]) / 2, cy(2) - 10, "no layer works", "ink", 13, "middle")
arrow(f"M{COL[1]+BW-40},{ROW[1]+BH} L{COL[2]+30},{ROW[2]-2}", bad=True)
text(COL[1] + BW + 52, ROW[1] + BH + 30, "training dies", "ink", 13)
arrow(f"M{cx(2)+40},{ROW[1]+BH} L{cx(2)+40},{ROW[2]-2}", bad=True)
text(cx(2) + 52, (ROW[1] + BH + ROW[2]) / 2 - 4, "deadline passes,", "ink", 13)
text(cx(2) + 52, (ROW[1] + BH + ROW[2]) / 2 + 14, "or training dies", "ink", 13)

# Self loops
for col, lines in ((2, ["retries failed layers", "until the deadline"]),
                   (3, ["a repeat hib checkpoint", "gets the same restore point"])):
    x0, x1, y = cx(col) - 34, cx(col) + 34, ROW[1]
    arrow(f"M{x0},{y} C{x0},{y-56} {x1},{y-56} {x1},{y-2}")
    text(cx(col), y - 78, lines[0], "muted", 13, "middle")
    text(cx(col), y - 61, lines[1], "muted", 13, "middle")

# Note under Restoring
text(cx(1), ROW[2] + BH + 22, "a hib checkpoint that arrives here", "muted", 13, "middle")
text(cx(1), ROW[2] + BH + 39, "waits until Running", "muted", 13, "middle")

# Legend
lx, ly = 760, 520
o.append(f'<rect x="{lx}" y="{ly-16}" width="34" height="22" rx="5" class="f-fin" stroke-width="2"/>')
o.append(f'<rect x="{lx+4}" y="{ly-12}" width="26" height="14" rx="3" class="f-fin" stroke-width="1.5" style="fill:none"/>')
text(lx + 44, ly, "final state: nothing leaves it", "muted", 13)
o.append(f'<path d="M{lx},{ly+26} L{lx+34},{ly+26}" class="edgebad" stroke-width="2" marker-end="url(#h-edgebad)"/>')
text(lx + 44, ly + 30, "leads to Failed", "muted", 13)

o.append("</svg>")
open(sys.argv[1], "w").write("\n".join(o))
