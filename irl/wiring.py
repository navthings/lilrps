import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RED, BLACK, ORANGE, BLUE = "#d6332a", "#333333", "#f08c28", "#2f6fd6"
fig, ax = plt.subplots(figsize=(12, 6.6))
ax.set_xlim(0, 120); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, label, fc, sub=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec="#444", lw=1.2))
    cx = x + w / 2 if label != "pico H" else x + 8
    ax.text(cx, y + h / 2 + (1.6 if sub else 0), label, ha="center", va="center", fontsize=13, fontweight="bold")
    if sub:
        ax.text(cx, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=9.5, color="#444")


def wire(pts, c, lw=3):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, lw=lw, solid_capstyle="round", zorder=1)


def dot(x, y, c):
    ax.plot(x, y, "o", color=c, ms=7, zorder=3)


# pico
box(1, 22, 29, 30, "pico H", "#cfe8cf", "usb -> your mac")
for i, (pin, label, y) in enumerate([("18", "GND", 44), ("19", "GP14", 37), ("20", "GP15", 30)]):
    ax.text(29.4, y, f"pin {pin}  {label}", va="center", ha="right", fontsize=10)
    dot(30.6, y, "#555")
# rails
ax.add_patch(FancyBboxPatch((50, 54), 64, 3, boxstyle="round,pad=0.3", fc="#f6d4d1", ec=RED))
ax.add_patch(FancyBboxPatch((50, 6), 64, 3, boxstyle="round,pad=0.3", fc="#d6dcf3", ec=BLUE))
ax.text(51, 59.5, "+ rail (breadboard)", color=RED, fontsize=10)
ax.text(51, 2, "- rail (breadboard)", color=BLUE, fontsize=10)
# battery
box(98, 26, 16, 14, "4x AA", "#f3e7c4", "4.8-6V")
wire([(110, 40), (110, 55.5)], RED); dot(110, 55.5, RED)
wire([(102, 26), (102, 7.5)], BLACK); dot(102, 7.5, BLACK)
ax.text(111, 47, "+", color=RED, fontsize=14, fontweight="bold")
ax.text(103, 16, "-", color=BLACK, fontsize=14, fontweight="bold")
# servos
for x, name, gp_y, label in [(56, "servo A", 37, "index + middle"), (76, "servo B", 30, "ring + pinky")]:
    box(x, 22, 14, 14, name, "#dfe6f7", label)
    # brown, red, orange
    wire([(x + 3, 22), (x + 3, 7.5)], "#7a4b2a"); dot(x + 3, 7.5, "#7a4b2a")
    wire([(x + 7, 36), (x + 7, 55.5)], RED); dot(x + 7, 55.5, RED)
    top, drop = (46, 43) if name == "servo A" else (49, 47)
    wire([(x + 11, 36), (x + 11, top), (drop, top), (drop, gp_y), (30.6, gp_y)], ORANGE)
# common ground
wire([(30.6, 44), (38, 44), (38, 7.5), (52, 7.5)], BLACK); dot(52, 7.5, BLACK)
ax.text(39, 18, "pico GND to\nbattery -\n(must share\nground)", fontsize=9, color="#333")
ax.text(60, 63.5, "servo wires: brown = -   red = +   orange = signal", fontsize=10.5, color="#333")
ax.text(6, 14, "dont power the servos from the pico.\nthey pull way more than it can give", fontsize=9.5, color="#a33")
fig.tight_layout()
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "wiring.png", dpi=110)
