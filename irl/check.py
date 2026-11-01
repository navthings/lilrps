import cadquery as cq

from cad import (BOSS_R, G, SH, SHAFT, SL, SW, TAB_AT, TAB_L, TAB_T, blk, flat, parts, xcyl, PRINT)

BED = 180


def servo():
    # dummy servo a
    body = blk(G, G + SH, -SW / 2, SW / 2, SHAFT - SL, SHAFT)
    z0 = SHAFT - SL / 2 - TAB_L / 2
    tab = blk(G + TAB_AT, G + TAB_AT + TAB_T, -SW / 2, SW / 2, z0, z0 + TAB_L)
    return body.union(tab).union(xcyl(BOSS_R, G + SH, G + SH + 4))


def vol(a, b):
    try:
        return a.intersect(b).val().Volume()
    except Exception:
        return 0.0


def turn(shape, deg):
    # fingers curl toward +y
    return shape.rotate((0, 0, 0), (1, 0, 0), -deg)


p = parts()
s = servo()
servos = s.union(s.mirror("YZ"))
fixed = p["palm"].union(p["cover"]).union(p["thumb"]).union(p["base"]).union(servos)
bad = []
print(f"servo vs palm: {vol(servos, p['palm']):.2f} mm3")
for name in ("yoke_a", "yoke_b"):
    worst = max((vol(turn(p[name], d), fixed), d) for d in range(-5, 106, 5))
    print(f"{name} vs fixed parts, -5..105 deg: worst {worst[0]:.2f} mm3 at {worst[1]} deg")
    bad += [name] if worst[0] > 0.5 else []
for da in (0, 50, 100):
    for db in (0, 50, 100):
        v = vol(turn(p["yoke_a"], da), turn(p["yoke_b"], db))
        if v > 0.5:
            bad.append(f"a{da} b{db}")
            print(f"yoke a {da} vs b {db}: {v:.1f} mm3")
for name, shape in p.items():
    bb = flat(shape, PRINT[name]).val().BoundingBox()
    ok = bb.xlen <= BED and bb.ylen <= BED and bb.zlen <= BED
    solids = len(shape.solids().vals())
    print(f"{name}: {bb.xlen:.0f} x {bb.ylen:.0f} x {bb.zlen:.0f} mm, {solids} solid{'s' if solids > 1 else ''}{'' if ok else '  TOO BIG'}")
    if not ok or solids != 1:
        bad.append(name)
print("ALL GOOD" if not bad else f"PROBLEMS: {bad}")
