from pathlib import Path

import cadquery as cq

OUT = Path(__file__).parent / "stl"

# servo, mg90s-ish (mm). fit_test.stl checks these
SL, SW, SH = 22.8, 12.2, 22.5        # length, width, bottom to top face
SHAFT = 5.9                          # shaft from short end
TAB_L, TAB_T, TAB_AT = 32.2, 2.5, 16.0
BOSS_R = 6.0
HORN_OUT = 30.8                      # horn outer face from servo bottom
HUB_D, ARM_W, ARM_L, ARM_D = 7.6, 6.4, 19.0, 1.8   # horn pocket
CL = 0.3
G = 1.0                              # centre to servo bottom
WALL = 2.0

# palm
X_OUT = G + SH + CL + WALL
Y_BACK, Y_FRONT = -(SW / 2 + CL), SW / 2 + CL + WALL
Z_TAB = (SHAFT - SL / 2 - TAB_L / 2, SHAFT - SL / 2 + TAB_L / 2)
Z_TOP, Z_BOT = Z_TAB[1] + CL + 1.6, Z_TAB[0] - CL - WALL
R_CLIP = 14.3                        # fingers sweep outside this
WRIST_X, WRIST_BOT, WRIST_HOLLOW = 19.0, -60.0, 13.0
COVER_T, COVER_BOT = 2.4, -50.0
SCREWS = [(22.9, 9.4), (-22.9, 9.4), (16.0, -45.0), (-16.0, -45.0)]

# yoke
XP = G + HORN_OUT - ARM_D            # plate inner face
PT = 4.0
R_IN, R_OUT = 16.0, 21.0
FT = 12.0                            # finger thickness
FW = 14.0
FINGERS = {"a": [(25.0, 46.0, 12.0), (8.5, 52.0, 2.0)],      # index, middle
           "b": [(25.0, 38.0, 12.0), (8.5, 48.0, 2.0)]}      # pinky, ring (mirrored)


def blk(x0, x1, y0, y1, z0, z1):
    return cq.Workplane().box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def xcyl(r, x0, x1, y=0.0, z=0.0):
    return cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0, y, z))


def servo_hole():
    # servo a, swept out the back so it slides in
    back = Y_BACK - 1
    front = SW / 2 + CL
    zb0, zb1 = SHAFT - SL - CL, SHAFT + CL
    body = blk(G - CL, G + SH + CL, back, front, zb0, zb1)
    tab = blk(G + TAB_AT - CL / 2, G + TAB_AT + TAB_T + CL / 2, back, front, Z_TAB[0] - CL, Z_TAB[1] + CL)
    r = BOSS_R + CL
    boss = xcyl(r, G + SH, X_OUT + 1).union(blk(G + SH, X_OUT + 1, back, 0, -r, r))
    cable = blk(-1, G + 9, back, 4, Z_BOT - 1, zb0 + 1)
    return body.union(tab).union(boss).union(cable)


def palm():
    p = blk(-X_OUT, X_OUT, Y_BACK, Y_FRONT, Z_BOT, Z_TOP)
    p = p.union(blk(-WRIST_X, WRIST_X, Y_BACK, Y_FRONT, WRIST_BOT, Z_BOT + 0.01))
    # sweep zone
    clip = blk(-X_OUT - 1, X_OUT + 1, 0, 40, 0, 40).cut(xcyl(R_CLIP, -X_OUT - 2, X_OUT + 2))
    p = p.cut(clip)
    hole = servo_hole()
    p = p.cut(hole).cut(hole.mirror("YZ"))
    p = p.cut(blk(-WRIST_HOLLOW, WRIST_HOLLOW, Y_BACK - 1, Y_FRONT - 2.5, WRIST_BOT - 1, Z_BOT + 1))
    for x, z in SCREWS:
        p = p.cut(cq.Workplane("XZ").circle(0.9).extrude(-8).translate((x, Y_BACK, z)))
    return p


def cover():
    c = blk(-X_OUT, X_OUT, Y_BACK - COVER_T, Y_BACK, Z_BOT, Z_TOP)
    c = c.union(blk(-WRIST_X, WRIST_X, Y_BACK - COVER_T, Y_BACK, COVER_BOT, Z_BOT + 0.01))
    for x, z in SCREWS:
        c = c.cut(cq.Workplane("XZ").circle(1.2).extrude(-10).translate((x, Y_BACK - COVER_T - 1, z)))
    return c


def finger(length, tilt):
    straight = length - FW / 2
    f = blk(-FW / 2, FW / 2, -FT / 2, FT / 2, -3, straight)
    f = f.union(cq.Workplane("XZ").circle(FW / 2).extrude(FT / 2, both=True).translate((0, 0, straight)))
    try:
        f = f.faces(">Y").edges().fillet(2.0)
    except Exception as e:
        print(f"finger fillet skipped: {e}")
    # knuckles
    for k in (0.38, 0.68):
        f = f.cut(xcyl(0.9, -FW, FW, y=FT / 2, z=straight * k))
    return f.rotate((0, 0, 0), (0, 1, 0), tilt).translate((0, 0, R_OUT))


def yoke(side):
    # built as servo a; b is mirrored
    plate = xcyl(7.0, XP, XP + PT).union(blk(XP, XP + PT, -FT / 2, FT / 2, 0, R_OUT))
    y = plate.union(blk(0.6, XP + PT, -FT / 2, FT / 2, R_IN, R_OUT))
    for x, length, tilt in FINGERS[side]:
        y = y.union(finger(length, tilt).translate((x, 0, 0)))
    # horn pocket
    pocket = xcyl(HUB_D / 2, XP - 1, XP + ARM_D).union(blk(XP - 1, XP + ARM_D, -ARM_W / 2, ARM_W / 2, 0, ARM_L))
    y = y.cut(pocket).cut(xcyl(2.6, XP, XP + PT + 1))
    y = y.intersect(blk(-1, XP + PT + 1, -FT / 2, FT / 2, -10, 100))
    return y if side == "a" else y.mirror("YZ")


def thumb():
    t = cq.Workplane("XZ").center(9.5, -17.5).slot2D(24, 11, -9).extrude(-11).translate((0, Y_FRONT, 0))
    try:
        t = t.faces(">Y").edges().fillet(3.0)
    except Exception as e:
        print(f"thumb fillet skipped: {e}")
    return t


def base():
    b = cq.Workplane().box(110, 85, 5, centered=(True, False, False)).translate((0, -30, 0))
    b = b.edges("|Z").fillet(8)
    ix, y0, y1 = WRIST_X + 0.3, Y_BACK - COVER_T - 0.3, Y_FRONT + 0.3
    sleeve = blk(-ix - 3, ix + 3, y0 - 3, y1 + 3, 5, 17).cut(blk(-ix, ix, y0, y1, 4, 18))
    # cable slot
    sleeve = sleeve.cut(blk(-8, 8, y0 - 4, y0 + 1, 4.99, 18))
    return b.union(sleeve).translate((0, 0, WRIST_BOT - 5))


def fit_test():
    part = palm().intersect(blk(G + TAB_AT - 4, X_OUT + 1, -20, 20, Z_TAB[0] - 3, Z_TOP + 1))
    coupon = blk(XP, XP + PT, -9, 9, -8, ARM_L + 3)
    coupon = coupon.cut(xcyl(HUB_D / 2, XP - 1, XP + ARM_D).union(
        blk(XP - 1, XP + ARM_D, -ARM_W / 2, ARM_W / 2, 0, ARM_L))).cut(xcyl(2.6, XP, XP + PT + 1))
    return part, coupon


def parts() -> dict:
    return {"palm": palm(), "cover": cover(), "yoke_a": yoke("a"), "yoke_b": yoke("b"),
            "thumb": thumb(), "base": base()}


def flat(shape, down: str):
    # print orientation
    turn = {"front": -90, "back": 90, "bottom": 0}[down]
    s = shape.rotate((0, 0, 0), (1, 0, 0), turn)
    bb = s.val().BoundingBox()
    return s.translate((-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin))


PRINT = {"palm": "front", "cover": "back", "yoke_a": "back", "yoke_b": "back", "thumb": "back", "base": "bottom"}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, shape in parts().items():
        cq.exporters.export(flat(shape, PRINT[name]), str(OUT / f"{name}.stl"), tolerance=0.05, angularTolerance=0.15)
        print(f"{name}: {shape.val().Volume() / 1000:.1f} cm3")
    part, coupon = fit_test()
    test = flat(part, "front").union(flat(coupon, "back").translate((30, 0, 0)))
    cq.exporters.export(test, str(OUT / "fit_test.stl"), tolerance=0.05, angularTolerance=0.15)
    print("fit_test done")
