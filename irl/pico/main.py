import sys

from machine import PWM, Pin

# a = index + middle, b = ring + pinky
SERVOS = {"A": PWM(Pin(14)), "B": PWM(Pin(15))}
for s in SERVOS.values():
    s.freq(50)


def go(name: str, deg: float):
    deg = max(0, min(180, deg))
    SERVOS[name].duty_ns(int(500_000 + deg / 180 * 2_000_000))


# lines like "A95 B10"
line = ""
while True:
    c = sys.stdin.read(1)
    if c != "\n":
        line += c
        continue
    for part in line.split():
        if part[:1] in SERVOS:
            try:
                go(part[0], float(part[1:]))
            except ValueError:
                print("bad angle:", part)
        else:
            print("bad command:", part)
    line = ""
