import argparse
import socket
import time
from pathlib import Path

PWM = Path("/sys/class/pwm")
PORT = 5005
PERIOD = 20_000_000  # ns
# a = index + middle on gpio18, b = ring + pinky on gpio19
CHANNELS = {"A": 2, "B": 3}


def find_chip(chip: str) -> Path:
    if chip != "auto":
        return PWM / chip
    # gpio18/19
    for c in sorted(PWM.glob("pwmchip*")):
        if "1f00098000" in str(c.resolve()):
            return c
    found = ", ".join(c.name for c in PWM.glob("pwmchip*")) or "none"
    raise SystemExit(f"no gpio18/19 pwm (chips: {found}). is dtoverlay=pwm-2chan in /boot/firmware/config.txt? or pass --chip")


class Servo:
    def __init__(self, chip: Path, ch: int):
        self.dir = chip / f"pwm{ch}"
        if not self.dir.exists():
            (chip / "export").write_text(str(ch))
        # udev
        for _ in range(50):
            if (self.dir / "period").exists():
                break
            time.sleep(0.02)
        (self.dir / "period").write_text(str(PERIOD))
        self.on = False

    def go(self, deg: float):
        deg = max(0, min(180, deg))
        (self.dir / "duty_cycle").write_text(str(int(500_000 + deg / 180 * 2_000_000)))
        if not self.on:
            (self.dir / "enable").write_text("1")
            self.on = True

    def off(self):
        if self.on:
            (self.dir / "enable").write_text("0")
            self.on = False


def main():
    ap = argparse.ArgumentParser(description="pi side of the irl hand: listens for lines like 'A95 B10' and moves the servos")
    ap.add_argument("--chip", default="auto", help="pwmchip name, or leave it to find the gpio18/19 one")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    chip = find_chip(args.chip)
    servos = {name: Servo(chip, ch) for name, ch in CHANNELS.items()}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", args.port))
    print(f"listening on udp {args.port}, servos on {chip.name}")

    try:
        while True:
            data, who = sock.recvfrom(64)
            line = data.decode(errors="replace").strip()
            # ping
            if line == "?":
                sock.sendto(b"ok", who)
                continue
            for part in line.split():
                if part[:1] in servos:
                    try:
                        servos[part[0]].go(float(part[1:]))
                    except ValueError:
                        print("bad angle:", part)
                else:
                    print("bad command:", part)
    except KeyboardInterrupt:
        pass
    finally:
        for s in servos.values():
            s.off()


if __name__ == "__main__":
    main()
