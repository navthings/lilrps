import argparse
import sys
import termios
import tty

from main import CHANNELS, Servo, find_chip

NAMES = {"r": "rock", "p": "paper", "s": "scissors"}


def key() -> str:
    # no enter
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ap = argparse.ArgumentParser(description="press r / p / s and the wheel spins to it. no camera, runs on the pi")
    ap.add_argument("--chip", default="auto", help="pwmchip name, or leave it to find the gpio18/19 one")
    ap.add_argument("--rock", type=float, default=0, help="degrees")
    ap.add_argument("--paper", type=float, default=90, help="degrees")
    ap.add_argument("--scissors", type=float, default=180, help="degrees")
    args = ap.parse_args()

    wheel = Servo(find_chip(args.chip), CHANNELS["A"])
    deg = {"r": args.rock, "p": args.paper, "s": args.scissors}
    a = 90.0
    # raw mode needs \r
    print("r / p / s = spin to it   [ ] = nudge by 5   q = quit\r")
    try:
        while True:
            k = key()
            if k in ("q", "\x03", ""):
                break
            if k in deg:
                a = deg[k]
                wheel.go(a)
                print(f"{NAMES[k]}   {a:.0f}\r")
            elif k in ("[", "]"):
                a = max(0.0, min(180.0, a + (5 if k == "]" else -5)))
                wheel.go(a)
                print(f"{a:.0f}\r")
    finally:
        wheel.off()


if __name__ == "__main__":
    main()
