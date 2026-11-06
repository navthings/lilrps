import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from irl.link import CONFIG, WHEEL, Hand, Wheel

STEPS = [("a", "open", "index + middle pointing straight up"),
         ("a", "closed", "index + middle pointing straight at you (don't go past, it hits the thumb)"),
         ("b", "open", "ring + pinky pointing straight up"),
         ("b", "closed", "ring + pinky pointing straight at you")]
DEMO = ("half", "rock", "paper", "scissors", "rock", "scissors", "paper")


def ask(deg: float, move) -> int:
    print("type an angle 0-180 and press enter to move it. empty enter = thats right")
    while True:
        move(deg)
        got = input(f"  {deg:.0f} > ").strip()
        if not got:
            return int(deg)
        if not got.lstrip("-").isdigit() or not 0 <= int(got) <= 180:
            print("  needs a whole number 0-180")
            continue
        deg = int(got)


def hand_cal(host: str):
    hand = Hand(host, need_config=False)
    print(f"talking to {hand.host}")
    cfg = hand.cfg or {"a": {"open": 90, "closed": 90}, "b": {"open": 90, "closed": 90}}
    for servo, pose, want in STEPS:
        print(f"\nservo {servo}, {pose}: {want}")
        if servo == "a":
            cfg[servo][pose] = ask(cfg[servo][pose], lambda d: hand.angles(d, cfg["b"]["open"]))
        else:
            cfg[servo][pose] = ask(cfg[servo][pose], lambda d: hand.angles(cfg["a"]["open"], d))
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"\nsaved {CONFIG}. demo:")
    hand.cfg = cfg
    return hand


def wheel_cal(host: str):
    wheel = Wheel(host, need_config=False)
    print(f"talking to {wheel.host}")
    cfg = wheel.cfg or {"rock": 0, "paper": 90, "scissors": 180}
    for face in ("rock", "paper", "scissors"):
        print(f"\n{face}: spin until the {face} face points at the player")
        cfg[face] = ask(cfg[face], wheel.angles)
    WHEEL.write_text(json.dumps(cfg, indent=2))
    print(f"\nsaved {WHEEL}. demo:")
    wheel.cfg = cfg
    return wheel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host", nargs="?", default="lilrps.local")
    ap.add_argument("--wheel", action="store_true", help="calibrate the spinning wheel instead of the finger hand")
    args = ap.parse_args()
    bot = wheel_cal(args.host) if args.wheel else hand_cal(args.host)
    for move in DEMO:
        print(" ", move)
        bot.pose(move)
        time.sleep(0.8)
    bot.close()


if __name__ == "__main__":
    main()
