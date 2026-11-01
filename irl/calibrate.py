import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from irl.link import CONFIG, Hand

STEPS = [("a", "open", "index + middle pointing straight up"),
         ("a", "closed", "index + middle pointing straight at you (don't go past, it hits the thumb)"),
         ("b", "open", "ring + pinky pointing straight up"),
         ("b", "closed", "ring + pinky pointing straight at you")]


def main():
    hand = Hand(sys.argv[1] if len(sys.argv) > 1 else "auto", need_config=False)
    print(f"talking to {hand.port}")
    cfg = hand.cfg or {"a": {"open": 90, "closed": 90}, "b": {"open": 90, "closed": 90}}
    for servo, pose, want in STEPS:
        deg = cfg[servo][pose]
        print(f"\nservo {servo}, {pose}: {want}")
        print("type an angle 0-180 and press enter to move it. empty enter = thats right")
        while True:
            a, b = (deg, cfg["b"]["open"]) if servo == "a" else (cfg["a"]["open"], deg)
            hand.angles(a, b)
            got = input(f"  {deg} > ").strip()
            if not got:
                break
            if not got.lstrip("-").isdigit() or not 0 <= int(got) <= 180:
                print("  needs a whole number 0-180")
                continue
            deg = int(got)
        cfg[servo][pose] = deg
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"\nsaved {CONFIG}. demo:")
    hand.cfg = cfg
    for move in ("half", "rock", "paper", "scissors", "rock", "scissors", "paper"):
        print(" ", move)
        hand.pose(move)
        time.sleep(0.8)


if __name__ == "__main__":
    main()
