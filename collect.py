import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

import hand
from play import keypoints, open_camera, put

HERE = Path(__file__).parent
CROPS = HERE / "data" / "crops"
KEYS = {ord("r"): "rock", ord("p"): "paper", ord("s"): "scissors"}


def count() -> dict:
    return {m: len(list((CROPS / m).glob("*.jpg"))) for m in hand.MOVES}


def main():
    ap = argparse.ArgumentParser(description="record labelled hand crops for the classifier")
    ap.add_argument("--player", default="nav")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=4.0, help="how long each burst records")
    args = ap.parse_args()

    for m in hand.MOVES:
        (CROPS / m).mkdir(parents=True, exist_ok=True)
    counts = count()
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                     min_detection_confidence=0.3, min_tracking_confidence=0.3)
    boxer = hand.Boxer()
    clf = hand.load()
    cap = open_camera(args.camera)
    label, burst, until, n = None, 0, 0.0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera stopped giving frames")
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        kp, _ = keypoints(hands, frame)
        box = boxer.update(kp, frame, now)
        img = hand.crop(frame, box) if box is not None else None

        if label and now < until and img is not None:
            cv2.imwrite(str(CROPS / label / f"{args.player}_{burst}_{n:03d}.jpg"), img)
            n += 1
            counts[label] += 1
        elif label and now >= until:
            label = None

        if box is not None:
            hand.draw_box(frame, box, (80, 80, 255) if label else (255, 200, 80))
        if img is not None:
            frame[10:234, -234:-10] = cv2.resize(img, (224, 224))
            if clf:
                move, p = clf(img)
                put(frame, f"model: {move} {p:.0%}", 280, 0.9, (120, 200, 255))

        if label:
            put(frame, f"recording {label} {until - now:.1f}s", 60, 1.2, (80, 80, 255))
            put(frame, "move it around, tilt it, go head-on, near and far", 110, 0.7)
        else:
            put(frame, "hold a move then press r / p / s to record it   q = quit", 60, 0.7)
        c = counts
        put(frame, f"crops  rock {c['rock']}  paper {c['paper']}  scissors {c['scissors']}",
            frame.shape[0] - 25, 0.7, (180, 255, 180))

        cv2.imshow("lilrps collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEYS and not label:
            label, burst, until, n = KEYS[key], int(time.time() * 1000), now + args.seconds, 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
