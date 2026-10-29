import argparse
import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
import mlx.core as mx
import numpy as np

from train import Predictor, window

HERE = Path(__file__).parent
ROUNDS = HERE / "data" / "rounds"
CLIPS = HERE / "clips"

MOVES = ["rock", "paper", "scissors"]
COUNTER = {"rock": "paper", "paper": "scissors", "scissors": "rock"}  # move -> what beats it
WORDS = ["rock", "paper", "scissors", "shoot!"]

BEAT = 0.45        # seconds between each word of the countdown
STILL = 0.015      # max hand movement per frame that still counts as stopped (tune with the debug number)
STABLE = 3         # frames in a row that must be still with the same pose
LOOK_FROM = -0.2   # start looking for the finished pose this long before shoot (people land early)
TIMEOUT = 1.5      # give up on the round this long after shoot
UP_RATIO = 1.1     # a finger is straight if its tip is this much further from the wrist than its knuckle

# (tip, middle knuckle) for index, middle, ring, pinky. thumb is skipped, it points everywhere
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]
# wrist + the base of each finger. fingertips jitter a lot (worst on scissors) but the palm stays steady
PALM = [0, 5, 9, 13, 17]


def keypoints(hands, frame):
    """21 hand points as an (21, 3) array, all nan if no hand. also returns mediapipe's result for drawing."""
    h, w = frame.shape[:2]
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return np.full((21, 3), np.nan, dtype=np.float32), res
    lm = res.multi_hand_landmarks[0].landmark
    # x is scaled by the aspect ratio so a distance means the same thing sideways and up/down
    return np.array([[p.x * w / h, p.y, p.z] for p in lm], dtype=np.float32), res


def read_pose(kp: np.ndarray) -> str | None:
    """The referee: which move a hand is making, or None if its mid-motion or a weird shape."""
    if np.isnan(kp).any():
        return None
    wrist = kp[0, :2]
    up = [bool(np.linalg.norm(kp[tip, :2] - wrist) > UP_RATIO * np.linalg.norm(kp[knuckle, :2] - wrist))
          for tip, knuckle in FINGERS]
    index, middle, ring, pinky = up
    # ring + pinky decide it (same split as the 2 servo hand). up = paper, down = rock or scissors
    if ring and pinky:
        return "paper"
    if not ring and not pinky:
        return "scissors" if index or middle else "rock"
    return None


def motion(a: np.ndarray, b: np.ndarray) -> float:
    """How far the palm moved between two frames."""
    a, b = a[PALM], b[PALM]
    if np.isnan(a).any() or np.isnan(b).any():
        return float("inf")
    return float(np.linalg.norm(b[:, :2] - a[:, :2], axis=1).mean())


def robot_move(model, times: list, frames: list, cutoff: float):
    """The robot throws the counter to whatever the model thinks you're throwing. Random if there's no model."""
    if model is None:
        return random.choice(MOVES), None
    x = window({"t": np.array(times), "kp": np.stack(frames)}, cutoff)
    guess = MOVES[int(model(mx.array(x[None])).argmax(axis=1).item())]
    return COUNTER[guess], guess


def winner(you: str, robot: str) -> str:
    if you == robot:
        return "draw"
    return "robot wins" if COUNTER[you] == robot else "you win"


def put(frame, text, y, scale=1.0, color=(255, 255, 255)):
    # black outline under the text so it reads on any background
    for c, t in (((0, 0, 0), 6), (color, 2)):
        cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, t, cv2.LINE_AA)


def count_rounds() -> dict:
    counts = {m: 0 for m in MOVES}
    for f in ROUNDS.glob("*.npz"):
        label = str(np.load(f)["label"])
        if label in counts:
            counts[label] += 1
    return counts


def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    waited = 0
    while not cap.isOpened() and waited < 30:
        # opencv asks macos for permission then gives up straight away, so keep retrying while the popup is up
        if waited == 0:
            print("waiting for camera permission, click allow on the popup...")
        time.sleep(1)
        waited += 1
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("no camera. run this from apple terminal, or check system settings > privacy > camera")
    cap.set(cv2.CAP_PROP_FPS, 60)  # most webcams ignore this, the real fps shows on screen
    return cap


def main():
    ap = argparse.ArgumentParser(description="play rock paper scissors against lilrps and collect training rounds")
    ap.add_argument("--player", default="nav", help="whose hand this is, saved with every round")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--record", action="store_true", help="save the whole session as a video in clips/")
    ap.add_argument("--cutoff", type=float, default=0.3, help="seconds after shoot the robot has to commit")
    ap.add_argument("--random", action="store_true", help="ignore the trained model and throw randomly")
    args = ap.parse_args()

    ROUNDS.mkdir(parents=True, exist_ok=True)
    counts = count_rounds()

    # the classic cpu hand model (mediapipe 0.10.14). the newer tasks api crashes on some macs trying to use metal.
    # lite model: the full one dropped to 15fps and lost fists. low confidences so it grabs a fist back quickly
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                     min_detection_confidence=0.3, min_tracking_confidence=0.3)
    draw = mp.solutions.drawing_utils
    cap = open_camera(args.camera)

    model = None
    weights = HERE / "results" / "predictor.safetensors"
    if weights.exists() and not args.random:
        model = Predictor()
        model.load_weights(str(weights))
        model.eval()
    print(f"robot: {'trained model, commits ' + format(args.cutoff, '+.2f') + 's after shoot' if model else 'random'}")

    writer = None
    state = "idle"
    debug = True
    fps, last = 0.0, time.perf_counter()
    started = shoot_at = 0.0
    times, frames = [], []
    seen, last_kp = [], None
    robot = guess = finish = None
    rnd = {}

    def start_round():
        nonlocal state, started, shoot_at, times, frames, seen, last_kp, robot, guess, finish
        state = "count"
        started = time.perf_counter()
        shoot_at = started + 3 * BEAT
        times, frames = [], []
        seen, last_kp = [], None
        robot = guess = finish = None

    def save_round():
        name = f"{args.player}_{int(time.time() * 1000)}.npz"
        np.savez_compressed(
            ROUNDS / name,
            t=np.array(times, dtype=np.float32),   # seconds relative to shoot, negative = before
            kp=np.stack(frames),                   # (frames, 21, 3)
            label=rnd["label"], referee=rnd["referee"] or "",
            done_t=np.float32(rnd["done_t"] if rnd["done_t"] is not None else np.nan),
            fps=np.float32(fps), beat=np.float32(BEAT), player=args.player,
            robot=rnd["robot"], guess=rnd["guess"] or "", cutoff=np.float32(args.cutoff),
        )
        counts[rnd["label"]] += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera stopped giving frames")
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        kp, res = keypoints(hands, frame)
        if res.multi_hand_landmarks:
            draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

        if state == "idle":
            put(frame, "space = play   q = quit", 60)

        elif state == "count":
            t = now - shoot_at
            times.append(t)
            frames.append(kp)
            put(frame, WORDS[min(int((now - started) / BEAT), 3)], 90, 2.2)

            # the round is done once the palm has stopped and the move held for a few frames.
            # frames where mediapipe lost the hand are skipped, not counted as moving
            landed = None
            if not np.isnan(kp).any():
                pose = read_pose(kp)
                still = last_kp is not None and motion(last_kp, kp) < STILL
                seen = seen + [(pose, t)] if t >= LOOK_FROM and still and pose else []
                last_kp = kp
                recent = [p for p, _ in seen[-STABLE:]]
                if len(recent) == STABLE:
                    top = max(set(recent), key=recent.count)
                    if recent.count(top) >= STABLE - 1:  # one glitchy frame is allowed
                        landed = top

            # the robot has to commit at the cutoff, using only the frames it's seen so far
            if robot is None and t >= args.cutoff:
                robot, guess = robot_move(model, times, frames, args.cutoff)
            if robot:
                put(frame, f"robot: {robot}", 170, 1.4, (120, 200, 255))

            if finish is None and (landed or t > TIMEOUT):
                finish = (landed, seen[-STABLE][1] if landed else None)
            if finish and robot:
                rnd = {"referee": finish[0], "label": finish[0], "done_t": finish[1],
                       "robot": robot, "guess": guess}
                state = "result"

        elif state == "result":
            if rnd["label"]:
                put(frame, f"you: {rnd['label']}   robot: {rnd['robot']}", 60)
                put(frame, winner(rnd["label"], rnd["robot"]), 110, 1.6)
                if rnd["done_t"] is not None:
                    put(frame, f"landed {rnd['done_t'] * 1000:+.0f}ms from shoot", 160, 0.8)
                if rnd["guess"]:
                    put(frame, f"it guessed {rnd['guess']} at {args.cutoff * 1000:+.0f}ms", 260, 0.7)
                put(frame, "wrong move? r / p / s   x = bin it   space = next", 210, 0.7)
            else:
                put(frame, "couldnt read your hand", 60)
                put(frame, "r / p / s = tell me   x = bin it", 110, 0.7)

        if debug:
            m = motion(frames[-2], frames[-1]) if state == "count" and len(frames) > 1 else 0.0
            live = read_pose(kp) or "-"
            put(frame, f"fps {fps:.0f}   motion {m:.3f}   sees {live}", frame.shape[0] - 60, 0.6, (180, 255, 180))
            c = counts
            put(frame, f"saved  rock {c['rock']}  paper {c['paper']}  scissors {c['scissors']}",
                frame.shape[0] - 25, 0.6, (180, 255, 180))

        if args.record:
            if writer is None:
                CLIPS.mkdir(exist_ok=True)
                path = CLIPS / f"session_{int(time.time())}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        cv2.imshow("lilrps", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug
        if state == "result":
            if key in (ord("r"), ord("p"), ord("s")):
                rnd["label"] = {"r": "rock", "p": "paper", "s": "scissors"}[chr(key)]
            elif key == ord("x"):
                state = "idle"
            elif key == ord(" ") and rnd["label"]:
                save_round()
                start_round()
        elif state == "idle" and key == ord(" "):
            start_round()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
