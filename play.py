import argparse
import random
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import mlx.core as mx
import numpy as np

import hand
from train import Predictor, window

HERE = Path(__file__).parent
ROUNDS = HERE / "data" / "rounds"
CLIPS = HERE / "clips"
CROPS = HERE / "data" / "crops"

MOVES = ["rock", "paper", "scissors"]
COUNTER = {"rock": "paper", "paper": "scissors", "scissors": "rock"}  # beats
WORDS = ["rock", "paper", "scissors", "shoot!"]

BEAT = 0.45        # seconds
STILL = 0.015      # tune
STABLE = 3         # frames
LOOK_FROM = -0.2   # seconds
TIMEOUT = 1.5      # seconds
UP_RATIO = 1.1
SPREAD = 0.8       # v gap / palm width
FOLDED = 0.9       # curled tips fold back
DARK = 70          # brightness
REC_FPS = 15
SURE = 0.6         # classifier confidence
KEEP = 6           # crops saved per round
ROCK_AFTER = 0.45  # seconds
VOTES = 3          # frames

# tip/knuckle
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]
PALM = [0, 5, 9, 13, 17]


def keypoints(hands, frame):
    h, w = frame.shape[:2]
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return np.full((21, 3), np.nan, dtype=np.float32), res
    lm = res.multi_hand_landmarks[0].landmark
    # aspect
    return np.array([[p.x * w / h, p.y, p.z] for p in lm], dtype=np.float32), res


def read_pose(kp: np.ndarray) -> str | None:
    if np.isnan(kp).any():
        return None
    wrist = kp[0, :2]
    ratio = [np.linalg.norm(kp[tip, :2] - wrist) / np.linalg.norm(kp[knuckle, :2] - wrist)
             for tip, knuckle in FINGERS]
    index, middle, ring, pinky = [r > UP_RATIO for r in ratio]
    if ring and pinky:
        return "paper"
    # v, not fist
    spread = np.linalg.norm(kp[8, :2] - kp[12, :2]) / np.linalg.norm(kp[5, :2] - kp[17, :2])
    if spread > SPREAD and min(ratio[:2]) > FOLDED:
        return "scissors"
    if not ring and not pinky:
        return "scissors" if index or middle else "rock"
    return None


def see(kp: np.ndarray, img, clf) -> tuple[str | None, float | None]:
    if clf is None:
        return read_pose(kp), None
    if img is None:
        return None, None
    move, p = clf(img)
    return (move if p >= SURE and move in MOVES else None), p


def overrule(landed: str | None, votes: list) -> str | None:
    # everyone starts on a fist
    if landed not in (None, "rock"):
        return landed
    other = [m for m in votes if m != "rock"]
    top = max(set(other), key=other.count) if other else None
    return top if top and other.count(top) >= max(VOTES, len(votes) / 3) else landed


def motion(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[PALM], b[PALM]
    if np.isnan(a).any() or np.isnan(b).any():
        return float("inf")
    return float(np.linalg.norm(b[:, :2] - a[:, :2], axis=1).mean())


def robot_move(model, times: list, frames: list, cutoff: float):
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
        # permission
        if waited == 0:
            print("waiting for camera permission, click allow on the popup...")
        time.sleep(1)
        waited += 1
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("no camera. run this from apple terminal, or check system settings > privacy > camera")
    cap.set(cv2.CAP_PROP_FPS, 60)
    print(f"camera says {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} at {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
    return cap


class Camera:
    # grabs frames on its own thread so slow frames dont make us miss the next one
    def __init__(self, cap):
        self.cap, self.frame, self.n, self.fps = cap, None, 0, 0.0
        self.cond = threading.Condition()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        last = time.perf_counter()
        while True:
            ok, f = self.cap.read()
            now = time.perf_counter()
            with self.cond:
                self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
                last = now
                self.frame, self.n = (f, self.n + 1) if ok else (None, -1)
                self.cond.notify_all()
            if not ok:
                return

    def read(self, seen: int):
        with self.cond:
            self.cond.wait_for(lambda: self.n != seen)
            return self.frame, self.n


def main():
    ap = argparse.ArgumentParser(description="play rock paper scissors against lilrps and collect training rounds")
    ap.add_argument("--player", default="nav", help="whose hand this is, saved with every round")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--record", action="store_true", help="save the whole session as a video in clips/")
    ap.add_argument("--cutoff", type=float, default=0.3, help="seconds after shoot the keypoint guess is made")
    ap.add_argument("--predict", action="store_true", help="old mode: robot commits to the keypoint guess at the cutoff instead of reacting")
    ap.add_argument("--random", action="store_true", help="ignore the trained model and throw randomly")
    ap.add_argument("--rules", action="store_true", help="referee uses finger rules instead of the hand classifier")
    args = ap.parse_args()

    ROUNDS.mkdir(parents=True, exist_ok=True)
    counts = count_rounds()

    # legacy
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                     min_detection_confidence=0.3, min_tracking_confidence=0.3)
    draw = mp.solutions.drawing_utils
    cap = open_camera(args.camera)
    cam, seen_n = Camera(cap), 0
    boxer = hand.Boxer()
    clf = None if args.rules else hand.load()
    print(f"referee: {'hand classifier' if clf else 'finger rules'}")

    model = None
    weights = HERE / "results" / "predictor.safetensors"
    if weights.exists() and not args.random:
        model = Predictor()
        model.load_weights(str(weights))
        model.eval()
    react = not args.predict
    if react:
        print("robot: reacts to the referee the frame it locks in")
    else:
        print(f"robot: {'trained model, commits ' + format(args.cutoff, '+.2f') + 's after shoot' if model else 'random'}")

    writer = None
    state = "idle"
    debug = not args.record
    rec_t = 0.0
    fps, work, last = 0.0, 0.0, time.perf_counter()
    started = shoot_at = 0.0
    times, frames, crops = [], [], []
    seen, last_kp, votes = [], None, []
    robot = guess = finish = pick = None
    rnd = {}

    def start_round():
        nonlocal state, started, shoot_at, times, frames, crops, seen, last_kp, votes, robot, guess, finish, pick
        state = "count"
        started = time.perf_counter()
        shoot_at = started + 3 * BEAT
        times, frames, crops = [], [], []
        seen, last_kp, votes = [], None, []
        robot = guess = finish = pick = None

    def save_round():
        name = f"{args.player}_{int(time.time() * 1000)}.npz"
        np.savez_compressed(
            ROUNDS / name,
            t=np.array(times, dtype=np.float32),   # seconds
            kp=np.stack(frames),
            label=rnd["label"], referee=rnd["referee"] or "",
            done_t=np.float32(rnd["done_t"] if rnd["done_t"] is not None else np.nan),
            fps=np.float32(fps), beat=np.float32(BEAT), player=args.player,
            robot=rnd["robot"], guess=rnd["guess"] or "", cutoff=np.float32(args.cutoff),
            react=react, lock_t=np.float32(rnd["lock_t"]),
        )
        counts[rnd["label"]] += 1
        # landed crops
        start = rnd["done_t"] if rnd["done_t"] is not None else args.cutoff
        keep = [c for c, t in zip(crops, times) if t >= start and c is not None][:KEEP]
        (CROPS / rnd["label"]).mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(keep):
            cv2.imwrite(str(CROPS / rnd["label"] / f"{name[:-4]}_{i}.jpg"), c)

    while True:
        frame, seen_n = cam.read(seen_n)
        if frame is None:
            raise SystemExit("camera stopped giving frames")
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        kp, res = keypoints(hands, frame)
        box = boxer.update(kp, frame, now)
        img = hand.crop(frame, box) if box is not None else None
        live, live_p = see(kp, img, clf)
        if res.multi_hand_landmarks:
            draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
        if box is not None:
            hand.draw_box(frame, box)

        if state == "idle":
            put(frame, "space = play   q = quit", 60)

        elif state == "count":
            t = now - shoot_at
            times.append(t)
            frames.append(kp)
            crops.append(img)
            put(frame, WORDS[min(int((now - started) / BEAT), 3)], 90, 2.2)

            # landing
            landed = None
            lost = np.isnan(kp).any()
            if live and t >= 0:
                votes.append(live)
            if not lost or (clf and img is not None):
                # held box = assume still
                still = (lost and clf is not None) or (last_kp is not None and motion(last_kp, kp) < STILL)
                seen = seen + [(live, t)] if t >= LOOK_FROM and still and live else []
                if not lost:
                    last_kp = kp
                recent = [p for p, _ in seen[-STABLE:]]
                if len(recent) == STABLE:
                    top = max(set(recent), key=recent.count)
                    if recent.count(top) >= STABLE - 1 and (top != "rock" or t >= ROCK_AFTER):  # glitch
                        landed = top

            # guess
            if pick is None and t >= args.cutoff:
                pick, guess = robot_move(model, times, frames, args.cutoff)
                if not react:
                    robot = pick
            if robot:
                put(frame, f"robot: {robot}", 170, 1.4, (120, 200, 255))

            if finish is None and (landed or t > TIMEOUT):
                finish = (overrule(landed, votes), seen[-STABLE][1] if landed else None, t)
            # react
            if react and finish:
                if pick is None:
                    pick, guess = robot_move(model, times, frames, args.cutoff)
                robot = COUNTER[finish[0]] if finish[0] else pick
            if finish and robot:
                rnd = {"referee": finish[0], "label": finish[0], "done_t": finish[1],
                       "robot": robot, "guess": guess, "lock_t": finish[2]}
                state = "result"

        elif state == "result":
            if rnd["label"]:
                put(frame, f"you: {rnd['label']}   robot: {rnd['robot']}", 60)
                put(frame, winner(rnd["label"], rnd["robot"]), 110, 1.6)
                if rnd["done_t"] is not None:
                    put(frame, f"landed {rnd['done_t'] * 1000:+.0f}ms from shoot", 160, 0.8)
                if react:
                    put(frame, f"robot locked in {rnd['lock_t'] * 1000:+.0f}ms from shoot", 260, 0.7)
                if rnd["guess"]:
                    put(frame, f"keypoint guess was {rnd['guess']}", 300 if react else 260, 0.7)
                put(frame, "wrong move? r / p / s   x = bin it   space = next", 210, 0.7)
            else:
                put(frame, "couldnt read your hand", 60)
                put(frame, "r / p / s = tell me   x = bin it", 110, 0.7)

        if debug:
            m = motion(frames[-2], frames[-1]) if state == "count" and len(frames) > 1 else 0.0
            sees = (live or "-") + (f" {live_p:.0%}" if live_p is not None else "")
            light = frame.mean()
            put(frame, f"cam {cam.fps:.0f}fps   loop {fps:.0f}fps   work {work:.0f}ms   motion {m:.3f}   sees {sees}   light {light:.0f}{' (too dark)' if light < DARK else ''}",
                frame.shape[0] - 60, 0.6, (180, 255, 180) if light >= DARK else (80, 80, 255))
            c = counts
            put(frame, f"saved  rock {c['rock']}  paper {c['paper']}  scissors {c['scissors']}",
                frame.shape[0] - 25, 0.6, (180, 255, 180))

        if args.record:
            if writer is None:
                CLIPS.mkdir(exist_ok=True)
                path = CLIPS / f"session_{int(time.time())}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS,
                                         (frame.shape[1], frame.shape[0]))
                rec_t = now
            # realtime
            while rec_t <= now:
                writer.write(frame)
                rec_t += 1 / REC_FPS

        work = 0.9 * work + 0.1 * (time.perf_counter() - now) * 1000
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
