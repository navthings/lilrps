import argparse
import random
import time

import cv2
import mlx.core as mx
import numpy as np

import hand
from hand import COUNTER, MOVES, put

CLIPS = hand.HERE / "clips"
WORDS = ["rock", "paper", "scissors", "shoot!"]

BEAT = 0.45        # seconds
STILL = 0.015      # tune
STABLE = 3         # frames
LOOK_FROM = -0.2   # seconds
TIMEOUT = 1.5      # seconds
DARK = 70          # brightness
REC_FPS = 15
SURE = 0.6         # confidence
KEEP = 6           # crops per round
ROCK_AFTER = 0.45  # seconds
VOTES = 3          # frames
PALM = [0, 5, 9, 13, 17]


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
    x = hand.window({"t": np.array(times), "kp": np.stack(frames)}, cutoff)
    guess = MOVES[int(model(mx.array(x[None])).argmax(axis=1).item())]
    return COUNTER[guess], guess


def winner(you: str, robot: str) -> str:
    if you == robot:
        return "draw"
    return "robot wins" if COUNTER[you] == robot else "you win"


def count_rounds() -> dict:
    counts = {m: 0 for m in MOVES}
    for f in hand.ROUNDS.glob("*.npz"):
        label = str(np.load(f)["label"])
        if label in counts:
            counts[label] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="play rock paper scissors against lilrps and collect training rounds")
    ap.add_argument("--player", default="nav", help="whose hand this is, saved with every round")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--record", action="store_true", help="save the whole session as a video in clips/")
    ap.add_argument("--cutoff", type=float, default=0.3, help="seconds after shoot the keypoint guess is made")
    ap.add_argument("--predict", action="store_true", help="old mode: robot commits to the keypoint guess at the cutoff instead of reacting")
    ap.add_argument("--random", action="store_true", help="ignore the early guess model and throw randomly")
    ap.add_argument("--hand", nargs="?", const="lilrps.local", help="drive the irl hand on the pi over wifi (hostname, or leave empty for lilrps.local)")
    ap.add_argument("--wheel", nargs="?", const="lilrps.local", help="drive the spinning wheel on the pi over wifi (hostname, or leave empty for lilrps.local)")
    args = ap.parse_args()

    clf = hand.load_classifier()
    if clf is None:
        raise SystemExit("no classifier yet. collect bursts then run: .venv/bin/python train.py classifier")
    hand.ROUNDS.mkdir(parents=True, exist_ok=True)
    counts = count_rounds()

    import mediapipe as mp
    hands = hand.tracker()
    draw = mp.solutions.drawing_utils
    cap = hand.open_camera(args.camera)
    cam, seen_n = hand.Camera(cap), 0
    boxer = hand.Boxer()

    model = None if args.random else hand.load_predictor()
    react = not args.predict
    if react:
        print("robot: reacts to the referee the frame it locks in")
    else:
        print(f"robot: {'early guess, commits ' + format(args.cutoff, '+.2f') + 's after shoot' if model else 'random'}")

    arm = None
    if args.hand and args.wheel:
        raise SystemExit("pick one: --hand or --wheel")
    if args.hand:
        from irl.link import Hand
        arm = Hand(args.hand)
        arm.pose("paper")
        print(f"irl hand on {arm.host}")
    elif args.wheel:
        from irl.link import Wheel
        arm = Wheel(args.wheel)
        arm.pose("paper")
        print(f"wheel on {arm.host}")

    writer = None
    state = "idle"
    debug = not args.record
    rec_t = 0.0
    fps, work, last = 0.0, 0.0, time.perf_counter()
    started = shoot_at = 0.0
    times, frames, crops = [], [], []
    last_kp, votes = None, []
    reads, lives, ps = [], [], []
    robot = guess = finish = pick = None
    rnd = {}

    def start_round():
        nonlocal state, started, shoot_at, times, frames, crops, last_kp, votes, robot, guess, finish, pick
        nonlocal reads, lives, ps
        state = "count"
        started = time.perf_counter()
        shoot_at = started + 3 * BEAT
        times, frames, crops = [], [], []
        last_kp, votes = None, []
        reads, lives, ps = [], [], []
        robot = guess = finish = pick = None
        if arm:
            arm.pose("half")

    def save_round():
        name = f"{args.player}_{int(time.time() * 1000)}"
        np.savez_compressed(
            hand.ROUNDS / f"{name}.npz",
            t=np.array(times, dtype=np.float32),   # seconds
            kp=np.stack(frames),
            label=rnd["label"], referee=rnd["referee"] or "",
            done_t=np.float32(rnd["done_t"] if rnd["done_t"] is not None else np.nan),
            fps=np.float32(fps), beat=np.float32(BEAT), player=args.player,
            robot=rnd["robot"], guess=rnd["guess"] or "", cutoff=np.float32(args.cutoff),
            react=react, lock_t=np.float32(rnd["lock_t"]),
            live=np.array(lives), live_p=np.array(ps, dtype=np.float32),  # per frame
        )
        counts[rnd["label"]] += 1
        # landed crops
        start = rnd["done_t"] if rnd["done_t"] is not None else args.cutoff
        keep = [c for c, t in zip(crops, times) if t >= start and c is not None][:KEEP]
        out = hand.REAL / rnd["label"]
        out.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(keep):
            cv2.imwrite(str(out / f"{name}_{i}.jpg"), c)

    while True:
        frame, seen_n = cam.read(seen_n)
        if frame is None:
            raise SystemExit("camera stopped giving frames")
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        kp, res = hand.keypoints(hands, frame)
        box = boxer.update(kp, frame, now)
        img = hand.crop(frame, box) if box is not None else None
        live, live_p = clf(img) if img is not None else (None, None)
        if live not in MOVES or live_p < SURE:
            live = None
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
            lives.append(live or "")
            ps.append(np.nan if live_p is None else live_p)
            put(frame, WORDS[min(int((now - started) / BEAT), 3)], 90, 2.2)

            # landing
            landed = land_t = None
            lost = np.isnan(kp).any()
            if live and t >= 0:
                votes.append(live)
            # held box = assume still
            still = lost or (last_kp is not None and motion(last_kp, kp) < STILL)
            if not lost:
                last_kp = kp
            if t >= LOOK_FROM:
                reads.append((live, t, still))
            # 3 of last 4, gaps ok
            last4 = [r for r in reads[-(STABLE + 1):] if r[0]]
            moves = [m for m, _, _ in last4]
            if moves:
                top = max(set(moves), key=moves.count)
                hits = [r for r in last4 if r[0] == top]
                # rock = fist stopped pumping
                if len(hits) >= STABLE and (top != "rock" or (t >= ROCK_AFTER and all(r[2] for r in hits))):
                    landed, land_t = top, hits[0][1]

            # guess
            if pick is None and t >= args.cutoff:
                pick, guess = robot_move(model, times, frames, args.cutoff)
                if not react:
                    robot = pick
                # head start
                if arm and finish is None:
                    arm.pose(pick)
            if robot:
                put(frame, f"robot: {robot}", 170, 1.4, (120, 200, 255))

            if finish is None and (landed or t > TIMEOUT):
                finish = (overrule(landed, votes), land_t, t)
            # react
            if react and finish:
                if pick is None:
                    pick, guess = robot_move(model, times, frames, args.cutoff)
                robot = COUNTER[finish[0]] if finish[0] else pick
            if finish and robot:
                rnd = {"referee": finish[0], "label": finish[0], "done_t": finish[1],
                       "robot": robot, "guess": guess, "lock_t": finish[2]}
                state = "result"
                if arm:
                    arm.pose(robot)

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
    if arm:
        arm.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
