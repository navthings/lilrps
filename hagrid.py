import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import hand

HERE = Path(__file__).parent
ROOT = HERE / "data" / "hagrid" / "hagrid-sample-120k-384p"
OUT = HERE / "data" / "hagrid" / "crops"
GESTURES = {"fist": "rock", "palm": "paper", "peace": "scissors", "peace_inverted": "scissors"}


def box_of(b, w: int, h: int, pad: float) -> np.ndarray:
    x, y, bw, bh = b[0] * w, b[1] * h, b[2] * w, b[3] * h
    return np.array([x + bw / 2, y + bh / 2, max(bw, bh) * pad])


# box slid off the hand, like when tracking drops
def drifted(box: np.ndarray, w: int, h: int, rng) -> np.ndarray | None:
    for _ in range(10):
        ang = rng.uniform(np.pi / 3, 2 * np.pi / 3) if rng.random() < 0.6 else rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.7, 1.3) * box[2]
        cx, cy = box[0] + d * np.cos(ang), box[1] + d * np.sin(ang)
        if 0 < cx < w and 0 < cy < h:
            return np.array([cx, cy, box[2] * rng.uniform(0.8, 1.3)])
    return None


def main():
    ap = argparse.ArgumentParser(description="cut hagrid photos into lilrps style hand crops")
    ap.add_argument("--per-move", type=int, default=4000)
    ap.add_argument("--none", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for c in hand.CLASSES:
        (OUT / c).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in hand.CLASSES}
    none_left = args.none
    for g, move in GESTURES.items():
        ann = json.loads((ROOT / "ann_train_val" / f"{g}.json").read_text())
        have = sorted(f.stem for f in (ROOT / "hagrid_120k" / f"train_val_{g}").glob("*.jpg") if f.stem in ann)
        share = sum(m == move for m in GESTURES.values())
        ids = rng.permutation(have)[:args.per_move // share]
        none_here = none_left // (len(GESTURES) - list(GESTURES).index(g))
        for n, i in enumerate(ids):
            a = ann[i]
            img = cv2.imread(str(ROOT / "hagrid_120k" / f"train_val_{g}" / f"{i}.jpg"))
            if img is None or g not in a["labels"]:  # broken file
                continue
            h, w = img.shape[:2]
            box = box_of(a["bboxes"][a["labels"].index(g)], w, h, rng.uniform(1.3, 1.9))
            user = a["user_id"][:12]
            cv2.imwrite(str(OUT / move / f"{user}_{g}_{n}.jpg"), hand.crop(img, box))
            counts[move] += 1
            if n < none_here:
                off = drifted(box, w, h, rng)
                if off is not None:
                    cv2.imwrite(str(OUT / "none" / f"{user}_{g}_{n}.jpg"), hand.crop(img, off))
                    counts["none"] += 1
                    none_left -= 1
        print(g, "->", move, counts)
    print("done", counts)


if __name__ == "__main__":
    main()
