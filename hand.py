from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

MOVES = ["rock", "paper", "scissors"]
SAVE = 112   # px stored
SIZE = 96    # px model
PAD = 2.2    # box / landmarks
HOLD = 1.0   # seconds
SMOOTH = 0.5
WEIGHTS = Path(__file__).parent / "results" / "classifier.safetensors"


# held when lost
class Boxer:
    def __init__(self):
        self.box, self.seen = None, -1e9

    def update(self, kp: np.ndarray, frame: np.ndarray, now: float):
        h = frame.shape[0]
        if not np.isnan(kp).any():
            xs, ys = kp[:, 0] * h, kp[:, 1] * h
            new = np.array([(xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2,
                            max(np.ptp(xs), np.ptp(ys), 0.12 * h) * PAD])
            self.box = new if self.box is None else SMOOTH * self.box + (1 - SMOOTH) * new
            self.seen = now
        elif now - self.seen > HOLD:
            self.box = None
        return self.box


def crop(frame: np.ndarray, box) -> np.ndarray:
    cx, cy, side = box
    half = side / 2
    src = np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]])
    dst = np.float32([[0, 0], [SAVE, 0], [0, SAVE]])
    return cv2.warpAffine(frame, cv2.getAffineTransform(src, dst), (SAVE, SAVE), borderMode=cv2.BORDER_REPLICATE)


def draw_box(frame: np.ndarray, box, color=(255, 200, 80)):
    cx, cy, side = box
    half = side / 2
    cv2.rectangle(frame, (int(cx - half), int(cy - half)), (int(cx + half), int(cy + half)), color, 2)


def prep(imgs: np.ndarray) -> mx.array:
    return mx.array(imgs.astype(np.float32) / 255 - 0.5)


def center(imgs: np.ndarray) -> np.ndarray:
    o = (SAVE - SIZE) // 2
    return imgs[:, o:o + SIZE, o:o + SIZE]


class HandNet(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [3, 24, 48, 96, 160]
        self.convs = [nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans, chans[1:])]
        self.norms = [nn.BatchNorm(b) for b in chans[1:]]
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(chans[-1], len(MOVES))

    def __call__(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = self.pool(nn.relu(norm(conv(x))))
        return self.head(self.drop(x.mean(axis=(1, 2))))


class Classifier:
    def __init__(self, path: Path = WEIGHTS):
        self.net = HandNet()
        self.net.load_weights(str(path))
        self.net.eval()

    def __call__(self, img: np.ndarray) -> tuple[str, float]:
        p = mx.softmax(self.net(prep(center(img[None]))), axis=-1)[0]
        i = int(p.argmax().item())
        return MOVES[i], float(p[i].item())


def load() -> Classifier | None:
    return Classifier() if WEIGHTS.exists() else None
