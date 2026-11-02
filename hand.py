import threading
import time
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
BURSTS = DATA / "bursts"  # collect.py
ROUNDS = DATA / "rounds"  # play.py
REAL = DATA / "real"      # crops from real rounds
RESULTS = HERE / "results"
R18 = RESULTS / "classifier_r18.safetensors"
PREDICTOR = RESULTS / "predictor.safetensors"

MOVES = ["rock", "paper", "scissors"]
CLASSES = MOVES + ["none"]
COUNTER = {"rock": "paper", "paper": "scissors", "scissors": "rock"}  # beats

SAVE = 112   # px
PAD = 2.2    # box / landmarks
HOLD = 1.0   # seconds
SMOOTH = 0.5
W = 8        # frames
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # imagenet
STD = np.array([0.229, 0.224, 0.225], np.float32)


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
    # own thread so slow frames dont make us miss the next one
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


def tracker():
    import mediapipe as mp
    return mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                    min_detection_confidence=0.3, min_tracking_confidence=0.3)


def keypoints(hands, frame):
    h, w = frame.shape[:2]
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return np.full((21, 3), np.nan, dtype=np.float32), res
    lm = res.multi_hand_landmarks[0].landmark
    # aspect
    return np.array([[p.x * w / h, p.y, p.z] for p in lm], dtype=np.float32), res


def put(frame, text, y, scale=1.0, color=(255, 255, 255)):
    for c, t in (((0, 0, 0), 6), (color, 2)):
        cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, t, cv2.LINE_AA)


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


# same names as mlx-image
class Block(nn.Module):
    def __init__(self, a: int, b: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(a, b, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm(b)
        self.conv2 = nn.Conv2d(b, b, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm(b)
        if stride != 1 or a != b:
            self.downsample = nn.Sequential(nn.Conv2d(a, b, 1, stride, bias=False), nn.BatchNorm(b))

    def __call__(self, x):
        y = self.bn2(self.conv2(nn.relu(self.bn1(self.conv1(x)))))
        return nn.relu(y + (self.downsample(x) if "downsample" in self else x))


class ResNet18(nn.Module):
    def __init__(self, classes: int = len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        chans = [64, 64, 128, 256, 512]
        for i in range(4):
            a, b = chans[i], chans[i + 1]
            setattr(self, f"layer{i + 1}", nn.Sequential(Block(a, b, 1 if i == 0 else 2), Block(b, b, 1)))
        self.fc = nn.Linear(512, classes)

    def __call__(self, x):
        x = self.maxpool(nn.relu(self.bn1(self.conv1(x))))
        for i in range(1, 5):
            x = self[f"layer{i}"](x)
        return self.fc(x.mean(axis=(1, 2)))


def prep(imgs: np.ndarray) -> mx.array:
    # bgr
    x = imgs[..., ::-1].astype(np.float32) / 255
    return mx.array((x - MEAN) / STD)


class Classifier:
    def __init__(self):
        self.net = ResNet18()
        self.net.load_weights(str(R18))
        self.net.eval()

    def __call__(self, img: np.ndarray) -> tuple[str, float]:
        p = mx.softmax(self.net(prep(img[None])), axis=-1)[0]
        i = int(p.argmax().item())
        return CLASSES[i], float(p[i].item())


def load_classifier() -> Classifier | None:
    return Classifier() if R18.exists() else None


# early guess from keypoints
class Predictor(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W * 42, hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def __call__(self, x):
        return self.net(x)


def load_predictor() -> Predictor | None:
    if not PREDICTOR.exists():
        return None
    model = Predictor()
    model.load_weights(str(PREDICTOR))
    model.eval()
    return model


def normalise(kp: np.ndarray) -> np.ndarray:
    xy = kp[:, :2] - kp[0, :2]
    # head-on
    scale = max(np.linalg.norm(xy[9]), np.linalg.norm(xy[5] - xy[17]))
    return (xy / (scale + 1e-6)).ravel()


def window(rnd: dict, cutoff: float) -> np.ndarray:
    seen = [normalise(k) for k, t in zip(rnd["kp"], rnd["t"]) if t <= cutoff and not np.isnan(k).any()][-W:]
    x = np.zeros((W, 42), dtype=np.float32)
    if seen:
        x[W - len(seen):] = np.stack(seen)
    return x.ravel()
